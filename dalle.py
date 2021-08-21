#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from utils import *
from dvae import dVAE
from slot_attn import SlotAttentionEncoder,SlotAttentionEncoderDecomposed
from transformer import PositionalEncoding,PositionalEncodingStoch, TransformerDecoder
from einops import rearrange

class ConvOneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        :param x: B, C, G, G
        :return:
        """

        tokens = x.permute(0, 2, 3, 1)  # batch_size x G x G x vocab_size
        tokens = torch.argmax(tokens, dim=-1)   # batch_size x G x G
        token_embs = self.dictionary(tokens)    # batch_size x G x G x emb_size
        return token_embs.permute(0, 3, 1, 2)

class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        :param x: B, N, vocab_size
        :return:
        """

        tokens = torch.argmax(x, dim=-1)   # batch_size x N
        token_embs = self.dictionary(tokens)    # batch_size x N x emb_size
        return token_embs

class DALLE(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.num_slots = args.num_slots
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        self.dvae = dVAE(args.vocab_size, args.img_channels)

        self.positional_encoder = PositionalEncoding(1 + (args.image_size // 4) ** 2, args.d_model, args.dropout)
        self.positional_encoder_error = PositionalEncodingStoch(1 + (args.image_size // 4) ** 2, args.d_model, args.dropout)

        self.slot_attn = SlotAttentionEncoderDecomposed(
            args.num_iterations, args.num_slots,
            args.d_model, args.image_size // 4, args.slot_size, args.mlp_hidden_size, args.pos_channels)
        
        self.y_pos_enc = nn.Dropout(args.dropout)
        
        # self.tf_enc = TransformerEncoder(
        #     args.num_enc_blocks, args.slot_size, args.num_heads, args.dropout)

        self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
        self.make_transformer_input_enc = linear(args.slot_size, args.d_model, bias=False)

        self.tf_dec = TransformerDecoder(
            args.num_dec_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_heads, args.dropout)
        
        self.out = linear(args.d_model, args.vocab_size, bias=False)

    
    def forward(self, image, tau, hard, beta_dvae, attns_normalize=True):
        """
        image: batch_size x img_channels x 64 x 64
        """
        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)  # batch_size x vocab_size x 4 x 4
        _, _, H_enc, W_enc = z_logits.size()

        entropy = -(F.softmax(z_logits, dim=1) * z_logits).flatten(start_dim=1).sum(-1).mean() * beta_dvae
        z = gumbel_softmax(z_logits, tau, hard, dim=1)  # batch_size x vocab_size x 4 x 4

        # recon
        recon = self.dvae.decoder(z)

        # hard z
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # batch_size x vocab_size x 4 x 4
        # z_hard = torch.argmax(z_logits, axis=1)
        # z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()  # batch_size x vocab_size x 4 x 4

        # target for transformer
        z_transformer_target = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # batch_size x 16 x vocab_size

        # inputs for transformer with bos
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)  # batch_size x 16 x 1+vocab_size
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2)  # batch_size x 1+16 x 1+vocab_size
        z_transformer_input[:, 0, 0] = 1.0  # set bos # batch_size x 1+16 x vocab_size+1
        z_transformer_input_ = self.dictionary(z_transformer_input)  # batch_size x 16 x d_model
        z_transformer_input = self.positional_encoder(z_transformer_input_)
        z_transformer_input_content = self.positional_encoder_error(z_transformer_input_)
        z_transformer_input_pos = self.positional_encoder(torch.zeros_like(z_transformer_input_))

        # slot attention
        slots_,slots_pos,slots_content, attns = self.slot_attn(z_transformer_input[:, 1:],z_transformer_input_pos[:, 1:],z_transformer_input_content[:, 1:])
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attns` has shape: [batch_size, enc_height * enc_width, num_slots].
        attns = attns.transpose(-1, -2)
        attns = attns.reshape(B, self.num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)
        if attns_normalize:
            # attns = 1. * (H_enc * W_enc) * attns / (attns.sum(-1, keepdim=True).sum(-2, keepdim=True))
            attns = image.unsqueeze(1) * attns + 1. - attns
        else:
            # masked true image
            attns = image.unsqueeze(1) * attns
        # `attns` has shape: [batch_size, num_slots, 1, enc_height, enc_width].

        # transformer
        # encoder_output = self.tf_enc(y_concat)  # batch_size x num_slots x (num_attrs * attr_size)
        # encoder_output = self.tf_enc(slots)  # batch_size x num_slots x slot_size

        recon_transformer,kl_dvae = [],0
        # for slots in [slots_,slots_pos+slots_content]:
        for slots in [slots_]:
            encoder_output = self.make_transformer_input_enc(slots)  # batch_size x num_slots x d_model

            decoder_output = self.tf_dec(z_transformer_input[:, :-1], encoder_output)  # batch_size x 16 x vocab_size

            pred = self.out(decoder_output)   # batch_size x 16 x vocab_size
            cross_entropy = -(z_transformer_target * torch.log_softmax(pred, dim=-1)).flatten(start_dim=1).sum(-1).mean()
            kl_dvae += cross_entropy - entropy

            z_pred = F.one_hot(pred.argmax(dim=-1), self.vocab_size)  # batch_size x 16 x vocab_size
            z_pred = z_pred.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)

            with torch.no_grad():
                recon_transformer.append(self.dvae.decoder(z_pred))

        mse = ((image - recon) ** 2).sum() / B
        
        ret = {
            'recon': recon,
            'recon_transformer_org': recon_transformer[0],
            'recon_transformer': recon_transformer[0],
            'kl_dvae': kl_dvae,
            'mse': mse,
            'attns': attns,
            'z_cross_entropy': cross_entropy,
        }
        return ret

    def generate(self, video, num_generation_steps=10):
        """
        video: batch_size x T x img_channels x 64 x 64
        """

        B, T, C, H, W = video.size()

        # dvae encode
        # BT, D, G, G
        z_logits = F.log_softmax(self.dvae.encoder(video.reshape(B*T, C, H, W)), dim=1)

        # hard z
        # BT, D, G, G
        z_hard = torch.argmax(z_logits, axis=1)
        z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()

        for s in range(num_generation_steps):
            print('gen', s)
            # only trained on sample_length-1, so need that here as well
            z_hard = rearrange(z_hard, '(b t) d g1 g2 -> b t d g1 g2', b=B)
            z_hard = z_hard[:, -(self.sample_length-1):]
            z_hard = rearrange(z_hard, 'b t d g1 g2 -> (b t) d g1 g2')

            z_transformer_input = z_hard.new_zeros(z_hard.shape[0], 1, self.vocab_size + 1)
            z_transformer_input[..., 0] = 1.0  # batch_size x 1 x vocab_size+1

            # slot attention
            # BT, D, G, G -> B, T, D, G, G
            slot_input = rearrange(self.slot_input_maker(z_hard), '(b t) d gh gw -> b t d gh gw', b=B)

            # `slots` has shape: B, T-1, N, D
            slots, _ = self.slot_attn(slot_input)

            slots = rearrange(slots, 'b t n d -> (b t) n d')
            encoder_output = self.make_transformer_input_enc(slots)

            # generate image tokens auto-regressively
            z_gen = z_hard.new_zeros(0)
            for t in range(self.G**2):
                # import pdb; pdb.set_trace()
                decoder_output = self.tf_dec(
                    self.z_pos_enc(self.make_transformer_input_z(z_transformer_input)),
                    encoder_output
                )  # batch_size x ?? x vocab_size
                z_next = F.one_hot(self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)  # batch_size x 1 x vocab_size
                z_gen = torch.cat((z_gen, z_next), dim=1)  # batch_size x ?? x vocab_size
                z_transformer_input = torch.cat([
                    z_transformer_input,
                    torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
                ], dim=1)    # batch_size x ?? x vocab_size + 1

            z_gen = rearrange(z_gen.transpose(1, 2).float(), '(b t) d (g1 g2) -> b t d g1 g2', b=B, g1=self.G)
            z_hard = rearrange(z_hard, '(b t) d g1 g2 -> b t d g1 g2', b=B)
            z_hard = torch.cat([z_hard, z_gen[:, -1:]], dim=1)
            z_hard = rearrange(z_hard, 'b t d g1 g2 -> (b t) d g1 g2')

        gen_video = self.dvae.decoder(z_hard)
        gen_video = rearrange(gen_video, '(b t) c h w -> b t c h w', b=B)

        return gen_video.clamp(0., 1.)

# class DALLE(nn.Module):
    
#     def __init__(self, args):
#         super().__init__()
        
#         self.num_slots = args.num_slots
#         self.vocab_size = args.vocab_size
#         self.d_model = args.d_model

#         self.dvae = dVAE(args.vocab_size, args.img_channels)

#         self.positional_encoder = PositionalEncoding(1 + (args.image_size // 4) ** 2, args.d_model, args.dropout)

#         self.slot_attn = SlotAttentionEncoder(
#             args.num_iterations, args.num_slots,
#             args.d_model, args.image_size // 4, args.slot_size, args.mlp_hidden_size, args.pos_channels)
        
#         self.y_pos_enc = nn.Dropout(args.dropout)
        
#         # self.tf_enc = TransformerEncoder(
#         #     args.num_enc_blocks, args.slot_size, args.num_heads, args.dropout)

#         self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
#         self.make_transformer_input_enc = linear(args.slot_size, args.d_model, bias=False)

#         self.tf_dec = TransformerDecoder(
#             args.num_dec_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_heads, args.dropout)
        
#         self.out = linear(args.d_model, args.vocab_size, bias=False)

    
#     def forward(self, image, tau, hard, beta_dvae, attns_normalize=False):
#         """
#         image: batch_size x img_channels x 64 x 64
#         """
#         B, C, H, W = image.size()

#         # dvae encode
#         z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)  # batch_size x vocab_size x 4 x 4
#         _, _, H_enc, W_enc = z_logits.size()

#         entropy = -(F.softmax(z_logits, dim=1) * z_logits).flatten(start_dim=1).sum(-1).mean() * beta_dvae
#         z = gumbel_softmax(z_logits, tau, hard, dim=1)  # batch_size x vocab_size x 4 x 4

#         # recon
#         recon = self.dvae.decoder(z)

#         # hard z
#         z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # batch_size x vocab_size x 4 x 4
#         # z_hard = torch.argmax(z_logits, axis=1)
#         # z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()  # batch_size x vocab_size x 4 x 4

#         # target for transformer
#         z_transformer_target = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # batch_size x 16 x vocab_size

#         # inputs for transformer with bos
#         z_transformer_input = torch.cat([torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)  # batch_size x 16 x 1+vocab_size
#         z_transformer_input = torch.cat([torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2)  # batch_size x 1+16 x 1+vocab_size
#         z_transformer_input[:, 0, 0] = 1.0  # set bos # batch_size x 1+16 x vocab_size+1
#         z_transformer_input = self.dictionary(z_transformer_input)  # batch_size x 16 x d_model
#         z_transformer_input = self.positional_encoder(z_transformer_input)

#         # slot attention
#         slots, attns = self.slot_attn(z_transformer_input[:, 1:])
#         # `slots` has shape: [batch_size, num_slots, slot_size].
#         # `attns` has shape: [batch_size, enc_height * enc_width, num_slots].
#         attns = attns.transpose(-1, -2)
#         attns = attns.reshape(B, self.num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)
#         if attns_normalize:
#             # attns = 1. * (H_enc * W_enc) * attns / (attns.sum(-1, keepdim=True).sum(-2, keepdim=True))
#             attns = image.unsqueeze(1) * attns + 1. - attns
#         else:
#             # masked true image
#             attns = image.unsqueeze(1) * attns
#         # `attns` has shape: [batch_size, num_slots, 1, enc_height, enc_width].

#         # transformer
#         # encoder_output = self.tf_enc(y_concat)  # batch_size x num_slots x (num_attrs * attr_size)
#         # encoder_output = self.tf_enc(slots)  # batch_size x num_slots x slot_size
#         encoder_output = self.make_transformer_input_enc(slots)  # batch_size x num_slots x d_model

#         decoder_output = self.tf_dec(z_transformer_input[:, :-1], encoder_output)  # batch_size x 16 x vocab_size

#         pred = self.out(decoder_output)   # batch_size x 16 x vocab_size
#         cross_entropy = -(z_transformer_target * torch.log_softmax(pred, dim=-1)).flatten(start_dim=1).sum(-1).mean()
#         kl_dvae = cross_entropy - entropy

#         z_pred = F.one_hot(pred.argmax(dim=-1), self.vocab_size)  # batch_size x 16 x vocab_size
#         z_pred = z_pred.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)

#         with torch.no_grad():
#             recon_transformer = self.dvae.decoder(z_pred)

#         mse = ((image - recon) ** 2).sum() / B
        
#         ret = {
#             'recon': recon,
#             'recon_transformer': recon_transformer,
#             'kl_dvae': kl_dvae,
#             'mse': mse,
#             'attns': attns,
#             'z_cross_entropy': cross_entropy,
#         }
#         return ret

#     # def forward(self, video, tau, hard, beta_dvae, attns_normalize=True):
#     #     """
#     #     video: batch_size x img_channels x 64 x 64
#     #     """
#     #     B, T, C, H, W = video.shape
#     #     video = video.contiguous()

#     #     # dvae encode
#     #     # BT, D, G,  G
#     #     z_logits = F.log_softmax(self.dvae.encoder(video.view(B*T,C,H,W)), dim=1)

#     #     entropy = -(F.softmax(z_logits, dim=1) * z_logits).flatten(start_dim=1).sum(-1).mean() * beta_dvae
#     #     # BT, D, G,  G
#     #     z = gumbel_softmax(z_logits, tau, hard, dim=1)

#     #     # recon
#     #     recon = self.dvae.decoder(z)

#     #     # hard z
#     #     # z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # batch_size x vocab_size x 4 x 4
#     #     z_hard = torch.argmax(z_logits, axis=1)
#     #     # BT, D, G, G
#     #     z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()

#     #     # target for transformer are the codes for each frame
#     #     # move to batch dimension
#     #     # BT, D, G, G -> B(T-1), GG, D
#     #     z_transformer_target = rearrange(z_hard, '(b t) d h w -> b t d h w', b=B)
#     #     z_transformer_target = z_transformer_target[:, 1:]
#     #     z_transformer_target = rearrange(z_transformer_target, 'b t d h w -> (b t) (h w) d')

#     #     # inputs for transformer with bos
#     #     # B(T-1), GG, 1+D
#     #     z_transformer_input = torch.cat([torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)
#     #     # B(T-1), 1+GG, 1+D
#     #     z_transformer_input = torch.cat([torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2)
#     #     # set bos
#     #     z_transformer_input[:, 0, 0] = 1.0
#     #     # B(T-1), GG, D
#     #     z_transformer_input = self.make_transformer_input_z(z_transformer_input)[:, :-1]
#     #     z_transformer_input = self.z_pos_enc(z_transformer_input)

#     #     # slot attention
#     #     # BT, D, G, G -> B, T, D, G, G
#     #     slot_input = rearrange(self.slot_input_maker(z_hard), '(b t) d gh gw -> b t d gh gw', b=B)
#     #     # B, T-1, D, G, G
#     #     slot_input = slot_input[:, :-1]
#     #     # `slots` has shape: B, T-1, N, D
#     #     # `attns` has shape: B, T-1, T-1, GG, N
#     #     # first time is for output image, second is for each timestep it attends to
#     #     slots, attns = self.slot_attn(slot_input)
#     #     attns = rearrange(attns, 'b t1 t2 (gh gw) n -> b t1 t2 n 1 gh gw', gh=self.G)
#     #     attns = attns.repeat_interleave(H // self.G, dim=-2).repeat_interleave(W // self.G, dim=-1)

#     #     # transformer
#     #     # B(T-1), N, D
#     #     slots = rearrange(slots, 'b t n d -> (b t) n d')
#     #     encoder_output = self.make_transformer_input_enc(slots)

#     #     # B(T-1), GG, D
#     #     decoder_output = self.tf_dec(z_transformer_input, encoder_output)

#     #     # B(T-1), GG, D
#     #     pred = self.out(decoder_output)   # batch_size x 16 x vocab_size
#     #     cross_entropy = -(z_transformer_target  * torch.log_softmax(pred, dim=-1)).flatten(start_dim=1).sum(-1).mean()
#     #     kl_dvae = cross_entropy - entropy

#     #     z_pred = F.one_hot(pred.argmax(dim=-1), self.vocab_size)  # batch_size x 16 x vocab_size
#     #     #z_pred = z_pred.transpose(1, 2).float().reshape(B, -1, self.G, self.G)
#     #     z_pred = rearrange(z_pred.float(), '(b t) (h w) d -> (b t) d h w', b=B, h=self.G)

#     #     with torch.no_grad():
#     #         recon_transformer = self.dvae.decoder(z_pred)

#     #     recon = rearrange(recon.clamp(0., 1.), '(b t) c h w -> b t c h w', b=B)
#     #     recon_transformer = rearrange(recon_transformer.clamp(0., 1.), '(b t) c h w -> b t c h w', b=B)
#     #     mse = ((video - recon) ** 2).sum() / B

#     #     ret = {
#     #         'recon': recon,
#     #         'recon_transformer': recon_transformer,
#     #         'kl_dvae': kl_dvae,
#     #         'mse': mse,
#     #         'attns': attns,
#     #         'z_cross_entropy': cross_entropy,
#     #     }

#     #     return ret

#     def generate(self, video, num_generation_steps=10):
#         """
#         video: batch_size x T x img_channels x 64 x 64
#         """

#         B, T, C, H, W = video.size()

#         # dvae encode
#         # BT, D, G, G
#         z_logits = F.log_softmax(self.dvae.encoder(video.reshape(B*T, C, H, W)), dim=1)

#         # hard z
#         # BT, D, G, G
#         z_hard = torch.argmax(z_logits, axis=1)
#         z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()

#         for s in range(num_generation_steps):
#             print('gen', s)
#             # only trained on sample_length-1, so need that here as well
#             z_hard = rearrange(z_hard, '(b t) d g1 g2 -> b t d g1 g2', b=B)
#             z_hard = z_hard[:, -(self.sample_length-1):]
#             z_hard = rearrange(z_hard, 'b t d g1 g2 -> (b t) d g1 g2')

#             z_transformer_input = z_hard.new_zeros(z_hard.shape[0], 1, self.vocab_size + 1)
#             z_transformer_input[..., 0] = 1.0  # batch_size x 1 x vocab_size+1

#             # slot attention
#             # BT, D, G, G -> B, T, D, G, G
#             slot_input = rearrange(self.slot_input_maker(z_hard), '(b t) d gh gw -> b t d gh gw', b=B)

#             # `slots` has shape: B, T-1, N, D
#             slots, _ = self.slot_attn(slot_input)

#             slots = rearrange(slots, 'b t n d -> (b t) n d')
#             encoder_output = self.make_transformer_input_enc(slots)

#             # generate image tokens auto-regressively
#             z_gen = z_hard.new_zeros(0)
#             for t in range(self.G**2):
#                 # import pdb; pdb.set_trace()
#                 decoder_output = self.tf_dec(
#                     self.z_pos_enc(self.make_transformer_input_z(z_transformer_input)),
#                     encoder_output
#                 )  # batch_size x ?? x vocab_size
#                 z_next = F.one_hot(self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)  # batch_size x 1 x vocab_size
#                 z_gen = torch.cat((z_gen, z_next), dim=1)  # batch_size x ?? x vocab_size
#                 z_transformer_input = torch.cat([
#                     z_transformer_input,
#                     torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
#                 ], dim=1)    # batch_size x ?? x vocab_size + 1

#             z_gen = rearrange(z_gen.transpose(1, 2).float(), '(b t) d (g1 g2) -> b t d g1 g2', b=B, g1=self.G)
#             z_hard = rearrange(z_hard, '(b t) d g1 g2 -> b t d g1 g2', b=B)
#             z_hard = torch.cat([z_hard, z_gen[:, -1:]], dim=1)
#             z_hard = rearrange(z_hard, 'b t d g1 g2 -> (b t) d g1 g2')

#         gen_video = self.dvae.decoder(z_hard)
#         gen_video = rearrange(gen_video, '(b t) c h w -> b t c h w', b=B)

#         return gen_video.clamp(0., 1.)

