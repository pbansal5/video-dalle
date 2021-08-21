import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from utils import *
from dvae import dVAE
# from hungarian_algorithm import Hungarian
import numpy as np
from slot_attn import SlotAttentionEncoder
from transformer import PositionalEncoding, TransformerEncoder, TransformerDecoder

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

        self.warmup = 3
        self.positional_encoder = PositionalEncoding(1 + (args.image_size // 4) ** 2, args.d_model, args.dropout)
        self.positional_encoder_slots = PositionalEncoding(self.warmup, args.num_slots*args.d_model, args.dropout)

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.image_size // 4, args.slot_size, args.mlp_hidden_size, args.pos_channels)
        
        self.y_pos_enc = nn.Dropout(args.dropout)
        
        self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
        self.make_transformer_input_enc = linear(args.slot_size, args.d_model, bias=False)

        self.tf_dec = TransformerDecoder(
            args.num_dec_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_heads, args.dropout)
        
        # self.tf_enc = TransformerEncoder(
        #     args.num_dec_blocks, args.d_model, args.num_heads, args.dropout)
        
        self.out = linear(args.d_model, args.vocab_size, bias=False)

    def forward(self, image, tau, hard, beta_dvae, attns_normalize=True):
        """
        image: batch_size x img_channels x 64 x 64
        """
        B,T, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image.flatten(end_dim=1)), dim=1)  # batch_size x vocab_size x 4 x 4
        _, _, H_enc, W_enc = z_logits.size()

        entropy = -(F.softmax(z_logits, dim=1) * z_logits).flatten(start_dim=1).sum(-1).mean() * beta_dvae
        z = gumbel_softmax(z_logits, tau, hard, dim=1)  # batch_size x vocab_size x 4 x 4

        # recon
        recon = self.dvae.decoder(z)
        recon = recon.view(B,T,C,H,W)

        # hard z
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # batch_size x vocab_size x 4 x 4
        # z_hard = torch.argmax(z_logits, axis=1)
        # z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()  # batch_size x vocab_size x 4 x 4
        z_hard = z_hard.view((B,T)+z_hard.shape[1:])

        # target for transformer
        z_transformer_target = z_hard.permute(0, 1, 3, 4, 2).flatten(start_dim=2, end_dim=3)  # batch_size x T x 16 x vocab_size

        # inputs for transformer with bos
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)  # batch_size x T x 16 x 1+vocab_size
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2)  # batch_size x T x 1+16 x 1+vocab_size
        z_transformer_input[:,:, 0, 0] = 1.0  # set bos # batch_size x T x 1+16 x vocab_size+1
        z_transformer_input = self.dictionary(z_transformer_input)  # batch_size x T x 1+16 x d_model
        z_transformer_input = self.positional_encoder(z_transformer_input.flatten(end_dim=1)).view(z_transformer_input.shape)

        # slot attention
        slots, attns = self.slot_attn(z_transformer_input.flatten(end_dim=1)[:, 1:])
        slots = slots.view((B,T)+slots.shape[1:])
        attns = attns.view((B,T)+attns.shape[1:])
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attns` has shape: [batch_size, enc_height * enc_width, num_slots].
        attns = attns.transpose(-1, -2)
        attns = attns.reshape(B, T, self.num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)
        if attns_normalize:
            # attns = 1. * (H_enc * W_enc) * attns / (attns.sum(-1, keepdim=True).sum(-2, keepdim=True))
            attns = image.unsqueeze(2) * attns + 1. - attns
        else:
            # masked true image
            attns = image.unsqueeze(2) * attns
        # `attns` has shape: [batch_size, num_slots, 1, enc_height, enc_width].

        # transformer
        # encoder_output = self.tf_enc(y_concat)  # batch_size x num_slots x (num_attrs * attr_size)
        # encoder_output = self.tf_enc(slots)  # batch_size x num_slots x slot_size
        encoder_output = self.make_transformer_input_enc(slots.flatten(end_dim=1))  # batch_size*T x num_slots x d_model
        encoder_output = encoder_output.view((B,T)+encoder_output.shape[1:]).unsqueeze(2) # batch_size x T x 1 x num_slots x d_model
        len_ = T-(self.warmup)
        encoder_output = torch.cat([encoder_output[:,i:i+len_] for i in range(self.warmup)],dim=2) # batch_size x T-warmup x warmup x num_slots x d_model
        
        # self.warmup:self.warmup+T-1
        z_transformer_input = z_transformer_input[:,self.warmup:self.warmup+len_].flatten(end_dim=1)  # batch_size*(T-warmup) x 1+16 x d_model
        z_transformer_target = z_transformer_target[:,self.warmup:self.warmup+len_].flatten(end_dim=1)  # batch_size*(T-warmup) x 16 x vocab_size
        
        encoder_output = encoder_output.flatten(end_dim=1) #  batch_size*(T-warmup) x warmup x num_slots x d_model
        encoder_output = self.positional_encoder_slots(encoder_output.flatten(start_dim=2)).view(encoder_output.shape)
        encoder_output = encoder_output.flatten(start_dim=1,end_dim=2) #  batch_size*(T-warmup) x warmup*num_slots x d_model

        # encoder_output = self.tf_enc(encoder_output)
        # z_transformer_input, z_transformer_target, encoder_output, recon_transformer
        decoder_output = self.tf_dec(z_transformer_input[:, :-1], encoder_output)  # batch_size*(T-warmup) x 16 x vocab_size

        pred = self.out(decoder_output)   # batch_size*(T-warmup) x 16 x vocab_size
        cross_entropy = -(z_transformer_target * torch.log_softmax(pred, dim=-1)).flatten(start_dim=1).sum(-1).mean()
        kl_dvae = cross_entropy - entropy

        z_pred = F.one_hot(pred.argmax(dim=-1), self.vocab_size)  # batch_size*(T-warmup) x 16 x vocab_size
        z_pred = z_pred.transpose(1, 2).float().reshape(B*(T-self.warmup), -1, H_enc, W_enc)

        with torch.no_grad():
            recon_transformer = self.dvae.decoder(z_pred)

        mse = ((image - recon) ** 2).sum() / (B*T)
        
        forward_res = {'recon' : recon.view(B,T,C,H,W).clamp(0., 1.),
        'recon_transformer': recon_transformer.view(B,T-self.warmup,C,H,W).clamp(0., 1.),
        'kl_dvae' : kl_dvae, 'mse': mse, 'attns' : attns.view(B,T,self.num_slots,C,H,W)}

        return forward_res

    def generate(self, image):
        """
        image: batch_size x img_channels x 64 x 64
        """

        gen_len = (image.size(-1) // 4) ** 2

        B,T, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image.flatten(end_dim=1)), dim=1)  # batch_size x vocab_size x 4 x 4
        _, _, H_enc, W_enc = z_logits.size()

        # hard z
        z_hard = torch.argmax(z_logits, axis=1)
        z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()  # batch_size x vocab_size x 4 x 4

        # target for transformer
        one_hot_tokens = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # batch_size x 16 x vocab_size
        
        # inputs for transformer with bos
        token_embeddings = torch.cat([torch.zeros_like(one_hot_tokens[..., :1]), one_hot_tokens], dim=-1)  # batch_size x 16 x 1+vocab_size
        token_embeddings = torch.cat([torch.zeros_like(token_embeddings[..., :1, :]), token_embeddings], dim=-2)  # batch_size x 1+16 x 1+vocab_size
        token_embeddings[:, 0, 0] = 1.0  # set bos # batch_size x 1+16 x vocab_size+1
        token_embeddings = self.dictionary(token_embeddings)  # batch_size x 16 x d_model
        token_embeddings = self.positional_encoder(token_embeddings)

        z_transformer_input = z_hard.new_zeros(B*(T-self.warmup), 1, self.vocab_size + 1)
        z_transformer_input[..., 0] = 1.0  # batch_size x 1 x vocab_size+1

        # slot attention
        slots, attns = self.slot_attn(token_embeddings[:, 1:])
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attns` has shape: [batch_size, enc_height * enc_width, num_slots].
        attns = attns.transpose(-1, -2)
        attns = attns.reshape(B*T, self.num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)
        # `attns` has shape: [batch_size, num_slots, 1, enc_height, enc_width].
        # attns = image.unsqueeze(1) * attns
        attns = image.flatten(end_dim=1).unsqueeze(1) * attns + 1. - attns

        # transformer
        # encoder_output = self.tf_enc(slots)  # batch_size x num_slots x slot_size
        # encoder_output = self.make_transformer_input_enc(slots)  # batch_size x num_slots x d_model

        encoder_output = self.make_transformer_input_enc(slots)  # batch_size*T x num_slots x d_model
        encoder_output = encoder_output.view((B,T)+encoder_output.shape[1:]).unsqueeze(2) # batch_size x T x 1 x num_slots x d_model
        len_ = T-(self.warmup)
        encoder_output = torch.cat([encoder_output[:,i:i+len_] for i in range(self.warmup)],dim=2) # batch_size x T-warmup x warmup x num_slots x d_model
        
        encoder_output = encoder_output.flatten(end_dim=1) #  batch_size*(T-warmup) x warmup x num_slots x d_model
        encoder_output = self.positional_encoder_slots(encoder_output.flatten(start_dim=2)).view(encoder_output.shape)
        encoder_output = encoder_output.flatten(start_dim=1,end_dim=2) #  batch_size*(T-warmup) x warmup*num_slots x d_model

        # encoder_output = self.tf_enc(encoder_output)

        # generate image tokens auto-regressively
        z_gen = z_hard.new_zeros(0)
        for t in range(gen_len):
            # import pdb; pdb.set_trace()
            decoder_output = self.tf_dec(
                self.positional_encoder(self.dictionary(z_transformer_input)),
                encoder_output
            )  # batch_size x ?? x vocab_size
            z_next = F.one_hot(self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)  # batch_size x 1 x vocab_size
            z_gen = torch.cat((z_gen, z_next), dim=1)  # batch_size x ?? x vocab_size
            z_transformer_input = torch.cat([
                z_transformer_input,
                torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
            ], dim=1)    # batch_size x ?? x vocab_size + 1

        z_gen = z_gen.transpose(1, 2).float().reshape(B*(T-self.warmup), -1, H_enc, W_enc)
        recon_transformer = self.dvae.decoder(z_gen)
        recon_transformer = recon_transformer.view(B,T-self.warmup,C,H,W)

        return recon_transformer.clamp(0., 1.)

    def generate_combine(self, image):
        """
        image: batch_size x img_channels x 64 x 64
        """

        gen_len = (image.size(-1) // 4) ** 2

        B,T, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image.flatten(end_dim=1)), dim=1)  # batch_size x vocab_size x 4 x 4
        _, _, H_enc, W_enc = z_logits.size()

        # hard z
        z_hard = torch.argmax(z_logits, axis=1)
        z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()  # batch_size x vocab_size x 4 x 4

        # target for transformer
        one_hot_tokens = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # batch_size x 16 x vocab_size
        
        # inputs for transformer with bos
        token_embeddings = torch.cat([torch.zeros_like(one_hot_tokens[..., :1]), one_hot_tokens], dim=-1)  # batch_size x 16 x 1+vocab_size
        token_embeddings = torch.cat([torch.zeros_like(token_embeddings[..., :1, :]), token_embeddings], dim=-2)  # batch_size x 1+16 x 1+vocab_size
        token_embeddings[:, 0, 0] = 1.0  # set bos # batch_size x 1+16 x vocab_size+1
        token_embeddings = self.dictionary(token_embeddings)  # batch_size x 16 x d_model
        token_embeddings = self.positional_encoder(token_embeddings)

        z_transformer_input = z_hard.new_zeros(B*(T-self.warmup), 1, self.vocab_size + 1)
        z_transformer_input[..., 0] = 1.0  # batch_size x 1 x vocab_size+1

        # slot attention
        slots, attns = self.slot_attn(token_embeddings[:, 1:])
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attns` has shape: [batch_size, enc_height * enc_width, num_slots].
        attns = attns.transpose(-1, -2)
        attns = attns.reshape(B*T, self.num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)
        # `attns` has shape: [batch_size, num_slots, 1, enc_height, enc_width].
        # attns = image.unsqueeze(1) * attns
        attns = image.flatten(end_dim=1).unsqueeze(1) * attns + 1. - attns

        # transformer
        # encoder_output = self.tf_enc(slots)  # batch_size x num_slots x slot_size
        # encoder_output = self.make_transformer_input_enc(slots)  # batch_size x num_slots x d_model

        encoder_output = self.make_transformer_input_enc(slots)  # batch_size*T x num_slots x d_model
        encoder_output = encoder_output.view((B,T)+encoder_output.shape[1:]).unsqueeze(2) # batch_size x T x 1 x num_slots x d_model
        len_ = T-(self.warmup)
        encoder_output = torch.cat([encoder_output[:,i:i+len_] for i in range(self.warmup)],dim=2) # batch_size x T-warmup x warmup x num_slots x d_model
        
        encoder_output = encoder_output.flatten(end_dim=1) #  batch_size*(T-warmup) x warmup x num_slots x d_model

        encoder_output = self.positional_encoder_slots(encoder_output.flatten(start_dim=2)).view(encoder_output.shape) #  batch_size*(T-warmup) x warmup x num_slots x d_model
        encoder_output = torch.cat([encoder_output,torch.cat([encoder_output[-len_:],encoder_output[:-len_]],dim=0)],dim=2) #  batch_size*(T-warmup) x warmup x 2*num_slots x d_model
        encoder_output = encoder_output.flatten(start_dim=1,end_dim=2) #  batch_size*(T-warmup) x 2*warmup*num_slots x d_model

        # encoder_output = self.tf_enc(encoder_output)

        # generate image tokens auto-regressively
        z_gen = z_hard.new_zeros(0)
        for t in range(gen_len):
            # import pdb; pdb.set_trace()
            decoder_output = self.tf_dec(
                self.positional_encoder(self.dictionary(z_transformer_input)),
                encoder_output
            )  # batch_size x ?? x vocab_size
            z_next = F.one_hot(self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)  # batch_size x 1 x vocab_size
            z_gen = torch.cat((z_gen, z_next), dim=1)  # batch_size x ?? x vocab_size
            z_transformer_input = torch.cat([
                z_transformer_input,
                torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
            ], dim=1)    # batch_size x ?? x vocab_size + 1

        z_gen = z_gen.transpose(1, 2).float().reshape(B*(T-self.warmup), -1, H_enc, W_enc)
        recon_transformer = self.dvae.decoder(z_gen)
        recon_transformer = recon_transformer.view(B,T-self.warmup,C,H,W)

        return recon_transformer.clamp(0., 1.)

    # def swap_slots(self, image, n_swap=2):
    #     """
    #     image: batch_size x img_channels x 64 x 64
    #     """
    #     # take first two images.
    #     gen_len = (image.size(-1) // 4) ** 2
    #     image = image[:2]
    #     B, C, H, W = image.size()

    #     # dvae encode
    #     z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)  # batch_size x vocab_size x 4 x 4
    #     _, _, H_enc, W_enc = z_logits.size()

    #     # hard z
    #     z_hard = torch.argmax(z_logits, axis=1)
    #     z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()  # batch_size x vocab_size x 4 x 4

    #     z_transformer_input = z_hard.new_zeros(B, 1, self.vocab_size + 1)
    #     z_transformer_input[..., 0] = 1.0  # batch_size x 1 x vocab_size+1

    #     # slot attention
    #     slots, attns = self.slot_attn(self.slot_input_maker(z_hard))
    #     # `slots` has shape: [batch_size, num_slots, slot_size].
    #     # `attns` has shape: [batch_size, enc_height * enc_width, num_slots].
    #     attns = attns.transpose(-1, -2)
    #     attns = attns.reshape(B, self.num_slots, 1, H_enc, W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)

    #     attns_first = attns[0]
    #     attns_second = attns[1]

    #     region_mask = torch.zeros_like(attns_first)
    #     region_mask[:, :, int(0.6 * H):] = 1.0

    #     # find which slots have large attention regions
    #     important_slots_first = torch.argsort((attns_first * region_mask).flatten(start_dim=1).sum(-1), dim=0, descending=True)


    #     # important_slots_second = torch.argsort((attns_second * region_mask).flatten(start_dim=1).sum(-1), dim=0, descending=True)


    #     # apply hungarian to match swappable slots
    #     profit_matrix = (attns_first.unsqueeze(1) * attns_second.unsqueeze(0)).flatten(start_dim=2).sum(-1)  # (num_slots, num_slots)
    #     profit_matrix = profit_matrix.cpu().numpy()

    #     hungarian = Hungarian(np.asarray(profit_matrix), is_profit_matrix=True)
    #     hungarian.calculate()
    #     mappings = dict(hungarian.get_results())

    #     attns = image.unsqueeze(1) * attns

    #     for i_swap in range(1, 1 + n_swap):
    #         slot_idx_first = important_slots_first[i_swap].item()
    #         slot_idx_second = mappings[slot_idx_first]

    #         # swap
    #         temp = slots[0, slot_idx_first].clone()
    #         slots[0, slot_idx_first] = slots[1, slot_idx_second].clone()
    #         slots[1, slot_idx_second] = temp.clone()

    #         # show who got swapped
    #         attns[0, slot_idx_first, :, :2, :2] = image.new_tensor([1, 0, 0])[:, None, None]
    #         attns[1, slot_idx_second, :, :2, :2] = image.new_tensor([1, 0, 0])[:, None, None]

    #     # `attns` has shape: [batch_size, num_slots, 1, enc_height, enc_width].

    #     # transformer
    #     # encoder_output = self.tf_enc(slots)  # batch_size x num_slots x slot_size
    #     encoder_output = self.make_transformer_input_enc(slots)  # batch_size x num_slots x d_model

    #     # generate image tokens auto-regressively
    #     z_gen = z_hard.new_zeros(0)
    #     for t in range(gen_len):
    #         # import pdb; pdb.set_trace()
    #         decoder_output = self.tf_dec(
    #             self.positional_encoder(self.dictionary(z_transformer_input)),
    #             encoder_output
    #         )  # batch_size x ?? x vocab_size
    #         z_next = F.one_hot(self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)  # batch_size x 1 x vocab_size
    #         z_gen = torch.cat((z_gen, z_next), dim=1)  # batch_size x ?? x vocab_size
    #         z_transformer_input = torch.cat([
    #             z_transformer_input,
    #             torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
    #         ], dim=1)    # batch_size x ?? x vocab_size + 1

    #     z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
    #     recon_transformer = self.dvae.decoder(z_gen)

    #     return recon_transformer.clamp(0., 1.), attns
