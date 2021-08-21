#!/usr/bin/env python
# coding: utf-8


import os
import os.path
import argparse
import math
import torch
import torchvision.utils as vutils
from datetime import datetime
from einops import rearrange
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from balls import BallsImage
from dalle import DALLE
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--image_size', type=int, default=64)

parser.add_argument('--checkpoint_path', default='')
# parser.add_argument('--pretrained_dvae_path', default='/home/t-pbansal/video-dalle/checkpoints/dvae_model.pt')
parser.add_argument('--pretrained_dvae_path', default='/home/t-pbansal/video-dalle/checkpoints/best_model.pt')
parser.add_argument('--data_path', default='/home/t-pbansal/video-dalle/data/bouncing_3balls')
parser.add_argument('--log_path', default='/home/t-pbansal/video-dalle/logs')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--dvae_only_epochs', type=int, default=3)

parser.add_argument('--lr_slot', type=float, default=3e-5)
parser.add_argument('--lr_remaining', type=float, default=3e-5)
# parser.add_argument('--lr_remaining', type=float, default=3e-4)
parser.add_argument('--lr_warmup', type=int, default=2)


parser.add_argument('--beta_start', type=float, default=0.00)
parser.add_argument('--beta_final', type=float, default=0.00)
parser.add_argument('--beta_epochs', type=int, default=40)

parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=256)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

# parser.add_argument('--num_iterations', type=int, default=7)
parser.add_argument('--num_iterations', type=int, default=5)
parser.add_argument('--num_slots', type=int, default=4)
# parser.add_argument('--num_slots', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=32)
parser.add_argument('--slot_size', type=int, default=32)
# parser.add_argument('--slot_size', type=int, default=64)
parser.add_argument('--mlp_hidden_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)
parser.add_argument('--sample_length', type=int, default=20)
parser.add_argument('--num_gt_steps', type=int, default=10)
parser.add_argument('--num_generation_steps', type=int, default=10)
parser.add_argument('--num_vis', type=int, default=8)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_epochs', type=int, default=3)

parser.add_argument('--hard', action='store_true')
parser.add_argument('--sample_batch', action='store_true')

parser.add_argument('--memory_type', type=str, choices=['slots', 'conv', 'vector'], default='slots')

parser.add_argument('--fast_run', action='store_true', default=False)

args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)


train_dataset = BallsImage(root=args.data_path, mode='train')
val_dataset = BallsImage(root=args.data_path, mode='val')

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
    'num_workers':4,
}

train_loader = DataLoader(train_dataset, **loader_kwargs)
val_loader = DataLoader(val_dataset,  **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5


model = DALLE(args)

if os.path.isfile(args.checkpoint_path):
    print(f'Loading from checkpoint {args.checkpoint_path}')
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

if os.path.isfile(args.pretrained_dvae_path):
    print(f'Loading pretrained dvae {args.pretrained_dvae_path}')
    dvae_checkpoint = torch.load(args.pretrained_dvae_path, map_location='cpu')
    model.dvae.load_state_dict(dvae_checkpoint)

model = model.cuda()

# Divide params for different learning rates.
dvae_params = (x[1] for x in model.named_parameters() if 'dvae' in x[0])
slot_params = (x[1] for x in model.named_parameters() if 'slot_attn' in x[0])
remaining_params = (x[1] for x in model.named_parameters() if 'dvae' not in x[0] and 'slot_attn' not in x[0])

optimizer_dvae = Adam(dvae_params, lr=args.lr_dvae)
optimizer_slot = Adam(slot_params, lr=args.lr_slot)
optimizer_remaining = Adam(remaining_params, lr=args.lr_remaining, betas=(0.9, 0.95))
if checkpoint is not None:
    optimizer_dvae.load_state_dict(checkpoint['optimizer_dvae'])
    optimizer_slot.load_state_dict(checkpoint['optimizer_slot'])
    optimizer_remaining.load_state_dict(checkpoint['optimizer_remaining'])


def visualize(video, recon, recon_transformer=None, N=8):
    B, C, H, W = video.shape
    video = video[:N].unsqueeze(1)
    recon = recon[:N].unsqueeze(1)
    recon_diff = video - recon
    vis_list = [video, recon, recon_diff]
    if recon_transformer is not None:
        recon_transformer = recon_transformer[:N]
        # recon_transformer = torch.cat([torch.zeros(N, 1, C, H, W, device=recon_transformer.device), recon_transformer], dim=1)
        recon_transformer = recon_transformer.unsqueeze(1)
        recon_transformer_diff = video - recon_transformer
        vis_list.extend([recon_transformer, recon_transformer_diff])
    vis = torch.cat(vis_list, dim=1).view(-1, C, H, W)
    return vis

def visualize_attn(video, attn, N=8):
    # [[predicted frame, attended frame 1, slot1, slot2, etc...],
    #  [predicted frame, attended frame 2, slot1, slot2, etc...]
    #  ...]
    B, C, H, W = video.shape
    predicted = video[:N][ :, None, ...]
    attn = attn[:N]
    #attn = attn * attended
    # attn = attn * predicted + 1 - attn
    # attn = 1-attn(1-predicted)
    vis = torch.cat([predicted, predicted, attn], dim=-4)
    vis = vis.view(-1, C, H, W)
    return vis


# def visualize_attn(video, attn, N=2):
#     # [[predicted frame, attended frame 1, slot1, slot2, etc...],
#     #  [predicted frame, attended frame 2, slot1, slot2, etc...]
#     #  ...]
#     B, T, C, H, W = video.shape
#     predicted = video[:N, 1:][:, :, None, None, ...].repeat_interleave(T-1, dim=2)
#     attended = video[:N, :-1][:, None, :, None, ...].repeat_interleave(T-1, dim=1)
#     attn = attn[:N].expand(-1, -1, -1, -1, 3, H, W)
#     #attn = attn * attended
#     attn = attn * attended + 1 - attn
#     vis = torch.cat([predicted, attended, attn], dim=-4)
#     vis = vis.view(-1, C, H, W)
#     return vis


# def visualize_gen(video, recon, N=8):
#     _, _, H, W = image.shape
#     image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
#     recon = recon[:N].expand(-1, 3, H, W).unsqueeze(dim=1)

#     return torch.cat((image, recon), dim=1).view(-1, 3, H, W)

stagnation_counter = 0
lr_decay_factor = 1.0
for epoch in range(start_epoch, args.epochs):
    for batch, full_video in enumerate(train_loader):
        model.train()
        global_step = epoch * train_epoch_size + batch

        full_video = full_video.cuda(non_blocking=True)
        # just sample_length for now
        # video = full_video[:, :args.sample_length]
        video = full_video

        beta = linear_warmup(global_step,args.beta_start,args.beta_final,0,args.beta_epochs * train_epoch_size)
        tau = cosine_anneal(global_step,args.tau_start,args.tau_final,0,args.tau_epochs * train_epoch_size)

        if epoch < args.dvae_only_epochs:
            lr_remaining = 0.0
            lr_slot = 0.0
        if epoch < args.lr_warmup:
            lr_remaining = linear_warmup(max(global_step, 0),0.,args.lr_remaining,0,(args.lr_warmup) * train_epoch_size)
            lr_slot = linear_warmup(max(global_step, 0),0.,args.lr_slot,0,(args.lr_warmup) * train_epoch_size)
        else:
            lr_remaining = args.lr_remaining
            lr_slot = args.lr_slot

        for param_group in optimizer_remaining.param_groups:
            param_group['lr'] = lr_remaining
        for param_group in optimizer_slot.param_groups:
            param_group['lr'] = lr_slot

        optimizer_dvae.zero_grad()
        optimizer_slot.zero_grad()
        optimizer_remaining.zero_grad()

        forward_res = model(video, tau, args.hard, beta)
        recon = forward_res['recon']
        recon_transformer = forward_res['recon_transformer']
        kl_dvae = forward_res['kl_dvae']
        mse = forward_res['mse']
        attns = forward_res['attns']

        loss = mse + kl_dvae

        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer_dvae.step()
        optimizer_slot.step()
        optimizer_remaining.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))

                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/kl_dvae', kl_dvae.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)

                writer.add_scalar('TRAIN/beta', beta, global_step)
                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_slot', lr_slot, global_step)
                writer.add_scalar('TRAIN/lr_remaining', lr_remaining, global_step)
        if args.fast_run:
            break

    with torch.no_grad():
        vis = visualize(video, recon, recon_transformer, N=args.num_vis)
        vis_grid = vutils.make_grid(vis, nrow=video.shape[1], pad_value=1)
        writer.add_image('TRAIN_vis', vis_grid, global_step)

        attn_vis = visualize_attn(video, attns)
        attn_grid = vutils.make_grid(attn_vis, nrow=args.num_slots+2, pad_value=1)
        writer.add_image('TRAIN_attn', attn_grid, global_step)

    with torch.no_grad():
        model.eval()
        val_kl_dvae_relax,val_mse_relax,val_kl_dvae ,val_mse= 0,0,0,0

        for val_batch, full_video in enumerate(val_loader):
            full_video = full_video.cuda(non_blocking=True)

            # just sample_length for now
            # video = full_video[:, :args.sample_length]
            video = full_video

            forward_res_relax = model(video, tau, False, beta)
            recon_relax = forward_res['recon']
            recon_transformer_relax = forward_res['recon_transformer']
            kl_dvae_relax = forward_res['kl_dvae']
            mse_relax = forward_res['mse']
            attns_relax = forward_res['attns']

            forward_res = model(video, tau, True, beta)
            recon = forward_res['recon']
            recon_transformer = forward_res['recon_transformer']
            kl_dvae = forward_res['kl_dvae']
            mse = forward_res['mse']
            attns = forward_res['attns']

            val_kl_dvae_relax += kl_dvae_relax.item()
            val_mse_relax += mse_relax.item()

            val_kl_dvae += kl_dvae.item()
            val_mse += mse.item()

        val_kl_dvae_relax /= (val_epoch_size)
        val_mse_relax /= (val_epoch_size)

        val_kl_dvae /= (val_epoch_size)
        val_mse /= (val_epoch_size)

        val_loss_relax = val_mse_relax + val_kl_dvae_relax
        val_loss = val_mse + val_kl_dvae

        writer.add_scalar('VAL/loss_relax', val_loss_relax, global_step)
        writer.add_scalar('VAL/kl_dvae_relax', val_kl_dvae_relax, global_step)
        writer.add_scalar('VAL/mse_relax', val_mse_relax, global_step)
        writer.add_scalar('VAL/loss', val_loss, global_step)
        writer.add_scalar('VAL/kl_dvae', val_kl_dvae, global_step)
        writer.add_scalar('VAL/mse', val_mse, global_step)

        print('====> Epoch: {:3} [{:5}/{:5}] \t Loss = {:F} \t MSE = {:F}'.format(
            epoch+1, batch, train_epoch_size, val_loss, val_mse))

        vis = visualize(video, recon, recon_transformer, N=args.num_vis)
        vis_grid = vutils.make_grid(vis, nrow=8, pad_value=1)
        writer.add_image('VAL_vis', vis_grid, global_step)

        attn_vis = visualize_attn(video, attns)
        attn_grid = vutils.make_grid(attn_vis, nrow=args.num_slots+2, pad_value=1)
        writer.add_image('VAL_attn', attn_grid, global_step)

        # gen_video = model.generate(full_video[:, :args.num_gt_steps], num_generation_steps=args.num_generation_steps)
        # gt_video = full_video[:, :args.num_gt_steps+args.num_generation_steps]
        # gen_vis = visualize(gt_video, gen_video, N=args.num_vis)
        # gen_vis_grid = vutils.make_grid(gen_vis, nrow=gt_video.shape[1], pad_value=1)
        # writer.add_image('VAL_gen', gen_vis_grid, global_step)

        # gen_gif = torch.cat([full_video[:8, :args.num_gt_steps+args.num_generation_steps], gen_video[:8]], dim=-1)
        # writer.add_video('VAL_gen_vid', gen_gif, global_step)

        if val_loss < best_val_loss:
            stagnation_counter = 0
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

        else:
            stagnation_counter += 1
            if stagnation_counter >= 4:
                lr_decay_factor = lr_decay_factor / 2.0
                stagnation_counter = 0


        writer.add_scalar('VAL/best_loss', best_val_loss, global_step)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.state_dict(),
            'optimizer_dvae': optimizer_dvae.state_dict(),
            'optimizer_slot': optimizer_slot.state_dict(),
            'optimizer_remaining': optimizer_remaining.state_dict(),
            'lr_decay_factor': lr_decay_factor,
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {} [{:5}/{:5}]'.format(best_val_loss, best_epoch, batch, train_epoch_size))


writer.close()
