#!/usr/bin/env python
# coding: utf-8



import os.path
import argparse
import math
import torch
import torchvision.utils as vutils
from datetime import datetime
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from shapes import Shapes
# from shapes_3d import Shapes3D
from balls import Balls,BallsImage
from dvae import dVAE
from utils import *
# import torch.distributed as dist


parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--clip', type=float, default=1.0)

parser.add_argument('--checkpoint_path', default='/home/t-pbansal/video-dalle/checkpoints/checkpoint.pt.tar')
# parser.add_argument('--data_path', default='/common/home/gs790/idalle/datasets/3dshapes.h5')
parser.add_argument('--data_path', default='/home/t-pbansal/video-dalle/data/bouncing_3balls')
parser.add_argument('--log_path', default='/home/t-pbansal/video-dalle/logs')

parser.add_argument('--lr_start', type=float, default=1e-4)
parser.add_argument('--lr_final', type=float, default=1e-5)
parser.add_argument('--lr_epochs', type=int, default=200)

parser.add_argument('--beta_start', type=float, default=0.0)
parser.add_argument('--beta_final', type=float, default=1.0)
parser.add_argument('--beta_epochs', type=int, default=5)

parser.add_argument('--vocab_size', type=int, default=256)
# parser.add_argument('--vocab_size', type=int, default=64)
parser.add_argument('--img_channels', type=int, default=3)

parser.add_argument('--sigma', type=float, default=0.3)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_epochs', type=int, default=20)

parser.add_argument('--hard', action='store_true')

args = parser.parse_args()


torch.manual_seed(args.seed)


arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)


train_dataset = BallsImage(root=args.data_path,mode='train')
val_dataset = BallsImage(root=args.data_path,mode='val')
# train_dataset = Shapes3D(root=args.data_path, phase='train')
# val_dataset = Shapes3D(root=args.data_path, phase='val')

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset,  **loader_kwargs)
val_loader = DataLoader(val_dataset, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

model = dVAE(args.vocab_size, args.img_channels)

    
if os.path.isfile(args.checkpoint_path):
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

model = model.cuda()

optimizer = Adam(model.parameters(), lr=args.lr_start)
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])



def visualize(image, recon, N=8):
    
    _, _, H, W = image.shape
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon = recon[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    
    return torch.cat((image, recon), dim=1).view(-1, 3, H, W)


for epoch in range(start_epoch, args.epochs):
    model.train()
    
    for batch, image in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch
        
        beta = linear_warmup(global_step,args.beta_start,args.beta_final,0,args.beta_epochs * train_epoch_size)
        tau = cosine_anneal(global_step,args.tau_start,args.tau_final,0,args.tau_epochs * train_epoch_size)
        lr = cosine_anneal(global_step,args.lr_start,args.lr_final,0,args.lr_epochs * train_epoch_size)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        image = image.cuda()
        optimizer.zero_grad()
        (recon, neg_log_likelihood, kl, mse) = model(image, args.sigma, tau, args.hard)
        
        loss = neg_log_likelihood + beta * kl
        
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()
        
        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/neg_log_likelihood', neg_log_likelihood.item(), global_step)
                writer.add_scalar('TRAIN/kl', kl.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                
                writer.add_scalar('TRAIN/beta', beta, global_step)
                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr', lr, global_step)
    
    with torch.no_grad():
        if epoch < 50:
            vis_recon = visualize(image, recon, N=32)
            grid = vutils.make_grid(vis_recon, nrow=8, pad_value=1)[:, 2:-2, 2:-2]
            writer.add_image('TRAIN_recon/epoch={:03}'.format(epoch+1), grid)
    
    with torch.no_grad():
        model.eval()
        
        val_neg_log_likelihood_relax = 0.
        val_kl_relax = 0.
        val_mse_relax = 0.
        
        val_neg_log_likelihood = 0.
        val_kl = 0.
        val_mse = 0.
        
        for batch, image in enumerate(val_loader):
            image = image.cuda()
            
            (recon_relax, neg_log_likelihood_relax, kl_relax, mse_relax) = model(
                image, args.sigma, tau, False)
            
            (recon, neg_log_likelihood, kl, mse) = model(
                image, args.sigma, tau, True)
            
            val_neg_log_likelihood_relax += neg_log_likelihood_relax.item()
            val_kl_relax += kl_relax.item()
            val_mse_relax += mse_relax.item()
            
            val_neg_log_likelihood += neg_log_likelihood.item()
            val_kl += kl.item()
            val_mse += mse.item()
        
        val_neg_log_likelihood_relax /= (val_epoch_size )
        val_kl_relax /= (val_epoch_size )
        val_mse_relax /= (val_epoch_size )
        
        val_neg_log_likelihood /= (val_epoch_size)
        val_kl /= (val_epoch_size)
        val_mse /= (val_epoch_size)
        
        val_loss_relax = val_neg_log_likelihood_relax + args.beta_final * val_kl_relax
        val_loss = val_neg_log_likelihood + args.beta_final * val_kl
        
        writer.add_scalar('VAL/loss_relax', val_loss_relax, epoch+1)
        writer.add_scalar('VAL/neg_log_likelihood_relax', val_neg_log_likelihood_relax, epoch+1)
        writer.add_scalar('VAL/kl_relax', val_kl_relax, epoch+1)
        writer.add_scalar('VAL/mse_relax', val_mse_relax, epoch+1)
        
        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/neg_log_likelihood', val_neg_log_likelihood, epoch+1)
        writer.add_scalar('VAL/kl', val_kl, epoch+1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)
        
        print('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F}'.format(
            epoch+1, val_loss, val_mse))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
            
        if 50 <= epoch:
            vis_recon = visualize(image, recon, N=32)
            grid = vutils.make_grid(vis_recon, nrow=8, pad_value=1)[:, 2:-2, 2:-2]
            writer.add_image('VAL_recon/epoch={:03}'.format(epoch+1), grid)
        
        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)
        
        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))
        
        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
