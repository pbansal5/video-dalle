#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from shapes import Shapes, build_grid

from shapes_3d import Shapes3D
from dalle import DALLE

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--clip', type=float, default=1.0)

parser.add_argument('--checkpoint_path', default='/data/local/gs790/idalle/outputs/slot-joint-000/2021-05-01T18:39:05.552958/best_model.pt')
parser.add_argument('--log_path', default='/data/local/gs790/idalle/outputs/slot-joint-000')
parser.add_argument('--data_path', default='/common/home/gs790/idalle/datasets/3dshapes.h5')
parser.add_argument('--paste_path', default='/common/home/gs790/idalle/paste/')

parser.add_argument('--lr_start', type=float, default=3e-4)
parser.add_argument('--lr_final', type=float, default=3e-5)
parser.add_argument('--lr_epochs', type=int, default=5)  # warmup

parser.add_argument('--beta_start', type=float, default=0.0)
parser.add_argument('--beta_final', type=float, default=0.04)
parser.add_argument('--beta_epochs', type=int, default=0)

parser.add_argument('--num_attrs', type=int, default=8)
parser.add_argument('--attr_size', type=int, default=8)

parser.add_argument('--num_enc_blocks', type=int, default=4)
parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--max_len', type=int, default=16)
parser.add_argument('--vocab_size', type=int, default=256)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=3)
parser.add_argument('--cnn_hidden_size', type=int, default=32)
parser.add_argument('--slot_size', type=int, default=64)
parser.add_argument('--mlp_hidden_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_epochs', type=int, default=40)

parser.add_argument('--hard', action='store_true')

args = parser.parse_args()


# In[ ]:

use_ddp = (torch.cuda.device_count() > 1)

torch.manual_seed(args.seed)

val_dataset = Shapes3D(root=args.data_path, phase='val')

train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': (not use_ddp),
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

val_epoch_size = len(val_loader)

model = DALLE(args)
model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))
model = model.cuda(args.local_rank)

# def visualize(image, recon, attns, N=8):
#
#     _, _, H, W = image.shape
#     image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
#     recon = recon[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
#     attns = attns[:N].expand(-1, -1, 3, H, W)
#
#     return torch.cat((image, recon, attns), dim=1).view(-1, 3, H, W)


def visualize(image, recon, N=8):
    _, _, H, W = image.shape
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon = recon[:N].expand(-1, 3, H, W).unsqueeze(dim=1)

    return torch.cat((image, recon), dim=1).view(-1, 3, H, W)


# def visualize(image, recon_combined, recons_masked, attns, masks, recons, N=8):
#     _, _, H, W = image.size()
#     image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
#     recon_combined = recon_combined[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
#     recons_masked = recons_masked[:N].expand(-1, -1, 3, H, W)
#     attns = attns[:N].expand(-1, -1, 3, H, W)
#     masks = masks[:N].expand(-1, -1, 3, H, W)
#     recons = recons[:N].expand(-1, -1, 3, H, W)
#
#     return (torch.cat((image, recon_combined, recons_masked), dim=1).view(-1, 3, H, W),
#             torch.cat((image, recon_combined, attns), dim=1).view(-1, 3, H, W),
#             torch.cat((image, recon_combined, masks), dim=1).view(-1, 3, H, W),
#             torch.cat((image, recon_combined, recons), dim=1).view(-1, 3, H, W))


# In[ ]:

encoder_grid = build_grid(64).unsqueeze(0).repeat(args.batch_size, 1, 1, 1)


with torch.no_grad():
    model.eval()
    for batch, image in enumerate(val_loader):
        image = image.cuda(args.local_rank, non_blocking=True)
        encoder_grid = encoder_grid.cuda(args.local_rank, non_blocking=True)

        recon = model.generate(image)

        vis_recon = visualize(image, recon, N=32)
        grid = vutils.make_grid(vis_recon, nrow=8, pad_value=1)[:, 2:-2, 2:-2]

        vutils.save_image(grid, os.path.join(args.paste_path, 'generate.png'))

        break
