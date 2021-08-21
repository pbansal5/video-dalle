#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class dVAE(nn.Module):
    
    def __init__(self, vocab_size, img_channels):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Conv2dBlock(img_channels, 64, 4, 4),
            Conv2dBlock(64, 64, 1, 1),
            Conv2dBlock(64, 64, 1, 1),
            conv2d(64, vocab_size, 1)
        )
        
        # self.decoder = nn.Sequential(
        #     Conv2dBlock(vocab_size, 64, 1),
        #     Conv2dBlock(64, 64 * 2 * 2, 1),
        #     nn.PixelShuffle(2),
        #     Conv2dBlock(64, 64 * 2 * 2, 1),
        #     nn.PixelShuffle(2),
        #     conv2d(64, img_channels, 1),
        # )
    
        self.decoder = nn.Sequential(
            Conv2dBlock(vocab_size, 64, 1),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            Conv2dBlock(64, 64, 3, 1, 1),
            Conv2dBlock(64, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            conv2d(64, img_channels, 1),
        )
    
    def forward(self, image, sigma, tau, hard):
        """
        image: batch_size x img_channels x 64 x 64
        return: (batch_size x img_channels x 64 x 64, 1, 1, 1)
        """
        z_logits = F.log_softmax(self.encoder(image), dim=1)  # batch_size x vocab_size x 4 x 4
        B, V, H, W = z_logits.shape
        
        z = gumbel_softmax(z_logits, tau, hard, dim=1)  # batch_size x vocab_size x 4 x 4
        recon = self.decoder(z)  # batch_size x img_channels x 64 x 64
        
        log_likelihood = log_prob_gaussian(image, recon, sigma).sum() / B
        kl = ((z_logits.exp() * z_logits).sum() / B) + (math.log(V) * H * W)
        mse = ((image - recon) ** 2).sum() / B
        
        return (recon.clamp(0., 1.),
                -log_likelihood,
                kl,
                mse)



# class dVAE(nn.Module):
    
#     def __init__(self, vocab_size, img_channels):
#         super().__init__()
        
#         self.encoder = nn.Sequential(
#             Conv2dBlock(img_channels, 64, 4, 4),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             conv2d(64, vocab_size, 1)
#         )
        
#         self.decoder = nn.Sequential(
#             Conv2dBlock(vocab_size, 64, 1),
#             Conv2dBlock(64, 64, 3, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64 * 2 * 2, 1),
#             nn.PixelShuffle(2),
#             Conv2dBlock(64, 64, 3, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64, 1, 1),
#             Conv2dBlock(64, 64 * 2 * 2, 1),
#             nn.PixelShuffle(2),
#             conv2d(64, img_channels, 1),
#         )
    
    
#     def forward(self, image, sigma, tau, hard):
#         """
#         image: batch_size x img_channels x 64 x 64
#         return: (batch_size x img_channels x 64 x 64, 1, 1, 1)
#         """
#         z_logits = F.log_softmax(self.encoder(image), dim=1)  # batch_size x vocab_size x 4 x 4
#         B, V, H, W = z_logits.shape
        
#         z = gumbel_softmax(z_logits, tau, hard, dim=1)  # batch_size x vocab_size x 4 x 4
#         recon = self.decoder(z)  # batch_size x img_channels x 64 x 64
        
#         log_likelihood = log_prob_gaussian(image, recon, sigma).sum() / B
#         kl = ((z_logits.exp() * z_logits).sum() / B) + (math.log(V) * H * W)
#         mse = ((image - recon) ** 2).sum() / B
        
#         return (recon.clamp(0., 1.),
#                 -log_likelihood,
#                 kl,
#                 mse)

