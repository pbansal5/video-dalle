#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


# In[ ]:


class SlotAttention(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size, heads,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # Linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # Slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))

    def forward(self, inputs, slots):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].

        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k
        
        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))                             # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)                # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_vis = attn.sum(1)                                                      # Shape: [batch_size, num_inputs, num_slots].
            
            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)                              # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)                          # Shape: [batch_size, num_slots, slot_size].
            
            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis

class SlotAttentionDecomposed(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size, heads,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # Linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        self.project_v_pos = linear(input_size, slot_size, bias=False)
        self.project_v_content = linear(input_size, slot_size, bias=False)
        
        # Slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.gru_pos = gru_cell(slot_size, slot_size)
        self.gru_content = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.mlp_pos = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.mlp_content = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))

    def forward(self, inputs,inputs_pos,inputs_content, slots):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].

        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()
        
        pos_slots = torch.zeros_like(slots)
        content_slots = torch.zeros_like(slots)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v_pos = self.project_v_pos(inputs_pos).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v_content = self.project_v_content(inputs_content).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k
        
        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))                             # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)                # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_vis = attn.sum(1)                                                      # Shape: [batch_size, num_inputs, num_slots].
            
            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)                              # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)                          # Shape: [batch_size, num_slots, slot_size].

            updates_pos = torch.matmul(attn.transpose(-1, -2).clone().detach(), v_pos)                              # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates_pos = updates.transpose(1, 2).reshape(B, N_q, -1)                          # Shape: [batch_size, num_slots, slot_size].

            updates_content = torch.matmul(attn.transpose(-1, -2).clone().detach(), v_content)                              # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates_content = updates.transpose(1, 2).reshape(B, N_q, -1)                          # Shape: [batch_size, num_slots, slot_size].
            
            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

            pos_slots = self.gru_pos(updates_pos.view(-1, self.slot_size),
                             pos_slots.view(-1, self.slot_size))
            pos_slots = pos_slots.view(-1, self.num_slots, self.slot_size)
            pos_slots = pos_slots + self.mlp_pos(self.norm_mlp(pos_slots))
        
            content_slots = self.gru_content(updates_content.view(-1, self.slot_size),
                             content_slots.view(-1, self.slot_size))
            content_slots = content_slots.view(-1, self.num_slots, self.slot_size)
            content_slots = content_slots + self.mlp_content(self.norm_mlp(content_slots))
        
        return slots,pos_slots,content_slots, attn_vis


# In[ ]:

class SoftPositionEmbed(nn.Module):

    def __init__(self, in_channels, out_channels, G):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.grid = nn.Parameter(torch.zeros(1, out_channels, G, G), requires_grad=True)
        nn.init.trunc_normal_(self.grid)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        return self.dropout(inputs + self.grid)



class SlotAttentionEncoder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_channels, input_width, slot_size, mlp_hidden_size, pos_channels):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels

        self.encoder_pos = SoftPositionEmbed(pos_channels, input_channels, input_width)
        
        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        
        # Parameters for Gaussian init (shared by all slots).
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)
        
        self.slot_attention = SlotAttention(
            num_iterations, num_slots,
            input_channels, slot_size, mlp_hidden_size, 1)
    
    
    def forward(self, x):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        B, *_ = x.size()
        # Convolutional encoder with position embedding.
        # x = self.encoder_pos(x)  # Position embedding.
        # Flatten spatial dimensions (treat image as set).
        # x = x.permute(0, 2, 3, 1).reshape(B, G * G, D)
        # Feedforward network on set.
        x = self.mlp(self.layer_norm(x))
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].

        # Slot Attention module.
        slots = x.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots
        slots, attn = self.slot_attention(x, slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
        
        return slots, attn

class SlotAttentionEncoderDecomposed(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_channels, input_width, slot_size, mlp_hidden_size, pos_channels):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels

        self.encoder_pos = SoftPositionEmbed(pos_channels, input_channels, input_width)
        
        self.layer_norm = nn.LayerNorm(input_channels)
        self.layer_norm_pos = nn.LayerNorm(input_channels)
        self.layer_norm_content = nn.LayerNorm(input_channels)

        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        self.mlp_pos = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        self.mlp_content = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        
        # Parameters for Gaussian init (shared by all slots).
        self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)
        
        self.slot_attention = SlotAttentionDecomposed(
            num_iterations, num_slots,
            input_channels, slot_size, mlp_hidden_size, 1)
    
    
    def forward(self, x,pos_x,content_x):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        B, *_ = x.size()
        # Convolutional encoder with position embedding.
        # x = self.encoder_pos(x)  # Position embedding.
        # Flatten spatial dimensions (treat image as set).
        # x = x.permute(0, 2, 3, 1).reshape(B, G * G, D)
        # Feedforward network on set.
        x = self.mlp(self.layer_norm(x))
        pos_x = self.mlp_pos(self.layer_norm_pos(pos_x))
        content_x = self.mlp_content(self.layer_norm_content(content_x))
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].

        # Slot Attention module.
        slots = x.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots
        slots,slots_pos,slots_content,attn = self.slot_attention(x, pos_x,content_x,slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
        
        return slots,slots_pos,slots_content, attn

