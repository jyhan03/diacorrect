# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 08:36:43 2022

@author: Jyhan
"""

import math

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np        

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x

class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = ChannelWiseLayerNorm(conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = ChannelWiseLayerNorm(conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class SASEncoder(nn.Module):
    def __init__(self, odim=256, hidden=512):
        super(SASEncoder, self).__init__()
        self.conv = nn.Conv1d(1, odim, 1) 
        self.conv1d_block = Conv1DBlock(odim, hidden)
        
    def forward(self, x):
        """
        input: B, T, 1
        out: B, T, N
        """
        x = torch.einsum('ijk->ikj', x)
        x = F.relu(self.conv(x)) 
        x = self.conv1d_block(x)     
        return torch.einsum('ijk->ikj', x)  
    

class Conv2dEncoder(nn.Module):
    def __init__(self, idim, odim):
        """Construct an Conv2dEncoder object."""
        super(Conv2dEncoder, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, odim, (3, 7), (1, 5), (1, 0)),
            nn.ReLU(),
            nn.Conv2d(odim, odim, (3, 7), (1, 5), (1, 0)),
            nn.ReLU(),
        )
        self.down_dim = ((idim - 2)//5 - 2) // 5
        self.out = nn.Linear(odim * self.down_dim, odim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x   
    

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head=4, n_feat=256, dropout_rate=0.0):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None    # attention for plot
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(self, x):
        n_batch = x.size(0)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        
        x = torch.matmul(p_attn, v)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)        


class PositionwiseFeedForward(nn.Module):
    """ Positionwise feed-forward layer
    linear-->relu-->dropout-->linear
    """
    def __init__(self, idim, n_units, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, n_units)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(n_units, idim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(self, x):
        x = self.relu(self.w_1(x))
        x = self.w_2(self.dropout(x))
        
        return x


class TransfomerBlock(nn.Module):
    """
    Transfomer Encoder block.
    
    """
    def __init__(self, att_dim=256, n_units=2048, n_heads=4, dropout_rate=0.1):
        super(TransfomerBlock, self).__init__()
        self.ln_norm1 = nn.LayerNorm(att_dim)
        self.self_mha = MultiHeadedAttention(n_heads, att_dim, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.ln_norm2 = nn.LayerNorm(att_dim)
        self.ffn = PositionwiseFeedForward(att_dim, n_units, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def __call__(self, x):
        x = self.ln_norm1(x)
        x = x + self.dropout1(self.self_mha(x))
        x = self.ln_norm2(x)
        x = x + self.dropout2(self.ffn(x))
        
        return x


class TransfomerEncoder(nn.Module):
    """
    Transfomer Encoder layer for EEND.
    
    """
    def __init__(self, idim, n_blocks=2, att_dim=256, 
                 n_units=2048, n_heads=4, dropout_rate=0.1):
        super(TransfomerEncoder, self).__init__()
        self.linear = nn.Linear(idim, att_dim)
        self.transformer_blocks = self._build_blocks_layer(
                    n_blocks=n_blocks, att_dim=att_dim,
                    n_units=n_units, n_heads=n_heads, dropout_rate=dropout_rate)
        self.ln_norm = nn.LayerNorm(att_dim)
        
    def _build_blocks_layer(self, n_blocks, **block_kwargs):
        """
        build transformer blocks
        """
        blocks = [
             TransfomerBlock(**block_kwargs)
             for b in range(n_blocks)
        ]
        return nn.Sequential(*blocks)
    
    def __call__(self, x):
        x = self.linear(x)
        x = self.transformer_blocks(x)
        x = self.ln_norm(x)
        
        return x