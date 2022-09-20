# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

# Modified by: Jiangyu Han, 2022
#

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from .sub_nets import TransfomerEncoder

class NoamScheduler(_LRScheduler):
    """
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

        # the initial learning rate is set as step = 1
        if self.last_epoch == -1:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            self.last_epoch = 0
        print(self.d_model)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class TransformerModel(nn.Module):
    def __init__(self, idim=325, n_blocks=2, att_dim=256, 
                 n_units=2048, n_heads=4, dropout_rate=0.1, n_speakers=2):
        super(TransformerModel, self).__init__()
        self.eend_sa = TransfomerEncoder(idim=idim, n_blocks=n_blocks, att_dim=att_dim, 
                 n_units=n_units, n_heads=n_heads, dropout_rate=dropout_rate)
        self.decoder = nn.Linear(att_dim, n_speakers)
    
    def forward(self, src, activation=None):
        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)
        # src: (B, T, E)
        output = self.eend_sa(src)
        output = self.decoder(output)
        if activation:
            output = activation(output)

        output = [out[:ilen] for out, ilen in zip(output, ilens)]

        return output        

if __name__ == "__main__":
    model = TransformerModel(23)
    input = torch.randn(4, 200, 23)
    x = [x for x in input]
#    x = [input]
    y = model(x)
