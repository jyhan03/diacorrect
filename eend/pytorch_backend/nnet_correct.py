# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import torch
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler

from .sub_nets import TransfomerEncoder, Conv2dEncoder, SASEncoder

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


class TransformerCorrection(nn.Module):
    def __init__(self, idim, n_blocks=2, att_dim=256, 
                 n_units=2048, n_heads=4, dropout_rate=0.1, n_speakers=2):
        super(TransformerCorrection, self).__init__()
        self.speech_enc = Conv2dEncoder(idim, att_dim)
        self.sas_enc = SASEncoder(att_dim)
        self.sa_layer = TransfomerEncoder(idim=att_dim*(n_speakers+1), n_blocks=n_blocks, att_dim=att_dim, 
                 n_units=n_units, n_heads=n_heads, dropout_rate=dropout_rate)
        self.decoder = nn.Linear(att_dim, n_speakers)
    
    def forward(self, src, sas, activation=None):
        """
        rttm: B, T, C
        """
        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)
        src = self.speech_enc(src)
        
        sas = nn.utils.rnn.pad_sequence(sas, padding_value=-1, batch_first=True)
        spk_num = sas.shape[-1]
        sas = torch.cat([self.sas_enc(torch.unsqueeze(sas[:, :, c], -1)) 
                                                for c in range(spk_num)], -1)
        # src: (B, T, E)
        output = self.sa_layer(torch.cat([src, sas], -1))
        output = self.decoder(output)
        if activation:
            output = activation(output)

        output = [out[:ilen] for out, ilen in zip(output, ilens)]

        return output        

if __name__ == "__main__":
    model = TransformerCorrection(345)
    input = torch.randn(4, 200, 345)
    x = [x for x in input]
#    x = [input]
    rttm = torch.randn(4, 200, 2)
    y = model(x, rttm)
    print(y[0].shape)
#    torch.set_printoptions(
#        precision=2,    # 精度，保留小数点后几位，默认4
#        threshold=1000,
#        edgeitems=3,
#        linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
#        profile=None,
#        sci_mode=False  # 用科学技术法显示数据，默认True
#    )
#    
#    d_model = 256
#    posi_emb = PositionalEncoding(d_model, 0.1)
#    x = torch.randn(4, 200, d_model)
#    y = posi_emb(x)
