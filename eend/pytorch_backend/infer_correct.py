#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import h5py
import numpy as np
from scipy.ndimage import shift

import torch
import torch.nn as nn

from eend.pytorch_backend.nnet_correct import TransformerCorrection
from eend import feature
from eend import kaldi_data


def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


def infer(args):
    # Prepare model
    in_size = feature.get_input_dim(
            args.frame_size,
            args.context_size,
            args.input_transform)

    if args.model_type == 'Transformer':
        model = TransformerCorrection(
                idim=in_size,
                n_blocks=2,
                att_dim=args.hidden_size,
                n_units=2048,
                n_heads=args.transformer_encoder_n_heads,
                n_speakers=args.num_speakers,
                )
    else:
        raise ValueError('Unknown model type.')

    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > 0) else "cpu")
    if device.type == "cuda":
        model = nn.DataParallel(model, list(range(args.gpu)))
    model = model.to(device)

    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    kaldi_obj = kaldi_data.KaldiData(args.data_dir)
    sas_obj = kaldi_data.process_sas(args.sas_path)
    for recid in kaldi_obj.wavs:
        print("Process: {}".format(recid))
        data, rate = kaldi_obj.load_wav(recid)
        Y = feature.stft(data, args.frame_size, args.frame_shift)
        Y = feature.transform(Y, transform_type=args.input_transform)
        
        sas = sas_obj[recid]
        Y = feature.splice(Y, context_size=args.context_size)
        Y = Y[::args.subsampling]
        len_sas = sas.shape[0]
        len_y = Y.shape[0]
        assert sas.shape[0] == Y.shape[0] 
        with torch.no_grad():
            Y_chunked = torch.from_numpy(Y)
            Y_chunked.to(device)
            sas = torch.from_numpy(sas).float().to(device)
            ys = model([Y_chunked], [sas], activation=torch.sigmoid)
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        outdata = ys[0].cpu().detach().numpy()
        print(outdata.shape)
        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)

