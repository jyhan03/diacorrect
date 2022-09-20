#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import torch
import numpy as np
from eend import kaldi_data
from eend import feature

import h5py

def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


def my_collate(batch):
    data, target, init_sas = list(zip(*batch))
    return [data, target, init_sas]


def rttm2segments(rttm_path):
    rec2seg = {}
    with open(rttm_path, 'r') as frttm:
        for line in frttm:
            line = line.split()
            rec = line[1]
            spk = line[-2]
            start = float(line[3])
            end = start + float(line[4])
            utt = '{}_{}_{:07d}_{:07d}'.format(spk, rec, 
                     int(start * 100), int(end * 100))
            if rec not in rec2seg:
                rec2seg[rec] = []
            rec2seg[rec].append({'utt': utt, 'spk': spk, 'st': start, 'et': end})
    return rec2seg
        
        
def load_wav_scp(wav_scp_file):
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    return {x[0]: x[1] for x in lines}    

def load_sas(filepath):
    h5_data = h5py.File(filepath, 'r')
    data = h5_data["T_hat"][:]
    return data

class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            sas_path,
            chunk_size=2000,
            context_size=7,
            frame_size=200,
            frame_shift=80,
            subsampling=1,
            rate=8000,
            input_transform=None,
            use_last_samples=True,
            label_delay=0,
            n_speakers=2,
            ):
        self.data_dir = data_dir        
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay

        self.data = kaldi_data.KaldiData(self.data_dir)
        self.sas_scp = load_wav_scp(sas_path)      

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")
        
    def load_sas(self, recid):
        data = load_sas(self.sas_scp[recid])
        return data   

    def process_sas(self, recid):
        """
        sas is extraced from the whole utt, (T, C)
        """
        sas = self.load_sas(recid)   
        assert sas.shape[-1] == self.n_speakers
        sas = np.tile(sas, self.subsampling).reshape(-1, self.n_speakers)
        return sas

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        Y, T = feature.get_labeledSTFT(
            self.data, rec, st, ed, 
            self.frame_size, self.frame_shift, self.n_speakers)
        # Y: (frame, num_ceps)
        Y = feature.transform(Y, self.input_transform)
        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = feature.splice(Y, self.context_size)    
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)
        
        sas = self.process_sas(rec)[st:ed]
        sas = sas[::self.subsampling]
        
        Y_ss = torch.from_numpy(Y_ss).float()
        T_ss = torch.from_numpy(T_ss).float()
        sas = torch.from_numpy(sas).float()
        return Y_ss, T_ss, sas


if __name__ == '__main__':
    data_dir = r'C:\Users\Jyhan\Desktop\Speaker Diarization\projects\debug\sample\data\dev_clean_2_ns2_beta2_5'
    rttm_path=r'C:\Users\Jyhan\Desktop\Speaker Diarization\projects\debug\sample\data\dev_clean_2_ns2_beta2_5\rttm'
    dataset = KaldiDiarizationDataset(data_dir, rttm_path)
    A = dataset[1]
#    print(A[1])
#    print(A[2])
#    rttm_path=r'C:\Users\Jyhan\Desktop\Speaker Diarization\projects\debug\sample\data\dev_clean_2_ns2_beta2_5\rttm'
#    out_path = r'C:\Users\Jyhan\Desktop\喜马拉雅\work\diarization\ustc-ximalaya\ts-vad_ustc\revised\debug\segments'
#    a = rttm2segments(rttm_path, out_path)
