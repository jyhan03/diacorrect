# End-to-end error correction for speaker diarization
This repository provides a PyTorch implementation for our paper titled ["DiaCorrect: End-to-end error correction for speaker diarization"](https://arxiv.org/pdf/2210.17189.pdf).

Main training/inference framework in our work is based on [
EEND_PyTorch](https://github.com/Xflick/EEND_PyTorch).

# Usage
1. Prepare LibriSpeech-100 based simulation dataset. See the original implementation of EEND by [Hitachi Ltd](https://github.com/hitachi-speech/EEND). You can just replace the [script](https://github.com/hitachi-speech/EEND/tree/master/egs/mini_librispeech/v1) with the `run_prepare_shared.sh` provided here. 
2. Prepare initial speaker activity. `./run_eend.sh`
3. Refine the initial speaker activity. `./run_correct.sh`

# Results

| System | #Iter | FA | MISS | SA | DER | JER |
|:-------|:-----:|:--:|:----:|:--:|:---:|:---:|
|Baseline| - | 4.28 | 7.02 | 1.00 | 12.31 | 19.97|
|SAS<sub>p</sub> alone| - | 2.85 | 4.64 | 0.84 | 8.34 | 15.35 |
|DiaCorrect | - | 2.06 | 3.13 | 0.79 | 5.99 | 12.25 |
| | 1 | **1.26** | 2.74 | **0.83** | 4.83 | 10.70 |
| | 2| 1.32 | **2.46** | 0.86 | **4.63** | **10.25**

# Citation
    @article{diacorrect,
      title={DiaCorrect: End-to-end error correction for speaker diarization},  
      author={Han, J. and Cao, Y. and Lu, H. and Long, Y.},
      journal={arXiv preprint arXiv:2210.17189},
      year={2022}
    }

# Contact
Email: jyhan03@163.com



  


