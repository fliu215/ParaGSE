# ParaGSE:Parallel Generative Speech Enhancement with Group-Vector-Quantization-Based neural Speech Codec
### Fei Liu, Yang Ai
#### Published in ICASSP 2026

**Abstract:** 
Recently, generative speech enhancement has garnered considerable interest; however, existing approaches are hindered by excessive complexity, limited efficiency, and suboptimal speech quality. To overcome these challenges, this paper proposes a novel parallel
generative speech enhancement (ParaGSE) framework that leverages a group vector quantization (GVQ)-based neural speech codec. The GVQ-based codec adopts separate VQs to produce mutually independent tokens, enabling efficient parallel token prediction in
ParaGSE. Specifically, ParaGSE leverages the GVQ-based codec to encode degraded speech into distinct tokens, predicts the corresponding clean tokens through parallel branches conditioned 
on degraded spectral features, and ultimately reconstructs clean speech via the codec decoder. Experimental results demonstrate that ParaGSE consistently produces superior enhanced speech compared
to both discriminative and generative baselines, under a wide range of distortions including noise, reverberation, band-limiting, and their mixtures. Furthermore, empowered by parallel computation in token prediction, ParaGSE attains about a 1.5-fold improvement
in generation efficiency on CPU compared with serial generative speech enhancement approaches.

**We provide our implementation as open source in this repository. Audio samples can be found at the [demo website](https://anonymity225.github.io/ParaGSE/).**

## Model Structure
![model](figures/model.png)

## Pre-requisites
1. Clone this repository.
2. Install requirements.
```
conda create -n paragse python=3.9
conda activate paragse
cd ParaGSE
pip install -r requirements.txt
```
3. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942).Data processing instructions can be found in [this](https://github.com/fliu215/UDSE_Code/tree/main/data_prepare). Resample all wav files to 16 kHz.

## Training
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 python inference.py --test_noise_wav your/path --output_dir save/path
```

## Citation
```
@inproceedings{liu2026paragse,
  title={ParaGSE:Parallel Generative Speech Enhancement with Group-Vector-Quantization-Based neural Speech Codec},
  author={Liu, Fei and Ai, Yang},
  booktitle={Proc. ICASSP},
  year={2026},
}
```
