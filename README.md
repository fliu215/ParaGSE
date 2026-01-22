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
