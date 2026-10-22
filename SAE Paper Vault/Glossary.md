---
tags: [reference, glossary]
---

# Glossary

Back to [[00 Home]]

| Term | Meaning |
|---|---|
| **Activation** | The vector of numbers a layer outputs for an input (here: one 768-dim vector per image patch). |
| **ADC / DWI / T2W** | MRI sequences (apparent diffusion coefficient, diffusion-weighted, T2-weighted) used in prostate imaging. |
| **AUROC / AUPR** | Threshold-free ranking metrics. See [[Evaluation Metrics]]. |
| **BatchTopK** | SAE variant keeping the top k·B activations across a batch. See [[TopK and BatchTopK SAEs]]. |
| **Circuit** | A sub-network of weights that computes some features from others. See [[Mechanistic Interpretability]]. |
| **Cohen's d** | Effect size: difference of means in units of standard deviation. |
| **Concept** | In this paper: one SAE latent (dictionary element), often matching a clinical term. |
| **Concept bottleneck model (CBM)** | A model forced to predict through human-labeled concepts. This paper is "concept-then-predict" too, but post-hoc and unsupervised. |
| **c(x)** | Failure representation: concatenated SAE codes from 6 layers (9,216-dim). See [[04 Method - Failure Representation]]. |
| **Decoder direction $\mathbf{w}_i$** | Column $i$ of $\mathbf{W}_{dec}$: the activation-space direction of concept $i$. |
| **Dead latent** | An SAE unit that never activates. |
| **Dice (DSC)** | Overlap metric for masks. Often typo'd "DCS" in the paper's figures. |
| **Energy score** | $-\log\sum_j e^{f_j(x)}$, an OOD/uncertainty score from logits. |
| **Feature** | A property of the input that the network represents, ideally as a direction. |
| **Feeling of Error (FOE)** | Psychology: the spontaneous unease that flags a possible error. The paper's machine analog is an SAE concept signature. |
| **Focal loss** | Loss that down-weights easy pixels. Used to fine-tune MedSAM (α = 0.97, γ = 2). |
| **FPR95** | False-positive rate at 95 % true-positive rate. Lower is better. |
| **Linear representation hypothesis** | Activations ≈ sparse linear combinations of feature directions. |
| **MedSAM** | Segment Anything fine-tuned for medical images. See [[Vision Transformers and MedSAM]]. |
| **Monosemantic / polysemantic** | A unit meaning one thing vs many things. See [[Superposition and Polysemanticity]]. |
| **mp-MRI** | Multi-parametric MRI (several sequences combined). |
| **Overcomplete dictionary** | More dictionary elements than dimensions (D > d). |
| **Pareto front** | The set of best achievable trade-offs. Points beyond it dominate all baselines. |
| **Patch token** | The ViT's vector for one 16×16 image patch. |
| **Peripheral zone (PZ) / transition zone (TZ)** | Anatomical zones of the prostate. Most cancers arise in the PZ. |
| **Probe** | A simple classifier trained on internal activations to read out a property. |
| **SAE** | Sparse autoencoder. See [[Sparse Autoencoders]]. |
| **Silent failure** | A wrong output with no warning signal. |
| **Steering / amplification** | Editing a feature's activation to test its causal effect. See [[09 Causal Evidence]]. |
| **Superposition** | Packing more features than dimensions using nearly orthogonal directions. |
| **TTA** | Test-time augmentation: predict on augmented copies and measure disagreement. |
| **ViT** | Vision Transformer. |
| **XGBoost** | Gradient-boosted decision trees. The paper's failure classifier. |
