---
tags: [paper, ablation]
---

# 08 · Ablations, Baselines & Computational Overhead

Back to [[00 Home]] · Previous: [[07 Experiments and Results]] · Next: [[09 Causal Evidence]]

---

All ablations are on **PI-CAI**. The full method gives **DSC 50.1, F1 63.5**.

![[07_ablations.png]]

## 1. SAE size & sparsity (Table 2)

| | k = 4 | **k = 8** | k = 16 |
|---|---|---|---|
| **D = 1,536 (2×768)**: DSC | 49.7 | **50.1** | 49.9 |
| **D = 1,536**: F1 | 60.3 | **63.5** | 62.5 |
| D = 3,072 (4×768): DSC | 50.1 | 49.7 | 49.6 |
| D = 3,072: F1 | 58.4 | 57.4 | 62.6 |

- Best: **D = 1,536, k = 8**. Both are **smaller** than typical natural-image SAEs (k ≥ 16, D ≥ 10,000).
- The authors' hypothesis: medical images are **more structured** (fixed anatomy, standard views), so fewer concepts are needed.
- DSC barely moves (49.6–50.1). F1 varies more (57.4–63.5). The SAE configuration mostly affects **detection**.

## 2. Which layers? (Table 3)

| Layers | DSC | F1 |
|---|---|---|
| Early (1, 3) | 47.5 | 44.5 |
| Mid (5, 7) | 48.8 | 50.4 |
| Deep (9, 11) | 49.6 | 57.6 |
| **All six** | **50.1** | **63.5** |

- **Deep** concepts alone give the best detection (semantic patterns).
- **All layers combined** beat every subset by a wide margin (+5.9 F1 over deep-only), so layers carry **complementary** information. This supports [[04 Method - Failure Representation]].
- *(Text note: the paper's prose says mid-layers "excel at segmentation quality (DSC=48.8)", but deep layers have higher DSC (49.6) in the table. A minor inconsistency.)*

## 3. Which classifier? (Table 4)

| Classifier | DSC | F1 |
|---|---|---|
| **XGBoost** | **50.1** | **63.5** |
| Logistic regression | 47.8 | 59.2 |
| Decision tree | 48.5 | 58.7 |
| Random forest | 44.6 | 57.0 |

Even **logistic regression** (linear, very interpretable) gets F1 59.2. Most of the signal is in the **features**, not the classifier. XGBoost's non-linear interactions add about +4 F1.

## 4. vs. Test-Time Augmentation (Table 7, appendix)

TTA baseline: 6 transforms (identity, hflip, vflip, hflip+vflip, ±10 % brightness). Failure score = 1 − mean pairwise DSC.

| Method | F1 | Acc | DSC | Cost |
|---|---|---|---|---|
| MedSAM base | 0.359 | 79.55 % | 0.513 | 1× |
| TTA (K = 6) | 0.592 | 94.34 % | 0.434 | 6× |
| **Ours** | **0.635** | **95.85 %** | 0.501 | 1× + 6 SAEs |

- Better detection than TTA, at **one** forward pass instead of six.
- TTA hurts DSC badly (0.434). Ours keeps most of it (0.501 vs 0.513 base).
- TTA saturates: K = 4 gives F1 0.591.

## 5. Compute overhead (Table 5)

![[08_latency.png]]

| Configuration | Latency (ms) | Overhead |
|---|---|---|
| MedSAM alone | 126.3 ± 0.5 | — |
| + 6 SAE encoders (routine) | 130.9 ± 2.7 | **+3.7 %** |
| + 6 SAE enc+dec (intervention) | 134.3 ± 0.3 | +6.3 % |
| + XGBoost (full pipeline) | 176.5 ± 13.3 | +39.7 % |

- RTX A6000, 1024² input, mean ± std over 100 forwards
- +60 MB GPU memory (+1.8 %), 54 MB disk for all six SAE checkpoints
- Full pipeline ≈ **5.66 images/s**
- Relative overhead stable for batch size 1–16

> [!note] Where the time goes
> The SAEs are almost free. **XGBoost** is the bulk of the overhead (~42 ms), probably because it's applied per patch for correction. A GPU-based tree inference library or a linear head would shrink this.
