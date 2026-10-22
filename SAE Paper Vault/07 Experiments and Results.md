---
tags: [paper, results]
---

# 07 · Experiments & Results

Back to [[00 Home]] · Previous: [[06 Concept Localization and Failure Correction]] · Next: [[08 Ablations and Overhead]]

---

## 1. Datasets

| Dataset | Cancer / modality | Size | Split used | Role |
|---|---|---|---|---|
| **PI-CAI** | Prostate, mp-MRI (3 Dutch centers) | 1,500 scans | 1,200 / 60 / 240 | main |
| **Prostate158** | Prostate, mp-MRI (T2W, DWI, ADC) | 158 scans | all for testing | **zero-shot** transfer |
| **PanTS** | Pancreatic, CT (tumor, pancreas, head/body/tail) | 9,901 scans → 700 sampled | 448 / 112 / 140 | main |
| **MSD-Brain** | Brain tumor, MRI (complete/core/enhancing) | 484 scans | 310 / 78 / 96 | appendix |

## 2. Setup

- **Backbone:** MedSAM, automatic prompts, multi-class decoder, fine-tuned per dataset ([[Vision Transformers and MedSAM]])
- **SAEs:** BatchTopK, D = 1,536, k = 8, lr 1e-4, 5 epochs, layers {1, 3, 5, 7, 9, 11} ([[TopK and BatchTopK SAEs]])
- **Classifier:** XGBoost ([[05 Method - Failure Classifier]])
- **Baselines:** MaxProb, MeanProb, Entropy, Energy, direct probability thresholding (all post-hoc, training-free). TTA added in the appendix.
- **Metrics:** DSC (segmentation), F1 (primary, detection), AUROC, AUPR, FPR95 ([[Evaluation Metrics]])

> [!note] Deliberately excluded
> Methods that build failure detection into training. They need the vendor's training pipeline, which doesn't fit the "frozen model from a vendor" clinical setting.

---

## 3. Result 1: Breaking the Pareto trade-off (Fig. 5)

Each baseline is run at thresholds 0.5–0.8, giving a cloud of (DSC, F1) points. The gray Pareto front is the best any baseline achieves.

**Finding:** on both PI-CAI and PanTS, the concept-based method lies **beyond** the baseline Pareto front, with higher F1 **and** higher DSC at once. Baselines can only trade one for the other.

PI-CAI operating point (full method): **DSC 50.1, F1 63.5**. PanTS numbers are lower overall (the figure's axes run to ~35), and the method still sits above the front. Brain (Fig. 9, axes span F1 84–92 and DSC 64–72): the method again beats the front.

![[12_tradeoff_schematic.png]]

> [!question] Why can it improve DSC *and* F1?
> Because correction is **patch-level** ([[06 Concept Localization and Failure Correction]]). A baseline that flags a mask must discard all of it, true positives included. This method removes only the suspicious patches, so catching more failures doesn't destroy good segmentation.

## 4. Result 2: Zero-shot generalization (Table 1, Fig. 6)

Detectors trained on **PI-CAI**, tested **unchanged** on **Prostate158** (different scanners and hospital).

![[06_zero_shot_table1.png]]

| Method | FPR95 ↓ | AUROC ↑ | AUPR ↑ |
|---|---|---|---|
| MaxProb | 100.0 | 50.0 | 64.4 |
| MeanProb | 100.0 | 52.6 | 65.1 |
| Entropy | 97.8 | 53.1 | 65.6 |
| Energy | 100.0 | 48.0 | 63.6 |
| **Ours** | **64.4** | **73.6** | **74.1** |

- Confidence baselines collapse to **chance** (AUROC ≈ 50). Confidence calibration doesn't survive the domain shift.
- Concept-based detection keeps **AUROC 73.6**.
- Fig. 6: success vs failure still separate in concept space on unseen data (p < 1e-5, Cohen's **d ≈ 1.20 / 1.22** for count / mean activation, vs 1.8 in-domain).

> [!important] This is probably the strongest result
> In deployment, every hospital is "out of distribution". A failure detector that survives a dataset shift is worth much more than one that's only better in-domain.

## 5. Result 3: Interpretation (Figs. 2, 8)

- The model learns concepts that localize to anatomy: peripheral zone, gland, tumor; tumor core vs edge. Similar alignment in pancreatic CT.
- **Mechanistic explanation of failures:** successes strongly activate tumor concepts. Failures activate them weakly or not at all. Failures happen when the model *"misinterprets ambiguous image features, failing to activate the correct internal representations."*
- The most important detector features are **tumor-related** concepts, then anatomical-boundary concepts. See [[05 Method - Failure Classifier]].

## 6. Result 4: Brain tumors (Appendix A)

MSD-Brain contains **only positive cases** (every scan has a tumor), unlike PI-CAI/PanTS with many tumor-free cases. F1 is much higher here (the balanced data gives cleaner signals). The method still beats the baseline Pareto front, and concepts like *enhancing, necrotic, edema, margin* are found.

![[fig_paper_fig10_brain_concepts.png]]

## 7. Takeaways

> [!tip] Takeaways from §4.2–4.3
> - *"The model's internal concepts provide a mechanistic explanation for segmentation failures and a reliable signal for predicting them."*
> - *"Model's internal concepts contain richer signals about failures"* than output confidence.
