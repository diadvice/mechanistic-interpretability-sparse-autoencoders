---
tags: [paper, critique]
---

# 10 · Critical Analysis: Strengths, Weaknesses, Open Questions

Back to [[00 Home]] · Previous: [[09 Causal Evidence]]

---

## ✅ Strengths

1. **Applied mech int with a real payoff.** Most SAE papers stop at "here's a pretty feature". This one turns features into a tool with measurable clinical value: detect, explain, correct.
2. **Post-hoc and cheap.** Works on a frozen vendor model. +3.7 % latency for routine use, 54 MB of checkpoints.
3. **Zero-shot robustness.** AUROC 73.6 vs ~50 for confidence baselines on an unseen dataset. Strongest practical evidence that internals beat outputs.
4. **Escapes the trade-off** via patch-level correction rather than all-or-nothing rejection.
5. **Clinical grounding.** Concept labels checked automatically (IoU) and by two radiologists.
6. **Causal check.** Amplifying #469 is a real intervention with controls, more than many interpretability-for-medicine papers do.
7. **Three cancers, two modalities** (MRI, CT).

## ⚠️ Weaknesses & things to check

### Method clarity
- **Image vs patch level.** $\mathbf{c}(x)$ is defined per image (Eq. 2), but correction applies $g$ to per-patch vectors $\mathbf{c}(x)_{ij}$ (Eq. 5). How patch concepts are pooled for the image-level classifier, and how per-patch labels are obtained for training, isn't described.
- **Failure threshold τ** (Definition 1) isn't given. In the experiments, failures are framed mainly as TP vs **FP** masks, which narrows what "failure" means. Under-segmentation and missed tumors (FN) get less attention, and patch removal can't fix them.
- **How DSC is computed after detection** (corrected masks? rejected masks set to empty?) for the baselines isn't fully spelled out. This matters for the Pareto comparison.

### Evidence
- **"Fewer and weaker" is a summary statistic.** In Fig. 2's failure case, anatomical concepts (#968, #204) are actually *stronger* (0.70/0.67) than in the success case (0.15/0.15). The robust signal is the **missing tumor concept**, not a uniform dimming.
- **Selected qualitative examples.** The per-case DSC gains (e.g. +27.6) are hand-picked. There are no aggregate per-dataset DSC-gain tables with variance.
- **No error bars / seeds** for the main F1/DSC numbers or ablations. Differences like 63.5 vs 62.5 F1 may be noise.
- **Single-concept causal test** (one α, one layer, amplification only, no ablation). The random-latent control *improving* DSC (+0.035) isn't explained.
- **Baselines are weak-ish.** All training-free output statistics plus TTA. No comparison with Mahalanobis/feature-space OOD detectors, MC-dropout, deep ensembles, or a **probe on raw (non-SAE) activations**. That last one is the key missing control: is the **SAE** necessary, or would XGBoost on raw 768-dim features do as well, minus the interpretability?
- **Small zero-shot test** (158 scans), one transfer pair only.

### Small inconsistencies in the text
- Abstract mentions brain cancer; contributions list only prostate & pancreatic (brain is in the appendix).
- "Cohen's d = 1.8" vs "d > 1.8" in different places.
- FPR95 described as "TPR at 95 % FPR" (likely a wording slip).
- Ablation prose says mid-layers excel at DSC (48.8), but deep layers have 49.6.
- MedSAM cited as [47] in the main text and [48] in the appendix.
- Typos ("abailable", "Prostae158", "DCS" for DSC in figures).

## 🔭 Open questions & future directions

From the paper:
- **Causal mechanisms:** mediation analysis and causal abstraction to find *how* concepts interact, not just which co-occur.
- **Universal SAE** across cancers (brain, liver, lung) to find shared anatomical primitives.
- **Failure-mode classification** (boundary vs shape vs size errors) instead of binary.
- **Missing modalities:** use modality-specific concepts to tell whether the available scans are enough.

My additions:
- Can concepts **fix false negatives**, e.g. amplify the tumor concept *locally* where context concepts suggest a lesion?
- Does the "feeling of error" signal predict **inter-rater disagreement** (cases radiologists also find hard)?
- Would the same signal appear in **CNN** segmenters (nnU-Net), or is it ViT-specific?
- **Prospective** evaluation: does showing concept explanations to radiologists actually improve their accuracy or speed?

## 🧭 Bottom line

> [!summary] Verdict
> A clear, well-motivated demonstration that **SAE concepts inside a medical segmentation model carry a usable "something is wrong" signal**. It beats output-confidence methods, especially under dataset shift, and comes with human-readable explanations. The main gaps are method details (patch vs image level), statistical rigor (no variance, missing a raw-activation probe baseline), and depth of causal analysis. It's a strong proof of concept for **mech int as a safety tool in medical AI**.
