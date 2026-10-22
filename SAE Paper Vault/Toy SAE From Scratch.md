---
tags: [concept, sae, code, demo]
---

# Toy SAE From Scratch (numpy)

Back to [[00 Home]] · Previous: [[TopK and BatchTopK SAEs]] · Next: [[Vision Transformers and MedSAM]]

---

A self-contained experiment showing that an SAE **recovers hidden features from superposition**. This is the same principle the paper relies on, at toy scale.

## Setup

| | Toy | Paper |
|---|---|---|
| True features | 12 (known, random unit vectors) | unknown (clinical concepts) |
| Activation dim $d$ | 6 | 768 |
| Dictionary $D$ | 32 | 1,536 |
| Sparsity $k$ | 2 (TopK) | 8 (BatchTopK) |
| Data | each feature on with prob 8 %, magnitude U(0.5, 1.5) | MedSAM patch tokens |

12 features in 6 dimensions means they **must** overlap: superposition by construction.

## Result

![[02_toy_sae_recovery.png]]

Every one of the 12 true features is matched by a distinct SAE latent with |cos| ≥ 0.98 (mean 0.99). The SAE was **never told** what the features were. It found them only from the pressure to reconstruct with few active latents.

## Code

```python
import numpy as np

def run(seed=3, n_feat=12, d=6, D=32, k_top=2, steps=6000, p=0.08):
    rng = np.random.default_rng(seed)
    # ground-truth feature directions (unit vectors in R^d)
    T = rng.normal(size=(n_feat, d)); T /= np.linalg.norm(T, axis=1, keepdims=True)
    def sample(n):                       # sparse feature activations -> dense h
        a = rng.random((n, n_feat)) < p
        return (rng.uniform(0.5, 1.5, (n, n_feat)) * a) @ T
    X = sample(50000)

    Wd = rng.normal(size=(D, d)); Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    P = {"We": Wd.T.copy(), "be": np.zeros(D), "Wd": Wd, "bd": np.zeros(d)}
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    last_fire = np.zeros(D)
    for t in range(1, steps + 1):
        xb = X[rng.integers(0, len(X), 512)]
        pre = (xb - P["bd"]) @ P["We"] + P["be"]                 # encoder
        thr = np.sort(pre, 1)[:, -k_top][:, None]
        mask = (pre >= thr) & (pre > 0); z = pre * mask          # TopK + ReLU
        err = z @ P["Wd"] + P["bd"] - xb                         # decoder - input
        g = 2 * err / len(xb); gz = g @ P["Wd"].T * mask         # manual backprop
        G = {"Wd": z.T @ g, "bd": g.sum(0) - gz.sum(0) @ P["We"].T,
             "We": (xb - P["bd"]).T @ gz, "be": gz.sum(0)}
        for k in P:                                              # Adam
            m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
            P[k] -= 3e-3 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
        P["Wd"] /= np.linalg.norm(P["Wd"], axis=1, keepdims=True) + 1e-8   # unit-norm dictionary
        fired = mask.any(0); last_fire[fired] = 0; last_fire[~fired] += 1
        if t % 500 == 0 and t < steps * 0.7:     # resample dead latents onto badly reconstructed inputs
            dead = np.where(last_fire > 200)[0]
            if len(dead):
                worst = xb[np.argsort(-(err ** 2).sum(1))[:len(dead)]] - P["bd"]
                worst /= np.linalg.norm(worst, axis=1, keepdims=True) + 1e-8
                P["Wd"][dead] = worst; P["We"][:, dead] = worst.T * 0.5; P["be"][dead] = 0
                for k in m: m[k][...] = 0; v2[k][...] = 0
    cos = np.abs(T @ P["Wd"].T)          # (true features) x (learned latents)
    return cos

cos = run()
print("best |cos| per true feature:", cos.max(1).round(2))
```

## Lessons that carry over to the paper

1. **Unit-norm decoder columns:** each concept is a *direction*; magnitude lives in $z_i$.
2. **Dead latents** are a real problem. Without resampling, a first attempt with plain SGD produced one "catch-all" latent that matched many features. TopK/BatchTopK plus resampling fixes this.
3. **Overcompleteness:** with D = 32 > 12 features, extra latents stay unused or duplicate. The paper's ablation found that a *smaller* D (1,536 vs 3,072) was better for its medical data.
