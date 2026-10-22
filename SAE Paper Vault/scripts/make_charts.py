"""Generate all charts for the Obsidian vault. Numbers come from the paper's tables."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, LGRAY, INK, INK2, SURF = "#9a9994", "#d9d8d3", "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": LGRAY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.family": "DejaVu Sans",
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    print("wrote", name)

def label_bars(ax, bars, fmt="{:.1f}", dy=0.6):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom",
                fontsize=9, color=INK2)

# ---------------------------------------------------------------- 1. superposition toy
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
ax = axes[0]
ax.set_title("2 neurons, 2 features: one direction each")
for ang, c, lab in [(0, BLUE, "feature A"), (90, ORANGE, "feature B")]:
    v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=c))
    ax.text(*(v * 1.15), lab, ha="center", va="center", color=INK, fontsize=10)
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal")
ax.set_xlabel("neuron 1"); ax.set_ylabel("neuron 2")
ax = axes[1]
ax.set_title("2 neurons, 5 features: superposition")
cols = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]
names = ["tumor", "gland edge", "p-zone", "t-zone", "artifact"]
for k in range(5):
    ang = 90 + k * 72
    v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=cols[k]))
    ax.text(*(v * 1.22), names[k], ha="center", va="center", fontsize=10)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
ax.set_xlabel("neuron 1"); ax.set_ylabel("neuron 2")
fig.text(0.5, 0.005, "Each neuron now responds to several features (polysemantic). "
         "This works because features are sparse: rarely active together.",
         ha="center", fontsize=9.5, color=INK2)
save(fig, "01_superposition.png")

# ---------------------------------------------------------------- 2. toy SAE trained from scratch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_test import run
n_feat, d, D, k_top = 12, 6, 32, 2
true_dirs, P, cos, losses = run(3, n_feat=n_feat, d=d, D=D, k_top=k_top)
best_per_true = cos.max(1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.7, 1]})
ax = axes[0]
ax.plot(np.convolve(losses, np.ones(50) / 50, mode="valid"), color=BLUE, lw=2); ax.set_yscale("log")
ax.set_title("Training: reconstruction loss"); ax.set_xlabel("step"); ax.set_ylabel("MSE")
ax = axes[1]
match = cos.argmax(1)
im = ax.imshow(cos[:, match], cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_title("|cos| true feature vs its best SAE latent")
ax.set_xticks(range(n_feat), [f"#{m}" for m in match], fontsize=8, rotation=45)
ax.set_xlabel(f"best-matching learned latent (out of D = {D})"); ax.set_ylabel(f"true features ({n_feat})")
ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.04)
ax = axes[2]
bars = ax.bar(range(n_feat), best_per_true, color=AQUA, width=0.7)
ax.axhline(0.9, color=GRAY, ls="--", lw=1); ax.text(n_feat - 0.5, 0.91, "0.9", color=INK2, fontsize=9, ha="right")
ax.set_ylim(0, 1.05); ax.set_xticks(range(n_feat)); ax.set_title("Best match per true feature")
ax.set_xlabel("true feature"); ax.set_ylabel("max |cos|")
fig.suptitle(f"Toy experiment: {n_feat} sparse features squeezed into {d} dims, recovered by a TopK SAE (k={k_top})",
             fontsize=12, color=INK2)
save(fig, "02_toy_sae_recovery.png")
print("toy SAE: mean best |cos| =", best_per_true.mean().round(3), " recovered >0.9:", (best_per_true > 0.9).sum())

# ---------------------------------------------------------------- 3. dense vs sparse code
rng2 = np.random.default_rng(3)
h = rng2.normal(size=96)
z = np.zeros(192); idx = rng2.choice(192, 8, replace=False); z[idx] = rng2.uniform(0.6, 2.0, 8)
fig, axes = plt.subplots(2, 1, figsize=(11, 4.2))
axes[0].bar(range(96), h, color=GRAY, width=0.8)
axes[0].set_title("Dense ViT activation h  (d = 768 in the paper; 96 shown): every entry non-zero, meanings mixed")
axes[0].set_xticks([]); axes[0].set_ylabel("value")
axes[1].bar(range(192), z, color=BLUE, width=0.9)
for i in idx:
    axes[1].text(i, z[i] + 0.08, f"#{i}", ha="center", fontsize=8, color=INK2)
axes[1].set_title("Sparse SAE code z  (D = 1,536 in the paper; 192 shown): only k = 8 concepts on")
axes[1].set_xticks([]); axes[1].set_ylabel("activation"); axes[1].set_ylim(0, 2.4)
save(fig, "03_dense_vs_sparse.png")

# ---------------------------------------------------------------- 4. Cohen's d illustration
x = np.linspace(-4, 6, 500)
def npdf(x, m, s=1): return np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, dval, title in [(axes[0], 1.8, "In-distribution (PI-CAI): Cohen's d ≈ 1.8"),
                        (axes[1], 1.2, "Zero-shot (Prostate158): Cohen's d ≈ 1.2")]:
    ax.fill_between(x, npdf(x, 0), color=ORANGE, alpha=0.25); ax.plot(x, npdf(x, 0), color=ORANGE, lw=2, label="failure cases")
    ax.fill_between(x, npdf(x, dval), color=BLUE, alpha=0.25); ax.plot(x, npdf(x, dval), color=BLUE, lw=2, label="success cases")
    ax.set_title(title); ax.set_xlabel("# active SAE concepts / mean activation (standardised)")
    ax.set_yticks([])
axes[0].legend(frameon=False, loc="upper left")
fig.text(0.5, 0.005, "Illustrative normal curves with the effect sizes the paper reports, not the paper's raw data.",
         ha="center", fontsize=9, color=INK2)
save(fig, "04_cohens_d.png")

# ---------------------------------------------------------------- 5. per-concept AUROC (Fig 3)
layers = {
    "Layer 3 (early)": [("#1501", -0.063), ("#552", -0.063), ("#134", 0.079), ("#850", 0.091), ("#1300", 0.104)],
    "Layer 7 (mid)":   [("#698", -0.099), ("#240", -0.075), ("#1455", -0.057), ("#163", -0.056), ("#350", 0.070)],
    "Layer 11 (deep)": [("#1282", -0.118), ("#227", -0.090), ("#241", 0.178), ("#1029", 0.183), ("#469 'tumor'", 0.205)],
}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharex=True)
for ax, (name, items) in zip(axes, layers.items()):
    labs = [a for a, _ in items]; vals = [b for _, b in items]
    cs = [ORANGE if v < 0 else BLUE for v in vals]
    ax.barh(labs, vals, color=cs, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (0.006 if v > 0 else -0.006), i, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=1); ax.set_title(name); ax.set_xlim(-0.2, 0.27)
    ax.set_xlabel("AUROC − 0.5"); ax.invert_yaxis()
fig.suptitle("Top-5 most discriminative SAE concepts per layer (PI-CAI), re-plotted from Fig. 3\n"
             "orange = failure-predictive (AUROC < 0.5)    blue = success-predictive (AUROC > 0.5)",
             fontsize=12, color=INK2)
save(fig, "05_concept_auroc_by_layer.png")

# ---------------------------------------------------------------- 6. zero-shot table 1
methods = ["MaxProb", "MeanProb", "Entropy", "Energy", "Ours"]
fpr = [100.0, 100.0, 97.8, 100.0, 64.4]
auroc = [50.0, 52.6, 53.1, 48.0, 73.6]
aupr = [64.4, 65.1, 65.6, 63.6, 74.1]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, vals, t in [(axes[0], auroc, "AUROC ↑"), (axes[1], aupr, "AUPR ↑"), (axes[2], fpr, "FPR95 ↓")]:
    cs = [LGRAY] * 4 + [BLUE]
    b = ax.bar(methods, vals, color=cs, width=0.65)
    label_bars(ax, b, dy=1)
    ax.set_title(t); ax.set_ylim(0, 112); ax.tick_params(axis="x", rotation=20)
    if "AUROC" in t:
        ax.axhline(50, color=GRAY, ls="--", lw=1); ax.text(4.4, 51.5, "random", fontsize=8.5, color=INK2, ha="right")
fig.suptitle("Zero-shot failure detection on unseen Prostate158 (detectors trained on PI-CAI), Table 1",
             fontsize=12, color=INK2)
save(fig, "06_zero_shot_table1.png")

# ---------------------------------------------------------------- 7. ablations: layers, classifier, SAE size
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
groups = ["Early\n(1,3)", "Mid\n(5,7)", "Deep\n(9,11)", "All 6"]
dsc = [47.5, 48.8, 49.6, 50.1]; f1 = [44.5, 50.4, 57.6, 63.5]
xx = np.arange(4); w = 0.36
b1 = ax.bar(xx - w / 2 - 0.01, dsc, w, color=AQUA, label="seg. DSC")
b2 = ax.bar(xx + w / 2 + 0.01, f1, w, color=BLUE, label="failure F1")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(xx, groups); ax.set_ylim(0, 80); ax.set_title("Which layers? (Table 3)"); ax.legend(frameon=False, loc="upper left", ncol=2)
ax = axes[1]
groups = ["XGBoost", "Logistic", "Decision\nTree", "Random\nForest"]
dsc = [50.1, 47.8, 48.5, 44.6]; f1 = [63.5, 59.2, 58.7, 57.0]
b1 = ax.bar(xx - w / 2 - 0.01, dsc, w, color=AQUA, label="seg. DSC")
b2 = ax.bar(xx + w / 2 + 0.01, f1, w, color=BLUE, label="failure F1")
label_bars(ax, b1); label_bars(ax, b2)
ax.set_xticks(xx, groups); ax.set_ylim(0, 80); ax.set_title("Which classifier? (Table 4)"); ax.legend(frameon=False, loc="upper left", ncol=2)
ax = axes[2]
f1grid = np.array([[60.3, 63.5, 62.5], [58.4, 57.4, 62.6]])
dscgrid = np.array([[49.7, 50.1, 49.9], [50.1, 49.7, 49.6]])
im = ax.imshow(f1grid, cmap="Blues", vmin=55, vmax=65)
for i in range(2):
    for j in range(3):
        ax.text(j, i, f"F1 {f1grid[i, j]:.1f}\nDSC {dscgrid[i, j]:.1f}", ha="center", va="center",
                fontsize=9.5, color="white" if f1grid[i, j] > 62 else INK,
                fontweight="bold" if (i, j) == (0, 1) else "normal")
ax.set_xticks(range(3), ["k = 4", "k = 8", "k = 16"]); ax.set_yticks(range(2), ["D = 1,536\n(2×768)", "D = 3,072\n(4×768)"])
ax.grid(False); ax.set_title("SAE size & sparsity (Table 2)")
ax.spines[:].set_visible(False)
save(fig, "07_ablations.png")

# ---------------------------------------------------------------- 8. latency
fig, ax = plt.subplots(figsize=(9, 3.6))
cfg = ["MedSAM alone", "+ 6 SAE encoders\n(routine)", "+ 6 SAE enc+dec\n(intervention)", "+ XGBoost\n(full pipeline)"]
lat = [126.3, 130.9, 134.3, 176.5]; extra = ["", "+3.7%", "+6.3%", "+39.7%"]
b = ax.barh(cfg, lat, color=[LGRAY, BLUE, BLUE, BLUE], height=0.55)
for i, (v, e) in enumerate(zip(lat, extra)):
    ax.text(v + 2, i, f"{v:.1f} ms  {e}", va="center", fontsize=9.5, color=INK2)
ax.invert_yaxis(); ax.set_xlim(0, 215); ax.set_xlabel("per-image latency, ms (RTX A6000, 1024² input)")
ax.set_title("Computation overhead (Table 5)")
save(fig, "08_latency.png")

# ---------------------------------------------------------------- 9. causal intervention
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
modes = ["SAE round-trip\n(no edit)", "amplify random\nnon-tumor latent ×2", "amplify #469\n'tumor' ×2"]
vals = [0.013, 0.035, -0.173]
b = ax.bar(modes, vals, color=[LGRAY, LGRAY, ORANGE], width=0.55)
for bb, v in zip(b, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, v + (0.006 if v > 0 else -0.006), f"{v:+.3f}",
            ha="center", va="bottom" if v > 0 else "top", fontsize=10, color=INK2)
ax.axhline(0, color=INK2, lw=1); ax.set_ylim(-0.21, 0.07); ax.set_ylabel("Δ DSC vs. raw model")
ax.set_title("Steering a single concept (Table 6)")
ax = axes[1]
ex = ["example 1", "example 2"]; before = [0.409, 0.710]; after = [0.200, 0.339]
xx = np.arange(2); w = 0.34
b1 = ax.bar(xx - w / 2 - 0.01, before, w, color=BLUE, label="original")
b2 = ax.bar(xx + w / 2 + 0.01, after, w, color=ORANGE, label="#469 amplified")
label_bars(ax, b1, "{:.3f}", 0.01); label_bars(ax, b2, "{:.3f}", 0.01)
ax.set_xticks(xx, ex); ax.set_ylim(0, 0.85); ax.set_ylabel("DSC"); ax.legend(frameon=False)
ax.set_title("Per-image effect (Fig. 12)")
save(fig, "09_causal_intervention.png")

# ---------------------------------------------------------------- 10. correction examples + TTA
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
ax = axes[0]
cases = ["PI-CAI a", "PI-CAI b", "PanTS a", "PanTS b"]
before = [74.1, 78.5, 29.3, 54.5]; after = [86.5, 85.0, 40.7, 68.9]
for i, (bf, af) in enumerate(zip(before, after)):
    ax.plot([bf, af], [i, i], color=LGRAY, lw=3, zorder=1)
    ax.scatter(bf, i, s=80, color=GRAY, zorder=2, edgecolor=SURF, linewidth=2)
    ax.scatter(af, i, s=80, color=AQUA, zorder=3, edgecolor=SURF, linewidth=2)
    ax.text(af + 1.5, i, f"+{af - bf:.1f}", va="center", fontsize=9.5, color=INK2)
ax.set_yticks(range(4), cases); ax.invert_yaxis(); ax.set_xlim(20, 100); ax.set_xlabel("Dice (DSC)")
ax.scatter([], [], color=GRAY, label="before correction"); ax.scatter([], [], color=AQUA, label="after FP-patch removal")
ax.legend(frameon=False, loc="lower right"); ax.set_title("Patch-level failure correction (Fig. 7)")
ax = axes[1]
meth = ["MedSAM base", "TTA (K=6)", "Ours (SAE)"]
f1 = [0.359, 0.592, 0.635]; dsc = [0.513, 0.434, 0.501]
xx = np.arange(3)
b1 = ax.bar(xx - w / 2 - 0.01, f1, w, color=BLUE, label="failure F1")
b2 = ax.bar(xx + w / 2 + 0.01, dsc, w, color=AQUA, label="DSC after correction")
label_bars(ax, b1, "{:.3f}", 0.01); label_bars(ax, b2, "{:.3f}", 0.01)
ax.set_xticks(xx, ["MedSAM base\n1× cost", "TTA (K=6)\n6× cost", "Ours\n1× + 6 SAEs"]); ax.set_ylim(0, 0.8)
ax.legend(frameon=False, loc="upper left"); ax.set_title("vs. test-time augmentation (Table 7)")
save(fig, "10_correction_and_tta.png")

# ---------------------------------------------------------------- 11. importance (Fig 8)
fig, ax = plt.subplots(figsize=(9, 3.6))
cn = ["L11 #1029  'tumor' (partial)", "L11 #241  'tumor in transition zone'",
      "L11 #469  'tumor in TZ & PZ'", "L9 #1423  'gland margin'", "L5 #478  'outside the gland'"]
imp = [0.050, 0.045, 0.031, 0.025, 0.021]
b = ax.barh(cn, imp, color=[BLUE, BLUE, BLUE, AQUA, AQUA], height=0.55)
for i, v in enumerate(imp):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9.5, color=INK2)
ax.invert_yaxis(); ax.set_xlim(0, 0.06); ax.set_xlabel("XGBoost importance")
ax.set_title("Top-5 concepts the failure detector relies on (Fig. 8)")
ax.text(0.059, 4.3, "blue = tumor concepts, green = anatomy context", ha="right", fontsize=9, color=INK2)
save(fig, "11_concept_importance.png")

# ---------------------------------------------------------------- 12. tradeoff schematic
fig, ax = plt.subplots(figsize=(7.5, 4.6))
dsc_b = np.array([50, 48.5, 46.5, 44, 41.5, 40]); f1_b = np.array([36, 42, 48, 53, 56, 57])
ax.plot(dsc_b, f1_b, color=GRAY, lw=2, marker="o", ms=7, label="output-based baselines\n(sweep threshold 0.5→0.8)")
ax.scatter([50.1], [63.5], s=160, color=BLUE, zorder=5, edgecolor=SURF, linewidth=2, label="Ours (internal concepts)")
ax.annotate("", xy=(42, 57), xytext=(50, 37), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
ax.text(46.6, 44, "+20 F1 costs\n8–10 DSC", color=INK2, fontsize=9.5)
ax.set_xlabel("segmentation DSC →"); ax.set_ylabel("failure-detection F1 →")
ax.set_title("The sensitivity–quality trade-off (schematic)")
ax.legend(frameon=False, loc="lower left", fontsize=9)
fig.text(0.5, 0.005, "Schematic: baseline curve shape is illustrative; 'Ours' uses the paper's PI-CAI numbers (DSC 50.1, F1 63.5).",
         ha="center", fontsize=8.5, color=INK2)
save(fig, "12_tradeoff_schematic.png")
