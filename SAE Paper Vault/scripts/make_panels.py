import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
I, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def L(f): return Image.open(os.path.join(I, f)).convert("RGB")

def grid(rows, col_titles, row_titles, name, title, foot=None, size=2.3):
    nr, nc = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(nr, nc, figsize=(size * nc + (1.2 if row_titles else 0), size * nr + 0.9), squeeze=False)
    for i, r in enumerate(rows):
        for j in range(nc):
            ax = axes[i][j]; ax.axis("off")
            if j < len(r) and r[j]:
                f, cap = r[j] if isinstance(r[j], tuple) else (r[j], None)
                ax.imshow(L(f))
                if cap: ax.text(0.97, 0.04, cap, transform=ax.transAxes, ha="right", va="bottom", color="white",
                                fontsize=9, bbox=dict(fc="black", ec="none", alpha=0.65, pad=2))
            if i == 0 and col_titles: ax.set_title(col_titles[j], fontsize=10, color=INK)
        if row_titles:
            axes[i][0].text(-0.08, 0.5, row_titles[i], transform=axes[i][0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
    fig.suptitle(title, fontsize=12, color=INK2)
    if foot: fig.text(0.5, 0.005, foot, ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03 if foot else 0, 1, 1))
    fig.savefig(os.path.join(OUT, name), dpi=150); plt.close(fig); print("wrote", name)

# Fig 2 — success vs failure concept maps on PI-CAI
grid([["p3_0_Im1.jpg", "p3_1_Im10.jpg", "p3_5_Im3.jpg", "p3_4_Im2.jpg", "p3_6_Im4.jpg", "p3_2_Im11.png"],
      ["p3_7_Im5.jpg", "p3_8_Im6.jpg", "p3_9_Im7.jpg", "p3_11_Im9.jpg", "p3_10_Im8.jpg", "p3_3_Im12.png"]],
     ["predicted mask", '#469 "tumor"', '#968 "p-zone"', '#204 "t-zone"', '#1097 "anterior"', "activation bars"],
     ["Correct mask (TP)", "Wrong mask (FP)"], "fig_paper_fig2_success_vs_failure.png",
     "Paper Fig. 2 — concept activation maps for a success case and a failure case (PI-CAI prostate MRI)",
     "Re-assembled from images embedded in the PDF; panel-to-column assignment inferred from the layout. "
     "Key point: the 'tumor' concept #469 lights up only in the success case.")

# Fig 7 — failure correction
cols = ["prediction (red) + GT (yellow)", "SAE concept localization", "detected FP region", "corrected (green) + GT"]
grid([[("p12_10_Im4.jpg", "DSC 74.1"), "p12_0_Im1.jpg", "p12_8_Im2.jpg", ("p12_9_Im3.jpg", "DSC 86.5")],
      [("p12_11_Im5.jpg", "DSC 78.5"), "p12_12_Im6.jpg", "p12_13_Im7.jpg", ("p12_14_Im8.jpg", "DSC 85.0")]],
     cols, ["PI-CAI ex. 1", "PI-CAI ex. 2"], "fig_paper_fig7_correction_mri.png",
     "Paper Fig. 7 (left) — patch-level failure correction on prostate MRI")
grid([[("p12_15_Im9.jpg", "DSC 29.3"), "p12_1_Im10.jpg", "p12_2_Im11.jpg", ("p12_3_Im12.jpg", "DSC 40.7")],
      [("p12_4_Im13.jpg", "DSC 54.5"), "p12_5_Im14.jpg", "p12_6_Im15.jpg", ("p12_7_Im16.jpg", "DSC 68.9")]],
     cols, ["PanTS ex. 1", "PanTS ex. 2"], "fig_paper_fig7_correction_ct.png",
     "Paper Fig. 7 (right) — patch-level failure correction on pancreatic CT")

# Fig 8 — concept heatmaps used by the detector
grid([["p12_16_Im1.jpg", "p12_17_Im10.jpg", "p12_18_Im2.jpg", "p12_19_Im3.jpg", "p12_20_Im4.jpg"],
      ["p12_21_Im5.jpg", "p12_22_Im6.jpg", "p12_23_Im7.jpg", "p12_24_Im8.jpg", "p12_25_Im9.jpg"]],
     None, None, "fig_paper_fig8_concept_heatmaps.png",
     "Paper Fig. 8 — localization heatmaps of the top failure-predictive concepts (PI-CAI)",
     "Tumor concepts (#1029, #241, #469) focus on lesions; #1423 traces the gland margin; #478 fires outside the gland.")

# Fig 4 — pancreatic concept maps + correction
grid([["p6_15_Im7.jpg", "p6_16_Im8.jpg", "p6_17_Im9.jpg", "p6_1_Im10.jpg", "p6_2_Im11.jpg", "p6_3_Im12.jpg"],
      ["p6_4_Im13.jpg", "p6_5_Im14.jpg", "p6_6_Im15.jpg", "p6_7_Im16.jpg", "p6_8_Im17.jpg", "p6_9_Im18.jpg"]],
     None, None, "fig_paper_fig4_pants_concepts.png",
     "Paper Fig. 4 — SAE concept maps on pancreatic CT (concepts such as 'pancreatic tumor', 'pancreas head', 'kidneys', 'liver')",
     "Each heatmap is one SAE latent's activation over the patch grid, upsampled onto the CT slice.")
grid([[("p6_0_Im1.jpg", None), ("p6_10_Im2.jpg", "DSC 38.6"), "p6_11_Im3.jpg", ("p6_12_Im4.jpg", "DSC 66.2")]],
     ["input CT + mask", "prediction (red) vs GT (yellow)", "concept localization", "after correction (green)"],
     None, "fig_paper_fig4_pants_correction.png", "Paper Fig. 4 — end-to-end example on PanTS")

# Fig 12 — causal amplification
grid([[("p22_0_Im1.jpg", "DSC 0.409"), ("p22_1_Im2.jpg", "DSC 0.200"), ("p22_2_Im3.jpg", "DSC 0.710"), ("p22_3_Im4.jpg", "DSC 0.339")]],
     ["ex. 1 original", "ex. 1 #469 amplified ×2", "ex. 2 original", "ex. 2 #469 amplified ×2"], None,
     "fig_paper_fig12_amplification.png",
     "Paper Fig. 12 — amplifying the 'tumor' latent #469 causes extra false positives (GT yellow, prediction green/red)")

# Brain (Fig 10, Fig 13) crops from full-page JP2 renders
b = Image.open(os.path.join(I, "p20_0_X0.jp2")).convert("RGB"); s = 3840 / 1600
b.crop((int(295 * s), int(215 * s), int(875 * s), int(398 * s))).save(os.path.join(OUT, "fig_paper_fig10_brain_concepts.png")); print("wrote fig10")
b = Image.open(os.path.join(I, "p23_0_X0.jp2")).convert("RGB")
b.crop((int(100 * s), int(12 * s), int(712 * s), int(882 * s))).resize((900, 1280)).save(os.path.join(OUT, "fig_paper_fig13_brain_correction.png")); print("wrote fig13")
