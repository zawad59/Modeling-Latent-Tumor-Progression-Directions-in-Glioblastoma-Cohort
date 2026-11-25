"""Robustness figure for the drift pipeline, LUMIERE + MU-Glioma in one diagram.

Rows    : one per dataset.
Panel a : where does the variation in the selected weight lambda* come from?
          Box 1 = SD of lambda* across held-out patients, within each shuffle (one value per shuffle).
          Box 2 = SD of lambda* across shuffles, within each held-out patient (one value per patient).
Panel b : per-patient outcome stability across the 100 shuffles, binned: one stacked bar per
          patient group (progressors, stable), split by the share of shuffles in which the
          patient receives their most frequent clinical category.
sigma_ST stability is reported in a separate table; its numbers are printed below.

Usage:
    python robustness_figure_bc.py FullAudit_AllSeeds_LUMIERE.csv FullAudit_AllSeeds_MU-Glioma.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATASETS = [
    ("LUMIERE", sys.argv[1] if len(sys.argv) > 1 else "FullAudit_AllSeeds_LUMIERE.csv"),
    ("MU-Glioma", sys.argv[2] if len(sys.argv) > 2 else "FullAudit_AllSeeds_MU-Glioma.csv"),
]
BOX_COLORS = ["#5b8db8", "#e39c37"]   # across patients | across shuffles
RNG = np.random.default_rng(0)       # jitter only

# Agreement bands for panel (b): share of the 100 shuffles giving the patient's most frequent category
BANDS = [(1.00, 1.001, "100%"), (0.90, 1.00, "90–99%"), (0.75, 0.90, "75–89%"),
         (0.50, 0.75, "50–74%"), (0.00, 0.50, "<50%")]
BAND_COLORS = ["#1b5e6f", "#3f8f9f", "#8cc0c7", "#f0c27b", "#d9603b"]
DARK_BANDS = {BAND_COLORS[0], BAND_COLORS[1], BAND_COLORS[4]}   # use white text on these
INSIDE_MIN = 8   # segments narrower than this (% of bar) get their label above the bar


def load(path):
    """Keep only real result rows (drops blank/summary footer rows such as
    'Mean Lambda = ...') and coerce numeric columns."""
    d = pd.read_csv(path)
    d = d[d["Seed"].notna() & d["PatientID"].notna()].copy()
    d["Seed"] = d["Seed"].astype(int)
    for c in ("Best_k", "Std_St", "Threshold"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Is_Progressor"] = d["Is_Progressor"].astype(str).str.strip().str.lower().eq("true")
    dup = d.duplicated(["Seed", "PatientID"])
    if dup.any():
        raise ValueError(f"{path}: {dup.sum()} duplicate (Seed, PatientID) rows")
    return d


def stats(d):
    sig = d.groupby("PatientID")["Std_St"].first()
    modal = d.groupby("PatientID")["Clinical_Category"].agg(lambda x: x.value_counts(normalize=True).iloc[0])
    triggered = ~d["Clinical_Category"].isin(["Missed Tumor", "Correctly identified CR/SD"])
    trig_rate = triggered.groupby(d["PatientID"]).mean()
    return {
        "n_seeds": d["Seed"].nunique(),
        "n_folds": d["PatientID"].nunique(),
        "sigma_cv": 100 * sig.std() / sig.mean(),
        "sigma_seed_invariant": bool((d.groupby("PatientID")["Std_St"].nunique() == 1).all()),
        "sd_across_patients": d.groupby("Seed")["Best_k"].std(),        # one value per shuffle
        "sd_across_shuffles": d.groupby("PatientID")["Best_k"].std(),   # one value per patient
        "lam_mean": d["Best_k"].mean(),
        "modal": modal,
        "n_full": int((modal == 1).sum()),
        "n_trigger_consistent": int(((trig_rate == 0) | (trig_rate == 1)).sum()),
        "prog": d.groupby("PatientID")["Is_Progressor"].first(),
    }


data = [(name, load(path)) for name, path in DATASETS]
all_stats = [stats(d) for _, d in data]
ymax_a = 1.1 * max(max(s["sd_across_patients"].max(), s["sd_across_shuffles"].max()) for s in all_stats)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(len(data), 2, figsize=(10.5, 3.3 * len(data)),
                         gridspec_kw={"width_ratios": [0.8, 1], "wspace": 0.45, "hspace": 0.55})

for r, ((name, d), s) in enumerate(zip(data, all_stats)):
    ax_a, ax_b = axes[r]

    # ---- (a) sources of lambda* variation
    groups = [s["sd_across_patients"].values, s["sd_across_shuffles"].values]
    bp = ax_a.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                      medianprops={"color": "k", "lw": 1.2})
    for patch, col in zip(bp["boxes"], BOX_COLORS):
        patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
    for i, (vals, col) in enumerate(zip(groups, BOX_COLORS), start=1):
        ax_a.scatter(i + RNG.uniform(-0.12, 0.12, len(vals)), vals, s=7, color=col, alpha=0.7, lw=0)
    ax_a.set_xticks([1, 2])
    ax_a.set_xticklabels([f"Across held-out patients\n(within shuffle, n={s['n_seeds']})",
                          f"Across shuffles\n(within patient, n={s['n_folds']})"])
    ax_a.set_ylim(0, ymax_a)
    ax_a.set_ylabel(f"{name}\n\nSD of selected $\\lambda^*$")
    ax_a.set_title(("(a) Source of $\\lambda^*$ variation\n" if r == 0 else "") +
                   f"median SD {np.median(groups[0]):.2f} vs {np.median(groups[1]):.2f}", fontsize=9)

    # ---- (b) per-patient outcome stability, binned: one stacked bar per patient group
    groups_b = [("Progressors", s["modal"][s["prog"]]), ("Stable", s["modal"][~s["prog"]])]
    for yi, (gname, vals) in enumerate(groups_b):
        left = 0.0
        for (lo, hi, lab), col in zip(BANDS, BAND_COLORS):
            share = 100 * ((vals >= lo) & (vals < hi)).mean()
            if share > 0:
                ax_b.barh(yi, share, left=left, color=col, height=0.6, edgecolor="white", lw=0.8)
                x_mid = left + share / 2
                if share >= INSIDE_MIN:     # wide enough: label inside the segment
                    ax_b.text(x_mid, yi, f"{share:.0f}%", ha="center", va="center",
                              color="white" if col in DARK_BANDS else "black",
                              fontsize=8.5, fontweight="bold")
                else:                       # narrow segment: label above the bar with a leader line
                    ax_b.plot([x_mid, x_mid], [yi - 0.31, yi - 0.40], color="0.3", lw=0.7)
                    ax_b.text(x_mid, yi - 0.42, f"{share:.0f}%", ha="center", va="bottom",
                              color="0.15", fontsize=8, fontweight="bold")
            left += share
    ax_b.set_yticks([0, 1])
    ax_b.set_yticklabels([f"{g}\n(n={len(v)})" for g, v in groups_b])
    ax_b.set_ylim(1.5, -0.75)          # inverted, with room above bars for outside labels
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("% of patients")
    ax_b.spines["left"].set_visible(False); ax_b.tick_params(axis="y", length=0)
    ax_b.set_title(("(b) Per-patient outcome stability across 100 shuffles\n" if r == 0 else "") +
                   f"{s['n_full']}/{s['n_folds']} patients identical in all shuffles", fontsize=9)
    if r == 0:
        ax_b.legend(handles=[Patch(color=c, label=l) for (_, _, l), c in zip(BANDS, BAND_COLORS)],
                    title="Shuffles giving the patient's\nmost frequent category", frameon=False,
                    loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, title_fontsize=8.5)

    print(f"{name}: sigma_ST CV={s['sigma_cv']:.2f}% (seed-invariant: {s['sigma_seed_invariant']}); "
          f"lambda mean={s['lam_mean']:.2f}; SD across patients (median)={np.median(groups[0]):.3f}, "
          f"across shuffles (median)={np.median(groups[1]):.3f}; "
          f"category identical in all shuffles: {s['n_full']}/{s['n_folds']}; "
          f"trigger decision identical in all shuffles: {s['n_trigger_consistent']}/{s['n_folds']}")

fig.savefig("pipeline_robustness_bc.pdf", bbox_inches="tight")
fig.savefig("pipeline_robustness_bc.png", dpi=300, bbox_inches="tight")
print("Saved pipeline_robustness_bc.pdf / .png")