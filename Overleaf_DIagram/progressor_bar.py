import pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
U = "./"
rows = []
for c in ["LUMIERE", "MU-Glioma"]:
    a = pd.read_csv(f"{U}AggregateCounts_MeanStd_{c}.csv", index_col=0).loc["mean"]
    s = pd.read_csv(f"{U}SynchronousBreakdown_Wide_{c}.csv").fillna(0).drop(columns="Seed")
    n = a[["Early_Warning", "Synchronous", "Late_Detection", "Missed_Tumor"]].sum()
    first = s["MRI_Position_2"].mean()
    later = s.drop(columns="MRI_Position_2").sum(axis=1).mean()
    assert abs(first + later - a["Synchronous"]) < 1e-6
    rows.append((f"{c}\n(within-cohort)", [100*v/n for v in
                 (a["Early_Warning"], first, later, a["Late_Detection"], a["Missed_Tumor"])]))

labels = ["Early Warning", "Synchronous – 1st TP", "Synchronous – Later TPs", "Late Detection", "Missed"]
colors = ["#41783b", "#72c55f", "#b3cf98", "#e2a847", "#a94535"]
txtcol = ["white", "black", "black", "black", "white"]

fig, ax = plt.subplots(figsize=(11, 4))
for y, (name, vals) in zip([1, 0], rows):
    left = 0
    for i, v in enumerate(vals):
        ax.barh(y, v, left=left, height=0.55, color=colors[i],
                edgecolor="black" if i == 1 else "none", linewidth=2.2 if i == 1 else 0, zorder=3)
        ax.text(left + v/2, y, f"{v:.0f}%", ha="center", va="center", color=txtcol[i],
                fontsize=13 if i == 1 else 10, fontweight="bold", zorder=4)
        left += v
ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
ax.set_xlim(0, 100); ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("% of progressors across LOPO-held-out patients", fontsize=12)
ax.set_title("Mean Detection Timing (Within-Cohort)", fontsize=14, fontweight="bold")
ax.grid(axis="x", color="#dddddd", zorder=0)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [Patch(facecolor=c, edgecolor="black" if i == 1 else "none", linewidth=2 if i == 1 else 0, label=l)
           for i, (c, l) in enumerate(zip(colors, labels))]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig("Mean_Detection_Timing_WithinCohort.png", dpi=300, bbox_inches="tight")
plt.savefig("Mean_Detection_Timing_WithinCohort.pdf", bbox_inches="tight")
for r in rows: print(r[0].split()[0], [round(v, 1) for v in r[1]])