import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

U = os.path.dirname(os.path.abspath(__file__)) + os.sep

rows = []
for c in ["LUMIERE", "MU-Glioma"]:
    a = pd.read_csv(f"{U}AggregateCounts_MeanStd_{c}.csv", index_col=0).loc["mean"]
    n_stable = a["True_Negative"] + a["False_Positive"]
    tn = 100 * a["True_Negative"] / n_stable
    fp = 100 * a["False_Positive"] / n_stable
    rows.append((f"{c}\n(within-cohort)", [tn, fp], n_stable))

labels = ["Correctly Stable (TN)", "False Alarm (FP)"]
colors = ["#41783b", "#a94535"]

fig, ax = plt.subplots(figsize=(11, 3.2))
for y, (name, vals, n) in zip([1, 0], rows):
    left = 0
    for i, v in enumerate(vals):
        ax.barh(y, v, left=left, height=0.55, color=colors[i], zorder=3)
        ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                color="white", fontsize=13, fontweight="bold", zorder=4)
        left += v
    ax.text(101.5, y, f"n = {n:.0f}", ha="left", va="center", fontsize=11, style="italic")

ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
ax.set_xlim(0, 100); ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("% of Non-PD (stable) patients across LOPO-held-out patients", fontsize=12)
ax.set_title("Mean Stable Patient Outcomes (Within-Cohort)", fontsize=14, fontweight="bold")
ax.grid(axis="x", color="#dddddd", zorder=0)
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)],
          loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig(f"{U}Mean_Stable_Outcomes_WithinCohort.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{U}Mean_Stable_Outcomes_WithinCohort.pdf", bbox_inches="tight")
for name, vals, n in rows:
    print(name.split()[0], [round(v, 1) for v in vals], "n =", n)