"""Top radiomic features driving each learned progression vector, LUMIERE + MU-Glioma.

Layout: 2 x 2 (rows = cohorts, columns = progression vectors PV1, PV2).
Bars   : median rank of each feature across LOPO folds (lower = more important),
         with its bootstrap 95% CI as an error bar.
Colour : tumour sub-region of the feature (same colour in every panel).
Panel titles take the dominant sub-region from the data, because PV numbering
comes from K-Means cluster labels and is arbitrary between runs.

Usage:
    python feature_rank_figure.py            # expects the four RankStability_*.csv files here
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TOP_N = 12
# Shape features do not depend on the MRI sequence; the MU-Glioma extraction stores one
# identical copy per sequence. Keep one copy (sequence shown as "all") so they don't fill 4 rows.
DEDUPE_SHAPE = True
# Label abbreviations (define in the caption: GL = gray-level).
ABBREV = [("Gray Level", "GL"), ("Non Uniformity Normalized", "Non-Uniformity Norm."),
          ("Non Uniformity", "Non-Uniformity"), ("Large Dependence", "Large Dep."),
          ("Small Dependence", "Small Dep.")]
FILES = {
    ("LUMIERE", "PV1"): "RankStability_ProgressionVector1_LUMIERE.csv",
    ("LUMIERE", "PV2"): "RankStability_ProgressionVector2_LUMIERE.csv",
    ("MU-Glioma", "PV1"): "RankStability_ProgressionVector1_MU-Glioma.csv",
    ("MU-Glioma", "PV2"): "RankStability_ProgressionVector2_MU-Glioma.csv",
}
COHORTS = ["LUMIERE", "MU-Glioma"]
VECTORS = ["PV1", "PV2"]

REGIONS = {"Contrast-Enhancing": "#b03a2e", "ContrastEnhancing": "#b03a2e",
           "Necrosis": "#2e6f9e", "Edema": "#8e7d1f"}
REGION_DISPLAY = {"ContrastEnhancing": "Contrast-Enhancing"}
SEQUENCES = ("CT1", "T1", "T2", "FLAIR")
FAMILY_DISPLAY = {"firstorder": "First-order", "glcm": "GLCM", "glrlm": "GLRLM",
                  "glszm": "GLSZM", "gldm": "GLDM", "ngtdm": "NGTDM", "shape": "Shape"}

FONT = {"base": 9, "title": 10.5, "suptitle": 11.5}


def split_camel(name):
    """'HighGrayLevelRunEmphasis' -> 'High Gray Level Run Emphasis'; '90Percentile' -> '90th Percentile'."""
    name = re.sub(r"^(\d+)Percentile$", r"\1th Percentile", name)
    name = re.sub(r"(?<=[a-z])(?=[23]D)", " ", name)                  # 'Maximum3DDiameter' -> 'Maximum 3DDiameter'
    name = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)  # 'Imc1' stays 'Imc1'
    name = re.sub(r"([23]) D\b", r"\1D", name)
    for long, short in ABBREV:
        name = name.replace(long, short)
    return name


def parse_feature(raw):
    """original_<family>_<Feature>_<Region>_<Sequence> -> (family, feature, region, sequence)."""
    parts = raw.split("_")
    if parts[0] == "original":
        parts = parts[1:]
    family, feature, sequence = parts[0], parts[1], parts[-1]
    region = "_".join(parts[2:-1])
    if sequence not in SEQUENCES:
        raise ValueError(f"Unexpected sequence in feature name: {raw}")
    return FAMILY_DISPLAY.get(family.lower(), family.upper()), split_camel(feature), region, sequence


def load(path):
    d = pd.read_csv(path).sort_values(["median_rank", "ci_width"]).copy()
    parsed = d["feature"].map(parse_feature)
    d["family"], d["name"], d["region"], d["sequence"] = zip(*parsed)
    if DEDUPE_SHAPE:
        is_shape = d["family"] == "Shape"
        first = ~d[is_shape].duplicated(["name", "region"])
        n_dropped = int(is_shape.sum() - first.sum())
        d = pd.concat([d[~is_shape], d[is_shape][first].assign(sequence="all")]).sort_values(
            ["median_rank", "ci_width"])
        if n_dropped:
            print(f"{path}: collapsed {n_dropped} per-sequence duplicate shape feature(s)")
    return d.head(TOP_N).copy()


panels = {key: load(path) for key, path in FILES.items()}
xmax = 1.08 * max(p["ci_95_upper"].max() for p in panels.values())

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FONT["base"], "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": FONT["title"], "axes.titleweight": "bold",
})
fig = plt.figure(figsize=(7.5, 10.2), layout="constrained")   # final print size: full text width
# One shared grid so columns line up across cohorts: [header, panels] per cohort
gs = fig.add_gridspec(2 * len(COHORTS), 2, height_ratios=[0.05, 1] * len(COHORTS))

for r, cohort in enumerate(COHORTS):
    n_folds = int(panels[(cohort, "PV1")]["n_folds"].iloc[0])
    header = fig.add_subplot(gs[2 * r, :])
    header.axis("off")
    header.text(0.5, 0.0, f"{cohort}  ·  {n_folds} LOPO folds", ha="center", va="bottom",
                fontsize=FONT["suptitle"], fontweight="bold", transform=header.transAxes)
    for c, vec in enumerate(VECTORS):
        d = panels[(cohort, vec)].iloc[::-1]          # best rank at the top
        ax = fig.add_subplot(gs[2 * r + 1, c])
        regions = d["region"].map(lambda x: REGION_DISPLAY.get(x, x))
        dominant = regions.value_counts().idxmax()
        mixed = regions.nunique() > 1
        # two-line labels: feature name / family · sequence (keeps both columns readable at print width)
        labels = [f"{n}\n{f} · {s}" + (f" · {reg[:3]}" if mixed else "")
                  for f, n, s, reg in zip(d["family"], d["name"], d["sequence"], regions)]
        colors = [REGIONS.get(reg, "#777777") for reg in d["region"]]
        y = range(len(d))
        ax.barh(y, d["median_rank"], color=colors, height=0.72, alpha=0.9)
        ax.errorbar(d["median_rank"], y,
                    xerr=[d["median_rank"] - d["ci_95_lower"], d["ci_95_upper"] - d["median_rank"]],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_yticks(list(y)); ax.set_yticklabels(labels)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", color="0.88", lw=0.6); ax.set_axisbelow(True)
        ax.set_title(f"{vec} · {dominant}" + (" (mixed)" if mixed else ""), loc="left")
        if r == len(COHORTS) - 1:
            ax.set_xlabel("Median rank across LOPO folds\n(95% CI; lower = more important)")

used = sorted({REGION_DISPLAY.get(x, x) for p in panels.values() for x in p["region"]})
fig.legend(handles=[Patch(color=REGIONS[r], label=r) for r in used], loc="outside lower center",
           ncol=len(used), frameon=False, title="Tumour sub-region")

fig.savefig("feature_rank_progression_vectors.pdf", bbox_inches="tight")
fig.savefig("feature_rank_progression_vectors.png", dpi=300, bbox_inches="tight")
print("Saved feature_rank_progression_vectors.pdf / .png")