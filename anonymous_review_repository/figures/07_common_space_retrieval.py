"""Plot held-out PCA--CCA retrieval against random ranking."""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import figstyle as fs

ROOT = fs.HERE.parent
DATA_DIR = ROOT / "data"
PAPER_DIR = ROOT / "outputs" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = PAPER_DIR / "common_space_retrieval.pdf"


# Average the five folds and the models inside each architecture family.
data = pd.read_csv(DATA_DIR / "E4_common_space_by_model.csv")
data = data[data["cohort"] == "desi"].groupby("family", as_index=False).agg(
    mean_test_mrr=("mean_test_mrr", "mean")
)
data = data.sort_values("mean_test_mrr")
family_names = {
    "astropt": "AstroPT", "clip": "CLIP", "convnext": "ConvNeXt",
    "dino": "DINO", "dinov3": "DINOv3", "ijepa": "I-JEPA",
    "llava": "LLaVA", "paligemma": "PaliGemma", "vit": "ViT",
    "vit-mae": "ViT-MAE", "vjepa": "V-JEPA",
}

# A random counterpart has a uniform rank from 1 to the test-set size.  This is
# the analyzed DESI count after identifier and duplicate filtering.
objects = 20321
test_sizes = [sum(i % 10 in {(2 * fold) % 10, (2 * fold + 1) % 10}
                  for i in range(objects)) for fold in range(5)]
chance = np.mean([np.sum(1 / np.arange(1, n + 1)) / n for n in test_sizes])

values = data["mean_test_mrr"].to_numpy()
mean = values.mean()

# Resample whole families for the interval.
rng = np.random.default_rng(0)
draws = np.array([rng.choice(values, len(values), replace=True).mean()
                  for _ in range(200000)])
low, high = np.quantile(draws, [0.025, 0.975])

# Exact two-sided family sign-flip test against random ranking.
differences = values - chance
null = np.array([np.mean(differences * signs)
                 for signs in product((-1, 1), repeat=len(values))])
p_value = np.mean(np.abs(null) >= abs(differences.mean()))

fs.apply_style()

figure, axis = plt.subplots(figsize=(5.5, 3.15))
y = np.arange(len(data))
axis.scatter(values, y, s=25, color=fs.SURVEY_COLOR["desi"],
             edgecolor="white", linewidth=0.6,
             zorder=3)
axis.axvline(chance, color=fs.RED, linestyle=(0, (3, 2)), linewidth=1.1)
axis.text(chance + 0.0006, len(data) - 0.1, "random ranking", color=fs.RED,
          fontsize=7, va="top")

# The bottom row shows the family-equal mean and its family-bootstrap interval.
overall_y = -1.35
axis.errorbar(mean, overall_y, xerr=[[mean - low], [high - mean]], fmt="D",
              color=fs.INK, markerfacecolor=fs.SURVEY_COLOR["desi"], markeredgecolor="white",
              markersize=5.5, linewidth=1.4, capsize=3, zorder=4)
axis.axhline(-0.65, color=fs.GRID, linewidth=0.8)

axis.set_yticks(list(y) + [overall_y],
                [family_names[name] for name in data["family"]] + ["family-equal mean"])
axis.set_ylim(-1.9, len(data) - 0.45)
axis.set_xlim(0, 0.047)
axis.set_xlabel("held-out mean reciprocal rank")
axis.tick_params(axis="y", length=0)
fs.style_axis(axis, grid="x")
figure.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.98)
figure.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.03)
plt.close(figure)

print(f"Wrote {OUTPUT}")
print(f"mean={mean:.6f} chance={chance:.6f} CI=[{low:.6f}, {high:.6f}] p={p_value:.8f}")
