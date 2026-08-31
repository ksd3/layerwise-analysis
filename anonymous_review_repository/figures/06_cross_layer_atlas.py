"""Appendix atlas comparing every HSC block with every paired-survey block."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
PAPER_DIR = ROOT / "outputs" / "figures"

SURVEYS = ["cosmosweb", "jwst", "legacysurvey"]
METRICS = {
    "corrected_overlap_20": "MKNN",
    "cka": "CKA",
    "procrustes": "Procrustes",
}
BLOCK_PATTERN = re.compile(r"(?:^|\.)(?:h|layer|layers)\.\d+$")
def block_metadata(all_states):
    """Return the original layer number and block-relative depth."""
    states = all_states[all_states["layer_name"].map(
        lambda name: bool(BLOCK_PATTERN.search(str(name)))
    )].copy()
    states = states.sort_values(["cohort", "model", "layer"])
    states["block_number"] = states.groupby(["cohort", "model"]).cumcount()
    states["block_count"] = states.groupby(["cohort", "model"])["layer"].transform("size")
    states["block_depth"] = (
        states["block_number"] / (states["block_count"] - 1).clip(lower=1)
    )
    return states


def resize_matrix(matrix, old_depth, size=41):
    """Bilinearly interpolate one block map onto a common depth grid."""
    new_depth = np.linspace(0, 1, size)
    across_columns = np.array([
        np.interp(new_depth, old_depth, row) for row in matrix
    ])
    return np.array([
        np.interp(new_depth, old_depth, across_columns[:, column])
        for column in range(size)
    ]).T


def make_plot_data(maps, metadata):
    """Normalize each model map by its peak, then average models equally."""
    depth_grid = np.linspace(0, 1, 41)
    rows = []

    for survey in SURVEYS:
        for metric in METRICS:
            matrices = []
            for model, model_map in maps[maps["cohort"] == survey].groupby("model"):
                model_meta = metadata[
                    (metadata["cohort"] == survey) & (metadata["model"] == model)
                ].set_index("layer")
                block_layers = model_meta.index
                model_map = model_map[
                    model_map["left_layer"].isin(block_layers)
                    & model_map["right_layer"].isin(block_layers)
                ]
                layers = np.array(sorted(model_map["left_layer"].unique()))
                if len(layers) < 2:
                    continue
                old_depth = model_meta.loc[layers, "block_depth"].to_numpy()
                if old_depth[0] != 0 or old_depth[-1] != 1:
                    continue

                matrix = model_map.pivot(
                    index="right_layer", columns="left_layer", values=metric
                ).loc[layers, layers].to_numpy()
                matrix = resize_matrix(matrix, old_depth)
                matrix = matrix / np.nanmax(matrix)
                matrices.append(matrix)

            matrices = np.asarray(matrices)
            mean = matrices.mean(axis=0)
            error = matrices.std(axis=0, ddof=1) / np.sqrt(len(matrices))
            for right_index, survey_depth in enumerate(depth_grid):
                for left_index, hsc_depth in enumerate(depth_grid):
                    rows.append({
                        "cohort": survey,
                        "metric": metric,
                        "hsc_block_depth": hsc_depth,
                        "survey_block_depth": survey_depth,
                        "mean_relative_alignment": mean[right_index, left_index],
                        "standard_error": error[right_index, left_index],
                        "models": len(matrices),
                    })
    return pd.DataFrame(rows)


def main():
    fs.apply_style()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    maps = pd.read_csv(DATA_DIR / "E7_layer_pair_maps.csv")
    maps = maps[maps["cohort"].isin(SURVEYS)].copy()
    metadata = block_metadata(pd.read_csv(DATA_DIR / "E1_all_curves.csv"))
    plot_data = make_plot_data(maps, metadata)
    plot_data.to_csv(PAPER_DIR / "06_cross_layer_atlas_data.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(fs.TEXT_WIDTH, 5.35),
                             sharex=True, sharey=True, layout="constrained")
    image = None
    for row, survey in enumerate(SURVEYS):
        for column, (metric, metric_name) in enumerate(METRICS.items()):
            ax = axes[row, column]
            data = plot_data[
                (plot_data["cohort"] == survey) & (plot_data["metric"] == metric)
            ]
            matrix = data.pivot(
                index="survey_block_depth",
                columns="hsc_block_depth",
                values="mean_relative_alignment",
            ).sort_index().sort_index(axis=1).to_numpy()

            image = ax.imshow(matrix, origin="lower", extent=(0, 1, 0, 1),
                              vmin=0, vmax=1, cmap=fs.HEATMAP_COLORS, aspect="equal")
            ax.plot([0, 1], [0, 1], color="white", linewidth=0.75,
                    linestyle=(0, (3, 2)), alpha=0.9)
            peak = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            peak_depth = np.linspace(0, 1, matrix.shape[0])
            ax.scatter(peak_depth[peak[1]], peak_depth[peak[0]], marker="*", s=28,
                       color="white", edgecolor=fs.INK, linewidth=0.55,
                       zorder=3, clip_on=False)

            if row == 0:
                ax.set_title(metric_name, pad=5)
            if column == 0:
                ax.set_ylabel(f"{fs.SURVEY_LABEL[survey]} block depth")
            if row == 2:
                ax.set_xlabel("HSC block depth")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(length=2)
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_linewidth(0.6)

    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal",
                            fraction=0.04, pad=0.055, shrink=0.76, aspect=40)
    colorbar.set_label("mean alignment relative to each model's maximum")
    colorbar.set_ticks([0, 0.5, 1])

    fig.savefig(PAPER_DIR / "06_cross_layer_atlas.pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(PAPER_DIR / "06_cross_layer_atlas.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
