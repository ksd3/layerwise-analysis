"""Plot layerwise physical prediction beside layerwise embedding alignment."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import figstyle as fs


SCORES_FILE = ROOT / "data" / "layerwise_physics_scores.csv"
ALIGNMENT_FILE = ROOT / "data" / "E1_all_curves.csv"
OUTPUT_DIR = ROOT / "outputs" / "figures"

def model_family(name):
    if name.startswith("astropt"):
        return "AstroPT"
    if name.startswith("convnext"):
        return "ConvNeXt"
    if name.startswith("dinov3"):
        return "DINOv3"
    if name.startswith("dino"):
        return "DINO"
    if name.startswith("vit-mae"):
        return "ViT-MAE"
    if name.startswith("vjepa"):
        return "V-JEPA"
    if name.startswith("ijepa"):
        return "I-JEPA"
    if name.startswith("paligemma"):
        return "PaliGemma"
    if name.startswith("llava"):
        return "LLaVA"
    if name.startswith("clip"):
        return "CLIP"
    if name.startswith("vit"):
        return "ViT"
    return name


def pretty_name(name):
    # The family is already printed above the map, so only show the variant.
    prefixes = [
        "astropt_", "convnext_", "dinov3_", "dino_", "vit-mae_",
        "vjepa_", "ijepa_", "paligemma_", "llava_", "clip_", "vit_",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ")


def model_labels(models):
    """Name each family once, then use short variant labels within it."""
    labels = []
    previous_family = None
    for model in models:
        family = model_family(model)
        variant = pretty_name(model)
        labels.append(f"{family} {variant}" if family != previous_family else variant)
        previous_family = family
    return labels


def relative_columns(matrix, floor=None):
    """Scale every model curve from its own minimum to its own maximum."""
    values = matrix.copy()
    if floor is not None:
        values = np.maximum(values, floor)
    low = np.nanmin(values, axis=0)
    high = np.nanmax(values, axis=0)
    width = high - low
    width[width == 0] = 1
    return (values - low) / width


def curve_matrix(frame, models, value, depth_grid):
    matrix = np.full((len(depth_grid), len(models)), np.nan)
    maxima = np.full(len(models), np.nan)
    peak_depths = np.full(len(models), np.nan)

    for column, model in enumerate(models):
        curve = frame[frame["model"] == model]
        curve = curve.groupby("normalized_depth", as_index=False)[value].mean()
        curve = curve.replace([np.inf, -np.inf], np.nan).dropna().sort_values("normalized_depth")
        if curve.empty:
            continue

        x = curve["normalized_depth"].to_numpy()
        y = curve[value].to_numpy()
        interpolated = np.interp(depth_grid, x, y)
        interpolated[(depth_grid < x.min()) | (depth_grid > x.max())] = np.nan
        matrix[:, column] = interpolated

        best = int(np.argmax(y))
        maxima[column] = y[best]
        peak_depths[column] = x[best]

    return matrix, maxima, peak_depths


def add_family_separators(axes, models):
    families = [model_family(model) for model in models]
    starts = [0]
    for index in range(1, len(models)):
        if families[index] != families[index - 1]:
            starts.append(index)
            for axis in axes:
                axis.axvline(index - 0.5, color="white", linewidth=1.1, alpha=0.9)
    starts.append(len(models))



def draw_page(physics, alignment, cohort, target, direction):
    models = sorted(physics["model"].unique(), key=lambda name: (model_family(name), name))
    depth_grid = np.linspace(0, 1, 61)

    physics_matrix, physics_max, physics_depth = curve_matrix(
        physics, models, "test_r2", depth_grid
    )
    alignment_matrix, alignment_max, alignment_depth = curve_matrix(
        alignment, models, "corrected_overlap_20", depth_grid
    )

    figure = plt.figure(figsize=(11, 5.2))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.05, 4], hspace=0.06, wspace=0.16)

    physics_top = figure.add_subplot(grid[0, 0])
    physics_map = figure.add_subplot(grid[1, 0], sharex=physics_top)
    alignment_top = figure.add_subplot(grid[0, 1])
    alignment_map = figure.add_subplot(grid[1, 1], sharex=alignment_top)

    physics_relative = relative_columns(physics_matrix, floor=-0.5)
    alignment_relative = relative_columns(alignment_matrix)
    physical_image = physics_map.imshow(
        physics_relative,
        origin="lower",
        aspect="auto",
        extent=(-0.5, len(models) - 0.5, 0, 1),
        cmap=fs.HEATMAP_COLORS,
        vmin=0,
        vmax=1,
    )
    alignment_image = alignment_map.imshow(
        alignment_relative,
        origin="lower",
        aspect="auto",
        extent=(-0.5, len(models) - 0.5, 0, 1),
        cmap=fs.HEATMAP_COLORS,
        vmin=0,
        vmax=1,
    )

    physics_map.scatter(
        range(len(models)), physics_depth, marker="v", s=31,
        facecolor="white", edgecolor="black", linewidth=0.5, zorder=3,
    )
    alignment_map.scatter(
        range(len(models)), alignment_depth, marker="v", s=31,
        facecolor="white", edgecolor="black", linewidth=0.5, zorder=3,
    )

    survey_color = fs.SURVEY_COLOR[cohort]
    physics_top.scatter(range(len(models)), physics_max, color=survey_color, s=14,
                        edgecolor="white", linewidth=0.45)
    alignment_top.scatter(range(len(models)), alignment_max, color=survey_color, s=14,
                          edgecolor="white", linewidth=0.45)
    physics_top.axhline(0, color=fs.AXIS, linewidth=0.7)
    alignment_top.set_ylim(0, max(0.15, 1.08 * np.nanmax(alignment_max)))

    physics_top.set_ylabel("Held-out $R^2$")
    alignment_top.set_ylabel("Corrected MKNN")
    physics_top.text(-0.08, 1.12, "a", transform=physics_top.transAxes,
                     fontsize=14.5, fontweight="bold", color=fs.INK)
    alignment_top.text(-0.08, 1.12, "b", transform=alignment_top.transAxes,
                       fontsize=14.5, fontweight="bold", color=fs.INK)

    for axis in (physics_map, alignment_map):
        axis.set_ylabel("Normalized saved-stage index")
        axis.set_ylim(0, 1)
        axis.set_xticks(range(len(models)))
        axis.set_xticklabels(model_labels(models), rotation=58, ha="right", fontsize=8.5)

    for axis in (physics_top, alignment_top):
        axis.tick_params(axis="x", labelbottom=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=fs.GRID, linewidth=0.8)
        axis.set_axisbelow(True)

    for axis in (physics_map, alignment_map):
        for spine in axis.spines.values():
            spine.set_color(fs.AXIS)
            spine.set_linewidth(0.8)

    add_family_separators((physics_top, physics_map), models)
    add_family_separators((alignment_top, alignment_map), models)

    figure.colorbar(physical_image, ax=physics_map, pad=0.01,
                    label="relative physical score within model")
    figure.colorbar(alignment_image, ax=alignment_map, pad=0.01,
                    label="relative alignment within model")
    return figure, models, physics_max, alignment_max


def draw_summary(rows):
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.0), sharex=True)
    family_names = sorted({model_family(model) for row in rows for model in row["models"]})
    palette = [
        "#277DA1", "#F8961E", "#43AA8B", "#F94144", "#7B61A8", "#9C6644",
        "#E76F9A", "#6C757D", "#B5A800", "#00A6A6", "#577590",
    ]
    colors = dict(zip(family_names, palette))
    row_lookup = {(row["cohort"], row["target"], row["direction"]): row for row in rows}
    panels = [
        ("jwst", "mag_auto", "(a)  JWST magnitude"),
        ("jwst", "log_flux_radius", "(b)  JWST size"),
        ("legacysurvey", "g_minus_r", "(c)  Legacy Survey color"),
        ("legacysurvey", "log_shape", "(d)  Legacy Survey shape"),
    ]

    for axis, (cohort, target, title) in zip(axes.flat, panels):
        outward = row_lookup[(cohort, target, "hsc_to_other")]
        inward = row_lookup[(cohort, target, "other_to_hsc")]
        outward_values = dict(zip(outward["models"], outward["physics_max"]))
        inward_values = dict(zip(inward["models"], inward["physics_max"]))
        alignment_values = dict(zip(outward["models"], outward["alignment_max"]))
        models = outward["models"]

        # A vertical segment connects the two transfer directions for one model.
        for model in models:
            x = alignment_values[model]
            y_out = outward_values[model]
            y_in = inward_values[model]
            color = colors[model_family(model)]
            axis.plot([x, x], [y_out, y_in], color=color, alpha=0.26, linewidth=0.9, zorder=1)
            axis.scatter(
                x, y_out, marker="o", s=37, color=color, edgecolor="white",
                linewidth=0.55, alpha=0.94, zorder=3,
            )
            axis.scatter(
                x, y_in, marker="^", s=44, color=color, edgecolor="white",
                linewidth=0.55, alpha=0.94, zorder=3,
            )

        x = np.array([alignment_values[model] for model in models])
        y_out = np.array([outward_values[model] for model in models])
        y_in = np.array([inward_values[model] for model in models])
        rho_out = spearmanr(x, y_out).statistic
        rho_in = spearmanr(x, y_in).statistic

        axis.axhspan(-100, 0, color="#FBE9E7", alpha=0.75, zorder=0)
        axis.axhline(0, color="#8A5A55", linewidth=0.8, zorder=2)
        axis.axvline(np.median(x), color=fs.AXIS, linewidth=0.7, linestyle="--", zorder=0)
        axis.set_title(title, fontsize=9.2, weight="bold", loc="left", pad=15)
        axis.text(
            1.0, 1.035,
            f"● HSC → survey: ρ={rho_out:+.2f}     ▲ survey → HSC: ρ={rho_in:+.2f}",
            transform=axis.transAxes, ha="right", va="bottom", fontsize=7.3,
            color=fs.MUTED, clip_on=False,
        )
        axis.set_xlim(0.02, 0.26)
        axis.grid(color=fs.GRID, linewidth=0.55, alpha=0.9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)

        # The negative JWST outliers are real; a symmetric-log scale prevents them from flattening the rest of the model cloud around zero.
        if cohort == "jwst":
            axis.set_yscale("symlog", linthresh=0.15, linscale=0.9)
            axis.set_ylim(-3.4, 0.85)
            axis.set_yticks([-3, -1, -0.25, 0, 0.25, 0.5])
            axis.set_yticklabels(["−3", "−1", "−0.25", "0", "0.25", "0.5"])
        else:
            low = min(y_out.min(), y_in.min())
            high = max(y_out.max(), y_in.max())
            padding = max(0.03, 0.10 * (high - low))
            axis.set_ylim(low - padding, high + padding)

    for axis in axes[:, 0]:
        axis.set_ylabel("Maximum held-out $R^2$", fontsize=9)
    for axis in axes[-1, :]:
        axis.set_xlabel("Maximum corrected MKNN", fontsize=9)

    family_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=colors[name], markeredgecolor="white", label=name)
        for name in family_names
    ]
    figure.legend(
        handles=family_handles, loc="lower center", ncol=6, frameon=False,
        fontsize=7.5, columnspacing=1.2, handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.20, wspace=0.22, hspace=0.40)
    return figure


def main():
    fs.apply_style()
    # These pages are drawn at twice the paper width and reduced by LaTeX.
    plt.rcParams.update({
        "font.size": 12.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
    })
    scores = pd.read_csv(SCORES_FILE)
    alignment = pd.read_csv(ALIGNMENT_FILE)

    # E1 marks constant or tie-dominated representations as unavailable.
    alignment = alignment[
        alignment["available"].astype(bool)
        & alignment["corrected_overlap_20"].notna()
        & alignment["cohort"].isin(scores["cohort"].unique())
    ].copy()

    # Compare physics and geometry only at layers where geometry is valid.
    valid_layers = alignment[["cohort", "model", "layer"]].drop_duplicates()
    scores = scores.merge(valid_layers, on=["cohort", "model", "layer"], how="inner")

    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    combined_file = OUTPUT_DIR / "physics_alignment_depth_maps.pdf"

    with PdfPages(combined_file) as pdf:
        for (cohort, target, direction), physics in scores.groupby(
            ["cohort", "target", "direction"], sort=True
        ):
            matching = alignment[alignment["cohort"] == cohort]
            figure, models, physics_max, alignment_max = draw_page(
                physics, matching, cohort, target, direction
            )
            stem = f"{cohort}_{target}_{direction}"
            figure.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
            figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            rows.append(
                {
                    "cohort": cohort,
                    "target": target,
                    "direction": direction,
                    "models": models,
                    "physics_max": physics_max,
                    "alignment_max": alignment_max,
                }
            )

    summary = draw_summary(rows)
    summary.savefig(OUTPUT_DIR / "max_physics_vs_alignment.pdf", bbox_inches="tight")
    summary.savefig(OUTPUT_DIR / "max_physics_vs_alignment.png", dpi=220, bbox_inches="tight")
    plt.close(summary)

    print(f"Wrote {combined_file}")
    print(f"Wrote {OUTPUT_DIR / 'max_physics_vs_alignment.pdf'}")
    print("Every panel contains 36 models.")


if __name__ == "__main__":
    main()
