"""Figure 2: geometry as a stage-selection rule, and geometry versus physics.

a. Held-out R^2 lost by selecting the stage with geometry instead of labels.
b. Where each rule's stage ranks among all valid stages of the model.
c. Across-stage Spearman correlation between each geometry metric and R^2.
d-g. Mean within-model percentile rank of each stage by transfer R^2, MKNN,
     and MRR over normalized depth (the ranked curves behind c).

Inputs come from analysis_scripts/analyze_review_additions.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle as fs

ROOT = fs.HERE.parent
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "outputs" / "tables"
PAPER_DIR = ROOT / "outputs" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

fs.apply_style()

loss = pd.read_csv(DATA_DIR / "E3_inference_analysis.csv")
rank = pd.read_csv(TABLE_DIR / "review_selection_percentiles_summary.csv")
correlation = pd.read_csv(TABLE_DIR / "review_stage_covariation_summary.csv")
profiles = pd.read_csv(TABLE_DIR / "review_depth_profiles.csv")

ORDER = [
    ("legacysurvey", "g_minus_r", "hsc_to_other"),
    ("legacysurvey", "g_minus_r", "other_to_hsc"),
    ("legacysurvey", "log_shape", "hsc_to_other"),
    ("legacysurvey", "log_shape", "other_to_hsc"),
    ("jwst", "mag_auto", "hsc_to_other"),
    ("jwst", "mag_auto", "other_to_hsc"),
    ("jwst", "log_flux_radius", "hsc_to_other"),
    ("jwst", "log_flux_radius", "other_to_hsc"),
]
PANELS = [
    ("legacysurvey", "g_minus_r", "Legacy color"),
    ("legacysurvey", "log_shape", "Legacy shape"),
    ("jwst", "mag_auto", "JWST magnitude"),
    ("jwst", "log_flux_radius", "JWST size"),
]
TARGET = {"g_minus_r": "color", "log_shape": "shape", "mag_auto": "magnitude",
          "log_flux_radius": "size"}
SURVEY = {"legacysurvey": "Legacy", "jwst": "JWST"}
METRICS = ["MKNN", "CKA", "Procrustes", "MRR"]
WITHIN = "#71889b"
GEOMETRY = "#3b4650"


def label(survey, target, direction):
    ends = ("HSC", SURVEY[survey])
    left, right = ends if direction == "hsc_to_other" else ends[::-1]
    return f"{TARGET[target]} · {left} → {right}"


def lookup(frame, key, **match):
    rows = frame[(frame["cohort"] == key[0]) & (frame["target"] == key[1])
                 & (frame["direction"] == key[2])]
    for column, value in match.items():
        rows = rows[rows[column] == value]
    return rows.iloc[0]


def profile(cohort, target, direction, score):
    rows = profiles[(profiles["cohort"] == cohort) & (profiles["target"] == target)
                    & (profiles["direction"] == direction) & (profiles["score"] == score)]
    return rows.sort_values("depth")


fig = plt.figure(figsize=(fs.TEXT_WIDTH, 3.8))
top = fig.add_gridspec(
    2, 4, width_ratios=[1.0, 1.0, 1.02, 0.05], height_ratios=[1, 1],
    left=0.205, right=0.935, top=0.88, bottom=0.425, wspace=0.13, hspace=0.32,
)
bottom = fig.add_gridspec(1, 4, left=0.205, right=0.985, top=0.25, bottom=0.08,
                          wspace=0.16)
groups = [ORDER[:4], ORDER[4:]]

# a: R^2 lost, on a linear scale chosen separately for each survey.
loss_axes = []
for group_index, (group, limit, ticks) in enumerate(
    zip(groups, (1.12, 6.6), ([0, 0.25, 0.5, 0.75, 1.0], [0, 2, 4, 6]))
):
    axis = fig.add_subplot(top[group_index, 0])
    loss_axes.append(axis)
    for row, key in enumerate(group):
        record = lookup(loss, key)
        value = -record["mean_regret"]
        low, high = -record["family_bootstrap_high"], -record["family_bootstrap_low"]
        color = fs.SURVEY_COLOR[key[0]]
        axis.barh(row, value, height=0.5, color=color, alpha=0.85, linewidth=0, zorder=2)
        axis.errorbar(value, row, xerr=[[value - low], [high - value]], fmt="none",
                      ecolor=fs.INK, elinewidth=0.7, capsize=1.6, capthick=0.6, zorder=3)
    axis.set_xlim(0, limit)
    axis.set_xticks(ticks)
    axis.set_ylim(3.6, -0.6)
    axis.set_yticks(range(4), [label(*key) for key in group])
    axis.tick_params(axis="y", length=0)
    axis.text(0.98, 0.97, "Legacy Survey" if group_index == 0 else "JWST",
              transform=axis.transAxes, ha="right", va="top", fontsize=6.3,
              color=fs.SURVEY_COLOR[group[0][0]])
    fs.style_axis(axis, grid="x")
loss_axes[1].set_xlabel("R² lost by geometric selection")
fs.panel_letter(loss_axes[0], "a", x=-0.86, y=1.09)

# b: percentile of the chosen stage among all valid stages (0 worst, 1 best).
rank_axes = []
for group_index, group in enumerate(groups):
    axis = fig.add_subplot(top[group_index, 1])
    rank_axes.append(axis)
    axis.axvline(0.5, color=fs.INK, linewidth=0.6, linestyle=(0, (2.5, 2)), zorder=1)
    for row, key in enumerate(group):
        metric_values = [lookup(rank, key, rule=f"max_{name}")["family_mean"] for name in METRICS]
        axis.plot([min(metric_values), max(metric_values)], [row, row], color="#c9d2da",
                  linewidth=3.2, solid_capstyle="round", zorder=2)
        geometry = lookup(rank, key, rule="geometry_selected")
        axis.errorbar(
            geometry["family_mean"], row,
            xerr=[[geometry["family_mean"] - geometry["family_bootstrap_low"]],
                  [geometry["family_bootstrap_high"] - geometry["family_mean"]]],
            fmt="o", markersize=3.6, color=fs.SURVEY_COLOR[key[0]],
            markeredgecolor="white", markeredgewidth=0.4, elinewidth=0.8, capsize=0, zorder=4,
        )
        final = lookup(rank, key, rule="final_stage")["family_mean"]
        axis.plot(final, row, marker="s", markersize=3.2, markerfacecolor="white",
                  markeredgecolor=fs.INK, markeredgewidth=0.6, linestyle="none", zorder=3)
        labeled = lookup(rank, key, rule="label_selected")["family_mean"]
        axis.plot(labeled, row, marker="D", markersize=3.0, color=WITHIN,
                  linestyle="none", zorder=3)
    axis.set_xlim(-0.02, 1.02)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    axis.set_ylim(3.6, -0.6)
    axis.set_yticks(range(4), [""] * 4)
    axis.tick_params(axis="y", length=0)
    fs.style_axis(axis, grid="x")
rank_axes[0].tick_params(axis="x", labelbottom=False)
rank_axes[1].set_xlabel("percentile among all stages")
rank_axes[0].text(0.5, 1.02, "random stage", transform=rank_axes[0].get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=5.8, color=fs.INK)
fs.panel_letter(rank_axes[0], "b", x=-0.08, y=1.09)
selection_handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=3.6, color="#7d8b96",
           markeredgecolor="white", markeredgewidth=0.4, label="geometry-selected"),
    Line2D([], [], color="#c9d2da", linewidth=3.2, solid_capstyle="round",
           label="max of each metric (range)"),
    Line2D([], [], marker="s", linestyle="none", markersize=3.2, markerfacecolor="white",
           markeredgecolor=fs.INK, markeredgewidth=0.6, label="final stage"),
    Line2D([], [], marker="D", linestyle="none", markersize=3.0, color=WITHIN,
           label="label-selected"),
]
fig.legend(handles=selection_handles, loc="upper center", bbox_to_anchor=(0.57, 1.0),
           ncol=4, frameon=False, fontsize=6.0, handletextpad=0.3, columnspacing=1.0,
           handlelength=1.4)

# c: within-model correlation over stages between geometry and physics.
heat_axes = []
for group_index, group in enumerate(groups):
    axis = fig.add_subplot(top[group_index, 2])
    heat_axes.append(axis)
    values = np.array([[lookup(correlation, key, metric=name)["family_mean"] for name in METRICS]
                       for key in group])
    excludes_zero = np.array([[
        (lookup(correlation, key, metric=name)["family_bootstrap_low"] > 0)
        | (lookup(correlation, key, metric=name)["family_bootstrap_high"] < 0)
        for name in METRICS] for key in group])
    image = axis.imshow(values, cmap=fs.DIRECTION_COLORS.reversed(), vmin=-0.4, vmax=0.4,
                        aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.2f}".replace("-", "−"),
                      ha="center", va="center", fontsize=5.8,
                      fontweight="bold" if excludes_zero[row, column] else "normal",
                      color="white" if abs(values[row, column]) > 0.27 else fs.INK)
    axis.set_yticks([])
    axis.set_xticks(range(len(METRICS)), ["MKNN", "CKA", "Proc.", "MRR"])
    axis.tick_params(axis="x", length=0, labeltop=group_index == 0, labelbottom=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
fs.panel_letter(heat_axes[0], "c", x=-0.10, y=1.09)
colorbar_axis = fig.add_subplot(top[:, 3])
colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=[-0.4, -0.2, 0, 0.2, 0.4])
colorbar.ax.set_yticklabels(["−0.4", "−0.2", "0", "0.2", "0.4"])
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(length=1.5, labelsize=5.8)
colorbar.set_label("across-stage Spearman ρ", fontsize=6.3)

# d-g: ranked depth profiles. Geometry does not depend on the target, so the
# two panels of a survey share their MKNN and MRR curves.
for index, (cohort, target, title) in enumerate(PANELS):
    axis = fig.add_subplot(bottom[0, index])
    color = fs.SURVEY_COLOR[cohort]
    for direction, style in (("hsc_to_other", "-"), ("other_to_hsc", (0, (3, 1.6)))):
        line = profile(cohort, target, direction, "transfer R2")
        axis.fill_between(line["depth"], line["mean_rank"] - line["standard_error"],
                          line["mean_rank"] + line["standard_error"], color=color,
                          alpha=0.16, linewidth=0, zorder=1)
        axis.plot(line["depth"], line["mean_rank"], color=color, linestyle=style,
                  linewidth=1.3, zorder=3)
    for score, style in (("MKNN", "-"), ("MRR", (0, (1, 1.2)))):
        line = profile(cohort, target, "hsc_to_other", score)
        axis.plot(line["depth"], line["mean_rank"], color=GEOMETRY, linestyle=style,
                  linewidth=1.0, zorder=2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0.05, 0.95)
    axis.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    axis.set_yticks([0.25, 0.5, 0.75])
    axis.set_title(title, color=color, fontsize=6.6, pad=3, loc="left")
    if index:
        axis.tick_params(axis="y", labelleft=False)
    else:
        axis.set_ylabel("mean rank", fontsize=6.6)
    fs.style_axis(axis, grid="y")
    fs.panel_letter(axis, "defg"[index], x=-0.34 if index == 0 else -0.12, y=1.06)
fig.text(0.595, 0.012, "normalized depth", ha="center", va="bottom", fontsize=7.0,
         color=fs.MUTED)
profile_handles = [
    Line2D([], [], color="#7d8b96", linewidth=1.3, label="transfer R², HSC → survey"),
    Line2D([], [], color="#7d8b96", linewidth=1.3, linestyle=(0, (3, 1.6)),
           label="transfer R², survey → HSC"),
    Line2D([], [], color=GEOMETRY, linewidth=1.0, label="MKNN"),
    Line2D([], [], color=GEOMETRY, linewidth=1.0, linestyle=(0, (1, 1.2)), label="MRR"),
]
fig.legend(handles=profile_handles, loc="upper center", bbox_to_anchor=(0.595, 0.333),
           ncol=4, frameon=False, fontsize=6.0, handletextpad=0.4, columnspacing=1.2,
           handlelength=2.0)

fig.savefig(PAPER_DIR / "05_geometry_vs_physics.pdf", facecolor="white")
fig.savefig(PAPER_DIR / "05_geometry_vs_physics.png", dpi=300, facecolor="white")
plt.close(fig)
