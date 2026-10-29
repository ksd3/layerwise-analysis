"""Figure 3: final HSC--COSMOS-Web representations do not transfer.

a. Within- and cross-survey redshift R^2 of every model, on a linear axis.
b. For every target: unchanged cross-survey R^2, the best affine
   recalibration of the same predictions, and within-survey R^2.

Inputs come from E8 and analysis_scripts/analyze_review_additions.py.
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

redshift = pd.read_csv(DATA_DIR / "E8_model_analysis.csv")
models = pd.read_csv(TABLE_DIR / "review_final_diagnostics_by_model.csv")
WITHIN = "#71889b"
TARGETS = [
    ("redshift", "redshift"),
    ("magnitude", "r magnitude"),
    ("g_minus_r", "g − r color"),
    ("mass", "stellar mass"),
    ("sSFR", "sSFR"),
]

fig, (axis_a, axis_b) = plt.subplots(
    1, 2, figsize=(fs.TEXT_WIDTH, 1.78), gridspec_kw={"width_ratios": [1.0, 1.08]},
)
fig.subplots_adjust(left=0.115, right=0.985, top=0.80, bottom=0.215, wspace=0.42)

# a: every model's redshift R^2. Two models fall beyond the left edge and are
# drawn there as arrows with their values, so the axis can stay linear.
LEFT = -6.6
axis = axis_a
axis.axvspan(LEFT, 0, color=fs.WASH_RED, zorder=0)
axis.axvline(0, color=fs.INK, linewidth=0.7, zorder=1)
offsets = np.tile([-0.16, 0.0, 0.16, -0.08, 0.08], 7)
for row, column, color in ((1, "within_view_redshift_r2", WITHIN),
                           (0, "cross_view_redshift_r2", fs.RED)):
    values = np.sort(redshift[column].to_numpy())
    inside = values >= LEFT + 0.3
    y = row + offsets[: len(values)]
    axis.scatter(values[inside], y[inside], s=9, color=color, edgecolor="white",
                 linewidth=0.35, zorder=3)
    axis.plot([np.median(values)] * 2, [row - 0.3, row + 0.3], color=fs.INK,
              linewidth=0.9, zorder=4)
    # Off-scale models sit outside the jitter band so their labels stay clear.
    for number, value in enumerate(values[~inside]):
        height = row + 0.36 - 0.72 * number
        axis.plot(LEFT + 0.12, height, marker="<", markersize=3.2,
                  color=color, zorder=3, clip_on=False)
        axis.text(LEFT + 0.38, height, f"{value:.1f}".replace("-", "−"),
                  fontsize=5.6, va="center", ha="left", color=color)
axis.set_xlim(LEFT, 0.85)
axis.set_xticks([-6, -4, -2, 0], ["−6", "−4", "−2", "0"])
axis.set_ylim(-0.55, 1.55)
axis.set_yticks([0, 1], ["across\nsurveys", "within\nsurvey"])
axis.tick_params(axis="y", length=0)
axis.set_xlabel("redshift R² (one point per model; bar = median)")
fs.style_axis(axis, grid="x")
fs.panel_letter(axis, "a", x=-0.27, y=1.1)

# b: median over models, with interquartile ranges for the two scores that
# lie between zero and one. Recalibration fits a + b * prediction on the test
# labels themselves, so its R^2 is an upper bound for any affine correction.
axis = axis_b
LOW = -1.65
axis.axvspan(LOW, 0, color=fs.WASH_RED, zorder=0)
axis.axvline(0, color=fs.INK, linewidth=0.7, zorder=1)
for row, (target, _) in enumerate(TARGETS):
    cross = models[(models["target"] == target) & (models["kind"] == "cross")]
    within = models[(models["target"] == target) & (models["kind"] == "within")]
    unchanged = cross["pooled_r2"].median()
    bound = cross["affine_bound"]
    native = within["pooled_r2"]
    axis.plot([max(unchanged, LOW), bound.median()], [row, row], color=fs.RED,
              linewidth=0.7, linestyle=(0, (1.5, 1.2)), zorder=2)
    axis.plot(unchanged, row, marker="o", markersize=3.6, color=fs.RED,
              markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    for values, style in ((bound, dict(markerfacecolor="white", markeredgecolor=fs.RED)),
                          (native, dict(color=WITHIN, markeredgecolor="white"))):
        axis.plot([values.quantile(0.25), values.quantile(0.75)], [row, row],
                  color=style.get("markeredgecolor") if style.get("markerfacecolor") else WITHIN,
                  linewidth=1.0, zorder=3)
        axis.plot(values.median(), row, marker="o", markersize=3.6,
                  markeredgewidth=0.7, zorder=4, **style)
axis.set_xlim(LOW, 1.0)
axis.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1], ["−1.5", "−1", "−0.5", "0", "0.5", "1"])
axis.set_ylim(len(TARGETS) - 0.5, -0.5)
axis.set_yticks(range(len(TARGETS)), [name for _, name in TARGETS])
axis.tick_params(axis="y", length=0)
axis.set_xlabel("R² (median over 31 models; lines = IQR)")
fs.style_axis(axis, grid="x")
fs.panel_letter(axis, "b", x=-0.33, y=1.1)

handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=3.6, color=fs.RED,
           markeredgecolor="white", markeredgewidth=0.4, label="across surveys, unchanged"),
    Line2D([], [], marker="o", linestyle="none", markersize=3.6, markerfacecolor="white",
           markeredgecolor=fs.RED, markeredgewidth=0.7,
           label="across surveys, best affine recalibration"),
    Line2D([], [], marker="o", linestyle="none", markersize=3.6, color=WITHIN,
           markeredgecolor="white", markeredgewidth=0.4, label="within survey"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=3,
           frameon=False, fontsize=6.0, handletextpad=0.25, columnspacing=1.1)

fig.savefig(PAPER_DIR / "06_final_transfer_calibration.pdf", facecolor="white")
fig.savefig(PAPER_DIR / "06_final_transfer_calibration.png", dpi=300, facecolor="white")
plt.close(fig)
