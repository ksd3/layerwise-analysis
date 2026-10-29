"""Shared style for the four headline figures.

Figures are drawn at the NeurIPS text width (5.5 in) so every font size in the
PDF is the size that appears on the printed page.  The survey palette is fixed
across all figures (color follows the survey, never the panel) and was checked
for color-vision-deficiency separation.
"""

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
DATA = HERE

# For compatibility with both direct imports and saves
__all__ = ["HERE", "DATA", "TEXT_WIDTH"]

TEXT_WIDTH = 5.5  # NeurIPS \textwidth in inches

# Ink and chrome
INK = "#000000"      # titles, strong reference lines
MUTED = "#000000"    # secondary text, axis labels, ticks
AXIS = "#000000"     # spines
GRID = "#e4e9ed"     # hairline gridlines

# Reserved accents (never used for a survey)
RED = "#c13b3b"      # negative results / failures
WASH_RED = "#faf1ef"
WASH_BLUE = "#eef4fa"

SURVEYS = ["cosmosweb", "desi", "jwst", "legacysurvey"]
SURVEY_LABEL = {
    "cosmosweb": "COSMOS-Web",
    "desi": "DESI",
    "jwst": "JWST",
    "legacysurvey": "Legacy Survey",
}
SURVEY_COLOR = {
    "cosmosweb": "#2a78d6",
    "desi": "#4a3aa7",
    "jwst": "#eb6834",
    "legacysurvey": "#1baf7a",
}

# Continuous values use one palette everywhere. Survey colors above are used
# only when color identifies a survey.
HEATMAP_COLORS = LinearSegmentedColormap.from_list(
    "flare",
    ["#EDB081", "#E98768", "#DE5D5C", "#C14168", "#9A3670", "#722C6E", "#4B2362"],
)
HEATMAP_COLORS.set_bad("white")

DIRECTION_COLORS = LinearSegmentedColormap.from_list(
    "direction", ["#C66A4A", "#E9B18F", "#F4EEE7", "#C998B2", "#6C376B"]
)


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 7.0,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 6.3,
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax, grid="y"):
    """Recessive chrome: no top/right spines, hairline solid grid below the data."""
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def panel_letter(ax, letter, x=-0.24, y=1.12):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color=INK, ha="left", va="baseline")


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.pdf")
    fig.savefig(HERE / f"{stem}.png", dpi=300)
