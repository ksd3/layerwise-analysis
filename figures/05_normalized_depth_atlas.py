"""Appendix atlas of block states and block updates over normalized depth."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle as fs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
PAPER_DIR = ROOT / "outputs" / "figures"

SURVEYS = ["cosmosweb", "jwst", "legacysurvey"]
METRICS = {
    "cka": ("CKA", 1, "mean CKA"),
    "procrustes": ("Procrustes", 1, "mean similarity"),
    "mrr": ("MRR", 100, "mean MRR (%)"),
}
BLOCK_PATTERN = re.compile(r"(?:^|\.)(?:h|layer|layers)\.\d+$")


def block_states(all_states):
    """Keep computational block outputs and assign them block-relative depth."""
    states = all_states[all_states["layer_name"].map(
        lambda name: bool(BLOCK_PATTERN.search(str(name)))
    )].copy()
    states = states.sort_values(["cohort", "model", "layer"])
    states["block_number"] = states.groupby(["cohort", "model"]).cumcount()
    states["block_count"] = states.groupby(["cohort", "model"])["layer"].transform("size")
    states["block_depth"] = (
        states["block_number"] / (states["block_count"] - 1).clip(lower=1)
    )

    valid = states["available"] & states[list(METRICS)].notna().all(axis=1)
    endpoints = states[valid].groupby(["cohort", "model"])["block_depth"].agg(["min", "max"])
    complete = set(endpoints[(endpoints["min"] == 0) & (endpoints["max"] == 1)].index)
    keep = [(cohort, model) in complete for cohort, model in zip(states["cohort"], states["model"])]
    return states[np.asarray(keep) & valid].copy(), complete


def block_updates(all_updates, states, complete):
    """Place each valid update halfway between its two block outputs."""
    depth = {
        (row.cohort, row.model, row.layer_name): row.block_depth
        for row in states.itertuples()
    }
    updates = all_updates[
        all_updates["cohort"].isin(SURVEYS)
        & all_updates["available"]
        & all_updates["right_definition"].eq("consecutive_update")
        & all_updates[list(METRICS)].notna().all(axis=1)
    ].copy()

    updates["from_depth"] = [
        depth.get((row.cohort, row.model, row.from_layer), np.nan)
        for row in updates.itertuples()
    ]
    updates["to_depth"] = [
        depth.get((row.cohort, row.model, row.to_layer), np.nan)
        for row in updates.itertuples()
    ]
    updates = updates.dropna(subset=["from_depth", "to_depth"])
    updates = updates[
        [(cohort, model) in complete for cohort, model in zip(updates["cohort"], updates["model"])]
    ].copy()
    updates["block_depth"] = (updates["from_depth"] + updates["to_depth"]) / 2
    return updates


def average_curves(frame, kind):
    """Interpolate each model onto one depth grid, then average the models."""
    grid = np.linspace(0, 1, 41)
    rows = []

    for survey in SURVEYS:
        survey_data = frame[frame["cohort"] == survey]
        for metric in METRICS:
            model_curves = []
            for _, model_data in survey_data.groupby("model"):
                data = model_data[["block_depth", metric]].dropna()
                data = data.groupby("block_depth", as_index=False)[metric].mean()
                data = data.sort_values("block_depth")
                if len(data) < 2:
                    continue
                depth = data["block_depth"].to_numpy()
                values = np.interp(grid, depth, data[metric].to_numpy())
                values[(grid < depth.min()) | (grid > depth.max())] = np.nan
                model_curves.append(values)

            model_curves = np.asarray(model_curves)
            for index, block_depth in enumerate(grid):
                values = model_curves[:, index]
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                rows.append({
                    "cohort": survey,
                    "metric": metric,
                    "kind": kind,
                    "normalized_block_depth": block_depth,
                    "mean": values.mean(),
                    "standard_error": error,
                    "models": len(values),
                })
    return pd.DataFrame(rows)


def main():
    fs.apply_style()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    all_states = pd.read_csv(DATA_DIR / "E1_all_curves.csv")
    all_states = all_states[all_states["cohort"].isin(SURVEYS)].copy()
    states, complete = block_states(all_states)
    updates = block_updates(pd.read_csv(DATA_DIR / "E5_updates.csv"), states, complete)

    plot_data = pd.concat([
        average_curves(states, "block state"),
        average_curves(updates, "block update"),
    ], ignore_index=True)
    plot_data.to_csv(PAPER_DIR / "05_normalized_depth_atlas_data.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(fs.TEXT_WIDTH, 4.75), sharex=True)
    for row, survey in enumerate(SURVEYS):
        for column, (metric, (metric_name, scale, ylabel)) in enumerate(METRICS.items()):
            ax = axes[row, column]
            color = fs.SURVEY_COLOR[survey]

            for kind, line_color, linestyle, alpha, marker in [
                ("block state", fs.INK, "solid", 0.10, "o"),
                ("block update", color, (0, (3.2, 2.0)), 0.18, "^"),
            ]:
                data = plot_data[
                    (plot_data["cohort"] == survey)
                    & (plot_data["metric"] == metric)
                    & (plot_data["kind"] == kind)
                ]
                depth = data["normalized_block_depth"].to_numpy()
                mean = scale * data["mean"].to_numpy()
                error = scale * data["standard_error"].to_numpy()
                ax.fill_between(depth, mean - error, mean + error,
                                color=line_color, alpha=alpha, linewidth=0)
                ax.plot(depth, mean, color=line_color, linewidth=1.15,
                        linestyle=linestyle)
                peak = int(np.nanargmax(mean))
                ax.scatter(depth[peak], mean[peak], marker=marker, s=15,
                           color=line_color, edgecolor="white", linewidth=0.5,
                           zorder=3)

            fs.style_axis(ax)
            ax.margins(x=0.02, y=0.12)
            if row == 0:
                ax.set_title(metric_name, pad=5)
            if column == 0:
                ax.set_ylabel(f"{fs.SURVEY_LABEL[survey]}\n{ylabel}")
            if row == 2:
                ax.set_xlabel("normalized block depth")
                ax.set_xticks([0, 0.5, 1])
            else:
                ax.tick_params(labelbottom=False)

    handles = [
        Line2D([], [], color=fs.INK, linewidth=1.2, marker="o", markersize=3.2,
               label="block state"),
        Line2D([], [], color=fs.MUTED, linewidth=1.2,
               linestyle=(0, (3.2, 2.0)), marker="^", markersize=3.5,
               label="block update"),
    ]
    fig.legend(handles=handles, frameon=False, ncols=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.985), handlelength=2.2, columnspacing=1.5)
    fig.subplots_adjust(left=0.17, right=0.995, bottom=0.09, top=0.93,
                        wspace=0.45, hspace=0.42)

    fig.savefig(PAPER_DIR / "05_normalized_depth_atlas.pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(PAPER_DIR / "05_normalized_depth_atlas.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
