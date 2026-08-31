# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr

    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data"
    FINAL = ROOT / "outputs" / "figures"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figstyle as fs
    import plot_physics_alignment as physics_plot

    fs.apply_style()

    def save_figure(figure, name):
        FINAL.mkdir(parents=True, exist_ok=True)
        figure.savefig(FINAL / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white")
        figure.savefig(FINAL / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")

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

    return DATA, Line2D, ROOT, fs, model_family, np, pd, physics_plot, plt, save_figure, spearmanr


@app.cell
def _(DATA, Line2D, ROOT, fs, np, pd, plt, save_figure):
    # FIGURE 1: saved-state and block-update alignment through depth.
    def make_figure_1():
        import re

        updates = pd.read_csv(DATA / "E5_updates.csv")
        block_name = re.compile(r"(?:^|\.)(?:h|layer|layers)\.\d+$")
        updates = updates[
            updates["cohort"].isin(fs.SURVEYS)
            & updates["available"].astype(bool)
            & updates["corrected_overlap_20"].notna()
            & updates["full_state_after"].notna()
            & updates["from_layer"].map(
                lambda name: bool(block_name.search(str(name)))
            )
            & updates["to_layer"].map(
                lambda name: bool(block_name.search(str(name)))
            )
        ].copy()

        # Give every model equal weight: first interpolate each model onto one transition-depth grid, then average the interpolated curves.
        updates = updates.sort_values(["cohort", "model", "transition"])
        updates["block_transition"] = updates.groupby(
            ["cohort", "model"]
        ).cumcount()
        updates["block_transitions"] = updates.groupby(
            ["cohort", "model"]
        )["transition"].transform("size")
        updates = updates[updates["block_transitions"] > 1]
        updates["block_depth"] = updates["block_transition"] / (
            updates["block_transitions"] - 1
        )

        grid = np.linspace(0, 1, 41)
        curve_rows = []
        for survey in fs.SURVEYS:
            state_curves = []
            update_curves = []
            for _, model_data in updates[
                updates["cohort"] == survey
            ].groupby("model"):
                model_data = model_data.sort_values("block_depth")
                depth = model_data["block_depth"].to_numpy()
                state_curves.append(
                    np.interp(grid, depth, model_data["full_state_after"])
                )
                update_curves.append(
                    np.interp(
                        grid, depth, model_data["corrected_overlap_20"]
                    )
                )
            states = np.asarray(state_curves)
            block_updates = np.asarray(update_curves)
            for index, depth in enumerate(grid):
                curve_rows.append({
                    "cohort": survey,
                    "depth": depth,
                    "state_mean": states[:, index].mean(),
                    "state_se": states[:, index].std(ddof=1)
                    / np.sqrt(len(states)),
                    "update_mean": block_updates[:, index].mean(),
                    "update_se": block_updates[:, index].std(ddof=1)
                    / np.sqrt(len(block_updates)),
                })
        curves = pd.DataFrame(curve_rows)

        figure, axes = plt.subplots(
            2, 2, figsize=(5.5, 3.35), sharex=True, layout="constrained"
        )

        for axis, survey in zip(axes.flat, fs.SURVEYS):
            data = curves[curves["cohort"] == survey]
            depth = data["depth"].to_numpy()
            state = 100 * data["state_mean"].to_numpy()
            state_se = 100 * data["state_se"].to_numpy()
            update = 100 * data["update_mean"].to_numpy()
            update_se = 100 * data["update_se"].to_numpy()
            color = fs.SURVEY_COLOR[survey]

            axis.fill_between(depth, state - state_se, state + state_se,
                              color=fs.INK, alpha=0.10, linewidth=0)
            axis.plot(depth, state, color=fs.INK, linewidth=1.25)
            axis.fill_between(depth, update - update_se, update + update_se,
                              color=color, alpha=0.16, linewidth=0)
            axis.plot(depth, update, color=color, linewidth=1.25,
                      linestyle=(0, (3.2, 2.0)))

            state_peak = int(np.nanargmax(state))
            update_peak = int(np.nanargmax(update))
            axis.scatter(depth[state_peak], state[state_peak], s=15,
                         color=fs.INK, edgecolor="white", linewidth=0.5, zorder=3)
            axis.scatter(depth[update_peak], update[update_peak], s=17,
                         marker="D", color=color, edgecolor="white",
                         linewidth=0.5, zorder=3)

            axis.set_xlim(0, 1)
            axis.set_ylim(bottom=0)
            axis.set_title(fs.SURVEY_LABEL[survey], loc="left",
                           fontweight="normal", color=color, pad=5)
            fs.style_axis(axis)

        for axis in axes[-1, :]:
            axis.set_xlabel("normalized depth")
        for axis in axes[:, 0]:
            axis.set_ylabel("mean MKNN (%)")

        handles = [
            Line2D([], [], color=fs.INK, linewidth=1.25, marker="o",
                   markersize=3.2, label="saved block state"),
            Line2D([], [], color=fs.MUTED, linewidth=1.25,
                   linestyle=(0, (3.2, 2.0)), marker="D", markersize=3.2,
                   label="block update"),
        ]
        figure.legend(handles=handles, frameon=False, ncols=2,
                      loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      handlelength=2.0, columnspacing=1.4)
        return figure

    figure_1 = make_figure_1()
    save_figure(figure_1, "02_block_update_curves")
    figure_1
    return (figure_1,)


@app.cell
def _(ROOT, fs, np, pd, physics_plot, plt, save_figure):
    # FIGURE 2: maximum geometric alignment versus maximum physical prediction.
    def make_figure_2():
        scores = pd.read_csv(DATA / "layerwise_physics_scores.csv")
        alignment = pd.read_csv(DATA / "E1_all_curves.csv")
        alignment = alignment[
            alignment["available"].astype(bool)
            & alignment["corrected_overlap_20"].notna()
            & alignment["cohort"].isin(scores["cohort"].unique())
        ].copy()

        valid = alignment[["cohort", "model", "layer"]].drop_duplicates()
        scores = scores.merge(
            valid, on=["cohort", "model", "layer"], how="inner"
        )

        rows = []
        depth_grid = np.linspace(0, 1, 61)
        for (cohort, target, direction), physical in scores.groupby(
            ["cohort", "target", "direction"], sort=True
        ):
            models = sorted(
                physical["model"].unique(),
                key=lambda name: (physics_plot.model_family(name), name),
            )
            physical = physical.groupby(
                ["model", "layer", "normalized_depth"], as_index=False
            )["test_r2"].mean()
            matching = alignment[alignment["cohort"] == cohort]
            _, physics_max, _ = physics_plot.curve_matrix(
                physical, models, "test_r2", depth_grid
            )
            _, alignment_max, _ = physics_plot.curve_matrix(
                matching, models, "corrected_overlap_20", depth_grid
            )
            rows.append({
                "cohort": cohort,
                "target": target,
                "direction": direction,
                "models": models,
                "physics_max": physics_max,
                "alignment_max": alignment_max,
            })

        with plt.rc_context({
            "font.size": 12.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
        }):
            return physics_plot.draw_summary(rows)

    figure_2 = make_figure_2()
    save_figure(figure_2, "max_physics_vs_alignment")
    figure_2
    return (figure_2,)


@app.cell
def _(DATA, fs, np, pd, plt, save_figure):
    # FIGURE 3: saved-stage selection loss and redshift-transfer distributions.
    def make_figure_3():
        selection = pd.read_csv(DATA / "E3_inference_analysis.csv")
        redshift = pd.read_csv(DATA / "E8_model_analysis.csv")
        order = [
            ("legacysurvey", "g_minus_r", "hsc_to_other"),
            ("legacysurvey", "g_minus_r", "other_to_hsc"),
            ("legacysurvey", "log_shape", "hsc_to_other"),
            ("legacysurvey", "log_shape", "other_to_hsc"),
            ("jwst", "mag_auto", "hsc_to_other"),
            ("jwst", "mag_auto", "other_to_hsc"),
            ("jwst", "log_flux_radius", "hsc_to_other"),
            ("jwst", "log_flux_radius", "other_to_hsc"),
        ]
        target_name = {
            "g_minus_r": "color", "log_shape": "shape",
            "mag_auto": "magnitude", "log_flux_radius": "size",
        }
        survey_name = {"legacysurvey": "Legacy", "jwst": "JWST"}
        selection = selection.set_index(
            ["cohort", "target", "direction"]
        ).loc[order].reset_index()
        selection["loss"] = -selection["mean_regret"]
        selection["loss_low"] = -selection["family_bootstrap_high"]
        selection["loss_high"] = -selection["family_bootstrap_low"]

        labels = []
        for survey, target, direction in order:
            ends = ("HSC", survey_name[survey])
            left, right = ends if direction == "hsc_to_other" else ends[::-1]
            labels.append(f"{target_name[target]} · {left} → {right}")

        figure = plt.figure(figsize=(5.5, 2.9), layout="constrained")
        outer = figure.add_gridspec(
            1, 2, width_ratios=[1.15, 1.0], wspace=0.02
        )
        axis_a = figure.add_subplot(outer[0])
        right = outer[1].subgridspec(
            1, 2, width_ratios=[1.12, 0.88], wspace=0.06
        )
        axis_cross = figure.add_subplot(right[0])
        axis_within = figure.add_subplot(right[1], sharey=axis_cross)
        figure.get_layout_engine().set(w_pad=0.02)

        for row, record in selection.iterrows():
            color = fs.SURVEY_COLOR[record["cohort"]]
            left = record["loss"] - record["loss_low"]
            right_error = record["loss_high"] - record["loss"]
            axis_a.barh(row, record["loss"], height=0.44, color=color,
                        alpha=0.80, edgecolor="white", linewidth=0.5, zorder=2)
            axis_a.errorbar(
                record["loss"], row,
                xerr=np.array([[left], [right_error]]), fmt="none",
                ecolor=color, elinewidth=1.05, capsize=2.0,
                capthick=0.8, zorder=3,
            )

        axis_a.axvspan(0, 8, color=fs.WASH_RED, alpha=0.55, zorder=0)
        axis_a.axvline(0, color=fs.INK, linewidth=0.8)
        for divider in [1.5, 3.5, 5.5]:
            axis_a.axhline(divider, color=fs.GRID, linewidth=0.6)
        axis_a.set_xscale("symlog", linthresh=0.1, linscale=1.0)
        axis_a.set_xlim(-0.018, 7)
        axis_a.set_xticks([0, 0.05, 0.1, 0.3, 1, 3, 6],
                          ["0", "0.05", "0.1", "0.3", "1", "3", "6"])
        axis_a.minorticks_off()
        axis_a.set_yticks(np.arange(8), labels)
        axis_a.tick_params(axis="y", length=0)
        axis_a.set_ylim(7.7, -0.7)
        axis_a.set_xlabel("R² lost by selecting the saved stage using geometry")
        fs.style_axis(axis_a, grid="x")
        fs.panel_letter(axis_a, "a", x=-0.42, y=1.04)

        within = np.sort(redshift["within_view_redshift_r2"].to_numpy())
        across = np.sort(redshift["cross_view_redshift_r2"].to_numpy())
        fraction = np.arange(1, len(redshift) + 1) / len(redshift)

        axis_cross.axvspan(-32, -0.28, color=fs.WASH_RED, zorder=0)
        axis_cross.step(across, fraction, where="post", color=fs.RED,
                        linewidth=1.7)
        axis_cross.set_xscale("symlog", linthresh=1.0, linscale=1.4)
        axis_cross.set_xlim(-32, -0.28)
        axis_cross.set_xticks([-30, -3, -0.3], ["−30", "−3", "−0.3"])
        axis_cross.minorticks_off()
        axis_cross.set_ylim(0, 1.02)
        axis_cross.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axis_cross.set_xlabel("cross-survey R²")
        axis_cross.set_ylabel("fraction of models at or below score")
        axis_cross.text(0.5, 1.03, "across surveys",
                        transform=axis_cross.transAxes, ha="center",
                        va="bottom", fontsize=6.1, color=fs.RED)
        fs.style_axis(axis_cross)
        fs.panel_letter(axis_cross, "b", x=-0.22, y=1.04)

        axis_within.step(within, fraction, where="post", color="#71889b",
                         linewidth=2.0)
        axis_within.set_xlim(0.32, 0.61)
        axis_within.set_xticks([0.35, 0.45, 0.55])
        axis_within.set_xlabel("within-survey R²")
        axis_within.text(0.5, 1.03, "within survey",
                         transform=axis_within.transAxes, ha="center",
                         va="bottom", fontsize=6.1, color="#71889b")
        fs.style_axis(axis_within)
        axis_within.spines["left"].set_visible(False)
        axis_within.tick_params(axis="y", left=False, labelleft=False)

        mark = 0.018
        axis_cross.plot(
            (1 - mark, 1 + mark), (-mark, mark),
            transform=axis_cross.transAxes, color=fs.AXIS,
            clip_on=False, linewidth=0.8,
        )
        axis_within.plot(
            (-mark, mark), (-mark, mark),
            transform=axis_within.transAxes, color=fs.AXIS,
            clip_on=False, linewidth=0.8,
        )
        return figure

    figure_3 = make_figure_3()
    save_figure(figure_3, "04_scientific_transfer")
    figure_3
    return (figure_3,)


if __name__ == "__main__":
    app.run()
