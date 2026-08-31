"""Summarize E6 architecture, objective, and scale effects."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def analyze(results_dir, output_dir):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regression = pd.read_csv(results_dir / "E6_regression.csv")
    training = pd.read_csv(results_dir / "E6_training_effects.csv")
    leave_out = pd.read_csv(results_dir / "E6_leave_family_out.csv")

    # Require both the permutation test and family-bootstrap interval to agree.
    regression["interval_excludes_zero"] = (
        (regression["family_bootstrap_low"] > 0)
        | (regression["family_bootstrap_high"] < 0)
    )
    regression["supported_effect"] = (
        (regression["permutation_p"] <= 0.05)
        & regression["interval_excludes_zero"]
    )
    regression.to_csv(output_dir / "E6_regression_analysis.csv", index=False)

    training["interval_excludes_zero"] = (
        (training["family_bootstrap_low"] > 0)
        | (training["family_bootstrap_high"] < 0)
    )
    training.to_csv(output_dir / "E6_training_effect_analysis.csv", index=False)

    leave_out["model_beats_mean_baseline"] = leave_out["model_mse"] < leave_out["mean_baseline_mse"]
    leave_summary = leave_out.groupby("outcome").agg(
        held_out_families=("left_out_family", "nunique"),
        fraction_beating_baseline=("model_beats_mean_baseline", "mean"),
        mean_model_mse=("model_mse", "mean"),
        mean_baseline_mse=("mean_baseline_mse", "mean"),
    ).reset_index()
    leave_summary.to_csv(output_dir / "E6_leave_family_out_analysis.csv", index=False)

    # Plot the non-intercept coefficients for the primary maximum-alignment outcome.
    plot = regression[
        (regression["outcome"] == "maximum")
        & regression["family_bootstrap_low"].notna()
        & (regression["term"] != "intercept")
    ].copy()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot))))
    y = range(len(plot))
    lower = plot["coefficient"] - plot["family_bootstrap_low"]
    upper = plot["family_bootstrap_high"] - plot["coefficient"]
    ax.errorbar(
        plot["coefficient"].to_numpy(),
        list(y),
        xerr=[lower.to_numpy(), upper.to_numpy()],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(list(y), plot["term"])
    ax.set_xlabel("Coefficient with family-bootstrap interval")
    ax.set_title("E6: predictors of maximum alignment")
    fig.tight_layout()
    fig.savefig(output_dir / "E6_maximum_alignment_coefficients.png", dpi=160)
    plt.close(fig)

    supported = regression[regression["supported_effect"]]
    training_supported = training[training["interval_excludes_zero"]]
    text = (
        f"Regression effects supported by both permutation p <= 0.05 and the bootstrap interval: {len(supported)}.\n"
        f"Training comparisons whose family-bootstrap interval excludes zero: {len(training_supported)}.\n"
        "Use the leave-family-out table to judge whether these relationships generalize beyond known families.\n"
    )
    (output_dir / "E6_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
