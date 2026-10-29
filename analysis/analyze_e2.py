"""Summarize E2 tests of the universal-depth hypothesis."""

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

    depth = pd.read_csv(results_dir / "E2_depth_summary.csv")
    family = pd.read_csv(results_dir / "E2_family_cohort_summary.csv")
    sensitivity = pd.read_csv(results_dir / "E2_metric_sensitivity.csv")

    # The primary quantity is final-quarter alignment minus early-quarter alignment.
    cohort = depth.groupby("cohort").agg(
        models=("model", "nunique"),
        mean_final_minus_early=("final_minus_early", "mean"),
        median_final_minus_early=("final_minus_early", "median"),
        fraction_positive=("final_minus_early", lambda x: (x > 0).mean()),
        median_peak_depth=("peak_depth", "median"),
        fraction_final_in_peak_set=("final_in_peak_set", "mean"),
    ).reset_index()
    cohort.to_csv(output_dir / "E2_cohort_analysis.csv", index=False)

    metric = sensitivity.groupby(["cohort", "metric"]).agg(
        models=("model", "nunique"),
        mean_final_minus_early=("final_minus_early", "mean"),
        fraction_positive=("final_minus_early", lambda x: (x > 0).mean()),
        median_peak_depth=("peak_depth", "median"),
    ).reset_index()
    metric.to_csv(output_dir / "E2_metric_analysis.csv", index=False)

    # This table shows whether individual architecture families follow the overall sign.
    family.to_csv(output_dir / "E2_family_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = sorted(depth["cohort"].unique())
    values = [depth.loc[depth["cohort"] == name, "final_minus_early"] for name in names]
    ax.boxplot(values, tick_labels=names, showfliers=False)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Final-quarter minus early-quarter alignment")
    ax.set_title("E2: depth contrast by cohort")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "E2_depth_contrasts.png", dpi=160)
    plt.close(fig)

    all_positive = bool((cohort["mean_final_minus_early"] > 0).all())
    consistent_families = int(family["same_direction_as_overall"].sum())
    text = (
        f"All cohort means are positive: {all_positive}.\n"
        f"Families matching the overall direction: {consistent_families} of {len(family)} family/cohort groups.\n"
        "A universal increasing-depth claim requires consistent positive effects across cohorts, families, and metrics.\n"
    )
    (output_dir / "E2_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
