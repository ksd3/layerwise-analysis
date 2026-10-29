"""Summarize E5 consecutive-layer update geometry and transfer."""

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

    updates = pd.read_csv(results_dir / "E5_updates.csv")
    transfer = pd.read_csv(results_dir / "E5_transfer_cross_family_inference.csv")
    usable = updates[updates["available"] & updates["corrected_overlap_20"].notna()].copy()

    # Find the block transition with the strongest update alignment in each curve.
    peak_rows = usable.groupby(["cohort", "model"])["corrected_overlap_20"].idxmax()
    peaks = usable.loc[peak_rows].copy()
    columns = [
        "cohort", "model", "family", "transition", "normalized_depth",
        "from_layer", "to_layer", "corrected_overlap_20",
        "maximum_full_alignment", "maximum_update_alignment",
        "update_to_full_ratio", "update_change_correlation",
        "update_change_p_value", "curve_p_value_holm",
    ]
    peaks[columns].to_csv(output_dir / "E5_model_update_peaks.csv", index=False)

    cohort = peaks.groupby("cohort").agg(
        models=("model", "nunique"),
        mean_maximum_update_alignment=("maximum_update_alignment", "mean"),
        median_update_to_full_ratio=("update_to_full_ratio", "median"),
        median_peak_depth=("normalized_depth", "median"),
        mean_update_change_correlation=("update_change_correlation", "mean"),
        significant_curves=("curve_p_value_holm", lambda x: int((x <= 0.05).sum())),
    ).reset_index()
    cohort.to_csv(output_dir / "E5_cohort_analysis.csv", index=False)

    transfer["interval_excludes_zero"] = (
        (transfer["family_bootstrap_low"] > 0)
        | (transfer["family_bootstrap_high"] < 0)
    )
    transfer.to_csv(output_dir / "E5_transfer_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = sorted(peaks["cohort"].unique())
    values = [peaks.loc[peaks["cohort"] == name, "update_to_full_ratio"] for name in names]
    ax.boxplot(values, tick_labels=names, showfliers=False)
    ax.axhline(1, color="black", linewidth=1)
    ax.set_ylabel("Maximum update alignment / maximum full-state alignment")
    ax.set_title("E5: relative strength of layer updates")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "E5_update_ratios.png", dpi=160)
    plt.close(fig)

    significant = int((peaks["curve_p_value_holm"] <= 0.05).sum())
    correlated = int((peaks["update_change_p_value"] <= 0.05).sum())
    text = (
        f"Significant update-alignment curves: {significant} of {len(peaks)}.\n"
        f"Curves where update alignment predicts the next full-state change at p <= 0.05: {correlated}.\n"
        "Peak transitions identify the blocks most strongly writing shared geometry.\n"
    )
    (output_dir / "E5_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
