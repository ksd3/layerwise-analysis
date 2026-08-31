"""Summarize E1 layerwise cross-survey geometry."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze(results_dir, output_dir):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep usable layers and find the strongest k=20 layer for each model/cohort.
    curves = pd.read_csv(results_dir / "E1_all_curves.csv")
    curves = curves[curves["available"] & curves["corrected_overlap_20"].notna()].copy()
    peak_rows = curves.groupby(["cohort", "model"])["corrected_overlap_20"].idxmax()
    peaks = curves.loc[peak_rows].copy()
    peaks.to_csv(output_dir / "E1_model_peaks.csv", index=False)

    # Count statistically calibrated curves and summarize peak size and location.
    summary = peaks.groupby("cohort").agg(
        models=("model", "nunique"),
        median_peak_alignment=("corrected_overlap_20", "median"),
        mean_peak_alignment=("corrected_overlap_20", "mean"),
        median_peak_depth=("normalized_depth", "median"),
        significant_models=("curve_p_value_holm", lambda x: int((x <= 0.05).sum())),
    ).reset_index()
    summary["significant_fraction"] = summary["significant_models"] / summary["models"]
    summary.to_csv(output_dir / "E1_cohort_summary.csv", index=False)

    # Average curves in 20 normalized-depth bins so different architectures align.
    curves["depth_bin"] = np.minimum((curves["normalized_depth"] * 20).astype(int), 19)
    average_curve = curves.groupby(["cohort", "depth_bin"], as_index=False).agg(
        normalized_depth=("normalized_depth", "mean"),
        mean_alignment=("corrected_overlap_20", "mean"),
        standard_error=("corrected_overlap_20", lambda x: x.std() / np.sqrt(len(x))),
    )
    average_curve.to_csv(output_dir / "E1_average_depth_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cohort, group in average_curve.groupby("cohort"):
        ax.plot(group["normalized_depth"], group["mean_alignment"], marker="o", label=cohort)
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Mean corrected kNN overlap, k=20")
    ax.set_title("E1: mean alignment through depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "E1_depth_curves.png", dpi=160)
    plt.close(fig)

    total = len(peaks)
    significant = int((peaks["curve_p_value_holm"] <= 0.05).sum())
    text = (
        f"E1 contains {total} model/cohort curves.\n"
        f"{significant} curves ({significant / total:.1%}) have Holm-corrected p <= 0.05.\n"
        "Use E2 and E7 to decide whether the peak consistently moves toward late depth.\n"
    )
    (output_dir / "E1_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
