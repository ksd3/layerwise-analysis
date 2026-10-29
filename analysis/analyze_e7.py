"""Summarize E7 large-cohort layerwise replication."""

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

    curves = pd.read_csv(results_dir / "E7_large_cohort_curves.csv")
    maps = pd.read_csv(results_dir / "E7_layer_pair_maps.csv")
    audit = pd.read_csv(results_dir / "E7_extraction_shape_audit.csv")
    usable = curves[curves["available"] & curves["corrected_overlap_20"].notna()].copy()

    # Extract each model's best same-depth layer on the 56,322-object cohort.
    peak_rows = usable.groupby("model")["corrected_overlap_20"].idxmax()
    peaks = usable.loc[peak_rows].copy()
    peaks.to_csv(output_dir / "E7_model_peaks.csv", index=False)

    # Also locate the strongest left-layer/right-layer pair in every saved map.
    pair_rows = maps.groupby(["cohort", "model"])["corrected_overlap_20"].idxmax()
    best_pairs = maps.loc[pair_rows].copy()
    best_pairs.to_csv(output_dir / "E7_best_layer_pairs.csv", index=False)

    audit.to_csv(output_dir / "E7_shape_analysis.csv", index=False)

    usable["depth_bin"] = np.minimum((usable["normalized_depth"] * 20).astype(int), 19)
    average = usable.groupby("depth_bin", as_index=False).agg(
        normalized_depth=("normalized_depth", "mean"),
        mean_alignment=("corrected_overlap_20", "mean"),
        standard_error=("corrected_overlap_20", lambda x: x.std() / np.sqrt(len(x))),
    )
    average.to_csv(output_dir / "E7_average_depth_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(average["normalized_depth"], average["mean_alignment"], marker="o")
    ax.fill_between(
        average["normalized_depth"],
        average["mean_alignment"] - average["standard_error"],
        average["mean_alignment"] + average["standard_error"],
        alpha=0.2,
    )
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Mean corrected kNN overlap, k=20")
    ax.set_title("E7: large-cohort alignment through depth")
    fig.tight_layout()
    fig.savefig(output_dir / "E7_large_cohort_depth_curve.png", dpi=160)
    plt.close(fig)

    significant = int((peaks["curve_p_value_holm"] <= 0.05).sum())
    same_shape = int(audit["same_shape"].sum())
    text = (
        f"Large-cohort model curves: {len(peaks)}.\n"
        f"Holm-significant large-cohort curves: {significant}.\n"
        f"Layerwise/final extraction products with identical shape: {same_shape} of {len(audit)}.\n"
    )
    (output_dir / "E7_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
