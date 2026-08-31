"""Summarize E9 robustness and explanatory controls."""

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

    controls = pd.read_csv(results_dir / "E9_controls.csv")
    inference = pd.read_csv(results_dir / "E9_paired_control_inference.csv")

    # Differences are control minus reference. Their sign shows what each control changes.
    inference["interval_excludes_zero"] = (
        (inference["family_bootstrap_low"] > 0)
        | (inference["family_bootstrap_high"] < 0)
    )
    inference["holm_significant"] = inference["paired_p_holm"] <= 0.05
    inference.to_csv(output_dir / "E9_control_inference_analysis.csv", index=False)

    control_summary = controls.groupby("control").agg(
        models=("model", "nunique"),
        mean_corrected_overlap_20=("corrected_overlap_20", "mean"),
        mean_cka=("cka", "mean"),
        mean_procrustes=("procrustes", "mean"),
        mean_mrr=("mrr", "mean"),
    ).reset_index().sort_values("mean_corrected_overlap_20")
    control_summary.to_csv(output_dir / "E9_control_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(control_summary))))
    ax.barh(control_summary["control"], control_summary["mean_corrected_overlap_20"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean corrected kNN overlap, k=20")
    ax.set_title("E9: alignment after controls")
    fig.tight_layout()
    fig.savefig(output_dir / "E9_control_alignment.png", dpi=160)
    plt.close(fig)

    robust = inference[inference["interval_excludes_zero"] & inference["holm_significant"]]
    text = (
        f"Control/metric comparisons: {len(inference)}.\n"
        f"Comparisons supported by both the family interval and Holm p <= 0.05: {len(robust)}.\n"
        "Compare raw alignment with shuffled, projected, PC-removed, and residualized controls before accepting a primary claim.\n"
    )
    (output_dir / "E9_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
