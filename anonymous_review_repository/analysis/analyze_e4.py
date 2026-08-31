"""Summarize E4 image-spectrum common-space selection."""

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

    transfer = pd.read_csv(results_dir / "E4_spectrum_transfer.csv")
    inference = pd.read_csv(results_dir / "E4_cross_family_inference.csv")

    # DESI is geometry-only; SDSS also has redshift-transfer measurements.
    geometry = transfer.groupby(["cohort", "model", "family"]).agg(
        mean_test_mrr=("test_mrr", "mean"),
        selected_layer_median=("label_free_layer", "median"),
    ).reset_index()
    geometry.to_csv(output_dir / "E4_model_geometry.csv", index=False)

    inference["interval_excludes_zero"] = (
        (inference["family_bootstrap_low"] > 0)
        | (inference["family_bootstrap_high"] < 0)
    )
    inference["direction_of_regret"] = "uncertain"
    inference.loc[inference["family_bootstrap_high"] < 0, "direction_of_regret"] = "label_free_worse"
    inference.loc[inference["family_bootstrap_low"] > 0, "direction_of_regret"] = "label_free_better"
    inference.to_csv(output_dir / "E4_inference_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = sorted(geometry["cohort"].unique())
    values = [geometry.loc[geometry["cohort"] == name, "mean_test_mrr"] for name in names]
    ax.boxplot(values, tick_labels=names, showfliers=False)
    ax.set_ylabel("Mean held-out paired retrieval MRR")
    ax.set_title("E4: learned common-space retrieval")
    fig.tight_layout()
    fig.savefig(output_dir / "E4_common_space_retrieval.png", dpi=160)
    plt.close(fig)

    worse = int((inference["direction_of_regret"] == "label_free_worse").sum())
    uncertain = int((inference["direction_of_regret"] == "uncertain").sum())
    text = (
        f"E4 contains {len(geometry)} model/cohort geometry summaries.\n"
        f"SDSS transfer comparisons significantly worse than oracle: {worse}.\n"
        f"SDSS transfer comparisons whose interval includes zero: {uncertain}.\n"
    )
    (output_dir / "E4_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
