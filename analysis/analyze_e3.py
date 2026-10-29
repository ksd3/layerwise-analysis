"""Summarize E3 label-free image-to-image layer selection."""

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

    transfer = pd.read_csv(results_dir / "E3_image_transfer.csv")
    inference = pd.read_csv(results_dir / "E3_cross_family_inference.csv")

    # Negative regret means label-free selection transferred worse than the oracle.
    inference["interval_excludes_zero"] = (
        (inference["family_bootstrap_low"] > 0)
        | (inference["family_bootstrap_high"] < 0)
    )
    inference["direction_of_regret"] = "uncertain"
    inference.loc[inference["family_bootstrap_high"] < 0, "direction_of_regret"] = "label_free_worse"
    inference.loc[inference["family_bootstrap_low"] > 0, "direction_of_regret"] = "label_free_better"
    inference.to_csv(output_dir / "E3_inference_analysis.csv", index=False)

    model = transfer.groupby(["cohort", "model", "family", "target", "direction"]).agg(
        label_free_test_r2=("label_free_test_r2", "mean"),
        oracle_test_r2=("oracle_test_r2", "mean"),
        regret=("regret", "mean"),
    ).reset_index()
    model.to_csv(output_dir / "E3_model_transfer.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    for cohort, group in model.groupby("cohort"):
        ax.scatter(group["oracle_test_r2"], group["label_free_test_r2"], s=15, alpha=0.6, label=cohort)
    low = min(model["oracle_test_r2"].min(), model["label_free_test_r2"].min())
    high = max(model["oracle_test_r2"].max(), model["label_free_test_r2"].max())
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_xlabel("Oracle-selected test R squared")
    ax.set_ylabel("Label-free-selected test R squared")
    ax.set_title("E3: label-free selection versus oracle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "E3_label_free_vs_oracle.png", dpi=160)
    plt.close(fig)

    worse = int((inference["direction_of_regret"] == "label_free_worse").sum())
    better = int((inference["direction_of_regret"] == "label_free_better").sum())
    uncertain = len(inference) - worse - better
    text = (
        f"Cross-family comparisons with label-free selection worse than oracle: {worse}.\n"
        f"Comparisons with label-free selection better than oracle: {better}.\n"
        f"Comparisons whose interval includes zero: {uncertain}.\n"
    )
    (output_dir / "E3_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
