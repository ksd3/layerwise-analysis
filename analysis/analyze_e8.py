"""Summarize E8 final-representation scientific transfer."""

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

    models = pd.read_csv(results_dir / "E8_transfer_by_model.csv")
    regression = pd.read_csv(results_dir / "E8_alignment_transfer_regression.csv")
    all_targets = pd.read_csv(results_dir / "E8_all_target_regressions.csv")
    leave_out = pd.read_csv(results_dir / "E8_leave_family_out.csv")

    models["alignment_significant"] = models["alignment_p_holm"] <= 0.05
    models["transfer_gap"] = models["cross_view_redshift_r2"] - models["within_view_redshift_r2"]
    models.to_csv(output_dir / "E8_model_analysis.csv", index=False)

    regression["interval_excludes_zero"] = (
        (regression["family_bootstrap_low"] > 0)
        | (regression["family_bootstrap_high"] < 0)
    )
    regression.to_csv(output_dir / "E8_primary_regression_analysis.csv", index=False)

    all_targets["interval_excludes_zero"] = (
        (all_targets["family_bootstrap_low"] > 0)
        | (all_targets["family_bootstrap_high"] < 0)
    )
    all_targets["bh_significant"] = all_targets["secondary_p_bh"] <= 0.05
    all_targets.to_csv(output_dir / "E8_all_target_analysis.csv", index=False)

    leave_out["model_beats_mean_baseline"] = leave_out["model_mse"] < leave_out["mean_baseline_mse"]
    leave_out.to_csv(output_dir / "E8_leave_family_out_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for family, group in models.groupby("family"):
        ax.scatter(group["alignment"], group["cross_view_redshift_r2"], s=35, label=family)
    ax.set_xlabel("Corrected cross-view neighbor overlap")
    ax.set_ylabel("Cross-view redshift R squared")
    ax.set_title("E8: alignment versus scientific transfer")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "E8_alignment_vs_transfer.png", dpi=160)
    plt.close(fig)

    alignment_row = regression[regression["term"] == "alignment"].iloc[0]
    wins = int(leave_out["model_beats_mean_baseline"].sum())
    text = (
        f"Models with Holm-significant alignment: {int(models['alignment_significant'].sum())} of {len(models)}.\n"
        f"Adjusted alignment coefficient: {alignment_row['coefficient']:.6g}; permutation p={alignment_row['permutation_p']:.6g}.\n"
        f"Leave-family-out predictions beat the mean baseline for {wins} of {len(leave_out)} families.\n"
    )
    (output_dir / "E8_summary.txt").write_text(text)


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "results",
            sys.argv[2] if len(sys.argv) > 2 else "analysis_output")
