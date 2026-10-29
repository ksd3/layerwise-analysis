"""E2: summarize how alignment changes with normalized model depth.

This experiment does not reopen the embedding matrices. It reads the E1 curve
table, reduces each model/cohort curve to interpretable depth summaries, and
then asks whether the early-to-late change is consistent across architecture
families and cohorts.

The main blocks are:
1. Turn one layer curve into early, middle, final, peak, and area summaries.
2. Bootstrap architecture families rather than treating checkpoints as
   independent observations.
3. Fit cohort fixed effects with family and family-by-cohort random effects.
4. Repeat summaries for alternative metrics and leave one family out at a time.

Approximate runtime: ordinary curve summaries take seconds. The Powell mixed
model and its leave-family-out refits normally make the complete stage take a
few seconds to a few minutes; on a difficult fit it could take tens of minutes.
This stage is small and CPU-only, so GPU acceleration would not be useful.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# -----------------------------------------------------------------------------
# Reduce one depth curve to a short scientific summary
#
# Intention: describe where alignment appears in a network without comparing architectures that have different raw layer counts. Normalized depth maps every model to the common interval 0 to 1. This block takes milliseconds per curve and only works with the small E1 CSV table.
# -----------------------------------------------------------------------------

def summarize_curve(curve, value_column="corrected_overlap_20"):
    curve = curve.sort_values("normalized_depth")
    depth = curve["normalized_depth"].to_numpy()
    value = curve[value_column].to_numpy()
    finite = np.isfinite(value)
    if "available" in curve:
        finite &= curve["available"].astype(bool).to_numpy()
    depth = depth[finite]
    value = value[finite]
    if len(value) == 0:
        return None

    # Divide the normalized curve into coarse depth regions. The open middle interval prevents boundary layers from being counted twice.
    early = value[depth <= 0.25]
    middle = value[(depth > 0.25) & (depth < 0.75)]
    final = value[depth >= 0.75]
    peak = int(np.nanargmax(value))
    # The trapezoidal area preserves both the height and breadth of alignment. final_minus_early is the primary directional depth-change statistic.
    result = {
        "early_mean": float(np.mean(early)) if len(early) else np.nan,
        "middle_mean": float(np.mean(middle)) if len(middle) else np.nan,
        "final_mean": float(np.mean(final)) if len(final) else np.nan,
        "final_minus_early": float(np.mean(final) - np.mean(early)),
        "area_under_curve": float(np.trapezoid(value, depth)),
        "maximum": float(value[peak]),
        "final_layer_alignment": float(value[-1]),
        "peak_depth": float(depth[peak]),
    }

    # E1's simultaneous band defines a set of layers that cannot be separated from the peak. This is more cautious than declaring the raw maximum alone.
    if "simultaneous_low" in curve and "simultaneous_high" in curve:
        low = curve.loc[finite, "simultaneous_low"].to_numpy()
        high = curve.loc[finite, "simultaneous_high"].to_numpy()
        peak_set = high >= np.max(low)
        result["final_in_peak_set"] = bool(peak_set[-1])
    else:
        result["final_in_peak_set"] = bool(peak == len(value) - 1)
    return result


# -----------------------------------------------------------------------------
# Family-level uncertainty and the hierarchical cohort model
#
# Intention: avoid pretending that several sizes/checkpoints from one model family are independent replicates. The bootstrap samples whole families. The mixed model treats cohort as a fixed comparison and allows both a family intercept and a family-specific cohort departure. A bootstrap takes seconds; a mixed-model fit is usually seconds to minutes.
# -----------------------------------------------------------------------------

def family_bootstrap(summary, draws, seed):
    """Confidence interval with architecture family as the sampled unit."""
    family_values = summary.groupby("family")["final_minus_early"].mean().dropna()
    if len(family_values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        picked = rng.choice(family_values.to_numpy(), len(family_values), replace=True)
        values.append(np.mean(picked))
    return tuple(np.quantile(values, [0.025, 0.975]))


def hierarchical_table(summary):
    """Fit cohort effects with family and family-by-cohort random effects."""
    data = summary.dropna(subset=["final_minus_early", "family", "cohort"]).copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "term",
                "component",
                "coefficient",
                "standard_error",
                "p_value",
                "converged",
            ]
        )

    # A one-family synthetic smoke test cannot estimate random effects.
    if data["family"].nunique() < 2:
        means = data.groupby("cohort")["final_minus_early"].mean()
        return pd.DataFrame(
            {
                "term": [f"cohort_mean[{name}]" for name in means.index],
                "component": "smoke_only_mean",
                "coefficient": means.to_numpy(),
                "standard_error": np.nan,
                "p_value": np.nan,
                "converged": np.nan,
            }
        )

    # Families are the independent groups. The variance component lets every family have its own cohort departure without fitting a saturated OLS. Fixed effects estimate average cohort differences. The group intercept captures family-to-family shifts, while family_by_cohort captures a family reacting differently in different observational cohorts.
    model = smf.mixedlm(
        "final_minus_early ~ C(cohort)",
        data,
        groups=data["family"],
        re_formula="1",
        vc_formula={"family_by_cohort": "0 + C(cohort)"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        fitted = model.fit(reml=True, method="powell", maxiter=1000, disp=False)

    # Report fixed-effect coefficients and variance components in one simple long table so they can be inspected without a statsmodels result object.
    fixed = pd.DataFrame(
        {
            "term": fitted.fe_params.index,
            "component": "fixed_cohort_effect",
            "coefficient": fitted.fe_params.to_numpy(),
            "standard_error": fitted.bse_fe.to_numpy(),
            "p_value": fitted.pvalues.loc[fitted.fe_params.index].to_numpy(),
            "converged": fitted.converged,
        }
    )
    variances = pd.DataFrame(
        {
            "term": [
                "family_variance",
                "family_by_cohort_variance",
                "residual_variance",
            ],
            "component": "variance",
            "coefficient": [
                fitted.cov_re.iloc[0, 0],
                fitted.vcomp[0],
                fitted.scale,
            ],
            "standard_error": np.nan,
            "p_value": np.nan,
            "converged": fitted.converged,
        }
    )
    return pd.concat([fixed, variances], ignore_index=True)


# -----------------------------------------------------------------------------
# Complete E2 stage
#
# Intention: create the primary depth summary, metric sensitivity tables, hierarchical fit, leave-family-out sensitivity, and readable cohort/family aggregates. CSV reading and grouping are seconds; repeated mixed-model fits dominate the stage and usually finish within a few minutes.
# -----------------------------------------------------------------------------

def run(context):
    curves = pd.read_csv(Path(context["output_root"]) / "E1_all_curves.csv")
    rows = []
    for (cohort, model), curve in curves.groupby(["cohort", "model"]):
        numbers = summarize_curve(curve)
        if numbers is None:
            continue
        numbers.update(cohort=cohort, model=model, family=context["model_family"](model))
        numbers["maximum_p_value"] = curve["curve_p_value"].iloc[0]
        numbers["maximum_p_value_holm"] = curve["curve_p_value_holm"].iloc[0]
        rows.append(numbers)

    summary = pd.DataFrame(rows)
    low, high = family_bootstrap(summary, context["bootstraps"], context["seed"])
    summary["all_family_bootstrap_low"] = low
    summary["all_family_bootstrap_high"] = high
    summary.to_csv(Path(context["output_root"]) / "E2_depth_summary.csv", index=False)

    # Repeating the same curve reduction for CKA, Procrustes, and MRR checks that the qualitative depth conclusion is not peculiar to neighbor overlap.
    sensitivity_rows = []
    for metric in ("corrected_overlap_20", "cka", "procrustes", "mrr"):
        for (cohort, model), curve in curves.groupby(["cohort", "model"]):
            numbers = summarize_curve(curve, metric)
            if numbers is not None:
                numbers.update(
                    cohort=cohort,
                    model=model,
                    family=context["model_family"](model),
                    metric=metric,
                )
                sensitivity_rows.append(numbers)
    pd.DataFrame(sensitivity_rows).to_csv(
        Path(context["output_root"]) / "E2_metric_sensitivity.csv", index=False
    )

    hierarchical_table(summary).to_csv(
        Path(context["output_root"]) / "E2_hierarchical_regression.csv", index=False
    )

    # Refit after excluding each family. Large changes here reveal that the hierarchical result depends heavily on one architecture lineage.
    leave_out = []
    for family in sorted(summary["family"].dropna().unique()):
        remaining = summary[summary["family"] != family]
        refitted = hierarchical_table(remaining)
        if not refitted.empty:
            refitted.insert(0, "left_out_family", family)
            leave_out.append(refitted)
    leave_out_columns = [
        "left_out_family",
        "term",
        "component",
        "coefficient",
        "standard_error",
        "p_value",
        "converged",
    ]
    leave_out_table = (
        pd.concat(leave_out, ignore_index=True)
        if leave_out
        else pd.DataFrame(columns=leave_out_columns)
    )
    leave_out_table.to_csv(
        Path(context["output_root"]) / "E2_leave_family_out.csv", index=False
    )

    # These last two files contain descriptive means only. They are convenient for plots and manual review and normally take much less than one second.
    family_summary = (
        summary.groupby(["cohort", "family"], as_index=False)
        .agg(
            models=("model", "nunique"),
            mean_final_minus_early=("final_minus_early", "mean"),
            median_peak_depth=("peak_depth", "median"),
        )
    )
    overall_sign = np.sign(summary["final_minus_early"].mean())
    family_summary["same_direction_as_overall"] = (
        np.sign(family_summary["mean_final_minus_early"]) == overall_sign
    )
    family_summary.to_csv(
        Path(context["output_root"]) / "E2_family_cohort_summary.csv", index=False
    )

    cohort_summary = (
        summary.groupby("cohort", as_index=False)
        .agg(
            models=("model", "size"),
            mean_final_minus_early=("final_minus_early", "mean"),
            median_peak_depth=("peak_depth", "median"),
        )
    )
    cohort_summary.to_csv(
        Path(context["output_root"]) / "E2_cohort_summary.csv", index=False
    )
    return summary
