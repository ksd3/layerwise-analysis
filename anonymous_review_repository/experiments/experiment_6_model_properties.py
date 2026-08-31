"""E6: relate alignment to model family, training, width, and scale.

This is a model-level analysis, not an embedding-level calculation. It combines
E1, E2, and E5 summaries with hand-written model metadata and asks whether scale,
width, family, cohort, or training objective predicts representation alignment.

The main blocks are:
1. Build an ordinary least-squares design with scale, width, cohort, and family.
2. Calibrate the scale coefficient with within-family permutations and
   whole-family bootstraps.
3. Evaluate prediction after holding out an entire architecture family.
4. Summarize training-objective and astronomy-specific differences.

Approximate runtime: individual regressions are milliseconds because the table
has one row per model/cohort. Repeating them for 9,999 permutations, 2,000
bootstraps, and several outcomes normally takes a few minutes to roughly an hour.
This stage is CPU-only and is too small for useful GPU acceleration.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTCOMES = (
    "maximum",
    "final_layer_alignment",
    "area_under_curve",
    "peak_depth",
    "final_minus_early",
    "maximum_update_alignment",
)


# -----------------------------------------------------------------------------
# Regression design and one ordinary least-squares fit
#
# Intention: estimate a model-scale association while adjusting for embedding width, cohort, scale-by-cohort interactions, and optionally family indicators. Log10 makes multiplicative size differences linear. One fit normally takes milliseconds; it is repeated many times during inference below.
# -----------------------------------------------------------------------------

def regression_design(data, include_family=True):
    numeric = pd.DataFrame(
        {
            "intercept": np.ones(len(data)),
            "log_parameters": np.log10(data["parameters_m"].to_numpy()),
            "log_width": np.log10(data["width"].to_numpy()),
        },
        index=data.index,
    )
    # Drop one indicator from each categorical variable to avoid an exact linear dependency with the intercept.
    cohort = pd.get_dummies(data["cohort"], prefix="cohort", drop_first=True, dtype=float)
    interactions = cohort.mul(numeric["log_parameters"], axis=0)
    interactions.columns = [f"log_parameters_x_{name}" for name in interactions.columns]
    pieces = [numeric, cohort, interactions]
    if include_family:
        pieces.append(
            pd.get_dummies(data["family"], prefix="family", drop_first=True, dtype=float)
        )
    return pd.concat(pieces, axis=1)


def fit_regression(data, outcome, include_family=True):
    needed = [outcome, "parameters_m", "width", "family", "cohort"]
    data = data.dropna(subset=needed).copy()
    if len(data) < 4:
        return pd.DataFrame(columns=["term", "coefficient", "standard_error"]), np.nan
    design = regression_design(data, include_family)
    x = design.to_numpy()
    y = data[outcome].to_numpy()
    # Least squares returns the fitted coefficient vector and numerical rank. The rank is used in the residual-variance degrees of freedom.
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ coefficients
    variance = np.sum((y - prediction) ** 2) / max(1, len(y) - rank)
    # A pseudoinverse keeps the descriptive standard errors defined even when this small model table makes some columns nearly redundant.
    errors = np.sqrt(np.maximum(np.diag(variance * np.linalg.pinv(x.T @ x)), 0))
    frame = pd.DataFrame(
        {"term": design.columns, "coefficient": coefficients, "standard_error": errors}
    )
    parameter = frame.loc[frame["term"] == "log_parameters", "coefficient"]
    return frame, float(parameter.iloc[0]) if len(parameter) else np.nan


# -----------------------------------------------------------------------------
# Resampling-calibrated model-scale association
#
# Intention: preserve architecture-family structure. The null distribution shuffles parameter counts only among models in the same family. The confidence interval resamples entire families. Roughly 72,000 small fits are possible at the default settings, so this block usually takes minutes rather than seconds.
# -----------------------------------------------------------------------------

def calibrated_regressions(table, permutations, bootstraps, seed):
    """Use within-family permutations and whole-family bootstraps."""
    rng = np.random.default_rng(seed)
    output = []
    for outcome in OUTCOMES:
        if outcome not in table:
            continue
        coefficients, observed = fit_regression(table, outcome)
        if coefficients.empty:
            continue

        # Within-family shuffling asks whether the observed scale coefficient is unusual after respecting broad architectural lineage.
        null = []
        for _ in range(permutations):
            shuffled = table.copy()
            for _, family_rows in table.groupby("family"):
                models = family_rows[["model", "parameters_m"]].drop_duplicates("model")
                values = models["parameters_m"].to_numpy().copy()
                rng.shuffle(values)
                mapping = dict(zip(models["model"], values))
                shuffled.loc[family_rows.index, "parameters_m"] = family_rows["model"].map(mapping)
            _, value = fit_regression(shuffled, outcome)
            if np.isfinite(value):
                null.append(value)

        # Sample families with replacement so multiple checkpoints from one family remain together in each bootstrap dataset.
        families = table["family"].dropna().unique()
        draws = []
        for _ in range(bootstraps):
            picked = rng.choice(families, len(families), replace=True)
            sampled = pd.concat(
                [table[table["family"] == family] for family in picked],
                ignore_index=True,
            )
            _, value = fit_regression(sampled, outcome)
            if np.isfinite(value):
                draws.append(value)

        coefficients["outcome"] = outcome
        coefficients["permutation_p"] = np.nan
        coefficients["family_bootstrap_low"] = np.nan
        coefficients["family_bootstrap_high"] = np.nan
        parameter_row = coefficients["term"] == "log_parameters"
        if null and np.isfinite(observed):
            p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (
                len(null) + 1
            )
            coefficients.loc[parameter_row, "permutation_p"] = p_value
        if draws:
            low, high = np.quantile(draws, [0.025, 0.975])
            coefficients.loc[parameter_row, "family_bootstrap_low"] = low
            coefficients.loc[parameter_row, "family_bootstrap_high"] = high
        output.append(coefficients)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


# -----------------------------------------------------------------------------
# Out-of-family prediction and training-objective summaries
#
# Intention: check whether the scale/width/cohort relationship generalizes to a lineage not used for fitting, and separately compare training categories after family averaging. These blocks usually take seconds to a few minutes.
# -----------------------------------------------------------------------------

def leave_family_out(table):
    rows = []
    for outcome in OUTCOMES:
        if outcome not in table:
            continue
        for family in sorted(table["family"].dropna().unique()):
            train = table[table["family"] != family].dropna(
                subset=[outcome, "parameters_m", "width"]
            )
            test = table[table["family"] == family].dropna(
                subset=[outcome, "parameters_m", "width"]
            )
            if train.empty or test.empty:
                continue
            # Family indicators cannot describe an unseen family, so this predictive check uses only scale, width, cohort, and interactions.
            train_design = regression_design(train, include_family=False)
            test_design = regression_design(test, include_family=False).reindex(
                columns=train_design.columns, fill_value=0
            )
            coefficients, _, _, _ = np.linalg.lstsq(
                train_design.to_numpy(), train[outcome].to_numpy(), rcond=None
            )
            truth = test[outcome].to_numpy()
            prediction = test_design.to_numpy() @ coefficients
            baseline = train[outcome].mean()
            rows.append(
                {
                    "outcome": outcome,
                    "left_out_family": family,
                    "rows": len(test),
                    "model_mse": np.mean((truth - prediction) ** 2),
                    "mean_baseline_mse": np.mean((truth - baseline) ** 2),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "outcome",
            "left_out_family",
            "rows",
            "model_mse",
            "mean_baseline_mse",
        ],
    )


def training_effects(table, bootstraps, seed):
    """Compare objectives after first averaging checkpoints within each family."""
    # Average checkpoints first so a family with many released sizes does not have more influence merely because it contributes more rows.
    family = (
        table.groupby(["family", "objective", "astronomy_specific"], as_index=False)
        .agg(maximum=("maximum", "mean"))
        .dropna(subset=["maximum"])
    )
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [("objective", value) for value in sorted(family["objective"].unique())]
    comparisons.append(("astronomy_specific", True))
    for column, value in comparisons:
        selected = family[column] == value
        if not selected.any() or selected.all():
            continue
        observed = family.loc[selected, "maximum"].mean() - family.loc[~selected, "maximum"].mean()
        draws = []
        for _ in range(bootstraps):
            sample = family.iloc[rng.integers(0, len(family), len(family))]
            chosen = sample[column] == value
            if chosen.any() and not chosen.all():
                draws.append(
                    sample.loc[chosen, "maximum"].mean()
                    - sample.loc[~chosen, "maximum"].mean()
                )
        low, high = np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan)
        rows.append(
            {
                "comparison": column,
                "value": value,
                "difference_in_maximum": observed,
                "family_bootstrap_low": low,
                "family_bootstrap_high": high,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "value",
            "difference_in_maximum",
            "family_bootstrap_low",
            "family_bootstrap_high",
        ],
    )


# -----------------------------------------------------------------------------
# Complete E6 stage
#
# Intention: assemble one transparent model table, run every inferential check, and write simple CSV outputs. Input joins and output summaries take seconds; calibrated_regressions is responsible for nearly all runtime.
# -----------------------------------------------------------------------------

def run(context):
    root = Path(context["output_root"])
    summary = pd.read_csv(root / "E2_depth_summary.csv")
    curves = pd.read_csv(root / "E1_all_curves.csv")
    # Use the final HSC layer width as the model's representation-width covariate.
    widths = (
        curves.sort_values("normalized_depth")
        .groupby(["cohort", "model"], as_index=False)["left_width"]
        .last()
        .rename(columns={"left_width": "width"})
    )
    summary = summary.merge(widths, on=["cohort", "model"], how="left")

    # Metadata is produced by the orchestrator's explicit model-name table and family rules rather than inferred from result values.
    metadata_rows = []
    for model in summary["model"]:
        row = context["model_metadata"](model).copy()
        row["model"] = model
        metadata_rows.append(row)
    metadata = pd.DataFrame(metadata_rows).drop_duplicates("model")
    table = summary.drop(columns=["family"], errors="ignore").merge(
        metadata, on="model", how="left"
    )

    # E5 can add maximum update alignment as another outcome. E6 remains usable if that optional table is empty.
    updates_file = root / "E5_updates.csv"
    if updates_file.exists() and updates_file.stat().st_size:
        updates = pd.read_csv(updates_file)
        if not updates.empty:
            update_max = (
                updates.groupby(["cohort", "model"], as_index=False)["corrected_overlap_20"]
                .max()
                .rename(columns={"corrected_overlap_20": "maximum_update_alignment"})
            )
            table = table.merge(update_max, on=["cohort", "model"], how="left")

    table.to_csv(root / "E6_model_table.csv", index=False)
    calibrated_regressions(
        table, context["permutations"], context["bootstraps"], context["seed"]
    ).to_csv(root / "E6_regression.csv", index=False)
    leave_family_out(table).to_csv(root / "E6_leave_family_out.csv", index=False)

    objective_summary = (
        table.groupby(["objective", "astronomy_specific"], as_index=False)
        .agg(
            models=("model", "nunique"),
            mean_maximum=("maximum", "mean"),
            mean_final_minus_early=("final_minus_early", "mean"),
        )
    )
    objective_summary.to_csv(root / "E6_objective_summary.csv", index=False)
    training_effects(table, context["bootstraps"], context["seed"]).to_csv(
        root / "E6_training_effects.csv", index=False
    )
    return table
