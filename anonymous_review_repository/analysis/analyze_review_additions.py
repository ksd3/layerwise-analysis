"""Camera-ready analyses requested by the workshop reviews.

Everything here is computed from existing result files; no embedding is reread.
1. Stage selection against all stages: where does each label-free selection
   rule land among the valid saved stages of a model, ranked by held-out
   cross-survey R^2 (0 = worst stage, 1 = best, 0.5 = a random stage)?
2. Across-stage co-variation: within every model, does each geometry metric
   (MKNN, CKA, Procrustes, MRR) move with cross-survey R^2 over the saved
   stages, even when the two peaks differ?
3. Calibration: the R^2 of the best affine recalibration of each transferred
   predictor, fitted on the held-out labels themselves. This is an optimistic
   upper bound for any affine correction of an unchanged probe.
4. Final-representation agreement: correlation between a transferred
   predictor and a predictor fitted in the evaluated survey, on the same
   held-out objects.

Run from the repository root:
    python analysis/analyze_review_additions.py

Analyses 1 and 2 need only the CSV files in data/. Analyses 3 and 4 also need
the per-object held-out predictions of E3 and E8 (data/E3_predictions and
data/E8_predictions, written by experiments/experiment_3_image_transfer.py and
experiments/experiment_8_final_transfer.py). These archives are too large for
the repository; when they are absent, the released tables in outputs/tables
are kept unchanged.
"""

import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "tables"
SEED = 42
BOOTSTRAPS = 2000
METRICS = {
    "corrected_overlap_20": "MKNN",
    "cka": "CKA",
    "procrustes": "Procrustes",
    "mrr": "MRR",
}
MODEL_KEYS = ["cohort", "model", "family", "target", "direction"]


def family(model):
    for prefix in ("vit-mae", "llava", "paligemma"):
        if model.startswith(prefix):
            return prefix
    return model.split("_")[0]


def family_interval(values, rng):
    """Family-equal mean and 95% percentile interval from resampled families."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    draws = rng.choice(values, (BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return values.mean(), low, high


def summarize(frame, value, keys, rng):
    """Average models within a family first, then weight families equally."""
    rows = []
    for key, group in frame.groupby(keys):
        family_means = group.groupby("family")[value].mean()
        mean, low, high = family_interval(family_means.to_numpy(), rng)
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update(
            models=group["model"].nunique(),
            families=len(family_means),
            family_mean=mean,
            family_bootstrap_low=low,
            family_bootstrap_high=high,
            model_median=group[value].median(),
            fraction_models_positive=(group[value] > 0).mean(),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_stage_table():
    """Per-stage, per-fold cross-survey R^2 joined to E1 geometry.

    Only stages with valid E1 geometry are retained. These are exactly the
    candidate stages of the corrected E3 selection, so every rule below chooses
    from the same set.
    """
    physics = pd.read_csv(DATA / "layerwise_physics_scores.csv")
    geometry = pd.read_csv(DATA / "E1_all_curves.csv")
    geometry = geometry[
        geometry["cohort"].isin(physics["cohort"].unique())
        & geometry["available"].astype(bool)
        & geometry["corrected_overlap_20"].notna()
    ]
    columns = ["cohort", "model", "layer", *METRICS]
    return physics.merge(geometry[columns], on=["cohort", "model", "layer"], how="inner")


def selection_percentiles(stages, rng):
    """Percentile of each rule's stage among all stages of the same model/fold."""
    e3 = pd.read_csv(DATA / "E3_image_transfer.csv")
    e3 = e3.set_index(["cohort", "model", "fold", "target", "direction"])
    rows = []
    for key, curve in stages.groupby(["cohort", "model", "family", "fold", "target", "direction"]):
        curve = curve.sort_values("layer")
        layers = curve["layer"].to_numpy()
        r2 = curve["test_r2"].to_numpy()
        percentile = dict(zip(layers, (rankdata(r2) - 1) / (len(r2) - 1)))
        choice = e3.loc[(key[0], key[1], key[3], key[4], key[5])]
        row = dict(zip(["cohort", "model", "family", "fold", "target", "direction"], key))
        row.update(
            stages=len(r2),
            geometry_selected=percentile[choice["label_free_layer"]],
            label_selected=percentile[choice["oracle_layer"]],
            final_stage=percentile[layers[-1]],
            geometry_minus_final_r2=choice["label_free_test_r2"] - r2[-1],
        )
        # The full-sample geometry uses no labels, so its maximum is label free.
        for metric, name in METRICS.items():
            values = curve[metric].to_numpy()
            row[f"max_{name}"] = percentile[layers[np.nanargmax(values)]]
        rows.append(row)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(OUTPUT / "review_selection_percentiles_by_fold.csv", index=False)

    rules = ["geometry_selected", "label_selected", "final_stage",
             *[f"max_{name}" for name in METRICS.values()]]
    per_model = per_fold.groupby(MODEL_KEYS, as_index=False)[
        [*rules, "geometry_minus_final_r2"]
    ].mean()
    long = per_model.melt(
        id_vars=MODEL_KEYS, value_vars=rules, var_name="rule", value_name="percentile"
    )
    summary = summarize(long, "percentile", ["cohort", "target", "direction", "rule"], rng)
    summary.to_csv(OUTPUT / "review_selection_percentiles_summary.csv", index=False)

    # One overall number per rule: each family's mean over its models and all
    # eight target-direction comparisons, then families weighted equally.
    overall = []
    family_rows = long.groupby(["rule", "family"])["percentile"].mean().reset_index()
    for rule, group in family_rows.groupby("rule"):
        mean, low, high = family_interval(group["percentile"], rng)
        overall.append({"rule": rule, "family_mean": mean,
                        "family_bootstrap_low": low, "family_bootstrap_high": high})
    pd.DataFrame(overall).to_csv(OUTPUT / "review_selection_percentiles_overall.csv", index=False)

    difference = summarize(per_model, "geometry_minus_final_r2",
                           ["cohort", "target", "direction"], rng)
    difference.to_csv(OUTPUT / "review_geometry_minus_final_r2_summary.csv", index=False)
    return summary, pd.DataFrame(overall), difference


def covariation(stages, rng):
    """Spearman correlation over stages between geometry and fold-mean R^2."""
    curves = (
        stages.groupby([*MODEL_KEYS, "layer"])
        .agg(test_r2=("test_r2", "mean"), **{metric: (metric, "first") for metric in METRICS})
        .reset_index()
    )
    rows = []
    for key, curve in curves.groupby(MODEL_KEYS):
        for metric, name in METRICS.items():
            valid = curve[["test_r2", metric]].dropna()
            rho = spearmanr(valid["test_r2"], valid[metric]).statistic
            rows.append(dict(zip(MODEL_KEYS, key)) | {"metric": name, "stages": len(valid), "spearman": rho})
    per_model = pd.DataFrame(rows)
    per_model.to_csv(OUTPUT / "review_stage_covariation_by_model.csv", index=False)
    summary = summarize(per_model, "spearman", ["cohort", "target", "direction", "metric"], rng)
    summary.to_csv(OUTPUT / "review_stage_covariation_summary.csv", index=False)
    return summary


def depth_profiles(stages, points=41):
    """Mean within-model percentile rank of every score over normalized depth.

    Ranking each curve within its model is the transformation behind the
    Spearman correlations above, so these profiles show what they summarize.
    """
    curves = (
        stages.groupby([*MODEL_KEYS, "layer", "normalized_depth"])
        .agg(test_r2=("test_r2", "mean"), **{metric: (metric, "first") for metric in METRICS})
        .reset_index()
    )
    grid = np.linspace(0, 1, points)
    scores = {"test_r2": "transfer R2", **METRICS}
    rows = []
    for key, group in curves.groupby(["cohort", "target", "direction"]):
        profiles = {name: [] for name in scores.values()}
        for _, curve in group.groupby("model"):
            curve = curve.sort_values("normalized_depth")
            for column, name in scores.items():
                valid = curve[["normalized_depth", column]].dropna()
                ranks = (rankdata(valid[column]) - 1) / (len(valid) - 1)
                profiles[name].append(np.interp(grid, valid["normalized_depth"], ranks))
        for name, values in profiles.items():
            values = np.asarray(values)
            for depth, mean, error in zip(grid, values.mean(axis=0),
                                          values.std(axis=0, ddof=1) / np.sqrt(len(values))):
                rows.append(dict(zip(["cohort", "target", "direction"], key))
                            | {"score": name, "depth": depth, "mean_rank": mean,
                               "standard_error": error, "models": len(values)})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "review_depth_profiles.csv", index=False)
    return result


def affine_bound(y, prediction):
    """R^2 of the least-squares affine map a + b * prediction on these labels."""
    return np.corrcoef(y, prediction)[0, 1] ** 2


def pooled_r2(y, prediction):
    return 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)


def layerwise_calibration(rng):
    """Unchanged and affine-recalibrated R^2 for the two E3 selections."""
    pooled = defaultdict(list)
    pattern = r".*/(.+?)__(.+?)__(.+?)__(hsc_to_other|other_to_hsc)__fold(\d)\.npz$"
    for path in glob.glob(str(DATA / "E3_predictions" / "*.npz")):
        cohort, model, target, direction, _ = re.match(pattern, path).groups()
        data = np.load(path)
        pooled[(cohort, model, target, direction)].append(
            (data["y"], data["label_free_prediction"], data["oracle_prediction"])
        )
    rows = []
    for (cohort, model, target, direction), parts in pooled.items():
        y, geometry, label = (np.concatenate([part[i] for part in parts]) for i in range(3))
        good = np.isfinite(y) & np.isfinite(geometry) & np.isfinite(label)
        y, geometry, label = y[good], geometry[good], label[good]
        rows.append(
            {
                "cohort": cohort, "model": model, "family": family(model),
                "target": target, "direction": direction,
                "geometry_r2": pooled_r2(y, geometry),
                "label_r2": pooled_r2(y, label),
                "geometry_affine_bound": affine_bound(y, geometry),
                "label_affine_bound": affine_bound(y, label),
            }
        )
    per_model = pd.DataFrame(rows)
    per_model["affine_bound_gap"] = per_model["label_affine_bound"] - per_model["geometry_affine_bound"]
    per_model.to_csv(OUTPUT / "review_layerwise_calibration_by_model.csv", index=False)
    medians = per_model.groupby(["cohort", "target", "direction"])[
        ["geometry_r2", "label_r2", "geometry_affine_bound", "label_affine_bound"]
    ].median().reset_index()
    gap = summarize(per_model, "affine_bound_gap", ["cohort", "target", "direction"], rng)
    summary = medians.merge(
        gap[["cohort", "target", "direction", "family_mean", "family_bootstrap_low",
             "family_bootstrap_high"]],
        on=["cohort", "target", "direction"],
    ).rename(columns={
        "family_mean": "gap_family_mean",
        "family_bootstrap_low": "gap_family_low",
        "family_bootstrap_high": "gap_family_high",
    })
    summary.to_csv(OUTPUT / "review_layerwise_calibration_summary.csv", index=False)
    return summary


def load_final_predictions():
    """Pool the saved held-out E8 predictions over the five spatial folds."""
    pooled = defaultdict(list)
    for path in glob.glob(str(DATA / "E8_predictions" / "*.npz")):
        model, target, direction, _ = re.match(
            r".*/(.+?)__(.+?)__(.+?)__fold(\d)\.npz$", path
        ).groups()
        data = np.load(path)
        pooled[(model, target, direction)].append(
            (data["row_indices"], data["y"], data["prediction"])
        )
    answer = {}
    for key, parts in pooled.items():
        rows = np.concatenate([part[0] for part in parts])
        order = np.argsort(rows)
        answer[key] = tuple(np.concatenate([part[i] for part in parts])[order] for i in range(3))
    return answer


def final_diagnostics():
    predictions = load_final_predictions()
    native_view = {"hsc_to_jwst": "jwst_within", "jwst_to_hsc": "hsc_within"}
    rows = []
    for (model, target, direction), (row_index, y, prediction) in predictions.items():
        good = np.isfinite(y) & np.isfinite(prediction)
        record = {
            "model": model, "family": family(model), "target": target,
            "direction": direction,
            "pooled_r2": pooled_r2(y[good], prediction[good]),
            "pearson_r": np.corrcoef(y[good], prediction[good])[0, 1],
            "affine_bound": affine_bound(y[good], prediction[good]),
        }
        if direction in native_view:
            native_rows, _, native = predictions[(model, target, native_view[direction])]
            if not np.array_equal(native_rows, row_index):
                raise ValueError(f"Held-out rows differ for {model} {target} {direction}")
            both = good & np.isfinite(native)
            record["agreement_with_native"] = np.corrcoef(prediction[both], native[both])[0, 1]
        rows.append(record)
    per_direction = pd.DataFrame(rows)
    per_direction.to_csv(OUTPUT / "review_final_diagnostics_by_direction.csv", index=False)

    # As in E8, one value per model averages its two directions.
    per_direction["kind"] = np.where(per_direction["direction"].str.endswith("within"),
                                     "within", "cross")
    per_model = per_direction.groupby(["model", "family", "target", "kind"], as_index=False)[
        ["pooled_r2", "pearson_r", "affine_bound", "agreement_with_native"]
    ].mean()
    per_model.to_csv(OUTPUT / "review_final_diagnostics_by_model.csv", index=False)

    def quartiles(values):
        return values.quantile(0.25), values.median(), values.quantile(0.75)

    rows = []
    for (target, kind), group in per_model.groupby(["target", "kind"]):
        row = {"target": target, "kind": kind, "models": group["model"].nunique()}
        for column in ("pooled_r2", "affine_bound", "agreement_with_native"):
            low, middle, high = quartiles(group[column])
            row.update({f"{column}_q25": low, f"{column}_median": middle,
                        f"{column}_q75": high, f"{column}_max": group[column].max()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "review_final_diagnostics_summary.csv", index=False)
    return summary


def final_metric_association():
    """Across-model Spearman correlation of each metric with transfer."""
    geometry = pd.read_csv(DATA / "E8_final_transfer.csv").groupby("model")[list(METRICS)].first()
    # In this file the cross-survey column is named for redshift, but each row
    # holds the score of its own target.
    transfer = pd.read_csv(DATA / "E8_transfer_by_model_all_targets.csv")
    rows = []
    for target, group in transfer.groupby("target"):
        group = group.set_index("model").join(geometry)
        for metric, name in METRICS.items():
            rows.append({
                "target": target, "metric": name,
                "spearman_with_cross_r2":
                    spearmanr(group[metric], group["cross_view_redshift_r2"]).statistic,
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "review_final_metric_transfer_spearman.csv", index=False)
    return result


def main():
    rng = np.random.default_rng(SEED)
    stages = load_stage_table()
    percentiles, overall, difference = selection_percentiles(stages, rng)
    correlation = covariation(stages, rng)
    profiles = depth_profiles(stages)
    association = final_metric_association()
    have_predictions = all(
        any((DATA / name).glob("*.npz")) for name in ("E3_predictions", "E8_predictions")
    )
    if have_predictions:
        calibration = layerwise_calibration(rng)
        final = final_diagnostics()
    else:
        print("E3/E8 prediction archives not found in data/; keeping the released "
              "calibration and final-diagnostic tables in outputs/tables.")
        calibration = pd.read_csv(OUTPUT / "review_layerwise_calibration_summary.csv")
        final = pd.read_csv(OUTPUT / "review_final_diagnostics_summary.csv")

    pd.set_option("display.width", 220)
    for name, table in [
        ("Selection percentiles", percentiles),
        ("Overall percentiles", overall),
        ("Geometry minus final-stage R^2", difference),
        ("Across-stage Spearman correlation", correlation),
        ("Depth-profile peaks", profiles.loc[profiles.groupby(
            ["cohort", "target", "direction", "score"])["mean_rank"].idxmax()]),
        ("Layerwise calibration", calibration),
        ("Final-representation diagnostics", final),
        ("Final metric association", association),
    ]:
        print(f"\n{name}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
