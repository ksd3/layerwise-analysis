"""E8: geometry and scientific transfer for the validated final arrays.

This experiment uses the final 45,000-row HSC/JWST arrays rather than every
intermediate layer. For each model it:
1. Measures final-representation geometry and permutation uncertainty.
2. Splits the sky into right-ascension slabs to reduce spatial leakage.
3. Fits within-view and cross-view Ridge models for every scientific target.
4. Pools held-out sufficient statistics into one independent row per model.
5. Tests whether alignment predicts cross-view redshift transfer after model
   scale, width, family, and within-view performance are included.

Approximate full-data runtime: each model performs exact neighbor geometry,
9,999 permutations, and 5 folds x 5 targets x 4 transfer directions. One model
can take tens of minutes or more; all released models may require many hours.
Geometry, permutations, retrieval ranking, and Ridge fits use the assigned GPU.
The small model-level regressions and family bootstraps remain on the CPU.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_1_geometry import (
    configure_worker,
    curve_inference,
    geometry,
    holm_adjust,
)
from experiment_3_image_transfer import transfer_score


# -----------------------------------------------------------------------------
# Spatial folds and pooled model-level outcomes
#
# Intention: separate fit, validation, and test regions along right ascension so neighboring sky objects are less likely to straddle roles. Pooling stores SSE and target moments, allowing exact pooled R2 across folds without averaging fold R2 values. These table operations normally take seconds.
# -----------------------------------------------------------------------------

def spatial_roles(ra, fold):
    """Make fit/oracle/test roles from ten ordered right-ascension slabs."""
    order = np.argsort(ra)
    slabs = np.array_split(order, 10)
    test_slabs = {2 * fold, 2 * fold + 1}
    # Adjacent buffer slabs are discarded from fitting. They create a spatial gap around the two held-out test slabs.
    buffer_slabs = {
        number
        for number in (2 * fold - 1, 2 * fold + 2)
        if 0 <= number < 10
    }
    oracle_slabs = {(2 * fold + 5) % 10, (2 * fold + 6) % 10}
    fit_slabs = [
        number
        for number in range(10)
        if number not in buffer_slabs | test_slabs | oracle_slabs
    ]
    test = np.concatenate([slabs[number] for number in sorted(test_slabs)])
    oracle = np.concatenate([slabs[number] for number in sorted(oracle_slabs)])
    fit = np.concatenate([slabs[number] for number in fit_slabs])
    return {"fit": fit, "oracle": oracle, "test": test, "selector": oracle}


def transfer_by_model(results, context, target_name="redshift"):
    """Make one independent row per model for one scientific target."""
    def pooled_r2(frame):
        # Pool residual sums of squares and target moments before calculating R2. This gives each held-out object its proper weight across folds.
        scores = []
        for _, direction in frame.groupby("direction"):
            n = direction["test_n"].sum()
            total = direction["test_y_sum"].sum()
            squared = direction["test_y_squared_sum"].sum()
            sst = squared - total * total / n
            scores.append(1 - direction["test_sse"].sum() / sst if sst > 0 else np.nan)
        return float(np.nanmean(scores))

    # A model, rather than a fold or checkpoint row, is the independent unit for the model-property regression below.
    rows = []
    for model, model_rows in results.groupby("model"):
        target_rows = model_rows[model_rows["target"] == target_name]
        cross = target_rows[
            target_rows["direction"].isin(["hsc_to_jwst", "jwst_to_hsc"])
        ]
        within = target_rows[
            target_rows["direction"].isin(["hsc_within", "jwst_within"])
        ]
        first = model_rows.iloc[0]
        metadata = context["model_metadata"](model)
        rows.append(
            {
                "model": model,
                "target": target_name,
                "family": context["model_family"](model),
                "parameters_m": metadata["parameters_m"],
                "width": first["width"],
                "alignment": first["corrected_overlap_20"],
                "alignment_permutation_p": first["alignment_permutation_p"],
                "cross_view_redshift_r2": pooled_r2(cross),
                "within_view_redshift_r2": pooled_r2(within),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Multiple testing and the alignment/transfer regression
#
# Intention: estimate whether final alignment adds information about cross-view science beyond model scale, width, family, and within-view predictive quality. Within-family permutations calibrate the alignment coefficient; whole-family bootstraps give its interval. With the small model table, 9,999 permutations and 2,000 bootstraps usually take minutes rather than hours.
# -----------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """False-discovery-rate correction used for secondary scientific targets."""
    values = np.asarray(p_values, dtype=float)
    answer = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    running = 1.0
    for reverse_rank, position in enumerate(order[::-1], start=1):
        rank = len(order) - reverse_rank + 1
        running = min(running, values[position] * len(order) / rank)
        answer[position] = running
    return answer


def _regression_design(table, include_families=True):
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "alignment": table["alignment"],
            "log_parameters": np.log10(table["parameters_m"]),
            "log_width": np.log10(table["width"]),
            "within_view_redshift_r2": table["within_view_redshift_r2"],
        },
        index=table.index,
    )
    # Drop the first family indicator because the intercept represents the reference family.
    if include_families:
        indicators = pd.get_dummies(table["family"], prefix="family", drop_first=True, dtype=float)
        design = pd.concat([design, indicators], axis=1)
    return design


def _fit_transfer_regression(table, include_families=True):
    design = _regression_design(table, include_families)
    outcome = table["cross_view_redshift_r2"]
    good = np.isfinite(design).all(axis=1) & np.isfinite(outcome)
    if good.sum() <= design.shape[1]:
        return pd.DataFrame(columns=["term", "coefficient"]), np.nan
    # This is ordinary least squares on one row per model. The helper returns the alignment coefficient separately because inference focuses on it.
    coefficients, _, _, _ = np.linalg.lstsq(
        design.loc[good].to_numpy(), outcome.loc[good].to_numpy(), rcond=None
    )
    frame = pd.DataFrame({"term": design.columns, "coefficient": coefficients})
    alignment = frame.loc[frame["term"] == "alignment", "coefficient"].iloc[0]
    return frame, float(alignment)


def alignment_transfer_regression(table, context):
    """Fit once per model, then use family-blocked resampling."""
    coefficients, observed = _fit_transfer_regression(table)
    if coefficients.empty:
        coefficients["permutation_p"] = np.nan
        coefficients["family_bootstrap_low"] = np.nan
        coefficients["family_bootstrap_high"] = np.nan
        return coefficients

    rng = np.random.default_rng(context["seed"])
    # Shuffle alignment only within families, retaining family membership and all other model properties while breaking its model-specific association.
    null = []
    for _ in range(context["permutations"]):
        permuted = table.copy()
        for indices in table.groupby("family").groups.values():
            values = table.loc[indices, "alignment"].to_numpy().copy()
            rng.shuffle(values)
            permuted.loc[indices, "alignment"] = values
        _, value = _fit_transfer_regression(permuted)
        if np.isfinite(value):
            null.append(value)

    # Resample whole families so checkpoints in one lineage remain dependent.
    families = table["family"].dropna().unique()
    bootstrap = []
    for _ in range(context["bootstraps"]):
        picked = rng.choice(families, len(families), replace=True)
        sampled = pd.concat(
            [table[table["family"] == family] for family in picked],
            ignore_index=True,
        )
        _, value = _fit_transfer_regression(sampled)
        if np.isfinite(value):
            bootstrap.append(value)

    coefficients["permutation_p"] = np.nan
    coefficients["family_bootstrap_low"] = np.nan
    coefficients["family_bootstrap_high"] = np.nan
    alignment_row = coefficients["term"] == "alignment"
    if null:
        p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1)
        coefficients.loc[alignment_row, "permutation_p"] = p_value
    if bootstrap:
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        coefficients.loc[alignment_row, "family_bootstrap_low"] = low
        coefficients.loc[alignment_row, "family_bootstrap_high"] = high
    return coefficients


def leave_family_out(table):
    """Predict each held-out family and compare with a mean baseline."""
    rows = []
    for family in sorted(table["family"].dropna().unique()):
        train = table[table["family"] != family]
        test = table[table["family"] == family]
        # An unseen family has no learned indicator. The predictive version uses only alignment, scale, width, and within-view performance.
        train_design = _regression_design(train, include_families=False)
        test_design = _regression_design(test, include_families=False)
        train_good = np.isfinite(train_design).all(axis=1) & np.isfinite(
            train["cross_view_redshift_r2"]
        )
        test_good = np.isfinite(test_design).all(axis=1) & np.isfinite(
            test["cross_view_redshift_r2"]
        )
        if train_good.sum() <= train_design.shape[1] or not test_good.any():
            continue
        coefficients, _, _, _ = np.linalg.lstsq(
            train_design.loc[train_good].to_numpy(),
            train.loc[train_good, "cross_view_redshift_r2"].to_numpy(),
            rcond=None,
        )
        truth = test.loc[test_good, "cross_view_redshift_r2"].to_numpy()
        prediction = test_design.loc[test_good].to_numpy() @ coefficients
        baseline = train.loc[train_good, "cross_view_redshift_r2"].mean()
        rows.append(
            {
                "left_out_family": family,
                "models": len(truth),
                "model_mse": np.mean((truth - prediction) ** 2),
                "mean_baseline_mse": np.mean((truth - baseline) ** 2),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "left_out_family",
            "models",
            "model_mse",
            "mean_baseline_mse",
        ],
    )


# -----------------------------------------------------------------------------
# Per-model final geometry and transfer
#
# Intention: load one pair of arrays, calculate geometry once, and reuse it for all scientific targets. The geometry permutation test is commonly the longest part. The 100 Ridge evaluations per model can also take minutes for wide data.
# -----------------------------------------------------------------------------

def _run_group(arguments):
    gpu_id, items, catalog, root, cpu_threads, permutations, bootstraps, seed = arguments
    configure_worker(gpu_id, cpu_threads)
    rows = []
    prediction_dir = Path(root) / "E8_predictions"
    for item in items:
        print(f"E8  GPU {gpu_id}  {item['model']}", flush=True)
        # Memory mapping avoids an extra file-reading API, while np.asarray makes a normal float32 array for repeated numerical operations.
        left = np.asarray(np.load(item["hsc"], mmap_mode="r"), dtype=np.float32)
        right = np.asarray(np.load(item["jwst"], mmap_mode="r"), dtype=np.float32)
        lengths = {len(left), len(right), len(catalog["ra"])}
        if len(lengths) != 1:
            raise ValueError(f"Paired row counts differ for {item['model']}: {lengths}")
        n = len(left)
        # Geometry calculates exact neighbors, overlap, CKA, Procrustes, and retrieval once for this model and saves reusable neighbor arrays.
        numbers, per_row, left_neighbors, right_neighbors = geometry(
            left, right, Path(root) / "E8_neighbors" / f"{item['model']}.npz"
        )
        # With one final layer the permutation maximum is simply this model's alignment. The same routine is used for consistency with E1 and E5.
        p_value, low, high, _ = curve_inference(
            [(left_neighbors, right_neighbors)],
            [per_row],
            permutations,
            bootstraps,
            seed,
        )
        numbers["alignment_permutation_p"] = p_value
        numbers["alignment_low"] = low[0]
        numbers["alignment_high"] = high[0]

        # For each target, within-view directions establish an attainable baseline while cross-view directions measure representation transfer.
        for fold in range(5):
            roles = spatial_roles(np.asarray(catalog["ra"])[:n], fold)
            for target_name, label_values in catalog["targets"].items():
                labels = np.asarray(label_values)[:n]
                directions = {
                    "hsc_within": (left, left),
                    "jwst_within": (right, right),
                    "hsc_to_jwst": (left, right),
                    "jwst_to_hsc": (right, left),
                }
                for direction, (source, target) in directions.items():
                    score, alpha, y, prediction, test_rows = transfer_score(
                        source, target, labels, roles, return_predictions=True
                    )
                    prediction_file = (
                        prediction_dir
                        / f"{item['model']}__{target_name}__{direction}__fold{fold}.npz"
                    )
                    # Save predictions for independent recomputation of every test score without repeating the Ridge scan.
                    np.savez(
                        prediction_file,
                        row_indices=test_rows,
                        y=y,
                        prediction=prediction,
                    )
                    row = {
                        "model": item["model"],
                        "fold": fold,
                        "target": target_name,
                        "direction": direction,
                        "alpha": alpha,
                        "test_r2": score,
                        "test_n": len(y),
                        "test_sse": float(np.sum((y - prediction) ** 2)),
                        "test_y_sum": float(np.sum(y)),
                        "test_y_squared_sum": float(np.sum(y * y)),
                        "width": left.shape[1],
                    }
                    for name, value in numbers.items():
                        if np.isscalar(value):
                            row[name] = value
                    rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Multi-GPU scheduling, primary analysis, and secondary targets
#
# Intention: distribute models one-per-worker sequence, assemble one detailed CSV, run the primary redshift regression, then repeat it for all targets with false-discovery-rate correction. Scheduling/output takes minutes at most; _run_group dominates the full stage.
# -----------------------------------------------------------------------------

def run(context):
    root = Path(context["output_root"])
    catalog = context["final_catalog"]
    prediction_dir = root / "E8_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    neighbor_dir = root / "E8_neighbors"
    neighbor_dir.mkdir(parents=True, exist_ok=True)
    items = context["final_pairs"]
    workers = min(context.get("workers", 1), len(items))
    groups = [items[number::workers] for number in range(workers)]
    arguments = [
        (
            number if number < context.get("gpu_count", 0) else None,
            group,
            catalog,
            root,
            context.get("cpu_threads_per_worker", 1),
            context["permutations"],
            context["bootstraps"],
            context["seed"],
        )
        for number, group in enumerate(groups)
    ]
    # Spawn keeps each process's FAISS/CUDA state on its assigned GPU.
    if workers == 1:
        rows = _run_group(arguments[0])
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = [row for group_rows in pool.map(_run_group, arguments) for row in group_rows]
    result = pd.DataFrame(rows)
    result.to_csv(root / "E8_final_transfer.csv", index=False)
    model_table = transfer_by_model(result, context, "redshift")
    model_table["alignment_p_holm"] = holm_adjust(
        model_table["alignment_permutation_p"]
    )
    model_table.to_csv(root / "E8_transfer_by_model.csv", index=False)
    regression = alignment_transfer_regression(model_table, context)
    regression.to_csv(root / "E8_alignment_transfer_regression.csv", index=False)
    leave_family_out(model_table).to_csv(
        root / "E8_leave_family_out.csv", index=False
    )

    # Redshift is the prespecified primary outcome above. Other catalog targets are secondary and receive Benjamini-Hochberg correction as one family.
    all_tables = []
    all_regressions = []
    for target_name in sorted(result["target"].unique()):
        target_table = transfer_by_model(result, context, target_name)
        all_tables.append(target_table)
        target_regression = alignment_transfer_regression(target_table, context)
        target_regression["target"] = target_name
        all_regressions.append(target_regression)
    all_tables = pd.concat(all_tables, ignore_index=True)
    all_tables.to_csv(root / "E8_transfer_by_model_all_targets.csv", index=False)
    all_regressions = pd.concat(all_regressions, ignore_index=True)
    alignment_rows = all_regressions["term"] == "alignment"
    secondary = alignment_rows & (all_regressions["target"] != "redshift")
    all_regressions["secondary_p_bh"] = np.nan
    all_regressions.loc[secondary, "secondary_p_bh"] = benjamini_hochberg(
        all_regressions.loc[secondary, "permutation_p"]
    )
    all_regressions.to_csv(root / "E8_all_target_regressions.csv", index=False)
    return result
