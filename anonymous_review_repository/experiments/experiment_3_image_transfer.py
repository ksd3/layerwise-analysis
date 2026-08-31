"""E3: choose an image layer without labels, then test scientific transfer.

The question is whether cross-survey geometry can select a useful layer without
looking at the scientific target. For each of five folds, this file:
1. Splits objects into fit, geometry-selector, oracle-validation, and final-test
   roles.
2. Selects a layer from bidirectional paired-object retrieval alone.
3. Separately lets target labels select an oracle layer on validation rows.
4. Fits Ridge regression on one survey and evaluates it in the other survey.
5. Compares label-free and oracle predictions on the same untouched test rows.

Approximate full-data runtime on the intended machine: one model/cohort can take
minutes to hours because every layer is decoded and evaluated in five folds.
Retrieval, scaling, and Ridge decompositions use the assigned GPU; small R2 and
bootstrap reductions and Parquet reads remain CPU work. Across all image files,
budget several hours, with storage speed now likely to matter more. The smoke
test takes seconds.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

try:
    import cupy as cp
except ImportError:
    cp = None

from experiment_1_geometry import (
    assigned_gpu,
    configure_worker,
    layer_pairs,
    read_matrix,
    retrieval_ranks,
)

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


# -----------------------------------------------------------------------------
# Data splitting and label-free layer selection
#
# Intention: keep layer selection, hyperparameter selection, and final testing statistically separate. Each fold assignment takes milliseconds. Retrieval is the expensive part: normally seconds to minutes per layer/fold, depending mainly on selector-set size and whether FAISS is using the GPU.
# -----------------------------------------------------------------------------

def split_roles(n, fold, seed):
    """Create deterministic 60/10/10/20-ish fit/selector/oracle/test roles."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    buckets = np.empty(n, dtype=np.int8)
    buckets[order] = np.arange(n) % 10
    test_buckets = {(2 * fold) % 10, (2 * fold + 1) % 10}
    selector_bucket = (2 * fold + 2) % 10
    oracle_bucket = (2 * fold + 3) % 10
    # The selector sees geometry but no target values. The oracle set may see target values to choose Ridge alpha and the comparison oracle layer. Test rows are not consulted until all those choices are frozen.
    return {
        "fit": np.flatnonzero(~np.isin(buckets, list(test_buckets | {selector_bucket, oracle_bucket}))),
        "selector": np.flatnonzero(buckets == selector_bucket),
        "oracle": np.flatnonzero(buckets == oracle_bucket),
        "test": np.flatnonzero(np.isin(buckets, list(test_buckets))),
    }


def selection_mrr(left, right, rows):
    """Average reciprocal paired-object rank in both survey directions."""
    left_ranks = retrieval_ranks(left[rows], right[rows])
    right_ranks = retrieval_ranks(right[rows], left[rows])
    per_object = (1 / left_ranks + 1 / right_ranks) / 2
    return float(per_object.mean()), float(per_object.std(ddof=1) / np.sqrt(len(per_object)))


def one_standard_error_layer(scores):
    """Choose the earliest layer within one standard error of the best MRR."""
    best = max(scores, key=lambda item: item["mrr"])
    threshold = best["mrr"] - best["se"]
    eligible = [item for item in scores if item["mrr"] >= threshold]
    return min(eligible, key=lambda item: item["layer"])["layer"]


# -----------------------------------------------------------------------------
# Cross-view Ridge transfer and paired uncertainty
#
# Intention: fit labels in the source representation and use the same fitted scaler and Ridge coefficients in the target representation. Five alpha values are tried on the oracle rows using one shared GPU SVD. A layer normally takes seconds, although very wide matrices can take minutes. A 2,000-draw prediction bootstrap is normally seconds per comparison and remains on the CPU.
# -----------------------------------------------------------------------------

def _gpu_transfer_score(
    source,
    target,
    labels,
    fit,
    oracle,
    test,
    return_predictions,
    validation_only,
    gpu_id,
):
    """Fit all five Ridge choices with one GPU eigendecomposition."""
    with cp.cuda.Device(gpu_id):
        source_fit = cp.asarray(source[fit], dtype=cp.float64)
        target_oracle = cp.asarray(target[oracle], dtype=cp.float64)
        target_test = cp.asarray(target[test], dtype=cp.float64)

        # This is StandardScaler written out directly. The source fit rows set the mean and scale, and that exact transformation is used on target.
        mean = source_fit.mean(axis=0)
        scale = source_fit.std(axis=0)
        scale = cp.where(scale == 0, 1, scale)
        source_fit = (source_fit - mean) / scale
        target_oracle = (target_oracle - mean) / scale
        target_test = (target_test - mean) / scale

        y_fit = cp.asarray(labels[fit], dtype=cp.float64)
        y_mean = y_fit.mean()
        y_centered = y_fit - y_mean

        # All alpha values share one decomposition of the smaller square system. This avoids both five independent fits and a large rectangular SVD when there are many more objects than embedding coordinates.
        if source_fit.shape[1] <= source_fit.shape[0]:
            gram = source_fit.T @ source_fit
            vectors, values, _ = cp.linalg.svd(gram, full_matrices=False)
            projected = vectors.T @ (source_fit.T @ y_centered)

            def coefficients(alpha):
                return vectors @ (projected / (values + alpha))
        else:
            kernel = source_fit @ source_fit.T
            vectors, values, _ = cp.linalg.svd(kernel, full_matrices=False)
            projected = vectors.T @ y_centered

            def coefficients(alpha):
                dual = vectors @ (projected / (values + alpha))
                return source_fit.T @ dual

        best_alpha = None
        best_score = -np.inf
        best_coefficients = None
        for alpha in RIDGE_ALPHAS:
            coef = coefficients(alpha)
            prediction = cp.asnumpy(target_oracle @ coef + y_mean)
            score = r2_score(labels[oracle], prediction)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_coefficients = coef

        if validation_only:
            return best_score, best_alpha

        prediction = cp.asnumpy(target_test @ best_coefficients + y_mean)
        score = r2_score(labels[test], prediction)
    if return_predictions:
        return score, best_alpha, labels[test], prediction, test
    return score, best_alpha

def transfer_score(
    source,
    target,
    labels,
    roles,
    return_predictions=False,
    validation_only=False,
):
    """Fit on the source view and score validation or held-out test rows."""
    finite = np.isfinite(labels)
    fit = roles["fit"][finite[roles["fit"]]]
    oracle = roles["oracle"][finite[roles["oracle"]]]
    test = roles["test"][finite[roles["test"]]]
    if min(len(fit), len(oracle), len(test)) < 3 or source.shape[1] != target.shape[1]:
        if return_predictions:
            return np.nan, np.nan, np.array([]), np.array([]), np.array([], dtype=int)
        return np.nan, np.nan

    gpu_id = assigned_gpu()
    if gpu_id is not None and cp is not None:
        return _gpu_transfer_score(
            source,
            target,
            labels,
            fit,
            oracle,
            test,
            return_predictions,
            validation_only,
            gpu_id,
        )

    # Fit preprocessing only in the source view. Applying that exact scaler to the target view is part of the transfer test; no target-view refit occurs.
    scaler = StandardScaler().fit(source[fit])
    source_fit = scaler.transform(source[fit])
    target_oracle = scaler.transform(target[oracle])
    target_test = scaler.transform(target[test])

    # Labels on oracle rows choose regularization strength. They never choose coefficients, because coefficients are always learned on fit rows.
    best_alpha = None
    best_score = -np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(source_fit, labels[fit])
        score = r2_score(labels[oracle], model.predict(target_oracle))
        if score > best_score:
            best_score = score
            best_alpha = alpha

    # During layer scanning return only oracle performance. The final test is deliberately delayed until the label-free and oracle layers are fixed.
    if validation_only:
        return best_score, best_alpha

    model = Ridge(alpha=best_alpha).fit(source_fit, labels[fit])
    prediction = model.predict(target_test)
    score = r2_score(labels[test], prediction)
    if return_predictions:
        return score, best_alpha, labels[test], prediction, test
    return score, best_alpha


def paired_bootstrap_regret(y, label_free_prediction, oracle_prediction, draws, seed):
    """Bootstrap the test-R2 difference using paired test objects."""
    if len(y) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        rows = rng.integers(0, len(y), len(y))
        a = r2_score(y[rows], label_free_prediction[rows])
        b = r2_score(y[rows], oracle_prediction[rows])
        differences.append(a - b)
    return tuple(np.quantile(differences, [0.025, 0.975]))


# -----------------------------------------------------------------------------
# One model/cohort experiment
#
# Intention: make all fold-specific choices in one sequential pass through the layer archive, then reread only the two layers needed for final comparisons. For a real file this is the dominant block and can take minutes to hours.
# -----------------------------------------------------------------------------

def run_file(item, targets, output_root, bootstraps, seed):
    pairs = layer_pairs(item["path"], item["cohort"])
    if not pairs:
        return pd.DataFrame()
    row_indices = item.get("row_indices")
    n = read_matrix(item["path"], pairs[0][0], row_indices).shape[0]
    targets = {name: np.asarray(value)[:n] for name, value in targets.items()}
    result = []

    # Build all five fold states before reading layers so one decoded layer can supply every fold's selection and validation measurements.
    folds = []
    for fold in range(5):
        folds.append(
            {
                "fold": fold,
                "roles": split_roles(n, fold, seed),
                "unlabeled": [],
                "oracle": {
                    (name, direction): []
                    for name in targets
                    for direction in (0, 1)
                },
            }
        )

    # One archive pass supplies selection statistics for all five folds. Only equal-width image pairs can be used for direct retrieval and Ridge transfer.
    for layer, (left_name, right_name, _) in enumerate(pairs):
        left = read_matrix(item["path"], left_name, row_indices)
        right = read_matrix(item["path"], right_name, row_indices)
        if left.shape[1] != right.shape[1]:
            continue
        for state in folds:
            roles = state["roles"]
            mrr, se = selection_mrr(left, right, roles["selector"])
            state["unlabeled"].append({"layer": layer, "mrr": mrr, "se": se})
            for target_name, labels in targets.items():
                for direction, (source, target) in enumerate(((left, right), (right, left))):
                    score, alpha = transfer_score(
                        source, target, labels, roles, validation_only=True
                    )
                    state["oracle"][(target_name, direction)].append(
                        {"layer": layer, "score": score, "alpha": alpha}
                    )

    # The scan above has now frozen both choices. This second phase is the only place where final-test R2 is calculated.
    for state in folds:
        if not state["unlabeled"]:
            continue
        fold = state["fold"]
        roles = state["roles"]
        selected_layer = one_standard_error_layer(state["unlabeled"])

        for target_name, labels in targets.items():
            for direction in (0, 1):
                candidates = state["oracle"][(target_name, direction)]
                candidates = [row for row in candidates if np.isfinite(row["score"])]
                if not candidates:
                    continue
                oracle_layer = max(candidates, key=lambda row: row["score"])["layer"]

                # Evaluate both selected layers on identical test objects. This pairing makes their R2 difference meaningful and bootstrap-able.
                predictions = []
                alphas = []
                scores = []
                for layer in (selected_layer, oracle_layer):
                    left_name, right_name, _ = pairs[layer]
                    left = read_matrix(item["path"], left_name, row_indices)
                    right = read_matrix(item["path"], right_name, row_indices)
                    source, target = ((left, right), (right, left))[direction]
                    score, alpha, y, prediction, test_rows = transfer_score(
                        source, target, labels, roles, return_predictions=True
                    )
                    scores.append(score)
                    alphas.append(alpha)
                    predictions.append((y, prediction, test_rows))

                y = predictions[0][0]
                low, high = paired_bootstrap_regret(
                    y,
                    predictions[0][1],
                    predictions[1][1],
                    bootstraps,
                    seed + fold,
                )
                prediction_dir = Path(output_root) / "E3_predictions"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                direction_name = "hsc_to_other" if direction == 0 else "other_to_hsc"
                saved_rows = predictions[0][2]
                if row_indices is not None:
                    saved_rows = np.asarray(row_indices)[saved_rows]
                # Save row-level predictions so every reported R2 and regret can be independently recomputed after the long archive run.
                np.savez(
                    prediction_dir
                    / f"{item['cohort']}__{item['model']}__{target_name}__{direction_name}__fold{fold}.npz",
                    row_indices=saved_rows,
                    y=y,
                    label_free_prediction=predictions[0][1],
                    oracle_prediction=predictions[1][1],
                )
                result.append(
                    {
                        "cohort": item["cohort"],
                        "model": item["model"],
                        "fold": fold,
                        "target": target_name,
                        "direction": direction_name,
                        "label_free_layer": selected_layer,
                        "oracle_layer": oracle_layer,
                        "label_free_alpha": alphas[0],
                        "oracle_alpha": alphas[1],
                        "label_free_test_r2": scores[0],
                        "oracle_test_r2": scores[1],
                        "regret": scores[0] - scores[1],
                        "regret_low": low,
                        "regret_high": high,
                    }
                )
    return pd.DataFrame(result)


# -----------------------------------------------------------------------------
# Multi-GPU scheduling and cross-family summaries
#
# Intention: give each worker a disjoint group of files and at most one GPU. Worker startup and final CSV grouping take seconds; run_file dominates. The family bootstrap treats family means as replicates and is usually seconds.
# -----------------------------------------------------------------------------

def _run_group(arguments):
    gpu_id, items, old_targets, output_root, bootstraps, seed, cpu_threads = arguments
    configure_worker(gpu_id, cpu_threads)
    rows = []
    for item in items:
        print(f"E3  GPU {gpu_id}  {item['cohort']}/{item['model']}", flush=True)
        frame = run_file(
            item, old_targets[item["cohort"]], output_root, bootstraps, seed
        )
        rows.extend(frame.to_dict("records"))
    return rows


def run(context):
    items = [
        item for item in context["layerwise_files"]
        if item["cohort"] in ("legacysurvey", "jwst")
    ]
    workers = min(context.get("workers", 1), len(items))
    groups = [items[number::workers] for number in range(workers)]
    arguments = [
        (
            number if number < context.get("gpu_count", 0) else None,
            group,
            context["old_targets"],
            context["output_root"],
            context["bootstraps"],
            context["seed"],
            context.get("cpu_threads_per_worker", 1),
        )
        for number, group in enumerate(groups)
    ]
    # Spawn avoids inheriting initialized CUDA/FAISS state from the parent.
    if workers == 1:
        rows = _run_group(arguments[0])
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = [row for group_rows in pool.map(_run_group, arguments) for row in group_rows]
    result = pd.DataFrame(rows)
    if not result.empty:
        result["family"] = result["model"].map(context["model_family"])
    result.to_csv(Path(context["output_root"]) / "E3_image_transfer.csv", index=False)
    if not result.empty:
        # First average checkpoints within architecture family, then bootstrap those family means for the cross-family regret interval.
        family = (
            result.groupby(["cohort", "family", "target", "direction"], as_index=False)
            .agg(models=("model", "nunique"), mean_regret=("regret", "mean"))
        )
        family.to_csv(
            Path(context["output_root"]) / "E3_family_summary.csv", index=False
        )
        rng = np.random.default_rng(context["seed"])
        inference = []
        for keys, group in family.groupby(["cohort", "target", "direction"]):
            values = group["mean_regret"].dropna().to_numpy()
            if len(values) == 0:
                continue
            draws = [
                rng.choice(values, len(values), replace=True).mean()
                for _ in range(context["bootstraps"])
            ]
            low, high = np.quantile(draws, [0.025, 0.975])
            inference.append(
                {
                    "cohort": keys[0],
                    "target": keys[1],
                    "direction": keys[2],
                    "families": len(values),
                    "mean_regret": values.mean(),
                    "family_bootstrap_low": low,
                    "family_bootstrap_high": high,
                }
            )
        pd.DataFrame(inference).to_csv(
            Path(context["output_root"]) / "E3_cross_family_inference.csv",
            index=False,
        )
    return result
