"""E4: learn a small common space for HSC images and spectra.

Image and spectrum embeddings generally have different widths, so E3 cannot
compare them directly. E4 first learns a 64-dimensional-or-smaller common space:
1. Standardize each view using fit rows only.
2. Reduce each view with PCA.
3. Whiten the two PCA spaces and align them with regularized CCA.
4. Select a layer/ridge from label-free paired retrieval.
5. Compare that layer with a label-selected oracle using cross-view Ridge R2.

Approximate full-data runtime: PCA is fit once per layer/fold on the assigned GPU
and reused across all four CCA ridges. CCA, transforms, retrieval, and downstream
Ridge also use that GPU. A model/cohort file should generally take minutes to an
hour, while the complete stage can still take hours because every layer is read
in five folds. These are estimates; the small smoke test takes seconds.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import cupy as cp
except ImportError:
    cp = None

from experiment_1_geometry import assigned_gpu, configure_worker, layer_pairs, read_matrix
from experiment_3_image_transfer import (
    one_standard_error_layer,
    paired_bootstrap_regret,
    selection_mrr,
    split_roles,
    transfer_score,
)

CCA_RIDGES = (0.001, 0.01, 0.1, 1.0)


# -----------------------------------------------------------------------------
# Fit and apply the common image/spectrum space
#
# Intention: reduce unequal-width views to coordinates that maximize their regularized linear correlation without consulting scientific labels. PCA is the expensive part and can take seconds to minutes per fit on real matrices. It is done once before the four ridge choices. Later SVDs are at most 64 by 64.
# -----------------------------------------------------------------------------

def _inverse_sqrt(matrix):
    """Return a symmetric inverse square root for a covariance matrix."""
    if cp is not None and isinstance(matrix, cp.ndarray):
        # The input is symmetric positive definite. SVD gives the same positive eigenvalues here and is well supported by CuPy's CUDA libraries.
        vectors, values, _ = cp.linalg.svd(matrix, full_matrices=False)
        values = cp.maximum(values, 1e-10)
        return (vectors / cp.sqrt(values)) @ vectors.T
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-10)
    return (vectors / np.sqrt(values)) @ vectors.T


def _gpu_randomized_pca(values, rank, seed):
    """Find the leading PCA axes with a small randomized GPU decomposition."""
    mean = values.mean(axis=0)
    centered = values - mean
    sketch_width = min(rank + 12, min(centered.shape))

    # A fixed NumPy generator makes this approximation reproducible across GPU models. Three power steps sharpen the leading subspace before the small final SVD.
    rng = np.random.default_rng(seed)
    random_matrix = cp.asarray(
        rng.normal(size=(centered.shape[1], sketch_width)), dtype=cp.float64
    )
    q, _ = cp.linalg.qr(centered @ random_matrix, mode="reduced")
    for _ in range(3):
        q, _ = cp.linalg.qr(centered.T @ q, mode="reduced")
        q, _ = cp.linalg.qr(centered @ q, mode="reduced")
    small = q.T @ centered
    _, _, components = cp.linalg.svd(small, full_matrices=False)
    components = components[:rank]
    scores = centered @ components.T
    return mean, components, scores


def prepare_common_space(left, right, fit_rows, max_rank=64):
    """Standardize and PCA each view once before trying CCA ridges."""
    gpu_id = assigned_gpu()
    if gpu_id is not None and cp is not None:
        with cp.cuda.Device(gpu_id):
            left_fit = cp.asarray(left[fit_rows], dtype=cp.float64)
            right_fit = cp.asarray(right[fit_rows], dtype=cp.float64)
            left_mean = left_fit.mean(axis=0)
            right_mean = right_fit.mean(axis=0)
            left_scale = left_fit.std(axis=0)
            right_scale = right_fit.std(axis=0)
            left_scale = cp.where(left_scale == 0, 1, left_scale)
            right_scale = cp.where(right_scale == 0, 1, right_scale)
            left_scaled = (left_fit - left_mean) / left_scale
            right_scaled = (right_fit - right_mean) / right_scale
            left_rank = min(max_rank, left_scaled.shape[0] - 1, left_scaled.shape[1])
            right_rank = min(
                max_rank, right_scaled.shape[0] - 1, right_scaled.shape[1]
            )
            left_pca_mean, left_components, x = _gpu_randomized_pca(
                left_scaled, left_rank, 0
            )
            right_pca_mean, right_components, y = _gpu_randomized_pca(
                right_scaled, right_rank, 1
            )
        return {
            "backend": "gpu",
            "gpu_id": gpu_id,
            "left_mean": left_mean,
            "right_mean": right_mean,
            "left_scale": left_scale,
            "right_scale": right_scale,
            "left_pca_mean": left_pca_mean,
            "right_pca_mean": right_pca_mean,
            "left_components": left_components,
            "right_components": right_components,
            "left_scores": x,
            "right_scores": y,
        }

    # CPU fallback is the original sklearn calculation.
    left_scaler = StandardScaler().fit(left[fit_rows])
    right_scaler = StandardScaler().fit(right[fit_rows])
    left_scaled = left_scaler.transform(left[fit_rows])
    right_scaled = right_scaler.transform(right[fit_rows])

    # Rank cannot exceed the number of centered samples minus one or the input feature width. The hard cap keeps the CCA matrices small and stable.
    left_rank = min(max_rank, left_scaled.shape[0] - 1, left_scaled.shape[1])
    right_rank = min(max_rank, right_scaled.shape[0] - 1, right_scaled.shape[1])
    left_pca = PCA(n_components=left_rank, random_state=0).fit(left_scaled)
    right_pca = PCA(n_components=right_rank, random_state=0).fit(right_scaled)
    x = left_pca.transform(left_scaled)
    y = right_pca.transform(right_scaled)
    return {
        "backend": "cpu",
        "left_scaler": left_scaler,
        "right_scaler": right_scaler,
        "left_pca": left_pca,
        "right_pca": right_pca,
        "left_scores": x,
        "right_scores": y,
    }


def fit_common_space(left, right, fit_rows, ridge, max_rank=64, prepared=None):
    """PCA followed by a small regularized CCA calculation."""
    # PCA does not depend on the CCA ridge, so a caller scanning all four ridge values can prepare it once and pass the result into each fit.
    if prepared is None:
        prepared = prepare_common_space(left, right, fit_rows, max_rank)
    x = prepared["left_scores"]
    y = prepared["right_scores"]
    gpu = prepared["backend"] == "gpu"
    xp = cp if gpu else np

    # Regularized within-view covariances are whitened. SVD of the whitened cross-covariance supplies paired axes with descending shared correlation.
    xx = x.T @ x / max(1, len(x) - 1) + ridge * xp.eye(x.shape[1])
    yy = y.T @ y / max(1, len(y) - 1) + ridge * xp.eye(y.shape[1])
    xy = x.T @ y / max(1, len(x) - 1)
    left_inverse = _inverse_sqrt(xx)
    right_inverse = _inverse_sqrt(yy)
    whitened = left_inverse @ xy @ right_inverse
    u, _, vt = xp.linalg.svd(whitened, full_matrices=False)
    rank = min(max_rank, u.shape[1], vt.shape[0])
    mapping = dict(prepared)
    mapping.update({
        "left_map": left_inverse @ u[:, :rank],
        "right_map": right_inverse @ vt.T[:, :rank],
    })
    return mapping


def transform_common(mapping, left, right):
    """Apply a fitted scaler, PCA, and CCA map to both complete views."""
    if mapping["backend"] == "gpu":
        with cp.cuda.Device(mapping["gpu_id"]):
            def transform(values, side):
                answer = np.empty(
                    (len(values), mapping[f"{side}_map"].shape[1]),
                    dtype=np.float32,
                )
                for start in range(0, len(values), 8192):
                    stop = min(start + 8192, len(values))
                    block = cp.asarray(values[start:stop], dtype=cp.float64)
                    block = (
                        block - mapping[f"{side}_mean"]
                    ) / mapping[f"{side}_scale"]
                    block = (
                        block - mapping[f"{side}_pca_mean"]
                    ) @ mapping[f"{side}_components"].T
                    block = block @ mapping[f"{side}_map"]
                    answer[start:stop] = cp.asnumpy(block).astype(np.float32)
                return answer

            return transform(left, "left"), transform(right, "right")

    left = mapping["left_scaler"].transform(left)
    right = mapping["right_scaler"].transform(right)
    left = mapping["left_pca"].transform(left) @ mapping["left_map"]
    right = mapping["right_pca"].transform(right) @ mapping["right_map"]
    return left.astype(np.float32), right.astype(np.float32)


# -----------------------------------------------------------------------------
# One model/cohort experiment
#
# Intention: select a layer and CCA ridge with geometry alone, then compare its scientific transfer with a validation-label oracle. The first layer pass fits up to 4 ridges x 5 folds and dominates runtime. Final evaluation rereads and refits only selected layers. A real file generally takes minutes to hours.
# -----------------------------------------------------------------------------

def run_file(item, labels, bootstraps, seed, output_root=None):
    pairs = layer_pairs(item["path"], item["cohort"])
    row_indices = item.get("row_indices")
    n = read_matrix(item["path"], pairs[0][0], row_indices).shape[0]
    labels = None if labels is None else np.asarray(labels)[:n]
    rows = []

    # Construct all fold roles up front so each decoded layer can be evaluated for all folds before moving to the next Parquet column.
    folds = []
    for fold in range(5):
        folds.append(
            {
                "fold": fold,
                "roles": split_roles(n, fold, seed),
                "choices": [],
                "supervised": {0: [], 1: []},
            }
        )

    # One archive pass supplies selection statistics for all five folds. For every fold, the best CCA ridge is chosen only by selector-set retrieval.
    for layer, (left_name, right_name, _) in enumerate(pairs):
        left = read_matrix(item["path"], left_name, row_indices)
        right = read_matrix(item["path"], right_name, row_indices)
        for state in folds:
            roles = state["roles"]
            best = None
            prepared = prepare_common_space(left, right, roles["fit"])
            for cca_ridge in CCA_RIDGES:
                mapping = fit_common_space(
                    left,
                    right,
                    roles["fit"],
                    cca_ridge,
                    prepared=prepared,
                )
                selector = roles["selector"]
                left_common, right_common = transform_common(
                    mapping, left[selector], right[selector]
                )
                local_rows = np.arange(len(selector))
                mrr, se = selection_mrr(left_common, right_common, local_rows)
                if best is None or mrr > best["mrr"]:
                    best = {
                        "layer": layer,
                        "mrr": mrr,
                        "se": se,
                        "cca_ridge": cca_ridge,
                    }
            state["choices"].append(best)

            # SDSS has redshift labels, so an oracle-validation score is also recorded for later comparison. DESI remains geometry-only here.
            if labels is not None:
                mapping = fit_common_space(
                    left,
                    right,
                    roles["fit"],
                    best["cca_ridge"],
                    prepared=prepared,
                )
                left_common, right_common = transform_common(mapping, left, right)
                for direction, (source, target) in enumerate(
                    ((left_common, right_common), (right_common, left_common))
                ):
                    score, _ = transfer_score(
                        source, target, labels, roles, validation_only=True
                    )
                    state["supervised"][direction].append(
                        {"layer": layer, "score": score}
                    )

    # All layer/ridge choices are now frozen. Only this phase touches final test rows, first for retrieval and then (where labels exist) for transfer R2.
    for state in folds:
        fold = state["fold"]
        roles = state["roles"]
        layer_choices = state["choices"]
        supervised = state["supervised"]

        selected_layer = one_standard_error_layer(layer_choices)
        selected_cca_ridge = next(
            row["cca_ridge"] for row in layer_choices if row["layer"] == selected_layer
        )

        left_name, right_name, _ = pairs[selected_layer]
        left = read_matrix(item["path"], left_name, row_indices)
        right = read_matrix(item["path"], right_name, row_indices)
        selected_mapping = fit_common_space(
            left, right, roles["fit"], selected_cca_ridge
        )
        selected_left, selected_right = transform_common(selected_mapping, left, right)
        selected_test_mrr, _ = selection_mrr(
            selected_left, selected_right, roles["test"]
        )

        if labels is None:
            rows.append(
                {
                    "cohort": item["cohort"],
                    "model": item["model"],
                    "fold": fold,
                    "label_free_layer": selected_layer,
                    "cca_ridge": selected_cca_ridge,
                    "test_mrr": selected_test_mrr,
                    "target": "geometry_only",
                }
            )
            continue

        for direction in (0, 1):
            valid = [row for row in supervised[direction] if np.isfinite(row["score"])]
            if not valid:
                continue
            oracle_layer = max(valid, key=lambda row: row["score"])["layer"]
            predictions = []
            scores = []
            alphas = []
            # Fit both mappings independently on the same fit rows and evaluate both predictions on the same test rows for paired regret.
            for layer in (selected_layer, oracle_layer):
                left_name, right_name, _ = pairs[layer]
                left = read_matrix(item["path"], left_name, row_indices)
                right = read_matrix(item["path"], right_name, row_indices)
                cca_ridge = next(
                    row["cca_ridge"] for row in layer_choices if row["layer"] == layer
                )
                mapping = fit_common_space(left, right, roles["fit"], cca_ridge)
                left_common, right_common = transform_common(mapping, left, right)
                source, target = (
                    (left_common, right_common),
                    (right_common, left_common),
                )[direction]
                score, alpha, y, prediction, test_rows = transfer_score(
                    source, target, labels, roles, return_predictions=True
                )
                scores.append(score)
                alphas.append(alpha)
                predictions.append((y, prediction, test_rows))
            low, high = paired_bootstrap_regret(
                predictions[0][0],
                predictions[0][1],
                predictions[1][1],
                bootstraps,
                seed + fold,
            )
            direction_name = (
                "spectrum_to_image" if direction == 1 else "image_to_spectrum"
            )
            # Row-level predictions make the final R2 and bootstrap comparison reproducible without rerunning the expensive PCA/CCA scan.
            if output_root is not None:
                prediction_dir = Path(output_root) / "E4_predictions"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                saved_rows = predictions[0][2]
                if row_indices is not None:
                    saved_rows = np.asarray(row_indices)[saved_rows]
                np.savez(
                    prediction_dir
                    / f"{item['cohort']}__{item['model']}__redshift__{direction_name}__fold{fold}.npz",
                    row_indices=saved_rows,
                    y=predictions[0][0],
                    label_free_prediction=predictions[0][1],
                    oracle_prediction=predictions[1][1],
                )
            rows.append(
                {
                    "cohort": item["cohort"],
                    "model": item["model"],
                    "fold": fold,
                    "target": "redshift",
                    "direction": direction_name,
                    "test_mrr": selected_test_mrr,
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
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Worker scheduling and family-level output
#
# Intention: distribute independent files across at most eight processes, then combine their small result tables. Process scheduling and aggregation take seconds; PCA/CCA work inside run_file determines total runtime.
# -----------------------------------------------------------------------------

def _run_group(arguments):
    gpu_id, items, old_targets, bootstraps, seed, cpu_threads, output_root = arguments
    configure_worker(gpu_id, cpu_threads)
    rows = []
    for item in items:
        print(f"E4  GPU {gpu_id}  {item['cohort']}/{item['model']}", flush=True)
        # Only SDSS has labels used by this experiment. DESI still contributes the label-free common-space retrieval result.
        labels = None
        if item["cohort"] == "sdss":
            labels = old_targets["sdss"]["redshift"]
        frame = run_file(item, labels, bootstraps, seed, output_root)
        rows.extend(frame.to_dict("records"))
    return rows


def run(context):
    items = [
        item for item in context["layerwise_files"]
        if item["cohort"] in ("desi", "sdss")
    ]
    workers = min(context.get("workers", 1), len(items))
    groups = [items[number::workers] for number in range(workers)]
    arguments = [
        (
            number if number < context.get("gpu_count", 0) else None,
            group,
            context["old_targets"],
            context["bootstraps"],
            context["seed"],
            context.get("cpu_threads_per_worker", 1),
            context["output_root"],
        )
        for number, group in enumerate(groups)
    ]
    # Spawn creates independent numerical-library and FAISS/CUDA state.
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
    result.to_csv(Path(context["output_root"]) / "E4_spectrum_transfer.csv", index=False)
    if not result.empty:
        # Average checkpoints inside families before drawing cross-family bootstrap intervals, matching the independence assumption used in E3.
        family = (
            result.groupby(
                ["cohort", "family", "target", "direction"],
                as_index=False,
                dropna=False,
            )
            .agg(
                models=("model", "nunique"),
                mean_test_mrr=("test_mrr", "mean"),
                mean_regret=("regret", "mean"),
            )
        )
        family.to_csv(
            Path(context["output_root"]) / "E4_family_summary.csv", index=False
        )
        labeled_family = family.dropna(subset=["direction", "mean_regret"])
        rng = np.random.default_rng(context["seed"])
        inference = []
        for keys, group in labeled_family.groupby(["cohort", "target", "direction"]):
            values = group["mean_regret"].to_numpy()
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
        pd.DataFrame(
            inference,
            columns=[
                "cohort",
                "target",
                "direction",
                "families",
                "mean_regret",
                "family_bootstrap_low",
                "family_bootstrap_high",
            ],
        ).to_csv(
            Path(context["output_root"]) / "E4_cross_family_inference.csv",
            index=False,
        )
    return result
