"""E9: positive, negative, projection, PCA, and catalog-nuisance controls.

E9 tries to explain what could create the final HSC/JWST alignment seen in E8.
It measures deliberately aligned and deliberately broken pairs, low-dimensional
random projections, removal of dominant PCs, and removal of catalog variables.
Held-out transformations are fit outside the right-ascension test slabs.

The main blocks are:
1. Construct projected, PC-removed, and nuisance-residualized representations.
2. Run the complete geometry calculation for each control.
3. Repeat leakage-sensitive controls in five held-out spatial folds.
4. Calibrate every control with a shared correspondence-permutation bank.
5. Test paired control-minus-reference changes across architecture families.

Approximate full-data runtime: 31 models still produce about 868 control rows,
so exact neighbor searches and representation geometry remain substantial.
Controls with the same rows are calibrated together: they reuse each generated
permutation and run the overlap comparisons in GPU batches. Projection, PC
removal, residualization, CKA, Procrustes, overlap, and retrieval ranking also
use the assigned GPUs. The stage can still take many hours, but it avoids the
old independent Python permutation loops for all 868 rows.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

try:
    import cupy as cp
except ImportError:
    cp = None

from experiment_1_geometry import (
    assigned_gpu,
    configure_worker,
    geometry,
    holm_adjust,
    single_layer_inference,
)
from experiment_8_final_transfer import spatial_roles


# -----------------------------------------------------------------------------
# Control transformations
#
# Intention: alter selected aspects of the representations while preserving row correspondence. Random projection asks whether alignment survives aggressive dimension reduction. PC removal asks whether a few dominant global directions explain it. Residualization removes linear catalog trends. Projection/PCA can take seconds to minutes for a full wide matrix; residualization is usually seconds because the nuisance design has only three columns.
# -----------------------------------------------------------------------------

def random_projection(x, width, rng):
    """Project all rows through a Gaussian Johnson-Lindenstrauss-style matrix."""
    matrix = rng.normal(size=(x.shape[1], width)).astype(np.float32)
    matrix /= np.sqrt(width)
    return _matrix_projection(x, matrix)


def _matrix_projection(x, matrix):
    """Multiply a large matrix in GPU row blocks, with a NumPy fallback."""
    gpu_id = assigned_gpu()
    if gpu_id is None:
        return x @ matrix
    answer = np.empty((len(x), matrix.shape[1]), dtype=np.float32)
    with cp.cuda.Device(gpu_id):
        matrix_gpu = cp.asarray(matrix)
        for start in range(0, len(x), 8192):
            stop = min(start + 8192, len(x))
            answer[start:stop] = cp.asnumpy(cp.asarray(x[start:stop]) @ matrix_gpu)
    return answer


def random_projection_pair(left, right, width, rng):
    """Use the same projection when the two feature widths match."""
    # A shared matrix preserves any coordinate-by-coordinate relation when the views have equal width. Unequal widths necessarily use separate matrices.
    if left.shape[1] == right.shape[1]:
        matrix = rng.normal(size=(left.shape[1], width)).astype(np.float32)
        matrix /= np.sqrt(width)
        return _matrix_projection(left, matrix), _matrix_projection(right, matrix)
    return random_projection(left, width, rng), random_projection(right, width, rng)


def remove_leading_pcs(x, fit_rows, count=5):
    """Fit leading PCs on fit rows and subtract their projection from all rows."""
    count = min(count, x.shape[1], len(fit_rows) - 1)
    gpu_id = assigned_gpu()
    if gpu_id is not None:
        with cp.cuda.Device(gpu_id):
            fit = cp.asarray(x[fit_rows], dtype=cp.float64)
            mean = fit.mean(axis=0)
            centered = fit - mean

            # Only five directions are needed. A small randomized subspace is much cheaper than decomposing the entire rows-by-features matrix.
            rng = np.random.default_rng(0)
            sketch_width = min(count + 8, min(centered.shape))
            random_matrix = cp.asarray(
                rng.normal(size=(centered.shape[1], sketch_width)),
                dtype=cp.float64,
            )
            q, _ = cp.linalg.qr(centered @ random_matrix, mode="reduced")
            for _ in range(3):
                q, _ = cp.linalg.qr(centered.T @ q, mode="reduced")
                q, _ = cp.linalg.qr(centered @ q, mode="reduced")
            _, _, components = cp.linalg.svd(q.T @ centered, full_matrices=False)
            components = components[:count]

            answer = np.empty_like(x, dtype=np.float32)
            for start in range(0, len(x), 8192):
                stop = min(start + 8192, len(x))
                block = cp.asarray(x[start:stop], dtype=cp.float64) - mean
                block -= (block @ components.T) @ components
                answer[start:stop] = cp.asnumpy(block).astype(np.float32)
        return answer

    pca = PCA(n_components=count, random_state=0).fit(x[fit_rows])
    # Use the fit-set mean and components everywhere. Test rows influence neither the directions nor the centering.
    centered = x - pca.mean_
    return centered - (centered @ pca.components_.T) @ pca.components_


def residualize(x, nuisance, fit_rows):
    """Remove the linear prediction from catalog nuisance variables."""
    good_fit = fit_rows[np.isfinite(nuisance[fit_rows]).all(axis=1)]
    good_all = np.isfinite(nuisance).all(axis=1)
    answer = np.full_like(x, np.nan, dtype=np.float32)
    gpu_id = assigned_gpu()
    if gpu_id is not None:
        with cp.cuda.Device(gpu_id):
            fit_nuisance = cp.asarray(nuisance[good_fit], dtype=cp.float64)
            fit_values = cp.asarray(x[good_fit], dtype=cp.float64)
            nuisance_mean = fit_nuisance.mean(axis=0)
            values_mean = fit_values.mean(axis=0)
            nuisance_centered = fit_nuisance - nuisance_mean
            values_centered = fit_values - values_mean
            coefficients = cp.linalg.pinv(nuisance_centered) @ values_centered

            good_rows = np.flatnonzero(good_all)
            for start in range(0, len(good_rows), 8192):
                rows = good_rows[start : start + 8192]
                block_nuisance = cp.asarray(nuisance[rows], dtype=cp.float64)
                block_values = cp.asarray(x[rows], dtype=cp.float64)
                prediction = (
                    (block_nuisance - nuisance_mean) @ coefficients + values_mean
                )
                answer[rows] = cp.asnumpy(block_values - prediction).astype(np.float32)
        return answer

    # This is a multivariate regression with one target per embedding feature. It is fit only on spatial training rows, then applied to all complete rows.
    model = LinearRegression().fit(nuisance[good_fit], x[good_fit])
    answer[good_all] = x[good_all] - model.predict(nuisance[good_all])
    return answer


# -----------------------------------------------------------------------------
# One control's geometry and grouped correspondence calibration
#
# Intention: apply exactly the E8 geometry definition to every control, retain its neighbor arrays briefly, and calibrate controls with compatible row sets together. Geometry can take seconds to minutes per control. One shared GPU permutation pass can take minutes for a real group at 9,999 draws.
# -----------------------------------------------------------------------------

def add_control(
    rows,
    item,
    name,
    left,
    right,
    fold=None,
):
    # geometry performs exact neighbors, overlap, CKA, Procrustes, and retrieval.
    numbers, per_row, left_neighbors, right_neighbors = geometry(left, right)
    numbers["alignment_permutation_p"] = np.nan
    numbers["alignment_low"] = np.nan
    numbers["alignment_high"] = np.nan
    row = {"model": item["model"], "control": name, "fold": fold}
    row.update({key: value for key, value in numbers.items() if np.isscalar(value)})
    rows.append(row)
    return {
        "row": row,
        "per_row": per_row,
        # Permutation inference uses k=20, so do not retain the unused k=21..100 columns while the other controls in this group are being calculated.
        "neighbors": (
            left_neighbors[:, :20].copy(),
            right_neighbors[:, :20].copy(),
        ),
    }


def add_shared_inference(records, permutations, bootstraps, seed):
    """Add one correspondence test per control using a shared random bank."""
    # The primary k=20 correction is undefined when a tiny smoke-test fold has 21 or fewer usable rows. Real held-out folds contain thousands of rows.
    records = [
        record
        for record in records
        if len(record["per_row"]) > 21
        and np.isfinite(record["row"]["corrected_overlap_20"])
    ]
    if not records:
        return
    p_values, low, high = single_layer_inference(
        [record["neighbors"] for record in records],
        [record["per_row"] for record in records],
        permutations,
        bootstraps,
        seed,
    )
    for record, p_value, lower, upper in zip(records, p_values, low, high):
        record["row"]["alignment_permutation_p"] = p_value
        record["row"]["alignment_low"] = lower
        record["row"]["alignment_high"] = upper


# -----------------------------------------------------------------------------
# All controls for one worker's models
#
# Intention: build intuitive full-sample controls first, then fit PC/nuisance transformations outside each held-out spatial test fold. One model adds seven full-sample controls and 20 held-out rows; its raw full-sample row comes from E8. Exact geometry still makes this block take minutes to hours per real model.
# -----------------------------------------------------------------------------

def _run_group(arguments):
    gpu_id, items, nuisance, ra, permutations, bootstraps, seed, cpu_threads = arguments
    configure_worker(gpu_id, cpu_threads)
    rows = []
    for item in items:
        print(f"E9  GPU {gpu_id}  {item['model']}", flush=True)
        # Derive a stable model-specific random seed without Python's randomized hash function, so shuffled and projected controls reproduce exactly.
        model_number = sum(
            (position + 1) * ord(character)
            for position, character in enumerate(item["model"])
        )
        rng = np.random.default_rng(seed + model_number)
        left = np.asarray(np.load(item["hsc"], mmap_mode="r"), dtype=np.float32)
        right = np.asarray(np.load(item["jwst"], mmap_mode="r"), dtype=np.float32)
        lengths = {len(left), len(right), len(nuisance), len(ra)}
        if len(lengths) != 1:
            raise ValueError(f"Control row counts differ for {item['model']}: {lengths}")
        n = len(left)

        # Full-sample controls establish expected behavior: raw is the reference; shuffled pairs should break alignment; identical views should maximize it; noise/Gaussian/projection controls probe less extreme alternatives.
        full_controls = []
        full_controls.append(add_control(
            rows,
            item,
            "shuffled_correspondence",
            left,
            right[rng.permutation(n)],
        ))
        full_controls.append(add_control(
            rows, item, "both_views_shuffled",
            left[rng.permutation(n)], right[rng.permutation(n)],
        ))
        full_controls.append(add_control(rows, item, "identical", left, left))
        scale = np.std(left) * 0.1
        full_controls.append(add_control(
            rows,
            item,
            "identical_plus_noise",
            left,
            left + rng.normal(0, scale, left.shape).astype(np.float32),
        ))

        gaussian_left = rng.normal(size=left.shape).astype(np.float32)
        gaussian_right = rng.normal(size=right.shape).astype(np.float32)
        full_controls.append(add_control(
            rows, item, "matched_gaussian", gaussian_left, gaussian_right
        ))

        for width in (32, 64):
            projection_width = min(width, left.shape[1], right.shape[1])
            projected_left, projected_right = random_projection_pair(
                left, right, projection_width, rng
            )
            full_controls.append(add_control(
                rows,
                item,
                f"random_projection_{width}",
                projected_left,
                projected_right,
            ))
        add_shared_inference(full_controls, permutations, bootstraps, seed)
        del full_controls

        # Held-out controls use exactly the same RA folds as E8. PC directions and nuisance coefficients are learned from fit rows, then geometry is measured only among test rows.
        for fold in range(5):
            roles = spatial_roles(np.asarray(ra)[:n], fold)
            fit_rows = roles["fit"]
            test_rows = roles["test"]
            heldout_controls = []
            heldout_controls.append(add_control(
                rows,
                item,
                "raw_heldout",
                left[test_rows],
                right[test_rows],
                fold,
            ))

            # Refit separately in each fold; reusing a full-data PCA would allow held-out test rows to influence the removed directions.
            left_no_pc = remove_leading_pcs(left, fit_rows)
            right_no_pc = remove_leading_pcs(right, fit_rows)
            heldout_controls.append(add_control(
                rows,
                item,
                "leading_pcs_removed_heldout",
                left_no_pc[test_rows],
                right_no_pc[test_rows],
                fold,
            ))
            add_shared_inference(
                heldout_controls, permutations, bootstraps, seed
            )
            del heldout_controls

            # Report raw complete cases beside residualized complete cases so a change is not confused with merely dropping missing-catalog rows.
            left_residual = residualize(left, nuisance[:n], fit_rows)
            right_residual = residualize(right, nuisance[:n], fit_rows)
            good = (
                np.isfinite(left_residual[test_rows]).all(axis=1)
                & np.isfinite(right_residual[test_rows]).all(axis=1)
            )
            complete_case_controls = []
            complete_case_controls.append(add_control(
                rows,
                item,
                "raw_complete_case_heldout",
                left[test_rows][good],
                right[test_rows][good],
                fold,
            ))
            complete_case_controls.append(add_control(
                rows,
                item,
                "catalog_residualized_heldout",
                left_residual[test_rows][good],
                right_residual[test_rows][good],
                fold,
            ))
            add_shared_inference(
                complete_case_controls, permutations, bootstraps, seed
            )
            del complete_case_controls
    return rows


# -----------------------------------------------------------------------------
# Multi-GPU scheduling and family summaries
#
# Intention: distribute models across one process per available GPU and combine scalar control statistics. Scheduling and CSV grouping take seconds to minutes; repeated add_control calls account for almost the entire E9 runtime.
# -----------------------------------------------------------------------------

GEOMETRY_COLUMNS = [
    "n",
    "left_width",
    "right_width",
    "left_zero_fraction",
    "right_zero_fraction",
    "overlap_5",
    "corrected_overlap_5",
    "overlap_10",
    "corrected_overlap_10",
    "overlap_20",
    "corrected_overlap_20",
    "overlap_50",
    "corrected_overlap_50",
    "overlap_100",
    "corrected_overlap_100",
    "cka",
    "procrustes",
    "mrr",
    "median_rank",
    "tie_fraction",
    "recall_1",
    "recall_5",
    "recall_10",
    "recall_50",
    "tie_dominated",
    "available",
    "alignment_permutation_p",
    "alignment_low",
    "alignment_high",
]

CONTROL_COMPARISONS = {
    "shuffle_correspondence": ("raw", "shuffled_correspondence"),
    "shuffle_both_views": ("raw", "both_views_shuffled"),
    "make_views_identical": ("raw", "identical"),
    "add_noise_to_identical": ("identical", "identical_plus_noise"),
    "replace_with_gaussian": ("raw", "matched_gaussian"),
    "project_to_32": ("raw", "random_projection_32"),
    "project_to_64": ("raw", "random_projection_64"),
    "remove_leading_pcs": ("raw_heldout", "leading_pcs_removed_heldout"),
    "remove_catalog_nuisance": (
        "raw_complete_case_heldout",
        "catalog_residualized_heldout",
    ),
}

INFERENCE_METRICS = (
    "corrected_overlap_20",
    "cka",
    "procrustes",
    "mrr",
)


def _raw_rows_from_e8(root, items):
    """Reuse the raw geometry already calculated by E8 for these same arrays."""
    filename = Path(root) / "E8_final_transfer.csv"
    if not filename.exists():
        raise FileNotFoundError("E9 needs E8_final_transfer.csv from the E8 stage")
    e8 = pd.read_csv(filename).drop_duplicates("model")
    by_model = e8.set_index("model")
    rows = []
    for item in items:
        model = item["model"]
        if model not in by_model.index:
            raise ValueError(f"E8 has no raw geometry row for {model}")
        source = by_model.loc[model]
        row = {"model": model, "control": "raw", "fold": np.nan}
        for column in GEOMETRY_COLUMNS:
            if column in source.index:
                row[column] = source[column]
        rows.append(row)
    return rows


def paired_control_inference(result, context):
    """Test paired control-minus-reference changes across model families."""
    model_means = (
        result.groupby(["model", "family", "control"], as_index=False)[
            list(INFERENCE_METRICS)
        ]
        .mean()
    )
    rng = np.random.default_rng(context["seed"])
    detail_rows = []
    inference_rows = []

    for comparison, (reference_name, control_name) in CONTROL_COMPARISONS.items():
        reference = model_means[model_means["control"] == reference_name]
        control = model_means[model_means["control"] == control_name]
        paired = reference.merge(
            control,
            on=["model", "family"],
            suffixes=("_reference", "_control"),
        )
        for metric in INFERENCE_METRICS:
            difference = paired[f"{metric}_control"] - paired[f"{metric}_reference"]
            good = np.isfinite(difference)
            for index in paired.index[good]:
                detail_rows.append(
                    {
                        "comparison": comparison,
                        "reference": reference_name,
                        "control": control_name,
                        "metric": metric,
                        "model": paired.loc[index, "model"],
                        "family": paired.loc[index, "family"],
                        "reference_value": paired.loc[index, f"{metric}_reference"],
                        "control_value": paired.loc[index, f"{metric}_control"],
                        "difference": difference.loc[index],
                    }
                )

            usable = paired.loc[good, ["family"]].copy()
            usable["difference"] = difference[good].to_numpy()
            family_values = usable.groupby("family")["difference"].mean().to_numpy()
            if len(family_values) == 0:
                continue
            observed = float(family_values.mean())

            # Random signs give the paired null hypothesis of no systematic control effect. Family means are the independent units.
            extreme = 0
            for _ in range(context["permutations"]):
                signs = rng.choice((-1.0, 1.0), size=len(family_values))
                if abs(np.mean(signs * family_values)) >= abs(observed):
                    extreme += 1
            p_value = (extreme + 1) / (context["permutations"] + 1)

            bootstrap = [
                rng.choice(family_values, len(family_values), replace=True).mean()
                for _ in range(context["bootstraps"])
            ]
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            inference_rows.append(
                {
                    "comparison": comparison,
                    "reference": reference_name,
                    "control": control_name,
                    "metric": metric,
                    "models": int(good.sum()),
                    "families": len(family_values),
                    "mean_control_minus_reference": observed,
                    "family_bootstrap_low": low,
                    "family_bootstrap_high": high,
                    "paired_permutation_p": p_value,
                }
            )

    inference = pd.DataFrame(inference_rows)
    if not inference.empty:
        inference["paired_p_holm"] = holm_adjust(inference["paired_permutation_p"])
    return pd.DataFrame(detail_rows), inference

def run(context):
    root = Path(context["output_root"])
    catalog = context["final_catalog"]
    # These are the three catalog explanations removed together: brightness, broad color, and redshift.
    nuisance = np.column_stack(
        [
            catalog["targets"].get("magnitude"),
            catalog["targets"].get("g_minus_r"),
            catalog["targets"].get("redshift"),
        ]
    )

    items = context["final_pairs"]
    workers = min(context.get("workers", 1), len(items))
    groups = [items[number::workers] for number in range(workers)]
    arguments = [
        (
            number if number < context.get("gpu_count", 0) else None,
            group,
            nuisance,
            catalog["ra"],
            context["permutations"],
            context["bootstraps"],
            context["seed"],
            context.get("cpu_threads_per_worker", 1),
        )
        for number, group in enumerate(groups)
    ]
    # Spawn prevents workers from inheriting the parent's CUDA/FAISS state.
    if workers == 1:
        rows = _run_group(arguments[0])
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rows = [row for group_rows in pool.map(_run_group, arguments) for row in group_rows]
    rows = _raw_rows_from_e8(root, items) + rows
    result = pd.DataFrame(rows)
    if not result.empty:
        result["family"] = result["model"].map(context["model_family"])
        # E9 is one robustness-testing family: raw and every adjusted/null row receive a single Holm correction together.
        result["alignment_p_holm"] = holm_adjust(
            result["alignment_permutation_p"]
        )
    result.to_csv(root / "E9_controls.csv", index=False)
    if not result.empty:
        # Average control responses within architecture family for a compact, readable comparison. The detailed per-model rows remain in E9_controls.
        family = (
            result.groupby(["family", "control"], as_index=False)
            .agg(
                models=("model", "nunique"),
                mean_corrected_overlap_20=("corrected_overlap_20", "mean"),
                mean_cka=("cka", "mean"),
                mean_procrustes=("procrustes", "mean"),
                mean_mrr=("mrr", "mean"),
            )
        )
        family.to_csv(root / "E9_family_summary.csv", index=False)
        detail, inference = paired_control_inference(result, context)
        detail.to_csv(root / "E9_paired_control_differences.csv", index=False)
        inference.to_csv(root / "E9_paired_control_inference.csv", index=False)
    return result
