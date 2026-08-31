"""E7: collect the large COSMOS-Web replication and full layer-pair maps.

E1 compares layers only at matching depths. E7 forms a complete two-dimensional
map: every HSC layer is compared with every layer in the paired image survey.
It reuses E1's saved neighbor lists for local overlap, but reads the underlying
matrices again for off-diagonal CKA and Procrustes similarity. It also verifies
that the large layerwise COSMOS-Web files and the final E8 arrays have matching
row counts and final feature widths.

The optimized implementation keeps the same exhaustive statistics but avoids
the largest repeated work:
- There are about 151,286 layer-pair cells across the released image cohorts.
- Every layer's CKA denominator and centered norm are calculated once instead
  of once per partner layer, removing roughly 98 percent of the old matrix-
  multiplication work.
- Parquet columns are decoded once into a simple NPY cache beside the results.
  Layers are then loaded in blocks of eight to reduce repeated storage reads.
- Cross-products and singular values use CuPy float64 on the assigned GPU.
  A CPU fallback keeps the smoke test and CPU-only machines usable.

The remaining work is one cross-product and one dense SVD per off-diagonal cell.
On eight A100s the full stage should now be thought of as hours to perhaps a day
or more, rather than weeks. Storage speed, matrix widths, and SVD throughput can
still move that estimate substantially; it is not a benchmarked promise.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cupy as cp
except ImportError:
    cp = None

from experiment_1_geometry import (
    cka_and_procrustes,
    configure_worker,
    corrected_overlap,
    gpu_nuclear_norm,
    layer_pairs,
    overlap_per_row,
    read_matrix,
)


# Eight layers per side keeps host memory bounded while letting one disk read supply up to 64 layer-pair calculations. A worker holds at most two blocks.
LAYER_BLOCK_SIZE = 8

# Matrices are copied to a GPU in row chunks. This avoids requiring two complete float64 activation matrices to fit in GPU memory at the same time.
GPU_ROW_BLOCK_SIZE = 8192


# -----------------------------------------------------------------------------
# Complete layer-by-layer map for one model/cohort
#
# Intention: reveal whether the best cross-survey match lies on the same depth diagonal or between different depths. Saved k-nearest-neighbor arrays make the overlap part cheap. Cached self-statistics, blockwise NPY loading, and GPU cross-products remove the old repeated covariance and Parquet work. Dense SVDs remain the dominant cost and can still take hours for a wide model file.
# -----------------------------------------------------------------------------

def _all_rows_good(matrix):
    """Return True when the fast shared-row calculation matches E1's row set."""
    return bool(
        np.isfinite(matrix).all()
        and np.all(np.linalg.norm(matrix, axis=1) > 0)
    )


def _cpu_layer_statistics(matrix):
    """Calculate reusable centering and normalization terms with NumPy."""
    x = np.asarray(matrix, dtype=np.float64)
    n = len(x)
    total = x.sum(axis=0)
    self_cross = x.T @ x - np.outer(total, total) / n
    centered_squared = np.sum(x * x) - np.dot(total, total) / n
    return {
        "all_good": True,
        "sum": total,
        "cka_norm": float(np.linalg.norm(self_cross)),
        "matrix_norm": float(np.sqrt(max(centered_squared, 0.0))),
    }


def _gpu_layer_statistics(matrix, gpu_id):
    """Calculate the same reusable terms in float64 on one GPU."""
    n, width = matrix.shape
    with cp.cuda.Device(gpu_id):
        total = cp.zeros(width, dtype=cp.float64)
        self_cross = cp.zeros((width, width), dtype=cp.float64)
        squared = cp.zeros((), dtype=cp.float64)

        # Accumulating row blocks avoids keeping the complete float64 matrix on the GPU. Only one block and the width-by-width covariance are resident.
        for start in range(0, n, GPU_ROW_BLOCK_SIZE):
            stop = min(start + GPU_ROW_BLOCK_SIZE, n)
            block = cp.asarray(matrix[start:stop], dtype=cp.float64)
            total += block.sum(axis=0)
            squared += cp.dot(block.ravel(), block.ravel())
            self_cross += block.T @ block

        self_cross -= cp.outer(total, total) / n
        centered_squared = squared - cp.dot(total, total) / n
        statistics = {
            "all_good": True,
            "sum": cp.asnumpy(total),
            "cka_norm": float(cp.linalg.norm(self_cross).get()),
            "matrix_norm": float(cp.sqrt(cp.maximum(centered_squared, 0)).get()),
        }
    return statistics


def layer_statistics(matrix, gpu_id):
    """Calculate terms that depend on one layer rather than on a layer pair."""
    if not _all_rows_good(matrix):
        # If released data ever contain a bad row, cross_metrics below falls back to E1's pair-specific filtering so behavior remains correct.
        return {"all_good": False}
    if cp is not None and gpu_id is not None:
        return _gpu_layer_statistics(matrix, gpu_id)
    return _cpu_layer_statistics(matrix)


def _cpu_cross_metrics(left, right, left_stats, right_stats):
    """Calculate one centered cross-product and its two metrics with NumPy."""
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    cross = left64.T @ right64
    cross -= np.outer(left_stats["sum"], right_stats["sum"]) / len(left64)
    cka_denominator = left_stats["cka_norm"] * right_stats["cka_norm"]
    proc_denominator = left_stats["matrix_norm"] * right_stats["matrix_norm"]
    cka = (
        float(np.sum(cross * cross) / cka_denominator)
        if cka_denominator > 0
        else np.nan
    )
    procrustes = (
        float(np.linalg.svd(cross, compute_uv=False).sum() / proc_denominator)
        if proc_denominator > 0
        else np.nan
    )
    return cka, procrustes


def _gpu_cross_metrics(left, right, left_stats, right_stats, gpu_id):
    """Calculate one centered cross-product and SVD in float64 on one GPU."""
    n = len(left)
    with cp.cuda.Device(gpu_id):
        cross = cp.zeros((left.shape[1], right.shape[1]), dtype=cp.float64)
        for start in range(0, n, GPU_ROW_BLOCK_SIZE):
            stop = min(start + GPU_ROW_BLOCK_SIZE, n)
            left_block = cp.asarray(left[start:stop], dtype=cp.float64)
            right_block = cp.asarray(right[start:stop], dtype=cp.float64)
            cross += left_block.T @ right_block

        left_sum = cp.asarray(left_stats["sum"])
        right_sum = cp.asarray(right_stats["sum"])
        cross -= cp.outer(left_sum, right_sum) / n

        cka_denominator = left_stats["cka_norm"] * right_stats["cka_norm"]
        proc_denominator = left_stats["matrix_norm"] * right_stats["matrix_norm"]
        cka = (
            float((cp.sum(cross * cross) / cka_denominator).get())
            if cka_denominator > 0
            else np.nan
        )
        procrustes = (
            float((gpu_nuclear_norm(cross) / proc_denominator).get())
            if proc_denominator > 0
            else np.nan
        )
    return cka, procrustes


def cross_metrics(left, right, left_stats, right_stats, gpu_id):
    """Return exact CKA/Procrustes, using cached self-terms when possible."""
    if not left_stats["all_good"] or not right_stats["all_good"]:
        good = (
            np.isfinite(left).all(axis=1)
            & np.isfinite(right).all(axis=1)
            & (np.linalg.norm(left, axis=1) > 0)
            & (np.linalg.norm(right, axis=1) > 0)
        )
        return cka_and_procrustes(left[good], right[good])
    if cp is not None and gpu_id is not None:
        return _gpu_cross_metrics(left, right, left_stats, right_stats, gpu_id)
    return _cpu_cross_metrics(left, right, left_stats, right_stats)


def _stage_matrix(item, column, cache_file):
    """Decode one Parquet column once, then reuse its simple NPY file."""
    if cache_file.exists():
        return np.load(cache_file, allow_pickle=False)
    matrix = read_matrix(item["path"], column, item.get("row_indices"))
    # Rename only after np.save finishes, so an interrupted first run does not leave a partial file that looks like a completed cache entry.
    temporary = cache_file.with_suffix(".tmp.npy")
    np.save(temporary, matrix)
    temporary.replace(cache_file)
    return matrix


def prepare_layer_cache(item, pairs, root, gpu_id):
    """Create NPY layer files and calculate every layer's reusable statistics."""
    cache_dir = (
        Path(root).parent
        / "e7_matrix_cache"
        / item["cohort"]
        / item["model"]
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    left_files = []
    right_files = []
    left_statistics = []
    right_statistics = []

    # This is one sequential Parquet pass on the first run. Later runs load the straightforward NPY files. The cache intentionally has no manifest logic.
    for layer, (left_column, right_column, _) in enumerate(pairs):
        left_file = cache_dir / f"left_{layer:03d}.npy"
        right_file = cache_dir / f"right_{layer:03d}.npy"
        left = _stage_matrix(item, left_column, left_file)
        right = _stage_matrix(item, right_column, right_file)
        left_files.append(left_file)
        right_files.append(right_file)
        left_statistics.append(layer_statistics(left, gpu_id))
        right_statistics.append(layer_statistics(right, gpu_id))
    return left_files, right_files, left_statistics, right_statistics


def _load_block(files, start):
    """Load at most eight complete float32 layers into ordinary host memory."""
    stop = min(start + LAYER_BLOCK_SIZE, len(files))
    return [np.load(files[layer], allow_pickle=False) for layer in range(start, stop)]


def _neighbor_overlap_map(neighbors, gpu_id):
    """Calculate the exact mean top-20 overlap for every layer pair."""
    layer_count = len(neighbors)
    values = np.empty((layer_count, layer_count), dtype=np.float64)
    row_counts = {
        len(array)
        for left_right in neighbors
        for array in left_right
    }
    if len(row_counts) != 1:
        raise ValueError(f"Layer-map neighbor row counts differ: {row_counts}")

    if cp is None or gpu_id is None:
        for left_layer in range(layer_count):
            for right_layer in range(layer_count):
                left = neighbors[left_layer][0]
                right = neighbors[right_layer][1]
                k = min(20, left.shape[1], right.shape[1])
                values[left_layer, right_layer] = overlap_per_row(left, right, k).mean()
        return values

    # Neighbor matrices are small compared with embedding matrices, so all layers fit on one A100. CuPy then performs the 20-by-20 ID comparisons without repeatedly moving the same neighbor lists across PCIe.
    with cp.cuda.Device(gpu_id):
        left_gpu = [cp.asarray(pair[0][:, :20]) for pair in neighbors]
        right_gpu = [cp.asarray(pair[1][:, :20]) for pair in neighbors]
        for left_layer, left in enumerate(left_gpu):
            for right_layer, right in enumerate(right_gpu):
                k = min(left.shape[1], right.shape[1])
                matches = left[:, :k, None] == right[:, None, :k]
                counts = matches.any(axis=2).sum(axis=1)
                values[left_layer, right_layer] = float(
                    (counts.mean(dtype=cp.float64) / k).get()
                )
        del left, right, matches, counts, left_gpu, right_gpu
        cp.get_default_memory_pool().free_all_blocks()
    return values


def make_layer_map(item, root, gpu_id=None):
    pairs = layer_pairs(item["path"], item["cohort"])
    neighbor_dir = root / "E1" / item["cohort"] / item["model"] / "neighbors"
    same_depth = pd.read_csv(neighbor_dir.parent / "curve.csv")

    # E1 already paid for exact neighbor search. Copy these small integer arrays into memory once and combine any left depth with any right depth below.
    neighbors = []
    for layer in range(len(pairs)):
        with np.load(neighbor_dir / f"layer_{layer:03d}.npz") as saved:
            neighbors.append(
                (saved["left_neighbors"].copy(), saved["right_neighbors"].copy())
            )

    # Calculate the complete, inexpensive neighbor-overlap map in one GPU phase before allocating the much larger float64 covariance/SVD workspaces.
    overlap_values = _neighbor_overlap_map(neighbors, gpu_id)

    left_files, right_files, left_stats, right_stats = prepare_layer_cache(
        item, pairs, root, gpu_id
    )
    backend = f"CuPy GPU {gpu_id}" if cp is not None and gpu_id is not None else "NumPy CPU"
    print(f"E7  {item['cohort']}/{item['model']}  algebra: {backend}", flush=True)

    # Preallocate positions so blockwise calculation still writes the familiar left-layer-major, right-layer-minor CSV order.
    rows = [None] * (len(pairs) * len(pairs))
    for left_start in range(0, len(pairs), LAYER_BLOCK_SIZE):
        left_block = _load_block(left_files, left_start)
        for right_start in range(0, len(pairs), LAYER_BLOCK_SIZE):
            right_block = _load_block(right_files, right_start)
            for left_offset, left_matrix in enumerate(left_block):
                left_layer = left_start + left_offset
                for right_offset, right_matrix in enumerate(right_block):
                    right_layer = right_start + right_offset
                    left_neighbors = neighbors[left_layer][0]
                    right_neighbors = neighbors[right_layer][1]
                    if len(left_neighbors) != len(right_neighbors):
                        raise ValueError(
                            f"Layer-map row counts differ for {item['model']}"
                        )
                    n = len(left_neighbors)
                    k = min(20, left_neighbors.shape[1], right_neighbors.shape[1])
                    raw = overlap_values[left_layer, right_layer]

                    if left_layer == right_layer:
                        # E1 already calculated same-depth cells, so reuse them.
                        diagonal = same_depth.iloc[left_layer]
                        cka = diagonal["cka"]
                        procrustes = diagonal["procrustes"]
                    else:
                        # Cached self-terms mean this pair now needs only one cross-product and one singular-value decomposition.
                        cka, procrustes = cross_metrics(
                            left_matrix,
                            right_matrix,
                            left_stats[left_layer],
                            right_stats[right_layer],
                            gpu_id,
                        )

                    position = left_layer * len(pairs) + right_layer
                    rows[position] = {
                        "cohort": item["cohort"],
                        "model": item["model"],
                        "left_layer": left_layer,
                        "right_layer": right_layer,
                        "corrected_overlap_20": corrected_overlap(raw, k, n),
                        "cka": cka,
                        "procrustes": procrustes,
                    }
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Final-array shape audit
#
# Intention: establish that E7's large COSMOS-Web final layer describes the same row count and feature widths as E8's extracted NPY arrays. This reads metadata and NPY headers only and normally takes much less than one second per model.
# -----------------------------------------------------------------------------

def shape_audit(item, final_pair):
    pairs = layer_pairs(item["path"], item["cohort"])
    parquet = pq.ParquetFile(item["path"])
    left_field = parquet.schema_arrow.field(pairs[-1][0])
    right_field = parquet.schema_arrow.field(pairs[-1][1])
    left_final = np.load(final_pair["hsc"], mmap_mode="r")
    right_final = np.load(final_pair["jwst"], mmap_mode="r")
    return {
        "model": item["model"],
        "layerwise_rows": parquet.metadata.num_rows,
        "final_rows": left_final.shape[0],
        "layerwise_left_width": left_field.type.list_size,
        "layerwise_right_width": right_field.type.list_size,
        "final_left_width": left_final.shape[1],
        "final_right_width": right_final.shape[1],
        "same_shape": (
            parquet.metadata.num_rows == left_final.shape[0]
            and left_final.shape[0] == right_final.shape[0]
            and left_field.type.list_size == left_final.shape[1]
            and right_field.type.list_size == right_final.shape[1]
        ),
    }


# -----------------------------------------------------------------------------
# Worker scheduling and E7 outputs
#
# Intention: give each worker complete model/cohort files, combine their maps, and extract the COSMOS-Web replication curve from E1. Scheduling and CSV work are minutes at most; make_layer_map accounts for effectively all wall time.
# -----------------------------------------------------------------------------

def _run_group(arguments):
    gpu_id, items, root, final_by_model, cpu_threads = arguments
    configure_worker(gpu_id, cpu_threads)
    maps = []
    audits = []
    # Items run sequentially inside a worker, keeping memory use understandable and preventing multiple child processes from sharing one GPU assignment.
    for item in items:
        print(f"E7  GPU {gpu_id}  {item['model']}", flush=True)
        maps.extend(make_layer_map(item, Path(root), gpu_id).to_dict("records"))
        if item["cohort"] == "cosmosweb" and item["model"] in final_by_model:
            audits.append(shape_audit(item, final_by_model[item["model"]]))
    return maps, audits


def run(context):
    root = Path(context["output_root"])
    # Spectroscopic cohorts have a fixed spectrum rather than a second layer stack, so the full square map is defined only for paired image cohorts.
    image_items = [
        item
        for item in context["layerwise_files"]
        if item["cohort"] in ("legacysurvey", "jwst", "cosmosweb")
    ]
    final_by_model = {item["model"]: item for item in context.get("final_pairs", [])}
    workers = min(context.get("workers", 1), len(image_items))
    groups = [image_items[number::workers] for number in range(workers)]
    arguments = [
        (
            number if number < context.get("gpu_count", 0) else None,
            group,
            root,
            final_by_model,
            context.get("cpu_threads_per_worker", 1),
        )
        for number, group in enumerate(groups)
    ]
    # Spawn gives every process independent CuPy/CUDA state on its assigned GPU.
    if workers == 1:
        returned = [_run_group(arguments[0])]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            returned = list(pool.map(_run_group, arguments))
    maps = [row for group_maps, _ in returned for row in group_maps]
    audits = [row for _, group_audits in returned for row in group_audits]
    layer_map = pd.DataFrame(maps)
    layer_map.to_csv(root / "E7_layer_pair_maps.csv", index=False)
    pd.DataFrame(audits).to_csv(root / "E7_extraction_shape_audit.csv", index=False)

    # The large-cohort diagonal replication is already present in E1. This last extraction is a small CSV operation that normally takes seconds.
    curves = pd.read_csv(root / "E1_all_curves.csv")
    curves[curves["cohort"] == "cosmosweb"].to_csv(
        root / "E7_large_cohort_curves.csv", index=False
    )
    return layer_map
