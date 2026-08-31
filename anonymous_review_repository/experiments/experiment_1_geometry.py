"""E1: measure layer-wise geometry between HSC and a second survey.

The file has five main blocks:
1. Read corresponding layer matrices from a Parquet file.
2. Find exact cosine nearest neighbors with FAISS.
3. Measure neighbor overlap, linear CKA, Procrustes similarity, and retrieval.
4. Use permutations and bootstraps to put uncertainty around the depth curve.
5. Distribute model/cohort files across worker processes and write CSV/NPZ files.

Approximate full-data runtime on the intended 8-A100 machine:
- Reading one layer is normally seconds to minutes, depending on storage.
- FAISS neighbor searches and retrieval use the assigned GPU and are normally
  seconds to minutes per layer.
- CKA, Procrustes, overlap reductions, retrieval ranking, permutations, and
  bootstraps use the assigned GPU in bounded row batches.
- The 9,999-permutation curve calculation should normally take minutes per file,
  although large layer counts and slow storage can push it higher.
- The complete E1 stage is expected to take hours; Parquet decoding and exact
  all-pairs searches can still dominate despite the GPU reductions.

These are planning estimates, not benchmarks. Row count, feature width, storage
speed, and whether GPU FAISS imports successfully all change the runtime.
"""

import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cupy as cp
except ImportError:
    cp = None

# -----------------------------------------------------------------------------
# Shared settings and input helpers
#
# Intention: make every worker interpret the archive in the same way and limit each process to its assigned CPU/GPU resources. These operations are tiny; worker setup and schema inspection normally take well under a second, while reading a matrix is dominated by Parquet I/O and decoding.
# -----------------------------------------------------------------------------

K_VALUES = (5, 10, 20, 50, 100)
GPU_ROW_BLOCK_SIZE = 8192
GPU_PERMUTATION_BATCH_SIZE = 8
GPU_BOOTSTRAP_BATCH_SIZE = 8

# These model outputs are vocabulary logits.  Their apparent widths are the vocabulary sizes, but each is produced by a much narrower linear language head.  We use the released model's exact head weights to replace the logits by a small factor with the same row-by-row inner products.
LM_HEAD_MODELS = {
    "paligemma_3b": "google/paligemma2-3b-mix-224",
    "paligemma_10b": "google/paligemma2-10b-mix-224",
    "paligemma_28b": "google/paligemma2-28b-mix-224",
    "llava_15_7b": "llava-hf/llava-1.5-7b-hf",
    "llava_15_13b": "llava-hf/llava-1.5-13b-hf",
    "llava_ov_7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}
HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
)
HEAD_WEIGHT_ROW_BLOCK = 4096
HEAD_GRAM_ERROR_LIMIT = 0.01
FAISS_TEMP_MEMORY = 256 * 1024**2


def assigned_gpu():
    """Return this worker's GPU number, or None for the CPU fallback."""
    gpu_text = os.environ.get("EXPERIMENT_GPU_ID", "")
    if cp is None or gpu_text == "":
        return None
    return int(gpu_text)


def configure_worker(gpu_id, cpu_threads):
    """Give one process one GPU and a fixed share of the CPU threads."""
    os.environ["EXPERIMENT_GPU_ID"] = str(gpu_id) if gpu_id is not None else ""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(cpu_threads)
    faiss.omp_set_num_threads(cpu_threads)


def _read_stored_matrix(parquet_file, column, rows=None):
    """Read one stored Parquet column without special head handling."""
    array = pq.read_table(parquet_file, columns=[column]).column(0).combine_chunks()
    if hasattr(array.type, "list_size"):
        values = np.asarray(array.values).reshape(len(array), array.type.list_size)
    else:
        values = np.stack(array.to_pylist())
    values = np.asarray(values, dtype=np.float32)
    return values if rows is None else values[rows]


def _head_model_name(parquet_file):
    """Infer the normalized VLM name from its straightforward filename."""
    name = Path(parquet_file).name
    for model in LM_HEAD_MODELS:
        if name.startswith(model + "_"):
            return model
        # The released PaliGemma filenames repeat the size: 3b_3b, etc.
        if model.startswith("paligemma_"):
            size = model.removeprefix("paligemma_")
            if name.startswith(f"paligemma_{size}_{size}_"):
                return model
    return None


def _head_factor_file(parquet_file, model):
    """Put small reusable factors next to the downloaded data directory."""
    data_root = Path(parquet_file).parents[2]
    return data_root / "lm_head_factors" / f"{model}.npy"


def _project_head(hidden, factor):
    """Apply the exact head Gram factor in bounded GPU row blocks."""
    gpu_id = assigned_gpu()
    if gpu_id is None:
        return np.asarray(hidden @ factor, dtype=np.float32)

    output = np.empty((len(hidden), factor.shape[1]), dtype=np.float32)
    with cp.cuda.Device(gpu_id):
        factor_gpu = cp.asarray(factor, dtype=cp.float32)
        for start in range(0, len(hidden), GPU_ROW_BLOCK_SIZE):
            stop = min(start + GPU_ROW_BLOCK_SIZE, len(hidden))
            block = cp.asarray(hidden[start:stop], dtype=cp.float32)
            output[start:stop] = cp.asnumpy(block @ factor_gpu)
        del factor_gpu, block
        cp.get_default_memory_pool().free_all_blocks()
    return output


def read_matrix(parquet_file, column, rows=None):
    """Read a matrix, replacing enormous linear-head logits by an exact factor."""
    if not column.startswith("lm_head_"):
        return _read_stored_matrix(parquet_file, column, rows)

    model = _head_model_name(parquet_file)
    if model is None:
        return _read_stored_matrix(parquet_file, column, rows)

    suffix = column.removeprefix("lm_head_")
    hidden_column = f"hidden_states_last_{suffix}"
    names = pq.ParquetFile(parquet_file).schema_arrow.names
    if hidden_column not in names:
        raise RuntimeError(f"{column} has no matching {hidden_column}")

    factor_file = _head_factor_file(parquet_file, model)
    if not factor_file.exists():
        raise RuntimeError(f"Missing lm_head factor: {factor_file}")
    hidden = _read_stored_matrix(parquet_file, hidden_column, rows)
    factor = np.load(factor_file, mmap_mode="r", allow_pickle=False)
    if hidden.shape[1] != factor.shape[0]:
        raise RuntimeError(
            f"Head factor width {factor.shape[0]} does not match "
            f"{hidden_column} width {hidden.shape[1]}"
        )
    return _project_head(hidden, factor)


# -----------------------------------------------------------------------------
# Exact reduction for vocabulary-sized language-head outputs
#
# The stored head output is Z = H W.T, where H is hidden_states_last and W is the released bias-free lm_head weight.  If G = W.T W and G = L L.T, then (H L)(H L).T equals Z Z.T.  Cosine neighbors, retrieval, linear CKA, and normalized Procrustes therefore see the same geometry without a 257k-wide matrix.  Building the factors downloads one model shard apiece and normally takes a few minutes on an A100; later runs simply load the cached NPY files.
# -----------------------------------------------------------------------------

def _safetensor_info(path, tensor_name):
    """Return the shape, dtype, and byte location of one safetensor."""
    with open(path, "rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size))
    tensor = header[tensor_name]
    start, stop = tensor["data_offsets"]
    return tensor["shape"], tensor["dtype"], 8 + header_size + start, stop - start


def _weight_rows(path, tensor_name, start, stop):
    """Read a row block, including the bfloat16 used by PaliGemma."""
    shape, dtype, offset, byte_count = _safetensor_info(path, tensor_name)
    dtype_map = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8"}
    if dtype not in dtype_map:
        raise RuntimeError(f"Unsupported head-weight dtype: {dtype}")
    stored = np.memmap(
        path,
        mode="r",
        dtype=np.dtype(dtype_map[dtype]),
        offset=offset,
        shape=tuple(shape),
    )
    expected_bytes = int(np.prod(shape)) * stored.dtype.itemsize
    if expected_bytes != byte_count:
        raise RuntimeError(f"Unexpected tensor size for {tensor_name}")
    block = np.asarray(stored[start:stop])
    if dtype == "BF16":
        # NumPy has no native bfloat16.  Moving its 16 bits into the high half of float32 reconstructs exactly the value stored by safetensors.
        block = (block.astype(np.uint32) << 16).view(np.float32)
    return np.asarray(block, dtype=np.float32)


def _make_head_factor(weight_file, tensor_name, output_file):
    """Accumulate W.T W and save its small square-root factor."""
    shape, _, _, _ = _safetensor_info(weight_file, tensor_name)
    vocabulary, hidden_width = shape
    if vocabulary <= hidden_width:
        raise RuntimeError(f"Unexpected lm_head weight shape: {shape}")

    gpu_count = 0
    if cp is not None:
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
        except Exception:
            gpu_count = 0

    if gpu_count:
        with cp.cuda.Device(0):
            gram = cp.zeros((hidden_width, hidden_width), dtype=cp.float64)
            for start in range(0, vocabulary, HEAD_WEIGHT_ROW_BLOCK):
                stop = min(start + HEAD_WEIGHT_ROW_BLOCK, vocabulary)
                block = cp.asarray(
                    _weight_rows(weight_file, tensor_name, start, stop),
                    dtype=cp.float64,
                )
                gram += block.T @ block
            values, vectors = cp.linalg.eigh(gram)
            values = cp.maximum(values, 0)
            factor = cp.asnumpy(vectors * cp.sqrt(values)[None, :]).astype(np.float32)
            del gram, values, vectors, block
            cp.get_default_memory_pool().free_all_blocks()
    else:
        gram = np.zeros((hidden_width, hidden_width), dtype=np.float64)
        for start in range(0, vocabulary, HEAD_WEIGHT_ROW_BLOCK):
            stop = min(start + HEAD_WEIGHT_ROW_BLOCK, vocabulary)
            block = np.asarray(
                _weight_rows(weight_file, tensor_name, start, stop),
                dtype=np.float64,
            )
            gram += block.T @ block
        values, vectors = np.linalg.eigh(gram)
        factor = (vectors * np.sqrt(np.maximum(values, 0))[None, :]).astype(np.float32)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".tmp.npy")
    np.save(temporary, factor)
    temporary.replace(output_file)


def _first_parquet_rows(parquet_file, columns, count=16):
    """Read only a tiny validation batch rather than a complete huge column."""
    batch = next(
        pq.ParquetFile(parquet_file).iter_batches(
            batch_size=count, columns=columns, use_threads=False
        )
    )
    matrices = []
    for column in columns:
        array = batch.column(batch.schema.get_field_index(column))
        matrices.append(
            np.asarray(array.values, dtype=np.float32).reshape(
                len(array), array.type.list_size
            )
        )
    return matrices


def _validate_head_factor(parquet_file, factor_file):
    """Require the reduced and stored head Gram matrices to agree within 1%."""
    hidden, logits = _first_parquet_rows(
        parquet_file, ["hidden_states_last_hsc", "lm_head_hsc"]
    )
    factor = np.load(factor_file, allow_pickle=False)

    gpu_count = 0
    if cp is not None:
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
        except Exception:
            gpu_count = 0
    if gpu_count:
        with cp.cuda.Device(0):
            hidden_gpu = cp.asarray(hidden)
            factor_gpu = cp.asarray(factor)
            logits_gpu = cp.asarray(logits)
            reduced = hidden_gpu @ factor_gpu
            actual_gram = logits_gpu @ logits_gpu.T
            reduced_gram = reduced @ reduced.T
            error = float(
                (cp.linalg.norm(actual_gram - reduced_gram)
                 / cp.maximum(cp.linalg.norm(actual_gram), 1e-12)).get()
            )
    else:
        reduced = hidden @ factor
        actual_gram = logits @ logits.T
        reduced_gram = reduced @ reduced.T
        error = float(
            np.linalg.norm(actual_gram - reduced_gram)
            / max(np.linalg.norm(actual_gram), 1e-12)
        )
    if error > HEAD_GRAM_ERROR_LIMIT:
        raise RuntimeError(
            f"lm_head reduction validation failed for {parquet_file}: "
            f"relative Gram error {error:.6f} exceeds {HEAD_GRAM_ERROR_LIMIT}"
        )
    return error


def prepare_lm_head_factors(items, data_root, token):
    """Download one weight shard per affected model and prepare its factor."""
    from huggingface_hub import hf_hub_download

    needed = sorted({item["model"] for item in items if item["model"] in LM_HEAD_MODELS})
    if not needed:
        return

    factor_dir = Path(data_root) / "lm_head_factors"
    weight_root = Path(data_root) / "lm_head_weights"
    for model in needed:
        model_items = [item for item in items if item["model"] == model]
        sample = min(
            model_items,
            key=lambda item: pq.ParquetFile(item["path"]).metadata.num_rows,
        )
        factor_file = factor_dir / f"{model}.npy"
        if not factor_file.exists():
            repo = LM_HEAD_MODELS[model]
            local_dir = weight_root / model
            index_file = hf_hub_download(
                repo,
                "model.safetensors.index.json",
                token=token,
                local_dir=local_dir,
            )
            with open(index_file) as handle:
                weight_map = json.load(handle)["weight_map"]
            tensor_name = next(
                (name for name in HEAD_WEIGHT_NAMES if name in weight_map), None
            )
            if tensor_name is None:
                raise RuntimeError(f"No lm_head weight was found in {repo}")
            weight_file = hf_hub_download(
                repo,
                weight_map[tensor_name],
                token=token,
                local_dir=local_dir,
            )
            print(f"Preparing exact lm_head factor for {model}", flush=True)
            _make_head_factor(weight_file, tensor_name, factor_file)

        error = _validate_head_factor(sample["path"], factor_file)
        print(
            f"Validated lm_head factor for {model}: relative Gram error {error:.6f}",
            flush=True,
        )


def _layer_order(name):
    """Put ConvNeXt container hooks after the children which produce them."""
    stage = re.fullmatch(r"encoder\.stages\.(\d+)", name)
    downsample = re.fullmatch(
        r"encoder\.stages\.(\d+)\.downsampling_layer\.(\d+)", name
    )
    block = re.fullmatch(r"encoder\.stages\.(\d+)\.layers\.(\d+)", name)
    if name == "embeddings":
        return (0, 0, 0)
    if downsample:
        return (1 + int(downsample.group(1)), 0, int(downsample.group(2)))
    if block:
        return (1 + int(block.group(1)), 1, int(block.group(2)))
    if stage:
        return (1 + int(stage.group(1)), 2, 0)
    if name in ("layernorm", "pooler", "last_hidden_state"):
        return (1000, ("layernorm", "pooler", "last_hidden_state").index(name), 0)
    return None


def layer_pairs(parquet_file, cohort):
    """Return the columns which should be compared, in file order."""
    names = pq.ParquetFile(parquet_file).schema_arrow.names
    left = [name for name in names if name.endswith("_hsc")]

    if cohort in ("desi", "sdss"):
        fixed = f"{cohort}_embedding"
        pairs = [(name, fixed, name.removesuffix("_hsc")) for name in left]
    else:
        right_suffix = "_jwst" if cohort == "cosmosweb" else f"_{cohort}"
        right = {name for name in names if name.endswith(right_suffix)}
        pairs = []
        for left_name in left:
            base = left_name.removesuffix("_hsc")
            right_name = base + right_suffix
            if right_name in right:
                pairs.append((left_name, right_name, base))
    # Most files are already in forward order. ConvNeXt is the exception.
    if any(base.startswith("encoder.stages.") for _, _, base in pairs):
        original = {base: number for number, (_, _, base) in enumerate(pairs)}
        pairs.sort(
            key=lambda pair: _layer_order(pair[2])
            if _layer_order(pair[2]) is not None
            else (500, original[pair[2]], 0)
        )
    return pairs


# -----------------------------------------------------------------------------
# Exact nearest-neighbor overlap
#
# Intention: ask whether an object's local neighborhood is preserved between the two surveys. Rows are normalized, so inner product equals cosine similarity. FAISS performs the expensive all-neighbor search on the assigned GPU when GPU FAISS is available. A single large layer is usually seconds to minutes; the small NumPy overlap reductions are normally seconds.
# -----------------------------------------------------------------------------

def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(lengths, 1e-12)


def gpu_faiss_resources(gpu_id):
    """Create FAISS resources without its unnecessary 1.5 GiB reservation."""
    # A previous CuPy calculation can leave unused blocks in its memory pool. Return those blocks before FAISS asks CUDA for its separate allocation.
    if cp is not None:
        with cp.cuda.Device(gpu_id):
            cp.get_default_memory_pool().free_all_blocks()
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(FAISS_TEMP_MEMORY)
    return resources


def exact_neighbors(x, k=100):
    """Exact cosine neighbors. The row itself is removed by row number."""
    x = normalize(x)
    k = min(k, len(x) - 1)
    if k < 1:
        raise ValueError("At least two rows are needed for neighbor search")

    # IndexFlatIP is an exact search: it does not approximate or compress the vectors. Moving this index to a GPU changes speed, not the definition.
    index = faiss.IndexFlatIP(x.shape[1])
    resources = None
    gpu_text = os.environ.get("EXPERIMENT_GPU_ID", "")
    if gpu_text and hasattr(faiss, "StandardGpuResources"):
        gpu_id = int(gpu_text)
        resources = gpu_faiss_resources(gpu_id)
        index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
    index.add(x)
    _, found = index.search(x, min(len(x), k + 2))

    # FAISS also returns the query row itself. Remove it by row number rather than assuming it always occupies the first returned position.
    answer = np.empty((len(x), k), dtype=np.int32)
    for row in range(len(x)):
        keep = found[row][found[row] != row]
        answer[row] = keep[:k]
    return answer


def overlap_per_row(left_neighbors, right_neighbors, k, block_size=4096):
    """Return the fraction of shared top-k neighbors for every object."""
    left_neighbors = left_neighbors[:, :k]
    right_neighbors = right_neighbors[:, :k]

    # On a GPU, only the final one-number-per-row answer comes back to the CPU. The temporary k-by-k comparisons stay on the assigned device.
    gpu_id = assigned_gpu()
    if gpu_id is not None:
        with cp.cuda.Device(gpu_id):
            left_gpu = cp.asarray(left_neighbors)
            right_gpu = cp.asarray(right_neighbors)
            values = np.empty(len(left_neighbors), dtype=np.float64)
            for start in range(0, len(left_neighbors), block_size):
                stop = min(start + block_size, len(left_neighbors))
                matches = (
                    left_gpu[start:stop, :, None]
                    == right_gpu[start:stop, None, :]
                )
                counts = matches.any(axis=2).sum(axis=1)
                values[start:stop] = cp.asnumpy(counts) / k
            return values

    # Simple CPU fallback for machines without CuPy or an assigned GPU.
    values = np.empty(len(left_neighbors), dtype=np.float64)
    for start in range(0, len(left_neighbors), block_size):
        stop = min(start + block_size, len(left_neighbors))
        # This compares the k left IDs with the k right IDs for each row. The block keeps the temporary (rows x k x k) Boolean array modest.
        matches = (
            left_neighbors[start:stop, :, None]
            == right_neighbors[start:stop, None, :]
        )
        values[start:stop] = matches.any(axis=2).sum(axis=1) / k
    return values


def corrected_overlap(raw_overlap, k, n):
    """Remove the overlap expected from two random sets of k neighbors."""
    chance = k / (n - 1)
    if chance >= 1:
        return np.nan
    return (raw_overlap - chance) / (1 - chance)


def _center(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


# -----------------------------------------------------------------------------
# Whole-representation similarity
#
# Intention: complement the local-neighbor statistic with two global linear comparisons. CKA compares centered cross-covariance energy. Procrustes sums the singular values of the same cross-covariance and normalizes by the two matrix norms. These functions deliberately use float64 and accumulate row blocks on the assigned GPU. They usually take seconds to minutes for a wide layer. A direct NumPy implementation remains as the CPU fallback.
# -----------------------------------------------------------------------------

def cka_and_procrustes(x, y):
    """Centered linear CKA and normalized Procrustes similarity."""
    gpu_id = assigned_gpu()
    if gpu_id is not None:
        return _gpu_cka_and_procrustes(x, y, gpu_id)

    x = _center(x)
    y = _center(y)
    # cross contains every feature-to-feature covariance up to the common factor 1 / (n - 1), which cancels from both normalized statistics.
    cross = x.T @ y
    cross_squared = np.sum(cross * cross)
    xx = x.T @ x
    yy = y.T @ y
    cka_denominator = np.sqrt(np.sum(xx * xx) * np.sum(yy * yy))
    proc_denominator = np.linalg.norm(x) * np.linalg.norm(y)
    cka = float(cross_squared / cka_denominator) if cka_denominator > 0 else np.nan
    # The Procrustes numerator is the nuclear norm: the sum of all singular values of the centered cross-product.
    procrustes = (
        float(np.linalg.svd(cross, compute_uv=False).sum() / proc_denominator)
        if proc_denominator > 0
        else np.nan
    )
    return cka, procrustes


def gpu_nuclear_norm(matrix):
    """Sum singular values, with an equivalent GPU fallback for large shapes."""
    try:
        return cp.linalg.svd(matrix, compute_uv=False).sum()
    except cp.cuda.cusolver.CUSOLVERError:
        # Some cuSOLVER versions reject otherwise valid large rectangular matrices in gesvd_bufferSize.  The eigenvalues of A A.T (or A.T A) are the squared singular values, so this calculates the same number while sending only the smaller square matrix to the solver.
        print(
            f"cuSOLVER SVD rejected matrix {matrix.shape}; "
            "using the equivalent eigenvalue calculation",
            flush=True,
        )
        if matrix.shape[0] <= matrix.shape[1]:
            gram = matrix @ matrix.T
        else:
            gram = matrix.T @ matrix
        eigenvalues = cp.linalg.eigvalsh(gram)
        eigenvalues = cp.maximum(eigenvalues, 0)
        return cp.sqrt(eigenvalues).sum()


def _gpu_cka_and_procrustes(x, y, gpu_id):
    """Calculate the same two float64 statistics in GPU row blocks."""
    n = len(x)
    left_width = x.shape[1]
    right_width = y.shape[1]

    # Accumulate uncentered products and subtract the feature means at the end. This is exactly the centered formula, without storing two full float64 centered matrices on the GPU.
    with cp.cuda.Device(gpu_id):
        left_sum = cp.zeros(left_width, dtype=cp.float64)
        right_sum = cp.zeros(right_width, dtype=cp.float64)
        cross = cp.zeros((left_width, right_width), dtype=cp.float64)
        left_product = cp.zeros((left_width, left_width), dtype=cp.float64)
        right_product = cp.zeros((right_width, right_width), dtype=cp.float64)
        left_squared = cp.float64(0)
        right_squared = cp.float64(0)

        for start in range(0, n, GPU_ROW_BLOCK_SIZE):
            stop = min(start + GPU_ROW_BLOCK_SIZE, n)
            left_block = cp.asarray(x[start:stop], dtype=cp.float64)
            right_block = cp.asarray(y[start:stop], dtype=cp.float64)
            left_sum += left_block.sum(axis=0)
            right_sum += right_block.sum(axis=0)
            cross += left_block.T @ right_block
            left_product += left_block.T @ left_block
            right_product += right_block.T @ right_block
            left_squared += cp.sum(left_block * left_block)
            right_squared += cp.sum(right_block * right_block)

        cross -= cp.outer(left_sum, right_sum) / n
        left_product -= cp.outer(left_sum, left_sum) / n
        right_product -= cp.outer(right_sum, right_sum) / n
        left_squared -= cp.dot(left_sum, left_sum) / n
        right_squared -= cp.dot(right_sum, right_sum) / n

        cross_squared = cp.sum(cross * cross)
        cka_denominator = cp.sqrt(
            cp.sum(left_product * left_product)
            * cp.sum(right_product * right_product)
        )
        proc_denominator = cp.sqrt(
            cp.maximum(left_squared, 0) * cp.maximum(right_squared, 0)
        )

        cka_denominator_value = float(cka_denominator.item())
        proc_denominator_value = float(proc_denominator.item())
        cka = (
            float((cross_squared / cka_denominator).item())
            if cka_denominator_value > 0
            else np.nan
        )
        procrustes = (
            float(
                (
                    gpu_nuclear_norm(cross) / proc_denominator
                ).item()
            )
            if proc_denominator_value > 0
            else np.nan
        )
    return cka, procrustes


def linear_cka(x, y):
    """Centered linear CKA without cosine row normalization."""
    x = _center(x)
    y = _center(y)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = np.sum(xy * xy)
    denominator = np.sqrt(np.sum(xx * xx) * np.sum(yy * yy))
    return float(numerator / denominator) if denominator > 0 else np.nan


def procrustes_similarity(x, y):
    """Normalized Procrustes similarity without cosine row normalization."""
    x = _center(x)
    y = _center(y)
    cross = x.T @ y
    numerator = np.linalg.svd(cross, compute_uv=False).sum()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(numerator / denominator) if denominator > 0 else np.nan


# -----------------------------------------------------------------------------
# Paired-object retrieval
#
# Intention: test whether HSC row i retrieves the corresponding survey row i. Similarities are calculated in blocks so an n-by-n score matrix is never kept all at once. FAISS performs each block on the GPU when available. Runtime is usually seconds to minutes per direction for a large layer, and it scales approximately with n squared times the feature width.
# -----------------------------------------------------------------------------

def _gpu_inner_products(resources, queries, vectors, gpu_id):
    """Calculate one similarity block on a GPU and return it to CPU memory."""
    scores = np.empty((len(queries), len(vectors)), dtype=np.float32)
    arguments = faiss.GpuDistanceParams()
    arguments.metric = faiss.METRIC_INNER_PRODUCT
    arguments.k = -1
    arguments.dims = queries.shape[1]
    arguments.vectors = faiss.swig_ptr(vectors)
    arguments.vectorType = faiss.DistanceDataType_F32
    arguments.vectorsRowMajor = True
    arguments.numVectors = len(vectors)
    arguments.queries = faiss.swig_ptr(queries)
    arguments.queryType = faiss.DistanceDataType_F32
    arguments.queriesRowMajor = True
    arguments.numQueries = len(queries)
    arguments.outDistances = faiss.swig_ptr(scores)
    arguments.outIndices = None
    arguments.device = gpu_id
    faiss.bfKnn(resources, arguments)
    return scores


def retrieval_ranks(x, y, block_size=1024, return_ties=False):
    """Rank the paired row in the other view, using exact midranks for ties."""
    x = np.ascontiguousarray(normalize(x))
    y = np.ascontiguousarray(normalize(y))
    ranks = np.empty(len(x), dtype=np.float64)
    tie_flags = np.empty(len(x), dtype=bool)
    gpu_id = assigned_gpu()

    # The GPU computes ranks and ties as reductions. It does not send the block_size-by-n similarity matrix back through PCIe.
    if gpu_id is not None:
        with cp.cuda.Device(gpu_id):
            y_gpu = cp.asarray(y)
            for start in range(0, len(x), block_size):
                stop = min(start + block_size, len(x))
                scores = cp.asarray(x[start:stop]) @ y_gpu.T
                local_rows = cp.arange(stop - start)
                target_rows = cp.arange(start, stop)
                targets = scores[local_rows, target_rows]
                higher = cp.count_nonzero(scores > targets[:, None], axis=1)
                tied = cp.count_nonzero(scores == targets[:, None], axis=1)
                ranks[start:stop] = cp.asnumpy(higher + 1 + (tied - 1) / 2)
                tie_flags[start:stop] = cp.asnumpy(tied > 1)
        return (ranks, tie_flags) if return_ties else ranks

    gpu_text = os.environ.get("EXPERIMENT_GPU_ID", "")
    resources = None
    if gpu_text and hasattr(faiss, "GpuDistanceParams"):
        resources = gpu_faiss_resources(int(gpu_text))
    # Each block produces block_size x n scores. Only ranks and tie flags are retained after the block, keeping peak host memory predictable.
    for start in range(0, len(x), block_size):
        stop = min(start + block_size, len(x))
        if resources is None:
            scores = x[start:stop] @ y.T
        else:
            scores = _gpu_inner_products(
                resources, x[start:stop], y, int(gpu_text)
            )
        for local_row, row in enumerate(range(start, stop)):
            target = scores[local_row, row]
            higher = np.count_nonzero(scores[local_row] > target)
            tied = np.count_nonzero(scores[local_row] == target)
            ranks[row] = higher + 1 + (tied - 1) / 2
            tie_flags[row] = tied > 1
    return (ranks, tie_flags) if return_ties else ranks


def retrieval_summary(x, y):
    """Combine both retrieval directions into MRR, median rank, and recall."""
    left_ranks, left_ties = retrieval_ranks(x, y, return_ties=True)
    right_ranks, right_ties = retrieval_ranks(y, x, return_ties=True)
    ranks = np.concatenate([left_ranks, right_ranks])
    ties = np.concatenate([left_ties, right_ties])
    result = {
        "mrr": float(np.mean(1 / ranks)),
        "median_rank": float(np.median(ranks)),
        "tie_fraction": float(np.mean(ties)),
    }
    for k in (1, 5, 10, 50):
        result[f"recall_{k}"] = float(np.mean(ranks <= k))
    return result, left_ranks, right_ranks


# -----------------------------------------------------------------------------
# One complete layer-pair calculation
#
# Intention: collect every geometry statistic from one aligned pair of arrays and save reusable neighbors/ranks. A large layer can take minutes or longer; exact neighbors, CPU CKA/Procrustes, and optional retrieval dominate.
# -----------------------------------------------------------------------------

def geometry(x, y, save_neighbors=None):
    """Calculate all geometry numbers for one layer pair."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Row counts differ: {x.shape[0]} and {y.shape[0]}")
    # Use the same valid rows in both views. The zero fractions are reported so a degenerate representation is visible in the output rather than hidden.
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    left_zero = np.zeros(len(x), dtype=bool)
    right_zero = np.zeros(len(y), dtype=bool)
    left_zero[finite] = np.linalg.norm(x[finite], axis=1) == 0
    right_zero[finite] = np.linalg.norm(y[finite], axis=1) == 0
    finite_count = max(1, np.count_nonzero(finite))
    left_zero_fraction = np.count_nonzero(left_zero) / finite_count
    right_zero_fraction = np.count_nonzero(right_zero) / finite_count
    good = finite & ~left_zero & ~right_zero
    x = x[good]
    y = y[good]
    if len(x) < 3:
        raise ValueError("Too few finite rows")

    # First calculate neighbors once up to the largest requested k. Smaller-k overlaps are prefixes of these same exact neighbor lists.
    max_k = min(100, len(x) - 1)
    left_neighbors = exact_neighbors(x, max_k)
    right_neighbors = exact_neighbors(y, max_k)
    result = {
        "n": len(x),
        "left_width": x.shape[1],
        "right_width": y.shape[1],
        "left_zero_fraction": left_zero_fraction,
        "right_zero_fraction": right_zero_fraction,
    }
    per_row_20 = None
    for requested_k in K_VALUES:
        if requested_k > max_k:
            result[f"overlap_{requested_k}"] = np.nan
            result[f"corrected_overlap_{requested_k}"] = np.nan
            continue
        per_row = overlap_per_row(left_neighbors, right_neighbors, requested_k)
        raw = float(per_row.mean())
        result[f"overlap_{requested_k}"] = raw
        result[f"corrected_overlap_{requested_k}"] = corrected_overlap(
            raw, requested_k, len(x)
        )
        if requested_k == 20:
            per_row_20 = per_row

    if per_row_20 is None:
        k = min(20, max_k)
        per_row_20 = overlap_per_row(left_neighbors, right_neighbors, k)

    # CKA and Procrustes are defined even when the feature widths differ.
    result["cka"], result["procrustes"] = cka_and_procrustes(x, y)
    # Paired retrieval requires a shared feature width because it compares an HSC vector directly with every vector in the other view.
    if x.shape[1] == y.shape[1]:
        retrieval, left_ranks, right_ranks = retrieval_summary(x, y)
        result.update(retrieval)
        result["tie_dominated"] = result["tie_fraction"] >= 0.5
    else:
        left_ranks = np.array([], dtype=np.float64)
        right_ranks = np.array([], dtype=np.float64)
        result.update(
            mrr=np.nan,
            median_rank=np.nan,
            tie_fraction=np.nan,
            recall_1=np.nan,
            recall_5=np.nan,
            recall_10=np.nan,
            recall_50=np.nan,
            tie_dominated=False,
        )
    result["available"] = bool(
        not result["tie_dominated"]
        and result["left_zero_fraction"] < 0.5
        and result["right_zero_fraction"] < 0.5
    )

    # E5 and E7 reuse these neighbor arrays. Saving them prevents those later experiments from repeating the expensive exact neighbor search.
    if save_neighbors is not None:
        save_neighbors = Path(save_neighbors)
        save_neighbors.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_neighbors,
            left_neighbors=left_neighbors,
            right_neighbors=right_neighbors,
            left_ranks=left_ranks,
            right_ranks=right_ranks,
            finite_rows=np.flatnonzero(good),
        )
    return result, per_row_20, left_neighbors, right_neighbors


# -----------------------------------------------------------------------------
# Curve-level statistical inference
#
# Intention: test the strongest alignment anywhere on a depth curve while controlling that search over layers, then form simultaneous confidence bands. Every permutation changes cross-view object correspondence. Every bootstrap resamples objects and measures all layers on the same resample. At the full 9,999 permutations and 2,000 bootstraps this block often takes minutes per model/cohort on an A100. Small batches keep GPU memory use predictable.
# -----------------------------------------------------------------------------

def _permuted_overlap(left_neighbors, right_neighbors, permutation, k, rows=None):
    """Neighbor overlap after changing the cross-view correspondence."""
    inverse = np.empty(len(permutation), dtype=np.int32)
    inverse[permutation] = np.arange(len(permutation))
    if rows is None:
        rows = np.arange(len(permutation))
    moved_right = inverse[right_neighbors[permutation[rows], :k]]
    return overlap_per_row(left_neighbors[rows, :k], moved_right, k).mean()


def _gpu_curve_draws(
    neighbor_curves,
    row_curves,
    permutations,
    bootstraps,
    rng,
    permutation_rows,
    k,
    n,
    gpu_id,
    keep_null_curves=False,
):
    """Run E1 permutation and bootstrap draws in small GPU batches."""
    with cp.cuda.Device(gpu_id):
        gpu_neighbors = [
            (cp.asarray(left[:, :k]), cp.asarray(right[:, :k]))
            for left, right in neighbor_curves
        ]
        if permutation_rows is None:
            permutation_rows = np.arange(n, dtype=np.int32)
        else:
            permutation_rows = np.asarray(permutation_rows, dtype=np.int32)
        rows_gpu = cp.asarray(permutation_rows)
        left_rows = [left[rows_gpu] for left, _ in gpu_neighbors]
        chance = k / (n - 1)

        # Each batch contains independent complete permutations. Row chunks keep the temporary batch-by-rows-by-k-by-k Boolean array small.
        if keep_null_curves:
            null_values = np.empty(
                (permutations, len(gpu_neighbors)), dtype=np.float64
            )
        else:
            null_values = np.empty(permutations, dtype=np.float64)
        made = 0
        while made < permutations:
            batch = min(GPU_PERMUTATION_BATCH_SIZE, permutations - made)
            permutation_array = np.empty((batch, n), dtype=np.int32)
            inverse_array = np.empty((batch, n), dtype=np.int32)
            row_numbers = np.arange(n, dtype=np.int32)
            for draw in range(batch):
                permutation_array[draw] = rng.permutation(n)
                inverse_array[draw, permutation_array[draw]] = row_numbers

            permutation_gpu = cp.asarray(permutation_array)
            inverse_gpu = cp.asarray(inverse_array)
            permuted_rows = permutation_gpu[:, rows_gpu]
            batch_rows = cp.arange(batch)[:, None, None]
            curve_values = cp.empty(
                (batch, len(gpu_neighbors)), dtype=cp.float64
            )

            for layer, ((_, right), left_for_rows) in enumerate(
                zip(gpu_neighbors, left_rows)
            ):
                shared_total = cp.zeros(batch, dtype=cp.float64)
                for start in range(0, len(permutation_rows), GPU_ROW_BLOCK_SIZE):
                    stop = min(start + GPU_ROW_BLOCK_SIZE, len(permutation_rows))
                    moved_ids = right[permuted_rows[:, start:stop]]
                    moved_right = inverse_gpu[batch_rows, moved_ids]
                    matches = (
                        left_for_rows[None, start:stop, :, None]
                        == moved_right[:, :, None, :]
                    )
                    shared_total += matches.any(axis=3).sum(axis=(1, 2))
                raw = shared_total / (len(permutation_rows) * k)
                curve_values[:, layer] = (raw - chance) / (1 - chance)

            if keep_null_curves:
                null_values[made : made + batch] = cp.asnumpy(curve_values)
            else:
                null_values[made : made + batch] = cp.asnumpy(
                    cp.nanmax(curve_values, axis=1)
                )
            made += batch

        # Bootstrap all layers on the same sampled rows. This preserves the between-layer dependence needed by the simultaneous confidence band.
        del gpu_neighbors, left_rows, permutation_gpu, inverse_gpu, curve_values
        row_values = cp.asarray(np.stack(row_curves), dtype=cp.float64)
        draws = np.empty((bootstraps, len(row_curves)), dtype=np.float64)
        made = 0
        while made < bootstraps:
            batch = min(GPU_BOOTSTRAP_BATCH_SIZE, bootstraps - made)
            sampled_rows = cp.asarray(rng.integers(0, n, size=(batch, n)))
            raw = row_values[:, sampled_rows].mean(axis=2).T
            corrected = (raw - chance) / (1 - chance)
            draws[made : made + batch] = cp.asnumpy(corrected)
            made += batch
    return null_values, draws


def curve_inference(
    neighbor_curves,
    row_curves,
    permutations,
    bootstraps,
    seed,
    permutation_sample_size=None,
):
    """Shared-permutation p-value and simultaneous bootstrap interval."""
    observed = np.asarray([rows.mean() for rows in row_curves])
    n = len(row_curves[0])
    k = min(20, neighbor_curves[0][0].shape[1])
    observed_corrected = np.array([corrected_overlap(v, k, n) for v in observed])
    rng = np.random.default_rng(seed)
    permutation_rows = None
    if permutation_sample_size and n > permutation_sample_size:
        permutation_rows = np.sort(
            rng.choice(n, permutation_sample_size, replace=False)
        )

    gpu_id = assigned_gpu()
    if gpu_id is not None:
        null_max, draws = _gpu_curve_draws(
            neighbor_curves,
            row_curves,
            permutations,
            bootstraps,
            rng,
            permutation_rows,
            k,
            n,
            gpu_id,
        )
    else:
        # Store the maximum null alignment across layers. Comparing the observed maximum with this distribution accounts for choosing a peak.
        null_max = []
        for _ in range(permutations):
            permutation = rng.permutation(n)
            curve = []
            for left_neighbors, right_neighbors in neighbor_curves:
                raw = _permuted_overlap(
                    left_neighbors,
                    right_neighbors,
                    permutation,
                    k,
                    permutation_rows,
                )
                curve.append(corrected_overlap(raw, k, n))
            null_max.append(np.nanmax(curve))
        null_max = np.asarray(null_max)

        # Shared bootstrap row draws retain correlations between adjacent layers.
        draws = np.empty((bootstraps, len(row_curves)), dtype=np.float64)
        for draw in range(bootstraps):
            rows = rng.integers(0, n, n)
            for layer, values in enumerate(row_curves):
                raw = values[rows].mean()
                draws[draw, layer] = corrected_overlap(raw, k, n)

    p_value = (1 + np.count_nonzero(null_max >= np.nanmax(observed_corrected))) / (
        len(null_max) + 1
    )

    standard_error = draws.std(axis=0, ddof=1)
    safe_error = np.maximum(standard_error, 1e-12)
    maximum_t = np.max(np.abs((draws - observed_corrected) / safe_error), axis=1)
    cutoff = np.quantile(maximum_t, 0.95)
    low = observed_corrected - cutoff * standard_error
    high = observed_corrected + cutoff * standard_error
    peak_probability = np.bincount(
        np.nanargmax(draws, axis=1), minlength=len(row_curves)
    ) / bootstraps
    return p_value, low, high, peak_probability


def single_layer_inference(
    neighbor_curves,
    row_curves,
    permutations,
    bootstraps,
    seed,
    permutation_sample_size=None,
):
    """Calibrate several independent controls with one shared permutation bank."""
    n = len(row_curves[0])
    if any(len(values) != n for values in row_curves):
        raise ValueError("Grouped controls must have the same row count")
    k = min(20, neighbor_curves[0][0].shape[1])
    observed = np.asarray(
        [corrected_overlap(values.mean(), k, n) for values in row_curves]
    )

    # Sampling rows must not advance the permutation generator. This way the actual correspondence permutations are identical to E8's raw bank.
    permutation_rows = None
    if permutation_sample_size and n > permutation_sample_size:
        sample_rng = np.random.default_rng(seed + 1_000_003)
        permutation_rows = np.sort(
            sample_rng.choice(n, permutation_sample_size, replace=False)
        )
    rng = np.random.default_rng(seed)

    gpu_id = assigned_gpu()
    if gpu_id is not None:
        null_values, draws = _gpu_curve_draws(
            neighbor_curves,
            row_curves,
            permutations,
            bootstraps,
            rng,
            permutation_rows,
            k,
            n,
            gpu_id,
            keep_null_curves=True,
        )
    else:
        null_values = np.empty(
            (permutations, len(neighbor_curves)), dtype=np.float64
        )
        for draw in range(permutations):
            permutation = rng.permutation(n)
            for control, (left_neighbors, right_neighbors) in enumerate(
                neighbor_curves
            ):
                raw = _permuted_overlap(
                    left_neighbors,
                    right_neighbors,
                    permutation,
                    k,
                    permutation_rows,
                )
                null_values[draw, control] = corrected_overlap(raw, k, n)

        draws = np.empty((bootstraps, len(row_curves)), dtype=np.float64)
        for draw in range(bootstraps):
            sampled_rows = rng.integers(0, n, n)
            for control, values in enumerate(row_curves):
                draws[draw, control] = corrected_overlap(
                    values[sampled_rows].mean(), k, n
                )

    p_values = np.asarray(
        [
            (1 + np.count_nonzero(null_values[:, column] >= observed[column]))
            / (permutations + 1)
            if np.isfinite(observed[column])
            else np.nan
            for column in range(len(observed))
        ]
    )
    standard_error = draws.std(axis=0, ddof=1)
    safe_error = np.maximum(standard_error, 1e-12)
    studentized = np.abs((draws - observed) / safe_error)
    cutoffs = np.quantile(studentized, 0.95, axis=0)
    low = observed - cutoffs * standard_error
    high = observed + cutoffs * standard_error
    return p_values, low, high


def holm_adjust(p_values):
    """Holm-adjust a short list of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    order = finite[np.argsort(p_values[finite])]
    running = 0.0
    for rank, position in enumerate(order):
        value = (len(order) - rank) * p_values[position]
        running = max(running, value)
        adjusted[position] = min(1.0, running)
    return adjusted


# -----------------------------------------------------------------------------
# File-level and experiment-level orchestration
#
# Intention: calculate one depth curve per model/cohort, distribute independent files across workers, and combine their CSV files. One process receives one GPU and eight CPU threads in the real configuration. Scheduling and CSV work take seconds; the layer calculations and curve permutations dominate.
# -----------------------------------------------------------------------------

def run_file(
    parquet_file,
    cohort,
    model,
    output_dir,
    permutations=999,
    bootstraps=2000,
    seed=42,
    row_indices=None,
):
    """Run E1 for one model/cohort Parquet file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = layer_pairs(parquet_file, cohort)
    if not pairs:
        raise ValueError(f"No layer pairs found in {parquet_file}")

    # Spectroscopic cohorts compare every HSC layer with one fixed spectrum, so keep that one matrix. Image cohorts have a different right matrix at every layer; those are read one at a time and released instead of all being held.
    fixed_right = None
    if cohort in ("desi", "sdss"):
        fixed_right = read_matrix(parquet_file, pairs[0][1], row_indices)
    rows = []
    row_curves = []
    neighbor_curves = []
    for layer_number, (left_name, right_name, layer_name) in enumerate(pairs):
        left = read_matrix(parquet_file, left_name, row_indices)
        if fixed_right is None:
            right = read_matrix(parquet_file, right_name, row_indices)
        else:
            right = fixed_right
        neighbor_file = output_dir / "neighbors" / f"layer_{layer_number:03d}.npz"
        numbers, per_row, left_neighbors, right_neighbors = geometry(
            left, right, neighbor_file
        )
        numbers.update(
            cohort=cohort,
            model=model,
            layer=layer_number,
            normalized_depth=layer_number / max(1, len(pairs) - 1),
            left_column=left_name,
            right_column=right_name,
            layer_name=layer_name,
            representation_method=(
                "exact_lm_head_gram_factor"
                if left_name.startswith("lm_head_")
                or right_name.startswith("lm_head_")
                else "stored_features"
            ),
        )
        rows.append(numbers)
        row_curves.append(per_row)
        neighbor_curves.append((left_neighbors, right_neighbors))

    # All released layers normally have the same valid rows. The inference function requires this because one shared permutation is used at all depths. Legacy Survey uses an 8,192-row permutation sample for runtime.
    if len({len(values) for values in row_curves}) == 1:
        p_value, low, high, peak_probability = curve_inference(
            neighbor_curves,
            row_curves,
            permutations,
            bootstraps,
            seed,
            permutation_sample_size=8192 if cohort == "legacysurvey" else None,
        )
        for row, lo, hi, peak in zip(rows, low, high, peak_probability):
            row["curve_p_value"] = p_value
            row["simultaneous_low"] = lo
            row["simultaneous_high"] = hi
            row["peak_probability"] = peak
            row["permutation_rows"] = min(
                len(row_curves[0]), 8192 if cohort == "legacysurvey" else len(row_curves[0])
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "curve.csv", index=False)
    return frame


def _run_group(arguments):
    gpu_id, items, output_root, permutations, bootstraps, seed, overwrite, cpu_threads = arguments
    configure_worker(gpu_id, cpu_threads)
    files = []
    # A worker handles its assigned files sequentially, which avoids two processes competing for the same GPU.
    for item in items:
        out = Path(output_root) / "E1" / item["cohort"] / item["model"]
        csv_file = out / "curve.csv"
        if not csv_file.exists() or overwrite:
            print(f"E1  GPU {gpu_id}  {item['cohort']}/{item['model']}", flush=True)
            run_file(
                item["path"], item["cohort"], item["model"], out,
                permutations, bootstraps, seed, item.get("row_indices"),
            )
        files.append(str(csv_file))
    return files


def run(context):
    """Run E1 for every layerwise file listed by the master script."""
    items = context["layerwise_files"]
    workers = min(context.get("workers", 1), len(items))
    groups = [items[number::workers] for number in range(workers)]
    arguments = []
    for number, group in enumerate(groups):
        gpu_id = number if number < context.get("gpu_count", 0) else None
        arguments.append(
            (
                gpu_id,
                group,
                context["output_root"],
                context["permutations"],
                context["bootstraps"],
                context["seed"],
                context.get("overwrite", False),
                context.get("cpu_threads_per_worker", 1),
            )
        )
    # Spawn is used so FAISS/CUDA state is created independently in each child.
    if workers == 1:
        output_files = _run_group(arguments[0])
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            output_files = []
            for files in pool.map(_run_group, arguments):
                output_files.extend(files)

    all_rows = [pd.read_csv(path) for path in output_files]
    result = pd.concat(all_rows, ignore_index=True)

    # There is one primary p-value per model/cohort curve. Holm correction is applied once per curve rather than once per layer.
    tests = result.groupby(["cohort", "model"], as_index=False)["curve_p_value"].first()
    tests["curve_p_value_holm"] = holm_adjust(tests["curve_p_value"])
    result = result.merge(
        tests[["cohort", "model", "curve_p_value_holm"]],
        on=["cohort", "model"],
        how="left",
    )
    result.to_csv(Path(context["output_root"]) / "E1_all_curves.csv", index=False)
    return result
