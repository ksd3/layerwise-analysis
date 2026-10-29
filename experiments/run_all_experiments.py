#!/usr/bin/env python3
"""Download the released arrays and run experiments E1 through E9.

This is the single entry point for the project. It deliberately keeps the
workflow explicit:
1. Create a local uv environment and install the short dependency list.
2. Download released embeddings and labels with one Hugging Face worker.
3. Discover files and construct a shared context dictionary.
4. Run E1 through E9 in dependency order.
5. Make two basic figures and optionally upload the results directory.

Approximate full-run timing on the intended 8-A100/128-CPU machine:
- Environment creation/package installation: usually 2-10 minutes.
- The roughly 3.5 TB download with one transfer worker: about 10 hours at
  100 MB/s, 4 hours at 250 MB/s, or 2 hours at 500 MB/s, plus service overhead.
- Catalog/label preparation: minutes, with the streaming JWST filter potentially
  taking longer because it reads images sequentially.
- E1-E6 and E8: hours to possibly days in aggregate.
- Optimized E7: likely hours to a day or more; it now caches layer statistics,
  stages NPY matrices, and runs cross-products/SVDs on assigned GPUs.
- E9 remains geometry-heavy. Correspondence tests are restored for every control,
  but compatible controls share permutation generation and GPU batches.
- Upload: minutes to hours depending on result size and network speed.

The smoke test uses 60 synthetic rows, 19 permutations, and 30 bootstraps. It
should normally finish in roughly one to several minutes on a CPU-only computer.
All timings are planning estimates, not benchmarks.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
HF_TOKEN = os.environ.get("HF_TOKEN", "")
STORAGE_ROOT = Path(os.environ.get("EXPERIMENT_STORAGE_ROOT", HERE.parent / "experiment_workspace"))
DATA_ROOT = STORAGE_ROOT / "data"
OUTPUT_ROOT = STORAGE_ROOT / "results"
RESULTS_REPO = os.environ.get("HF_RESULTS_REPO", "")
WORKERS = 8
CPU_THREADS_PER_WORKER = 8
HF_TRANSFER_WORKERS = 1
SEED = 42
PERMUTATIONS = 9999
BOOTSTRAPS = 2000

# Set these before NumPy, FAISS, or scikit-learn are imported so those libraries use the intended thread count.
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = str(CPU_THREADS_PER_WORKER)

LAYER_REPO = os.environ.get("HF_LAYER_REPO", "")
FINAL_REPO = os.environ.get("HF_FINAL_REPO", "")
OLD_LABEL_REPO = os.environ.get("HF_LABEL_REPO", "")
FINAL_CATALOG_REPO = os.environ.get("HF_CATALOG_REPO", "")

PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "pyarrow",
    "matplotlib",
    "huggingface-hub",
    "datasets",
    "tqdm",
    "faiss-gpu",
    "cupy-cuda12x",
    "nvidia-cusolver-cu12",
]


# Model sizes are approximate millions of parameters used as a regression covariate in E6/E8. This is a plain lookup table, not an online metadata query.
PARAMETERS_M = {
    "vit_base": 86,
    "vit_large": 307,
    "vit_huge": 632,
    "clip_base": 86,
    "clip_large": 307,
    "dino_small": 22,
    "dino_base": 86,
    "dino_large": 307,
    "dino_giant": 1100,
    "dinov3_vits16": 22,
    "dinov3_vits16plus": 26,
    "dinov3_vitb16": 86,
    "dinov3_vitl16": 307,
    "dinov3_vith16plus": 650,
    "dinov3_vit7b16": 7000,
    "convnext_nano": 16,
    "convnext_tiny": 29,
    "convnext_base": 89,
    "convnext_large": 198,
    "ijepa_huge": 632,
    "ijepa_giant": 1011,
    "vjepa_large": 307,
    "vjepa_huge": 632,
    "vjepa_giant": 1011,
    "astropt_015M": 15,
    "astropt_095M": 95,
    "astropt_850M": 850,
    "vit-mae_base": 86,
    "vit-mae_large": 307,
    "vit-mae_huge": 632,
    "paligemma_3b": 3000,
    "paligemma_10b": 10000,
    "paligemma_28b": 28000,
    "llava_15_7b": 7000,
    "llava_15_13b": 13000,
    "llava_ov_7b": 7000,
    "toy_model": 10,
}


# -----------------------------------------------------------------------------
# Local uv environment
#
# Intention: make the script runnable directly from a base shell. On the first invocation this creates .venv, installs packages, and restarts this same file under .venv/bin/python. Installation normally takes 2-10 minutes. Later runs still ask uv to ensure packages are installed, usually taking seconds.
# -----------------------------------------------------------------------------

def _add_nvidia_library_paths():
    """Expose the CUDA libraries installed by the nvidia Python wheels."""
    library_root = HERE / ".venv" / "lib"
    nvidia_dirs = sorted(library_root.glob("python*/site-packages/nvidia/*/lib"))
    current_dirs = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    missing_dirs = [str(item.resolve()) for item in nvidia_dirs if str(item.resolve()) not in current_dirs]
    if missing_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing_dirs + current_dirs)
    return bool(missing_dirs)


def enter_local_environment():
    """Create .venv with uv, install packages, and restart inside it."""
    python = HERE / ".venv" / "bin" / "python"
    if python.exists() and Path(sys.executable).resolve() == python.resolve():
        # LD_LIBRARY_PATH is read when CUDA libraries are loaded. Restart once if the wheel-provided cuBLAS/cuSOLVER directories were just added.
        if _add_nvidia_library_paths():
            os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        return
    if not python.exists():
        subprocess.check_call(["uv", "venv", str(HERE / ".venv")])
    subprocess.check_call(["uv", "pip", "install", "--python", str(python), *PACKAGES])
    _add_nvidia_library_paths()
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


# -----------------------------------------------------------------------------
# Model-name normalization and explicit metadata
#
# Intention: give every experiment the same family, objective, parameter count, and training-data labels. These string/table lookups take far below one second.
# -----------------------------------------------------------------------------

def normalize_model_name(model):
    for size in ("3b", "10b", "28b"):
        model = model.replace(f"paligemma_{size}_{size}", f"paligemma_{size}")
    return model


def model_family(model):
    if model.startswith("vit-mae"):
        return "vit-mae"
    if model.startswith("llava"):
        return "llava"
    if model.startswith("paligemma"):
        return "paligemma"
    return model.split("_")[0]


def model_metadata(model):
    family = model_family(model)
    if family == "astropt":
        objective = "astronomy_self_supervised"
    elif family in ("dino", "dinov3"):
        objective = "self_distillation"
    elif family in ("ijepa", "vjepa"):
        objective = "joint_embedding"
    elif family == "vit-mae":
        objective = "masked_reconstruction"
    elif family in ("clip", "paligemma", "llava"):
        objective = "vision_language"
    else:
        objective = "supervised_images"
    return {
        "family": family,
        "checkpoint_lineage": family,
        "parameters_m": PARAMETERS_M.get(model, float("nan")),
        "objective": objective,
        "training_data": "astronomy" if family == "astropt" else "general",
        "astronomy_specific": family == "astropt",
    }


# -----------------------------------------------------------------------------
# Hugging Face downloads
#
# Intention: place the roughly 3.5 TB of inputs under the configured storage root and use Hugging Face's normal single-worker resume behavior without extra manifests or hash checks.
# -----------------------------------------------------------------------------

def download_data():
    """Straight downloads only. Hugging Face resumes partial files itself."""
    from huggingface_hub import hf_hub_download, snapshot_download

    token = HF_TOKEN
    print("Downloading layerwise embeddings. This is the very large download.")
    # This is by far the largest network transfer. allow_patterns prevents unrelated repository files from being downloaded.
    snapshot_download(
        LAYER_REPO,
        repo_type="dataset",
        token=token,
        local_dir=DATA_ROOT / "platonic-embeddings",
        allow_patterns=["*.parquet"],
        max_workers=HF_TRANSFER_WORKERS,
    )
    print("Downloading final HSC/JWST arrays.")
    # Final arrays are much smaller than the complete layerwise archives but are still large enough that their transfer can take minutes to hours.
    snapshot_download(
        FINAL_REPO,
        repo_type="dataset",
        token=token,
        local_dir=DATA_ROOT / "pu-regress-embeddings",
        allow_patterns=["*.npy"],
        max_workers=HF_TRANSFER_WORKERS,
    )
    # Label tables are small compared with embeddings and normally take seconds.
    for cohort in ("desi", "jwst", "legacysurvey", "sdss"):
        hf_hub_download(
            OLD_LABEL_REPO,
            f"static/labels_{cohort}.parquet",
            repo_type="dataset",
            token=token,
            local_dir=DATA_ROOT / "pu-umap-results",
        )


# -----------------------------------------------------------------------------
# Locate downloaded embedding files
#
# Intention: turn simple filename patterns into lists of model/cohort inputs. This scans directory entries and NPY filenames only, normally taking seconds.
# -----------------------------------------------------------------------------

def find_layerwise_files(data_root):
    items = []
    base = Path(data_root) / "platonic-embeddings"
    for cohort in ("desi", "sdss", "legacysurvey", "jwst", "cosmosweb"):
        for path in sorted((base / cohort).glob("*_blocks_layerwise.parquet")):
            raw_model = path.name.removesuffix("_blocks_layerwise.parquet")
            items.append(
                {"cohort": cohort, "model": normalize_model_name(raw_model), "path": path}
            )
    return items


def find_final_pairs(data_root):
    items = []
    base = Path(data_root) / "pu-regress-embeddings"
    for hsc in sorted(base.glob("emb_hsc_*_45000.npy")):
        raw_model = hsc.name.removeprefix("emb_hsc_").removesuffix("_45000.npy")
        jwst = base / f"emb_jwst_{raw_model}_45000.npy"
        if jwst.exists():
            items.append(
                {
                    "model": normalize_model_name(raw_model),
                    "hsc": hsc,
                    "jwst": jwst,
                }
            )
    return items


# -----------------------------------------------------------------------------
# Reconstruct old-cohort labels and a common object order
#
# Intention: match each layerwise archive to clean, unique HSC/counterpart object IDs and scientific targets. Parquet labels load in seconds to minutes. If the historical small-JWST row filter must be reproduced, streaming source images can take minutes to hours depending on network speed.
# -----------------------------------------------------------------------------

def _small_jwst_filter(expected_rows, label_rows):
    """Repeat the image filter used when the small JWST embeddings were made."""
    from datasets import load_dataset

    mask = []
    dataset = load_dataset(
        "Smith42/jwst_hsc_crossmatched",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    ).select_columns(["jwst_image"])
    # Reapply the original non-constant-image criterion in original row order. Stop as soon as the known label-table length has been examined.
    for row in dataset:
        image = row["jwst_image"]["flux"][3]
        low, high = np.nanpercentile(image, [5, 99])
        mask.append(low - high != 0)
        if len(mask) == label_rows:
            break
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == expected_rows:
        return mask
    raise RuntimeError(
        f"The small-JWST filter kept {mask.sum()} rows instead of {expected_rows}"
    )


def load_old_targets(data_root, layerwise_files):
    labels_root = Path(data_root) / "pu-umap-results" / "static"
    frames = {
        cohort: pd.read_parquet(labels_root / f"labels_{cohort}.parquet")
        for cohort in ("desi", "jwst", "legacysurvey", "sdss")
    }
    row_counts = {}
    for item in layerwise_files:
        if item["cohort"] != "cosmosweb" and item["cohort"] not in row_counts:
            row_counts[item["cohort"]] = pq.ParquetFile(item["path"]).metadata.num_rows

    jwst = frames["jwst"]
    n_jwst = row_counts.get("jwst", len(jwst))
    if len(jwst) != n_jwst:
        mask = _small_jwst_filter(n_jwst, len(jwst))
        jwst = jwst.loc[mask].reset_index(drop=True)

    frames["jwst"] = jwst
    for cohort in frames:
        frames[cohort] = frames[cohort].iloc[: row_counts.get(cohort, len(frames[cohort]))].reset_index(drop=True)

    # Remove every occurrence of a repeated object rather than choosing an arbitrary duplicate, then use one stable ID order for every model.
    counterpart = {
        "desi": "desi_object_id",
        "jwst": "jwst_object_id",
        "legacysurvey": "legacysurvey_object_id",
        "sdss": "sdss_object_id",
    }
    cohort_rows = {}
    for cohort, frame in frames.items():
        other = counterpart[cohort]
        good = (
            frame["hsc_object_id"].notna()
            & frame[other].notna()
            & ~frame["hsc_object_id"].duplicated(keep=False)
            & ~frame[other].duplicated(keep=False)
        )
        ordered = frame.loc[good, ["hsc_object_id", other]].sort_values(
            ["hsc_object_id", other]
        )
        cohort_rows[cohort] = ordered.index.to_numpy(dtype=np.int64)

    # Derive the few target quantities needed by E3-E5. Invalid logarithms and unreliable SDSS redshifts are represented as NaN and ignored downstream.
    legacy = frames["legacysurvey"]
    with np.errstate(divide="ignore", invalid="ignore"):
        legacy_color = -2.5 * np.log10(legacy["FLUX_G"].to_numpy() / legacy["FLUX_R"].to_numpy())
        legacy_shape = np.log10(legacy["SHAPE_R"].where(legacy["SHAPE_R"] > 0).to_numpy())
        jwst_radius = np.log10(jwst["flux_radius"].where(jwst["flux_radius"] > 0).to_numpy())

    sdss = frames["sdss"]
    redshift = sdss["Z"].to_numpy(dtype=float).copy()
    redshift[sdss["ZWARNING"].to_numpy() != 0] = np.nan
    targets = {
        "legacysurvey": {"g_minus_r": legacy_color, "log_shape": legacy_shape},
        "jwst": {"log_flux_radius": jwst_radius, "mag_auto": jwst["mag_auto"].to_numpy()},
        "sdss": {"redshift": redshift},
        "desi": {"redshift": frames["desi"]["Z"].to_numpy()},
    }
    for cohort, values in targets.items():
        rows = cohort_rows[cohort]
        targets[cohort] = {name: np.asarray(value)[rows] for name, value in values.items()}
    return targets, cohort_rows


# -----------------------------------------------------------------------------
# Final 45,000-row catalog
#
# Intention: stream and locally cache exactly the catalog columns aligned with the final NPY arrays. First use may take minutes to hours depending on remote dataset speed; subsequent reads of final_catalog.parquet normally take seconds.
# -----------------------------------------------------------------------------

def load_final_catalog(data_root):
    """Read the first 45,000 catalog rows, which match the final arrays."""
    local_file = Path(data_root) / "final_catalog.parquet"
    if not local_file.exists():
        from datasets import load_dataset

        columns = [
            "ra",
            "specz",
            "Confidence_level",
            "lp_mass",
            "lp_ssfr",
            "mag_model_hsc-g",
            "mag_model_hsc-r",
        ]
        dataset = load_dataset(
            FINAL_CATALOG_REPO,
            split="train",
            streaming=True,
            token=HF_TOKEN,
        ).select_columns(columns)
        values = {column: [] for column in columns}
        # The released final arrays contain the first 45,000 rows in this order. No join or reshuffling is performed here.
        for number, row in enumerate(dataset):
            for column in columns:
                values[column].append(row[column])
            if number + 1 == 45000:
                break
        local_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(values).to_parquet(local_file, index=False)

    frame = pd.read_parquet(local_file)
    # Apply the catalog's basic redshift validity definition once so all later experiments receive exactly the same target array.
    redshift = (
        pd.to_numeric(frame["specz"], errors="coerce").to_numpy(dtype=float).copy()
    )
    confidence = pd.to_numeric(
        frame.get("Confidence_level", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    redshift[(redshift <= 0) | (redshift >= 20) | (confidence < 0)] = np.nan
    magnitude = frame["mag_model_hsc-r"].to_numpy(dtype=float)
    color = frame["mag_model_hsc-g"].to_numpy(dtype=float) - magnitude
    return {
        "ra": frame["ra"].to_numpy(dtype=float),
        "targets": {
            "redshift": redshift,
            "mass": frame["lp_mass"].to_numpy(dtype=float),
            "sSFR": frame["lp_ssfr"].to_numpy(dtype=float),
            "g_minus_r": color,
            "magnitude": magnitude,
        },
    }


# -----------------------------------------------------------------------------
# Hardware discovery and synthetic smoke-test data
#
# Intention: count visible GPUs and construct tiny files with the same schemas, names, dependencies, and orchestration paths as the real data. Data generation takes well under a second; running all experiments on it usually takes one to several minutes.
# -----------------------------------------------------------------------------

def visible_gpus():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([line for line in output.splitlines() if line.startswith("GPU ")])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def _arrow_column(matrix):
    """Turn a dense smoke-test matrix into one fixed-size-list Arrow column."""
    import pyarrow as pa

    matrix = np.asarray(matrix, dtype=np.float32)
    return pa.array(matrix.tolist(), type=pa.list_(pa.float32(), matrix.shape[1]))


def make_smoke_context():
    """Make small arrays with the same column naming pattern as the real data."""
    import pyarrow as pa

    smoke_data = HERE / "smoke_data"
    smoke_output = HERE / "smoke_results"
    smoke_data.mkdir(exist_ok=True)
    smoke_output.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    n, width, spectrum_width, layers = 60, 12, 9, 4
    latent = rng.normal(size=(n, width)).astype(np.float32)
    spectrum_map = rng.normal(size=(width, spectrum_width)).astype(np.float32)
    spectrum = latent @ spectrum_map + rng.normal(0, 0.2, (n, spectrum_width))

    # Make five cohort archives, each with four progressively less noisy layers. The common latent matrix creates a known cross-view alignment signal.
    layerwise_files = []
    for cohort in ("desi", "sdss", "legacysurvey", "jwst", "cosmosweb"):
        directory = smoke_data / "platonic-embeddings" / cohort
        directory.mkdir(parents=True, exist_ok=True)
        columns = {}
        for layer in range(layers):
            noise = 0.9 / (layer + 1)
            left = latent + rng.normal(0, noise, latent.shape)
            columns[f"layer.{layer}_hsc"] = _arrow_column(left)
            if cohort not in ("desi", "sdss"):
                suffix = "jwst" if cohort == "cosmosweb" else cohort
                right = latent + rng.normal(0, noise, latent.shape)
                columns[f"layer.{layer}_{suffix}"] = _arrow_column(right)
        if cohort in ("desi", "sdss"):
            columns[f"{cohort}_embedding"] = _arrow_column(spectrum)
        path = directory / "toy_model_blocks_layerwise.parquet"
        pq.write_table(pa.table(columns), path)
        layerwise_files.append({"cohort": cohort, "model": "toy_model", "path": path})

    # E8/E9 use separate final NPY pairs rather than layerwise Parquet columns.
    final_dir = smoke_data / "pu-regress-embeddings"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_left = latent + rng.normal(0, 0.15, latent.shape)
    final_right = latent + rng.normal(0, 0.15, latent.shape)
    left_file = final_dir / "emb_hsc_toy_model_45000.npy"
    right_file = final_dir / "emb_jwst_toy_model_45000.npy"
    np.save(left_file, final_left.astype(np.float32))
    np.save(right_file, final_right.astype(np.float32))

    # Synthetic scientific targets are simple latent coordinates, ensuring that transfer code encounters a real signal rather than pure random noise.
    signal = latent[:, 0] + 0.5 * latent[:, 1]
    old_targets = {
        "legacysurvey": {"g_minus_r": signal, "log_shape": latent[:, 2]},
        "jwst": {"log_flux_radius": signal, "mag_auto": latent[:, 3]},
        "sdss": {"redshift": signal},
        "desi": {"redshift": signal},
    }
    final_catalog = {
        "ra": np.linspace(149.0, 151.0, n),
        "targets": {
            "redshift": signal,
            "mass": latent[:, 2],
            "sSFR": latent[:, 3],
            "g_minus_r": latent[:, 4],
            "magnitude": latent[:, 5],
        },
    }
    return {
        "layerwise_files": layerwise_files,
        "final_pairs": [{"model": "toy_model", "hsc": left_file, "jwst": right_file}],
        "old_targets": old_targets,
        "final_catalog": final_catalog,
        "output_root": smoke_output,
        "data_root": smoke_data,
        "hf_token": None,
        "seed": SEED,
        "permutations": 19,
        "bootstraps": 30,
        "overwrite": True,
        "workers": 2,
        "gpu_count": min(1, visible_gpus()),
        "cpu_threads_per_worker": 2,
        "model_family": model_family,
        "model_metadata": model_metadata,
    }


# -----------------------------------------------------------------------------
# Assemble the real context shared by all nine experiments
#
# Intention: validate that expected files exist, construct label row selections, and place all common settings/callables into one plain dictionary. This usually takes minutes; remote catalog/filter preparation can make first use longer.
# -----------------------------------------------------------------------------

def make_real_context(overwrite=False):
    layerwise_files = find_layerwise_files(DATA_ROOT)
    final_pairs = find_final_pairs(DATA_ROOT)
    if not layerwise_files:
        raise FileNotFoundError("No layerwise Parquet files were found")
    if not final_pairs:
        raise FileNotFoundError("No paired final NPY arrays were found")
    print(f"Found {len(layerwise_files)} layerwise files and {len(final_pairs)} final pairs.")
    old_targets, cohort_rows = load_old_targets(DATA_ROOT, layerwise_files)
    for item in layerwise_files:
        item["row_indices"] = cohort_rows.get(item["cohort"])
    return {
        "layerwise_files": layerwise_files,
        "final_pairs": final_pairs,
        "old_targets": old_targets,
        "final_catalog": load_final_catalog(DATA_ROOT),
        "output_root": OUTPUT_ROOT,
        "data_root": DATA_ROOT,
        "hf_token": HF_TOKEN,
        "seed": SEED,
        "permutations": PERMUTATIONS,
        "bootstraps": BOOTSTRAPS,
        "overwrite": overwrite,
        "workers": min(WORKERS, max(1, visible_gpus())),
        "gpu_count": visible_gpus(),
        "cpu_threads_per_worker": CPU_THREADS_PER_WORKER,
        "model_family": model_family,
        "model_metadata": model_metadata,
    }


# -----------------------------------------------------------------------------
# Lightweight figures and experiment ordering
#
# Intention: keep orchestration explicit and sequential at the experiment level. Each experiment performs its own model-level multiprocessing. Plotting takes seconds. run_all itself adds negligible overhead beyond E1-E9 runtimes.
# -----------------------------------------------------------------------------

def make_figures(context):
    import matplotlib.pyplot as plt

    root = Path(context["output_root"])
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    curves = pd.read_csv(root / "E1_all_curves.csv")
    plt.figure(figsize=(7, 4))
    for (cohort, model), curve in curves.groupby(["cohort", "model"]):
        plt.plot(curve["normalized_depth"], curve["corrected_overlap_20"], alpha=0.5)
    plt.xlabel("Normalized depth")
    plt.ylabel("Chance-corrected neighbor overlap")
    plt.tight_layout()
    plt.savefig(figure_dir / "all_depth_curves.png", dpi=160)
    plt.close()

    transfer = pd.read_csv(root / "E8_transfer_by_model.csv")
    if not transfer.empty:
        plt.figure(figsize=(5, 4))
        plt.scatter(
            transfer["alignment"], transfer["cross_view_redshift_r2"], alpha=0.7
        )
        plt.xlabel("Final representation alignment")
        plt.ylabel("Cross-view test R-squared")
        plt.tight_layout()
        plt.savefig(figure_dir / "alignment_vs_transfer.png", dpi=160)
        plt.close()


def run_all(context, only=None):
    from experiment_1_geometry import prepare_lm_head_factors, run as run_e1
    from experiment_2_depth import run as run_e2
    from experiment_3_image_transfer import run as run_e3
    from experiment_4_spectrum_transfer import run as run_e4
    from experiment_5_updates import run as run_e5
    from experiment_6_model_properties import run as run_e6
    from experiment_7_large_cohort import run as run_e7
    from experiment_8_final_transfer import run as run_e8
    from experiment_9_controls import run as run_e9

    experiments = {
        "E1": run_e1,
        "E2": run_e2,
        "E3": run_e3,
        "E4": run_e4,
        "E5": run_e5,
        "E6": run_e6,
        "E7": run_e7,
        "E8": run_e8,
        "E9": run_e9,
    }
    # Prepare each small exact head factor before any GPU worker pools start. Smoke data contain no language heads, so this returns immediately there.
    selected = set(only) if only else set(experiments)
    if selected.intersection({"E1", "E3", "E4", "E5", "E7"}):
        prepare_lm_head_factors(
            context["layerwise_files"],
            context["data_root"],
            context.get("hf_token"),
        )
    # Preserve numerical order because later stages read earlier CSV/NPZ files.
    for name, function in experiments.items():
        if only and name not in only:
            continue
        LOGGER.info("Starting %s", name)
        function(context)
        LOGGER.info("Finished %s", name)
    if not only or set(only) == set(experiments):
        make_figures(context)


def expand_experiment_dependencies(only):
    """Add the earlier experiment outputs needed by a requested stage."""
    valid = {f"E{number}" for number in range(1, 10)}
    unknown = only - valid
    if unknown:
        raise ValueError(f"Unknown experiment name(s): {', '.join(sorted(unknown))}")
    # Only direct file dependencies are listed. The small loop below adds their dependencies repeatedly until the requested set stops changing.
    dependencies = {
        "E2": {"E1"},
        "E5": {"E1"},
        "E6": {"E1", "E2", "E5"},
        "E7": {"E1"},
        "E9": {"E8"},
    }
    expanded = set(only)
    while True:
        before = set(expanded)
        for name in list(expanded):
            expanded.update(dependencies.get(name, set()))
        if expanded == before:
            return expanded


# -----------------------------------------------------------------------------
# Result upload
#
# Intention: create one dataset repository and send the completed result folder with the same single-transfer-worker policy as download. Runtime depends on the generated files and upstream bandwidth, normally minutes to hours.
# -----------------------------------------------------------------------------

def upload_results():
    """Create one Hugging Face dataset repository and upload the results folder."""
    from huggingface_hub import HfApi, create_repo

    token = HF_TOKEN
    if not token or not RESULTS_REPO:
        raise RuntimeError("Set HF_TOKEN and HF_RESULTS_REPO before uploading")
    create_repo(
        RESULTS_REPO,
        repo_type="dataset",
        token=token,
        exist_ok=True,
    )
    api = HfApi(token=token)
    print(f"Uploading {OUTPUT_ROOT} to {RESULTS_REPO}")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            folder_path=str(OUTPUT_ROOT),
            num_workers=HF_TRANSFER_WORKERS,
        )
    else:
        api.upload_folder(
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            folder_path=str(OUTPUT_ROOT),
        )


# -----------------------------------------------------------------------------
# Command-line entry point
#
# Intention: expose only a few understandable switches, prepare logging/storage, build smoke or real context, run requested stages plus dependencies, and upload only after an unfiltered full run. Argument handling takes under a second.
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run small synthetic data")
    parser.add_argument("--skip-download", action="store_true", help="use files already in DATA_ROOT")
    parser.add_argument("--overwrite", action="store_true", help="replace existing result CSVs")
    parser.add_argument("--no-upload", action="store_true", help="do not upload results to Hugging Face")
    parser.add_argument("--only", help="comma-separated experiments, for example E1,E2")
    args = parser.parse_args()

    output = HERE / "smoke_results" if args.smoke_test else OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    LOGGER.info(
        "Visible GPUs: %s; configured workers: %s; CPU threads per worker: %s",
        visible_gpus(),
        WORKERS,
        CPU_THREADS_PER_WORKER,
    )

    # Smoke mode never downloads or uploads and does not require replacing the Hugging Face token placeholder.
    if args.smoke_test:
        context = make_smoke_context()
    else:
        required_repositories = [LAYER_REPO, FINAL_REPO, OLD_LABEL_REPO, FINAL_CATALOG_REPO]
        if not HF_TOKEN or not all(required_repositories):
            raise RuntimeError("Set HF_TOKEN and the four HF source repository variables before downloading the full inputs")
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(STORAGE_ROOT / "huggingface_cache"))
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        if not args.skip_download:
            download_data()
        context = make_real_context(args.overwrite)

    # --only accepts explicit E-numbers. Required earlier output stages are
    # added automatically so a requested experiment does not start half-built.
    only = None
    if args.only:
        requested = {name.strip().upper() for name in args.only.split(",") if name.strip()}
        try:
            only = expand_experiment_dependencies(requested)
        except ValueError as error:
            parser.error(str(error))
        if only != requested:
            print(f"Also running required stages: {', '.join(sorted(only - requested))}")
    run_all(context, only)
    if not args.smoke_test and not args.no_upload and not only:
        upload_results()
    print(f"Finished. Results are in {context['output_root']}")


if __name__ == "__main__":
    # Environment setup must happen before importing NumPy/Pandas/PyArrow so the restarted interpreter sees the intended packages and thread settings.
    enter_local_environment()
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    main()
