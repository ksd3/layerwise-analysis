# Shared geometry reproduction

The compact experiment outputs needed to verify the paper are included in `data`. A reviewer does not need the multi-terabyte embedding archive or a GPU.

## One-command reproduction

```bash
git clone <anonymous-repository-url>
cd <repository-directory>
uv sync
uv run python reproduce.py
```

This regenerates every statistical summary in `outputs/tables` and every paper figure in `outputs/figures`.

## Run each stage separately

```bash
uv run python analyze_results.py
uv run python make_figures.py
```

The three main figures can also be edited interactively:

```bash
uv run marimo edit figures/main_figures_notebook.py
```

The appendix figures can be run individually:

```bash
uv run python figures/plot_physics_alignment.py
uv run python figures/05_normalized_depth_atlas.py
uv run python figures/06_cross_layer_atlas.py
```

## Full experiment computation

`generate_experiment_data.py` runs the nine GPU experiments, applies the final statistical corrections, and computes the layerwise physical-prediction scores. The source repository names, token, storage location, and optional result repository are supplied through environment variables so that no machine-specific or identifying values are stored in the code.

```bash
export HF_TOKEN=<token>
export HF_LAYER_REPO=<layerwise-embedding-dataset>
export HF_FINAL_REPO=<final-embedding-dataset>
export HF_LABEL_REPO=<label-dataset>
export HF_CATALOG_REPO=<catalog-dataset>
export HF_RESULTS_REPO=<result-dataset>
export EXPERIMENT_STORAGE_ROOT=<large-storage-directory>
uv run python generate_experiment_data.py
```

The small synthetic pipeline test is:

```bash
uv run python generate_experiment_data.py --smoke-test
```

## Repository layout

- `data`: compact released experiment CSVs used by the paper
- `analysis`: individual statistical-analysis scripts
- `experiments`: the full nine-experiment GPU implementation
- `figures`: editable figure code
- `outputs/tables`: regenerated machine-readable tables
- `outputs/figures`: regenerated PDF and PNG figures
