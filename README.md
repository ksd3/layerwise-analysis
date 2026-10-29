# Shared Geometry Is Not Shared Physics: A Layerwise Test of the PRH in Astronomy

Code and data for the NeurIPS 2026 Interpretability for Discovery workshop paper.

The experiment outputs needed to verify the paper are included in `data`. You can also generate all the data yourself. 

## One-command reproduction

```bash
git clone https://github.com/ksd3/layerwise-analysis.git
cd layerwise-analysis
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

## Camera-ready additions

The camera-ready version adds post hoc analyses that reuse the saved experiment outputs only; no model was re-run.

```bash
uv run python analysis/analyze_review_additions.py
uv run python figures/05_geometry_vs_physics.py
uv run python figures/06_final_transfer_calibration.py
uv run python figures/07_common_space_retrieval.py
```

`analyze_review_additions.py` writes the `review_*.csv` tables in `outputs/tables`: the percentile rank of geometry-, label- and final-stage selections among all stages, the across-stage Spearman correlation between alignment and physical performance, rank-normalized depth profiles, the affine-recalibration bound for E3 transfer, and the transferred-versus-native diagnostics and metric association for the final COSMOS-Web transfer (E8). The recalibration and E8 diagnostics need the per-object held-out predictions (`data/E3_predictions/*.npz` and `data/E8_predictions/*.npz`, about 1 GB together), which are written by the full experiment run and are not stored in this repository. Without them the script recomputes the other tables and keeps the released recalibration and diagnostic tables unchanged.

`05_geometry_vs_physics.py` and `06_final_transfer_calibration.py` draw main-text Figures 2 and 3; `07_common_space_retrieval.py` draws the held-out common-space retrieval figure in the appendix.

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
- `outputs/figures/04_scientific_transfer.*`: transfer figure from the reviewed version, replaced in the camera-ready by `05_geometry_vs_physics` and `06_final_transfer_calibration`
