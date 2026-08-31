"""Recompute only the layerwise physics scores needed for the new heatmaps."""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages

import run_all_experiments as runner

from experiment_1_geometry import (
    configure_worker,
    layer_pairs,
    prepare_lm_head_factors,
    read_matrix,
)
from experiment_3_image_transfer import split_roles, transfer_score
from run_all_experiments import (
    CPU_THREADS_PER_WORKER,
    DATA_ROOT,
    HF_TOKEN,
    OUTPUT_ROOT,
    SEED,
    WORKERS,
    find_layerwise_files,
    load_old_targets,
    make_smoke_context,
    model_family,
    visible_gpus,
)


# The main runner imports these only after its environment setup. This script already runs inside that environment, so make them available to its helpers.
runner.np = np
runner.pd = pd
runner.pa = pa
runner.pq = pq


FOLDS = 5


def scan_model(item, targets):
    """Read each layer once and test both physical labels in both directions."""
    pairs = layer_pairs(item["path"], item["cohort"])
    if not pairs:
        return pd.DataFrame()

    row_indices = item.get("row_indices")
    first = read_matrix(item["path"], pairs[0][0], row_indices)
    n = len(first)
    del first

    targets = {name: np.asarray(values)[:n] for name, values in targets.items()}
    folds = [split_roles(n, fold, SEED) for fold in range(FOLDS)]
    rows = []

    for layer, (left_name, right_name, layer_name) in enumerate(pairs):
        if layer == 0 or (layer + 1) % 5 == 0 or layer + 1 == len(pairs):
            print(
                f"  {item['cohort']}/{item['model']}: layer {layer + 1}/{len(pairs)}",
                flush=True,
            )

        left = read_matrix(item["path"], left_name, row_indices)
        right = read_matrix(item["path"], right_name, row_indices)
        if left.shape[1] != right.shape[1]:
            continue

        for fold, roles in enumerate(folds):
            for target_name, labels in targets.items():
                directions = (
                    ("hsc_to_other", left, right),
                    ("other_to_hsc", right, left),
                )
                for direction, source, target in directions:
                    score, alpha, y, prediction, test_rows = transfer_score(
                        source,
                        target,
                        labels,
                        roles,
                        return_predictions=True,
                    )
                    rows.append(
                        {
                            "cohort": item["cohort"],
                            "model": item["model"],
                            "family": model_family(item["model"]),
                            "layer": layer,
                            "layer_name": layer_name,
                            "normalized_depth": layer / max(1, len(pairs) - 1),
                            "fold": fold,
                            "target": target_name,
                            "direction": direction,
                            "alpha": alpha,
                            "test_r2": score,
                            "test_n": len(y),
                        }
                    )

        del left, right

    return pd.DataFrame(rows)


def run_worker(arguments):
    gpu_id, items, targets, part_dir, overwrite = arguments
    configure_worker(gpu_id, CPU_THREADS_PER_WORKER)
    files = []

    for item in items:
        part = part_dir / f"{item['cohort']}__{item['model']}.csv"
        files.append(part)
        if part.exists() and not overwrite:
            print(f"Using existing {part.name}", flush=True)
            continue

        print(f"GPU {gpu_id}: starting {item['cohort']}/{item['model']}", flush=True)
        frame = scan_model(item, targets[item["cohort"]])
        frame.to_csv(part, index=False)
        print(f"GPU {gpu_id}: finished {item['cohort']}/{item['model']}", flush=True)

    return [str(path) for path in files]


def interpolate_curves(frame, models, value, depth_grid):
    matrix = np.full((len(depth_grid), len(models)), np.nan)
    peaks = np.full(len(models), np.nan)
    peak_depths = np.full(len(models), np.nan)

    for column, model in enumerate(models):
        curve = frame[frame["model"] == model]
        curve = curve.groupby("normalized_depth", as_index=False)[value].mean()
        curve = curve.replace([np.inf, -np.inf], np.nan).dropna()
        if curve.empty:
            continue
        curve = curve.sort_values("normalized_depth")
        x = curve["normalized_depth"].to_numpy()
        y = curve[value].to_numpy()
        matrix[:, column] = np.interp(depth_grid, x, y)
        best = int(np.nanargmax(y))
        peaks[column] = y[best]
        peak_depths[column] = x[best]

    return matrix, peaks, peak_depths


def heatmap_limits(values, start_at_zero=False):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    low, high = np.quantile(finite, [0.02, 0.98])
    if start_at_zero:
        low = 0.0
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def draw_pair(pdf, physics, alignment, cohort, target, direction):
    """Draw one physics/alignment heatmap pair on one PDF page."""
    models = sorted(physics["model"].unique(), key=lambda name: (model_family(name), name))
    depth_grid = np.linspace(0, 1, 41)

    physics_matrix, physics_max, physics_depth = interpolate_curves(
        physics, models, "test_r2", depth_grid
    )
    alignment_matrix, alignment_max, alignment_depth = interpolate_curves(
        alignment, models, "corrected_overlap_20", depth_grid
    )

    figure = plt.figure(figsize=(18, 8))
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 4], hspace=0.05, wspace=0.23)

    panels = (
        (physics_matrix, physics_max, physics_depth, "Physical prediction", "Held-out $R^2$", False),
        (alignment_matrix, alignment_max, alignment_depth, "Embedding alignment", "Corrected MKNN", True),
    )

    for column, (matrix, maxima, peak_depths, title, color_label, zero) in enumerate(panels):
        top = figure.add_subplot(grid[0, column])
        heatmap = figure.add_subplot(grid[1, column], sharex=top)
        low, high = heatmap_limits(matrix, start_at_zero=zero)

        top.plot(range(len(models)), maxima, "o-", color="#202020", markersize=3)
        top.set_title(title, fontsize=14, weight="bold")
        top.set_ylabel("Maximum")
        top.grid(axis="y", alpha=0.2)
        top.tick_params(axis="x", labelbottom=False)

        image = heatmap.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=(-0.5, len(models) - 0.5, 0, 1),
            cmap="magma",
            vmin=low,
            vmax=high,
        )
        heatmap.scatter(
            range(len(models)),
            peak_depths,
            marker="v",
            s=25,
            color="white",
            edgecolor="black",
            linewidth=0.4,
        )
        heatmap.set_ylabel("Normalized model depth")
        heatmap.set_xticks(range(len(models)), models, rotation=65, ha="right", fontsize=7)
        figure.colorbar(image, ax=heatmap, label=color_label, pad=0.01)

    pretty_direction = direction.replace("_", " ").replace("hsc", "HSC")
    figure.suptitle(
        f"{cohort.upper()} — {target.replace('_', ' ')} — {pretty_direction}",
        fontsize=16,
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def make_maps(scores, output_root):
    alignment_file = output_root / "E1_all_curves.csv"
    if not alignment_file.exists():
        print(f"Scores were saved, but {alignment_file} was not found; skipping maps.")
        return

    alignment = pd.read_csv(alignment_file)
    alignment["corrected_overlap_20"] = pd.to_numeric(
        alignment["corrected_overlap_20"], errors="coerce"
    )
    output_file = output_root / "physics_alignment_maps.pdf"

    with PdfPages(output_file) as pdf:
        groups = scores.groupby(["cohort", "target", "direction"], sort=True)
        for (cohort, target, direction), physics in groups:
            matching_alignment = alignment[alignment["cohort"] == cohort]
            draw_pair(pdf, physics, matching_alignment, cohort, target, direction)

    print(f"Saved {output_file}")


def real_inputs():
    items = [
        item
        for item in find_layerwise_files(DATA_ROOT)
        if item["cohort"] in ("legacysurvey", "jwst")
    ]
    targets, cohort_rows = load_old_targets(DATA_ROOT, items)
    for item in items:
        item["row_indices"] = cohort_rows[item["cohort"]]
    return items, targets, Path(OUTPUT_ROOT), Path(DATA_ROOT), HF_TOKEN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        context = make_smoke_context()
        items = [
            item
            for item in context["layerwise_files"]
            if item["cohort"] in ("legacysurvey", "jwst")
        ]
        targets = context["old_targets"]
        output_root = Path(context["output_root"])
        data_root = Path(context["data_root"])
        token = context.get("hf_token")
        workers = 1
        gpu_count = 0
    else:
        items, targets, output_root, data_root, token = real_inputs()
        gpu_count = visible_gpus()
        workers = min(WORKERS, len(items), max(1, gpu_count))

    if not items:
        raise RuntimeError("No Legacy Survey or JWST layerwise files were found")

    output_root.mkdir(parents=True, exist_ok=True)
    part_dir = output_root / "layerwise_physics_parts"
    part_dir.mkdir(exist_ok=True)

    # These factors should already exist after E1; this call is then instantaneous.
    prepare_lm_head_factors(items, data_root, token)

    groups = [items[number::workers] for number in range(workers)]
    arguments = [
        (
            number if number < gpu_count else None,
            group,
            targets,
            part_dir,
            args.overwrite,
        )
        for number, group in enumerate(groups)
    ]

    if workers == 1:
        returned = [run_worker(arguments[0])]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            returned = list(pool.map(run_worker, arguments))

    part_files = [Path(name) for group in returned for name in group]
    scores = pd.concat([pd.read_csv(path) for path in part_files], ignore_index=True)
    score_file = output_root / "layerwise_physics_scores.csv"
    scores.to_csv(score_file, index=False)
    print(f"Saved {score_file}")
    make_maps(scores, output_root)


if __name__ == "__main__":
    main()
