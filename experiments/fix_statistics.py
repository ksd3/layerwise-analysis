"""Correct the few statistics that accidentally included unavailable layers."""

import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


HF_TOKEN = os.environ.get("HF_TOKEN", "")


# Make CUDA libraries installed in .venv visible before CuPy is imported.
def restart_with_cuda_libraries():
    library_root = Path(sys.prefix) / "lib"
    directories = sorted(library_root.glob("python*/site-packages/nvidia/*/lib"))
    current = [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(":") if value]
    missing = [str(path.resolve()) for path in directories if str(path.resolve()) not in current]
    if missing:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current)
        os.execv(sys.executable, [sys.executable, *sys.argv])


restart_with_cuda_libraries()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "8"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import experiment_1_geometry as e1
import experiment_2_depth as e2
import experiment_3_image_transfer as e3
import experiment_5_updates as e5
import experiment_6_model_properties as e6
import run_all_experiments as master


master.np = np
master.pd = pd
master.pq = pq
master.HF_TOKEN = HF_TOKEN


def configure_worker(gpu_id):
    """Give one process one GPU and eight CPU threads."""
    if gpu_id is None:
        os.environ.pop("EXPERIMENT_GPU_ID", None)
    else:
        os.environ["EXPERIMENT_GPU_ID"] = str(gpu_id)
        e1.cp.cuda.Device(gpu_id).use()
    e1.faiss.omp_set_num_threads(8)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "8"


# -----------------------------------------------------------------------------
# E1: reuse the saved neighbor arrays and repeat only curve inference.
# -----------------------------------------------------------------------------

def fix_e1_group(arguments):
    gpu_id, tasks, output_root, permutations, bootstraps, seed = arguments
    configure_worker(gpu_id)
    fixed = []

    for cohort, model in tasks:
        print(f"Fixing E1  GPU {gpu_id}  {cohort}/{model}", flush=True)
        model_dir = Path(output_root) / "E1" / cohort / model
        curve_file = model_dir / "curve.csv"
        curve = pd.read_csv(curve_file)
        valid_layers = curve.index[curve["available"].astype(bool)].to_list()

        neighbor_curves = []
        row_curves = []
        for layer in valid_layers:
            with np.load(model_dir / "neighbors" / f"layer_{layer:03d}.npz") as saved:
                left = saved["left_neighbors"]
                right = saved["right_neighbors"]
            neighbor_curves.append((left, right))
            row_curves.append(e1.overlap_per_row(left, right, min(20, left.shape[1])))

        sample_size = 8192 if cohort == "legacysurvey" else None
        p_value, low, high, peak = e1.curve_inference(
            neighbor_curves,
            row_curves,
            permutations,
            bootstraps,
            seed,
            permutation_sample_size=sample_size,
        )

        curve["curve_p_value"] = p_value
        curve["simultaneous_low"] = np.nan
        curve["simultaneous_high"] = np.nan
        curve["peak_probability"] = np.nan
        curve["permutation_rows"] = np.nan
        curve.loc[valid_layers, "simultaneous_low"] = low
        curve.loc[valid_layers, "simultaneous_high"] = high
        curve.loc[valid_layers, "peak_probability"] = peak
        curve.loc[valid_layers, "permutation_rows"] = min(
            len(row_curves[0]), sample_size or len(row_curves[0])
        )
        curve.to_csv(curve_file, index=False)
        fixed.append(str(curve_file))

    return fixed


def rebuild_e1(context):
    root = Path(context["output_root"])
    old = pd.read_csv(root / "E1_all_curves.csv")
    affected = [
        tuple(key)
        for key, group in old.groupby(["cohort", "model"])
        if (~group["available"].astype(bool)).any()
    ]
    # These two suffixed columns are the exact signature left by the first version after its expensive E1 calculations had already completed.
    inference_is_done = {
        "curve_p_value_x",
        "curve_p_value_y",
    }.issubset(old.columns)
    if not inference_is_done:
        inference_is_done = True
        for cohort, model in affected:
            curve = pd.read_csv(root / "E1" / cohort / model / "curve.csv")
            invalid = ~curve["available"].astype(bool)
            columns = ["simultaneous_low", "simultaneous_high", "peak_probability"]
            if curve.loc[invalid, columns].notna().any().any():
                inference_is_done = False
                break

    if inference_is_done:
        print("Corrected E1 inference is already finished; rebuilding its table", flush=True)
    else:
        groups = [affected[number::context["workers"]] for number in range(context["workers"])]
        arguments = [
            (
                gpu_id if gpu_id < context["gpu_count"] else None,
                group,
                root,
                context["permutations"],
                context["bootstraps"],
                context["seed"],
            )
            for gpu_id, group in enumerate(groups)
            if group
        ]
        with ProcessPoolExecutor(
            max_workers=len(arguments), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            list(pool.map(fix_e1_group, arguments))

    frames = []
    for item in context["layerwise_files"]:
        path = root / "E1" / item["cohort"] / item["model"] / "curve.csv"
        frames.append(pd.read_csv(path))
    result = pd.concat(frames, ignore_index=True).drop(columns="curve_p_value_holm", errors="ignore")
    tests = result.groupby(["cohort", "model"], as_index=False)["curve_p_value"].first()
    tests["curve_p_value_holm"] = e1.holm_adjust(tests["curve_p_value"])
    result = result.merge(
        tests[["cohort", "model", "curve_p_value_holm"]],
        on=["cohort", "model"],
        how="left",
    )
    result.to_csv(root / "E1_all_curves.csv", index=False)


# -----------------------------------------------------------------------------
# E3: make both label-free and oracle selection ignore unavailable layers.
# -----------------------------------------------------------------------------

E3_INVALID_LAYERS = set()
E3_COLUMN_LAYERS = {}
E3_CURRENT_LAYER = None
ORIGINAL_E3_READ = e3.read_matrix
ORIGINAL_E3_SELECTION = e3.selection_mrr
ORIGINAL_E3_TRANSFER = e3.transfer_score
ORIGINAL_E3_ONE_SE = e3.one_standard_error_layer


def fixed_e3_read(path, column, rows=None):
    global E3_CURRENT_LAYER
    if column in E3_COLUMN_LAYERS:
        E3_CURRENT_LAYER = E3_COLUMN_LAYERS[column]
    return ORIGINAL_E3_READ(path, column, rows)


def fixed_e3_selection(left, right, rows):
    if E3_CURRENT_LAYER in E3_INVALID_LAYERS:
        return np.nan, np.nan
    return ORIGINAL_E3_SELECTION(left, right, rows)


def fixed_e3_transfer(*args, **kwargs):
    if kwargs.get("validation_only", False) and E3_CURRENT_LAYER in E3_INVALID_LAYERS:
        return np.nan, np.nan
    return ORIGINAL_E3_TRANSFER(*args, **kwargs)


def fixed_e3_one_se(scores):
    usable = [row for row in scores if np.isfinite(row["mrr"])]
    return ORIGINAL_E3_ONE_SE(usable)


e3.read_matrix = fixed_e3_read
e3.selection_mrr = fixed_e3_selection
e3.transfer_score = fixed_e3_transfer
e3.one_standard_error_layer = fixed_e3_one_se


def fix_e3_group(arguments):
    gpu_id, items, invalid_by_item, old_targets, output_root, bootstraps, seed = arguments
    global E3_INVALID_LAYERS, E3_COLUMN_LAYERS
    configure_worker(gpu_id)
    rows = []

    for item in items:
        key = (item["cohort"], item["model"])
        print(f"Fixing E3  GPU {gpu_id}  {key[0]}/{key[1]}", flush=True)
        E3_INVALID_LAYERS = set(invalid_by_item[key])
        pairs = e1.layer_pairs(item["path"], item["cohort"])
        E3_COLUMN_LAYERS = {}
        for layer, (left, right, _) in enumerate(pairs):
            E3_COLUMN_LAYERS[left] = layer
            E3_COLUMN_LAYERS[right] = layer
        frame = e3.run_file(
            item,
            old_targets[item["cohort"]],
            output_root,
            bootstraps,
            seed,
        )
        rows.extend(frame.to_dict("records"))

    return rows


def write_e3_summaries(result, context):
    root = Path(context["output_root"])
    result["family"] = result["model"].map(context["model_family"])
    result.to_csv(root / "E3_image_transfer.csv", index=False)
    family = result.groupby(["cohort", "family", "target", "direction"], as_index=False).agg(
        models=("model", "nunique"), mean_regret=("regret", "mean")
    )
    family.to_csv(root / "E3_family_summary.csv", index=False)

    rng = np.random.default_rng(context["seed"])
    rows = []
    for keys, group in family.groupby(["cohort", "target", "direction"]):
        values = group["mean_regret"].dropna().to_numpy()
        if len(values) == 0:
            continue
        draws = [rng.choice(values, len(values), replace=True).mean() for _ in range(context["bootstraps"])]
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
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
    pd.DataFrame(rows).to_csv(root / "E3_cross_family_inference.csv", index=False)


def rebuild_e3(context):
    root = Path(context["output_root"])
    curves = pd.read_csv(root / "E1_all_curves.csv")
    invalid_by_item = {
        tuple(key): group.loc[~group["available"].astype(bool), "layer"].astype(int).to_list()
        for key, group in curves.groupby(["cohort", "model"])
        if key[0] in ("legacysurvey", "jwst") and (~group["available"].astype(bool)).any()
    }
    items = [
        item for item in context["layerwise_files"]
        if (item["cohort"], item["model"]) in invalid_by_item
    ]
    groups = [items[number::context["workers"]] for number in range(context["workers"])]
    arguments = [
        (
            gpu_id if gpu_id < context["gpu_count"] else None,
            group,
            invalid_by_item,
            context["old_targets"],
            root,
            context["bootstraps"],
            context["seed"],
        )
        for gpu_id, group in enumerate(groups)
        if group
    ]
    if arguments:
        with ProcessPoolExecutor(
            max_workers=len(arguments), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            new_rows = [row for rows in pool.map(fix_e3_group, arguments) for row in rows]
    else:
        new_rows = []

    old = pd.read_csv(root / "E3_image_transfer.csv")
    affected = set(invalid_by_item)
    keep = [
        (cohort, model) not in affected
        for cohort, model in zip(old["cohort"], old["model"])
    ]
    result = pd.concat([old.loc[keep], pd.DataFrame(new_rows)], ignore_index=True)
    write_e3_summaries(result, context)


# -----------------------------------------------------------------------------
# E5: repeat only curves containing tie-dominated update rows.
# -----------------------------------------------------------------------------

E5_FLAGS = []
ORIGINAL_E5_GEOMETRY = e5.geometry
ORIGINAL_E5_INFERENCE = e5.curve_inference


def fixed_e5_geometry(*args, **kwargs):
    answer = ORIGINAL_E5_GEOMETRY(*args, **kwargs)
    E5_FLAGS.append(bool(answer[0]["available"]))
    return answer


def fixed_e5_inference(neighbors, rows, *args, **kwargs):
    flags = np.asarray(E5_FLAGS[-len(rows):], dtype=bool)
    selected = np.flatnonzero(flags)
    p_value, low, high, peak = ORIGINAL_E5_INFERENCE(
        [neighbors[index] for index in selected],
        [rows[index] for index in selected],
        *args,
        **kwargs,
    )
    full_low = np.full(len(rows), np.nan)
    full_high = np.full(len(rows), np.nan)
    full_peak = np.full(len(rows), np.nan)
    full_low[selected] = low
    full_high[selected] = high
    full_peak[selected] = peak
    return p_value, full_low, full_high, full_peak


e5.geometry = fixed_e5_geometry
e5.curve_inference = fixed_e5_inference


def repair_e5_frame(frame, curve, permutations, bootstraps, seed):
    frame = frame.copy()
    tie = frame["tie_dominated"].fillna(False).astype(bool)
    left_ok = frame["left_zero_fraction"].fillna(0) < 0.5
    right_ok = frame["right_zero_fraction"].fillna(0) < 0.5
    frame["available"] = frame["corrected_overlap_20"].notna() & ~tie & left_ok & right_ok
    curve = curve.sort_values("layer").reset_index(drop=True)
    state_valid = curve["available"].astype(bool).to_numpy()
    state_value = curve["corrected_overlap_20"].to_numpy()

    full_max = curve.loc[curve["available"].astype(bool), "corrected_overlap_20"].max()
    update_max = frame.loc[frame["available"], "corrected_overlap_20"].max()
    frame["maximum_full_alignment"] = full_max
    frame["maximum_update_alignment"] = update_max
    frame["update_to_full_ratio"] = update_max / full_max if full_max != 0 else np.nan
    frame["full_state_change"] = np.nan
    frame["full_state_after"] = np.nan

    for index, row in frame.loc[frame["available"]].iterrows():
        transition = int(row["transition"])
        if transition + 1 < len(curve) and state_valid[transition] and state_valid[transition + 1]:
            frame.loc[index, "full_state_change"] = state_value[transition + 1] - state_value[transition]
            frame.loc[index, "full_state_after"] = state_value[transition + 1]

    paired = frame.loc[frame["available"], ["corrected_overlap_20", "full_state_change"]].dropna()
    frame["update_change_correlation"] = np.nan
    frame["update_change_p_value"] = np.nan
    frame["update_change_low"] = np.nan
    frame["update_change_high"] = np.nan
    if len(paired) >= 3:
        first = paired["corrected_overlap_20"].to_numpy()
        second = paired["full_state_change"].to_numpy()
        observed = e5.safe_correlation(first, second)
        rng = np.random.default_rng(seed)
        null = [e5.safe_correlation(first, rng.permutation(second)) for _ in range(permutations)]
        null = np.asarray([value for value in null if np.isfinite(value)])
        draws = []
        for _ in range(bootstraps):
            picked = rng.integers(0, len(first), len(first))
            value = e5.safe_correlation(first[picked], second[picked])
            if np.isfinite(value):
                draws.append(value)
        p_value = np.nan
        if len(null) and np.isfinite(observed):
            p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1)
        frame["update_change_correlation"] = observed
        frame["update_change_p_value"] = p_value
        if draws:
            low, high = np.quantile(draws, [0.025, 0.975])
            frame["update_change_low"] = low
            frame["update_change_high"] = high
    return frame


def fix_e5_group(arguments):
    gpu_id, items, output_root, permutations, bootstraps, seed = arguments
    configure_worker(gpu_id)
    rows = []
    for item in items:
        print(f"Fixing E5  GPU {gpu_id}  {item['cohort']}/{item['model']}", flush=True)
        E5_FLAGS.clear()
        frame = e5.run_file(item, output_root, permutations, bootstraps, seed)
        curve = pd.read_csv(Path(output_root) / "E1" / item["cohort"] / item["model"] / "curve.csv")
        frame = repair_e5_frame(frame, curve, permutations, bootstraps, seed)
        rows.extend(frame.to_dict("records"))
    return rows


def rebuild_e5(context):
    root = Path(context["output_root"])
    old = pd.read_csv(root / "E5_updates.csv")
    bad = old["tie_dominated"].fillna(False).astype(bool)
    bad |= old["left_zero_fraction"].fillna(0) >= 0.5
    bad |= old["right_zero_fraction"].fillna(0) >= 0.5
    affected = set(
        map(
            tuple,
            old.loc[bad, ["cohort", "model"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    items = [
        item for item in context["layerwise_files"]
        if (item["cohort"], item["model"]) in affected
    ]
    groups = [items[number::context["workers"]] for number in range(context["workers"])]
    arguments = [
        (
            gpu_id if gpu_id < context["gpu_count"] else None,
            group,
            root,
            context["permutations"],
            context["bootstraps"],
            context["seed"],
        )
        for gpu_id, group in enumerate(groups)
        if group
    ]
    if arguments:
        with ProcessPoolExecutor(
            max_workers=len(arguments), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            new_rows = [row for rows in pool.map(fix_e5_group, arguments) for row in rows]
    else:
        new_rows = []

    keep = [(cohort, model) not in affected for cohort, model in zip(old["cohort"], old["model"])]
    result = pd.concat([old.loc[keep], pd.DataFrame(new_rows)], ignore_index=True)
    result = result.drop(columns=["family", "curve_p_value_holm"], errors="ignore")
    result["family"] = result["model"].map(context["model_family"])
    tests = result.groupby(["cohort", "model"], as_index=False)["curve_p_value"].first()
    tests["curve_p_value_holm"] = e1.holm_adjust(tests["curve_p_value"])
    result = result.merge(
        tests[["cohort", "model", "curve_p_value_holm"]],
        on=["cohort", "model"],
        how="left",
    )
    result.to_csv(root / "E5_updates.csv", index=False)

    model = result.groupby(["cohort", "model", "family"], as_index=False).agg(
        maximum_update_alignment=("maximum_update_alignment", "first"),
        update_change_correlation=("update_change_correlation", "first"),
        update_change_p_value=("update_change_p_value", "first"),
    )
    family = model.groupby(["cohort", "family"], as_index=False).agg(
        models=("model", "nunique"),
        mean_maximum_update_alignment=("maximum_update_alignment", "mean"),
        mean_update_change_correlation=("update_change_correlation", "mean"),
    )
    family.to_csv(root / "E5_family_summary.csv", index=False)


# -----------------------------------------------------------------------------
# Small CPU reductions after E1 and E5 have been corrected.
# -----------------------------------------------------------------------------

def rebuild_e6(context):
    root = Path(context["output_root"])
    summary = pd.read_csv(root / "E2_depth_summary.csv")
    curves = pd.read_csv(root / "E1_all_curves.csv")
    widths = curves.sort_values("normalized_depth").groupby(["cohort", "model"], as_index=False)["left_width"].last()
    widths = widths.rename(columns={"left_width": "width"})
    summary = summary.merge(widths, on=["cohort", "model"], how="left")

    metadata = []
    for model in summary["model"]:
        row = context["model_metadata"](model).copy()
        row["model"] = model
        metadata.append(row)
    metadata = pd.DataFrame(metadata).drop_duplicates("model")
    table = summary.drop(columns="family", errors="ignore").merge(metadata, on="model", how="left")

    updates = pd.read_csv(root / "E5_updates.csv")
    updates = updates[updates["available"].astype(bool)]
    update_max = updates.groupby(["cohort", "model"], as_index=False)["corrected_overlap_20"].max()
    update_max = update_max.rename(columns={"corrected_overlap_20": "maximum_update_alignment"})
    table = table.merge(update_max, on=["cohort", "model"], how="left")
    table.to_csv(root / "E6_model_table.csv", index=False)
    e6.calibrated_regressions(table, context["permutations"], context["bootstraps"], context["seed"]).to_csv(root / "E6_regression.csv", index=False)
    e6.leave_family_out(table).to_csv(root / "E6_leave_family_out.csv", index=False)
    table.groupby(["objective", "astronomy_specific"], as_index=False).agg(
        models=("model", "nunique"),
        mean_maximum=("maximum", "mean"),
        mean_final_minus_early=("final_minus_early", "mean"),
    ).to_csv(root / "E6_objective_summary.csv", index=False)
    e6.training_effects(table, context["bootstraps"], context["seed"]).to_csv(root / "E6_training_effects.csv", index=False)


def filter_e7(context):
    root = Path(context["output_root"])
    curves = pd.read_csv(root / "E1_all_curves.csv")
    valid = {
        tuple(key): set(group.loc[group["available"].astype(bool), "layer"].astype(int))
        for key, group in curves.groupby(["cohort", "model"])
    }
    maps = pd.read_csv(root / "E7_layer_pair_maps.csv")
    keep = [
        int(left) in valid[(cohort, model)] and int(right) in valid[(cohort, model)]
        for cohort, model, left, right in zip(
            maps["cohort"], maps["model"], maps["left_layer"], maps["right_layer"]
        )
    ]
    maps.loc[keep].to_csv(root / "E7_layer_pair_maps.csv", index=False)
    curves[curves["cohort"] == "cosmosweb"].to_csv(root / "E7_large_cohort_curves.csv", index=False)


def upload_results(context):
    from huggingface_hub import HfApi

    root = Path(context["output_root"])
    patterns = [
        "E1_all_curves.csv",
        "E1/**/curve.csv",
        "E2_*.csv",
        "E3_*.csv",
        "E3_predictions/**",
        "E5_updates.csv",
        "E5_family_summary.csv",
        "E6_*.csv",
        "E7_layer_pair_maps.csv",
        "E7_large_cohort_curves.csv",
    ]
    print("Uploading corrected results", flush=True)
    HfApi(token=HF_TOKEN).upload_large_folder(
        repo_id=master.RESULTS_REPO,
        repo_type="dataset",
        folder_path=root,
        allow_patterns=patterns,
        num_workers=1,
    )


def main():
    if not HF_TOKEN or not master.RESULTS_REPO:
        raise RuntimeError("Set HF_TOKEN and HF_RESULTS_REPO before uploading")
    context = master.make_real_context(overwrite=False)
    context["workers"] = max(1, min(8, context["gpu_count"]))
    context["cpu_threads_per_worker"] = 8

    print("Starting corrected E1 inference", flush=True)
    rebuild_e1(context)
    print("Rebuilding E2", flush=True)
    e2.run(context)
    print("Starting corrected E3 selection", flush=True)
    rebuild_e3(context)
    print("Starting corrected E5 geometry", flush=True)
    rebuild_e5(context)
    print("Rebuilding E6", flush=True)
    rebuild_e6(context)
    print("Filtering E7 maps", flush=True)
    filter_e7(context)
    upload_results(context)
    print("Finished all corrections", flush=True)


if __name__ == "__main__":
    main()
