"""E5: compare the changes written by consecutive compatible layers.

E1-E4 study complete layer representations. E5 instead studies an update vector,
defined as the output of layer t+1 minus the output of layer t. Subtraction is
only meaningful when consecutive outputs belong to the same block stack and
have the same width.

This file performs three related calculations:
1. Geometry of consecutive updates, including E1-style uncertainty.
2. E3-style image transfer using update vectors instead of full states.
3. E4-style image/spectrum common-space transfer using update vectors.
It then compares strong update alignment with changes in full-state alignment.

Approximate full-data runtime: this is one of the long stages because it rereads
neighboring Parquet columns, performs geometry/permutation inference, and repeats
the five-fold transfer experiments. One model/cohort can take tens of minutes to
hours, and the complete E5 stage may take many hours. Geometry reductions,
PCA/CCA, Ridge, permutations, and bootstraps now use one assigned GPU per worker;
forming updates and decoding Parquet columns remain host work.
"""

import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_1_geometry import (
    configure_worker,
    curve_inference,
    geometry,
    holm_adjust,
    layer_pairs,
    read_matrix,
)
from experiment_3_image_transfer import (
    one_standard_error_layer,
    paired_bootstrap_regret,
    selection_mrr,
    split_roles,
    transfer_score,
)
from experiment_4_spectrum_transfer import (
    CCA_RIDGES,
    fit_common_space,
    prepare_common_space,
    transform_common,
)


# -----------------------------------------------------------------------------
# Update construction and small utilities
#
# Intention: decide when subtraction has a clear architectural meaning and read the two consecutive states needed to form it. Regex/shape checks take less than a millisecond. Reading and subtracting real matrices takes seconds to minutes and is governed mainly by Parquet speed and matrix size.
# -----------------------------------------------------------------------------

def read_item_matrix(item, column):
    """Read one matrix using the cohort's already-established clean row order."""
    return read_matrix(item["path"], column, item.get("row_indices"))


def compatible_transition(pairs, transition):
    """Only subtract neighboring outputs from the same numbered block stack."""
    first = re.fullmatch(r"(.+)\.(\d+)", pairs[transition][2])
    second = re.fullmatch(r"(.+)\.(\d+)", pairs[transition + 1][2])
    return first is not None and second is not None and first.group(1) == second.group(1)


def safe_correlation(first, second):
    """Return Pearson correlation unless either input is constant."""
    if np.std(first) == 0 or np.std(second) == 0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def read_update_pair(item, pairs, transition):
    """Read one consecutive update pair, or return None if widths change."""
    if not compatible_transition(pairs, transition):
        return None
    left_now = read_item_matrix(item, pairs[transition][0])
    left_next = read_item_matrix(item, pairs[transition + 1][0])
    if left_now.shape[1] != left_next.shape[1]:
        return None
    # An update is a row-wise displacement in representation space. It is not a relative percentage change or a difference between unmatched objects.
    left_update = left_next - left_now

    # Spectra have one fixed embedding rather than a spectrum layer stack, so image updates are compared directly with that fixed spectrum matrix.
    if item["cohort"] in ("desi", "sdss"):
        right_update = read_item_matrix(item, pairs[transition][1])
    else:
        right_now = read_item_matrix(item, pairs[transition][1])
        right_next = read_item_matrix(item, pairs[transition + 1][1])
        if right_now.shape[1] != right_next.shape[1]:
            return None
        right_update = right_next - right_now
    return left_update, right_update


# -----------------------------------------------------------------------------
# E3 repeated on image update vectors
#
# Intention: test whether label-free alignment of what consecutive layers add selects an update useful for cross-survey science. Selection and validation share one archive pass, while final test rows remain untouched until choices are fixed. A real model/cohort normally takes minutes to hours.
# -----------------------------------------------------------------------------

def image_update_transfer(item, targets, bootstraps, seed, output_root):
    """Repeat the E3 selection test with consecutive image updates."""
    pairs = layer_pairs(item["path"], item["cohort"])
    n = read_item_matrix(item, pairs[0][0]).shape[0]
    targets = {name: np.asarray(values)[:n] for name, values in targets.items()}
    # Hold all five folds in memory so one decoded update can supply every fold's geometry and oracle-validation measurements.
    folds = []
    for fold in range(5):
        folds.append(
            {
                "fold": fold,
                "roles": split_roles(n, fold, seed),
                "choices": [],
                "oracle": {
                    (name, direction): []
                    for name in targets
                    for direction in (0, 1)
                },
            }
        )

    # Read each transition once while collecting selection scores for all folds. Equal feature width is required for direct retrieval and Ridge transfer.
    for transition in range(len(pairs) - 1):
        update = read_update_pair(item, pairs, transition)
        if update is None:
            continue
        left, right = update
        if left.shape[1] != right.shape[1]:
            continue
        for state in folds:
            mrr, se = selection_mrr(left, right, state["roles"]["selector"])
            state["choices"].append({"layer": transition, "mrr": mrr, "se": se})
            for target_name, labels in targets.items():
                for direction, (source, target) in enumerate(((left, right), (right, left))):
                    score, _ = transfer_score(
                        source,
                        target,
                        labels,
                        state["roles"],
                        validation_only=True,
                    )
                    state["oracle"][(target_name, direction)].append(
                        {"layer": transition, "score": score}
                    )

    # The scan is complete: label-free and oracle transitions can now be frozen and evaluated on the final test rows.
    rows = []
    for state in folds:
        if not state["choices"]:
            continue
        selected = one_standard_error_layer(state["choices"])
        for target_name, labels in targets.items():
            for direction in (0, 1):
                valid = [
                    row
                    for row in state["oracle"][(target_name, direction)]
                    if np.isfinite(row["score"])
                ]
                if not valid:
                    continue
                oracle = max(valid, key=lambda row: row["score"])["layer"]
                predictions = []
                scores = []
                alphas = []
                # Use identical folds and test objects for the two predictions, allowing a paired bootstrap of their R2 difference.
                for transition in (selected, oracle):
                    left, right = read_update_pair(item, pairs, transition)
                    source, target = ((left, right), (right, left))[direction]
                    score, alpha, y, prediction, test_rows = transfer_score(
                        source,
                        target,
                        labels,
                        state["roles"],
                        return_predictions=True,
                    )
                    scores.append(score)
                    alphas.append(alpha)
                    predictions.append((y, prediction, test_rows))
                low, high = paired_bootstrap_regret(
                    predictions[0][0],
                    predictions[0][1],
                    predictions[1][1],
                    bootstraps,
                    seed + state["fold"],
                )
                direction_name = "hsc_to_other" if direction == 0 else "other_to_hsc"
                prediction_dir = Path(output_root) / "E5_image_predictions"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                saved_rows = predictions[0][2]
                if item.get("row_indices") is not None:
                    saved_rows = np.asarray(item["row_indices"])[saved_rows]
                # Persist predictions so the reported regret can be audited later without rereading large Parquet columns.
                np.savez(
                    prediction_dir
                    / f"{item['cohort']}__{item['model']}__{target_name}__{direction_name}__fold{state['fold']}.npz",
                    row_indices=saved_rows,
                    y=predictions[0][0],
                    label_free_prediction=predictions[0][1],
                    oracle_prediction=predictions[1][1],
                )
                rows.append(
                    {
                        "cohort": item["cohort"],
                        "model": item["model"],
                        "fold": state["fold"],
                        "target": target_name,
                        "direction": direction_name,
                        "label_free_transition": selected,
                        "oracle_transition": oracle,
                        "label_free_alpha": alphas[0],
                        "oracle_alpha": alphas[1],
                        "label_free_test_r2": scores[0],
                        "oracle_test_r2": scores[1],
                        "regret": scores[0] - scores[1],
                        "regret_low": low,
                        "regret_high": high,
                    }
                )
    return rows


# -----------------------------------------------------------------------------
# E4 repeated on image/spectrum update vectors
#
# Intention: make unequal-width image updates and spectra comparable with the same PCA-plus-CCA common space used by E4. Four CCA ridges are scanned in five folds per transition. PCA is now reused across the four ridges and runs on the assigned GPU; a real file can still take minutes to hours because of rereads.
# -----------------------------------------------------------------------------

def spectrum_update_transfer(item, labels, bootstraps, seed, output_root):
    """Repeat the E4 common-space test with image update vectors."""
    pairs = layer_pairs(item["path"], item["cohort"])
    n = read_item_matrix(item, pairs[0][0]).shape[0]
    labels = None if labels is None else np.asarray(labels)[:n]
    folds = []
    for fold in range(5):
        folds.append(
            {
                "fold": fold,
                "roles": split_roles(n, fold, seed),
                "choices": [],
                "oracle": {0: [], 1: []},
            }
        )

    # This is the selection pass. Selector retrieval chooses the CCA ridge for each transition; optional labels record an oracle-validation score only.
    for transition in range(len(pairs) - 1):
        update = read_update_pair(item, pairs, transition)
        if update is None:
            continue
        left, right = update
        for state in folds:
            best = None
            roles = state["roles"]
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
                        "layer": transition,
                        "mrr": mrr,
                        "se": se,
                        "cca_ridge": cca_ridge,
                    }
            state["choices"].append(best)

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
                        source,
                        target,
                        labels,
                        state["roles"],
                        validation_only=True,
                    )
                    state["oracle"][direction].append(
                        {"layer": transition, "score": score}
                    )

    # All choices are frozen before test retrieval or test R2 is calculated.
    rows = []
    for state in folds:
        if not state["choices"]:
            continue
        selected = one_standard_error_layer(state["choices"])
        selected_ridge = next(
            row["cca_ridge"] for row in state["choices"] if row["layer"] == selected
        )
        selected_left, selected_right = read_update_pair(item, pairs, selected)
        selected_mapping = fit_common_space(
            selected_left, selected_right, state["roles"]["fit"], selected_ridge
        )
        selected_left, selected_right = transform_common(
            selected_mapping, selected_left, selected_right
        )
        selected_test_mrr, _ = selection_mrr(
            selected_left, selected_right, state["roles"]["test"]
        )
        if labels is None:
            rows.append(
                {
                    "cohort": item["cohort"],
                    "model": item["model"],
                    "fold": state["fold"],
                    "target": "geometry_only",
                    "label_free_transition": selected,
                    "cca_ridge": selected_ridge,
                    "test_mrr": selected_test_mrr,
                }
            )
            continue

        for direction in (0, 1):
            valid = [row for row in state["oracle"][direction] if np.isfinite(row["score"])]
            if not valid:
                continue
            oracle = max(valid, key=lambda row: row["score"])["layer"]
            predictions = []
            scores = []
            alphas = []
            # Refit each chosen transition on fit rows and evaluate both on the same final test rows, exactly as in E4.
            for transition in (selected, oracle):
                left, right = read_update_pair(item, pairs, transition)
                cca_ridge = next(
                    row["cca_ridge"]
                    for row in state["choices"]
                    if row["layer"] == transition
                )
                mapping = fit_common_space(
                    left, right, state["roles"]["fit"], cca_ridge
                )
                left_common, right_common = transform_common(mapping, left, right)
                source, target = (
                    (left_common, right_common),
                    (right_common, left_common),
                )[direction]
                score, alpha, y, prediction, test_rows = transfer_score(
                    source,
                    target,
                    labels,
                    state["roles"],
                    return_predictions=True,
                )
                scores.append(score)
                alphas.append(alpha)
                predictions.append((y, prediction, test_rows))
            low, high = paired_bootstrap_regret(
                predictions[0][0],
                predictions[0][1],
                predictions[1][1],
                bootstraps,
                seed + state["fold"],
            )
            direction_name = (
                "image_to_spectrum" if direction == 0 else "spectrum_to_image"
            )
            prediction_dir = Path(output_root) / "E5_spectrum_predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            saved_rows = predictions[0][2]
            if item.get("row_indices") is not None:
                saved_rows = np.asarray(item["row_indices"])[saved_rows]
            # Save row-level outputs because regenerating this common space is one of the more expensive parts of the whole pipeline.
            np.savez(
                prediction_dir
                / f"{item['cohort']}__{item['model']}__redshift__{direction_name}__fold{state['fold']}.npz",
                row_indices=saved_rows,
                y=predictions[0][0],
                label_free_prediction=predictions[0][1],
                oracle_prediction=predictions[1][1],
            )
            rows.append(
                {
                    "cohort": item["cohort"],
                    "model": item["model"],
                    "fold": state["fold"],
                    "target": "redshift",
                    "direction": direction_name,
                    "test_mrr": selected_test_mrr,
                    "label_free_transition": selected,
                    "oracle_transition": oracle,
                    "label_free_alpha": alphas[0],
                    "oracle_alpha": alphas[1],
                    "label_free_test_r2": scores[0],
                    "oracle_test_r2": scores[1],
                    "regret": scores[0] - scores[1],
                    "regret_low": low,
                    "regret_high": high,
                }
            )
    return rows


# -----------------------------------------------------------------------------
# Update geometry and relation to the full-state curve
#
# Intention: calculate E1 geometry for every compatible transition, then ask whether highly aligned updates coincide with increases in full-state alignment. Geometry is seconds to minutes per transition. The 9,999 shared permutations can add minutes to hours per model/cohort.
# -----------------------------------------------------------------------------

def run_file(item, output_dir, permutations, bootstraps, seed):
    pairs = layer_pairs(item["path"], item["cohort"])
    full_curve_file = Path(output_dir) / "E1" / item["cohort"] / item["model"] / "curve.csv"
    full_curve = pd.read_csv(full_curve_file) if full_curve_file.exists() else None
    rows = []
    skipped = []
    row_curves = []
    neighbor_curves = []

    # Keep the current layer in memory so the next transition normally requires only one newly decoded layer per view.
    left_now = read_item_matrix(item, pairs[0][0])
    if item["cohort"] in ("desi", "sdss"):
        right_now = read_item_matrix(item, pairs[0][1])
    else:
        right_now = read_item_matrix(item, pairs[0][1])

    for transition in range(len(pairs) - 1):
        left_next = read_item_matrix(item, pairs[transition + 1][0])
        if not compatible_transition(pairs, transition):
            skipped.append(
                {
                    "cohort": item["cohort"],
                    "model": item["model"],
                    "transition": transition,
                    "normalized_depth": (transition + 0.5) / max(1, len(pairs) - 1),
                    "from_layer": pairs[transition][2],
                    "to_layer": pairs[transition + 1][2],
                    "available": False,
                    "skip_reason": "different_block_or_pooling_definition",
                }
            )
            left_now = left_next
            if item["cohort"] not in ("desi", "sdss"):
                right_now = read_item_matrix(item, pairs[transition + 1][1])
            continue
        if left_now.shape[1] != left_next.shape[1]:
            skipped.append(
                {
                    "cohort": item["cohort"],
                    "model": item["model"],
                    "transition": transition,
                    "normalized_depth": (transition + 0.5) / max(1, len(pairs) - 1),
                    "from_layer": pairs[transition][2],
                    "to_layer": pairs[transition + 1][2],
                    "available": False,
                    "skip_reason": "left_width_changed",
                }
            )
            left_now = left_next
            if item["cohort"] not in ("desi", "sdss"):
                right_now = read_item_matrix(item, pairs[transition + 1][1])
            continue
        left_update = left_next - left_now

        if item["cohort"] in ("desi", "sdss"):
            right_update = right_now
            right_description = "fixed_spectrum"
            right_next = right_now
        else:
            right_next = read_item_matrix(item, pairs[transition + 1][1])
            if right_now.shape[1] != right_next.shape[1]:
                skipped.append(
                    {
                        "cohort": item["cohort"],
                        "model": item["model"],
                        "transition": transition,
                        "normalized_depth": (transition + 0.5) / max(1, len(pairs) - 1),
                        "from_layer": pairs[transition][2],
                        "to_layer": pairs[transition + 1][2],
                        "available": False,
                        "skip_reason": "right_width_changed",
                    }
                )
                left_now = left_next
                right_now = right_next
                continue
            right_update = right_next - right_now
            right_description = "consecutive_update"

        # This is the expensive E1-style measurement on the two update arrays: exact neighbors, overlap, CKA, Procrustes, and possible retrieval.
        numbers, per_row, left_neighbors, right_neighbors = geometry(
            left_update, right_update
        )
        numbers.update(
            cohort=item["cohort"],
            model=item["model"],
            transition=transition,
            normalized_depth=(transition + 0.5) / max(1, len(pairs) - 1),
            from_layer=pairs[transition][2],
            to_layer=pairs[transition + 1][2],
            right_definition=right_description,
            available=True,
            skip_reason="",
        )
        # Pair the update statistic with the observed change between the two corresponding points on E1's complete-representation curve.
        if full_curve is not None and transition + 1 < len(full_curve):
            before = full_curve.iloc[transition]["corrected_overlap_20"]
            after = full_curve.iloc[transition + 1]["corrected_overlap_20"]
            numbers["full_state_change"] = after - before
            numbers["full_state_after"] = after
        rows.append(numbers)
        row_curves.append(per_row)
        neighbor_curves.append((left_neighbors, right_neighbors))
        left_now = left_next
        right_now = right_next

    # One shared set of permutations tests the maximum update alignment while respecting the search over all compatible transitions.
    if rows and len({len(values) for values in row_curves}) == 1:
        p_value, low, high, peak = curve_inference(
            neighbor_curves,
            row_curves,
            permutations,
            bootstraps,
            seed,
            permutation_sample_size=8192
            if item["cohort"] == "legacysurvey"
            else None,
        )
        for row, lo, hi, probability in zip(rows, low, high, peak):
            row["curve_p_value"] = p_value
            row["simultaneous_low"] = lo
            row["simultaneous_high"] = hi
            row["peak_probability"] = probability

    frame = pd.DataFrame(rows + skipped).sort_values("transition").reset_index(drop=True)
    if not frame.empty and full_curve is not None:
        full_max = full_curve["corrected_overlap_20"].max()
        update_max = frame.get("corrected_overlap_20", pd.Series(dtype=float)).max()
        frame["maximum_full_alignment"] = full_max
        frame["maximum_update_alignment"] = update_max
        frame["update_to_full_ratio"] = update_max / full_max if full_max != 0 else np.nan
        # Permuting full-state changes tests whether the update/full-state correlation is stronger than arbitrary transition matching. The bootstrap supplies its interval. These small vectors usually take seconds even with 9,999 permutations.
        if "full_state_change" in frame:
            paired = frame[["corrected_overlap_20", "full_state_change"]].dropna()
            if len(paired) >= 3:
                observed = safe_correlation(
                    paired["corrected_overlap_20"], paired["full_state_change"]
                )
                rng = np.random.default_rng(seed)
                null = []
                for _ in range(permutations):
                    shuffled = rng.permutation(paired["full_state_change"].to_numpy())
                    value = safe_correlation(paired["corrected_overlap_20"], shuffled)
                    if np.isfinite(value):
                        null.append(value)
                draws = []
                for _ in range(bootstraps):
                    picked = rng.integers(0, len(paired), len(paired))
                    value = safe_correlation(
                        paired["corrected_overlap_20"].to_numpy()[picked],
                        paired["full_state_change"].to_numpy()[picked],
                    )
                    if np.isfinite(value):
                        draws.append(value)
                p_value = np.nan
                if null and np.isfinite(observed):
                    p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (
                        len(null) + 1
                    )
                frame["update_change_correlation"] = observed
                frame["update_change_p_value"] = p_value
                if draws:
                    low, high = np.quantile(draws, [0.025, 0.975])
                    frame["update_change_low"] = low
                    frame["update_change_high"] = high
    return frame


# -----------------------------------------------------------------------------
# Worker scheduling and final summaries
#
# Intention: run all three update analyses for each item inside one assigned worker, then write detailed and family-averaged CSV files. Scheduling and CSV aggregation take seconds to minutes; the geometry and transfer routines above account for almost all wall time.
# -----------------------------------------------------------------------------

def _run_group(arguments):
    (
        gpu_id,
        items,
        output_root,
        old_targets,
        permutations,
        bootstraps,
        seed,
        cpu_threads,
    ) = arguments
    configure_worker(gpu_id, cpu_threads)
    rows = []
    image_transfer_rows = []
    spectrum_transfer_rows = []
    # Keeping an item's three analyses in the same worker gives it one stable GPU assignment and avoids concurrent work on that GPU.
    for item in items:
        print(f"E5  GPU {gpu_id}  {item['cohort']}/{item['model']}", flush=True)
        frame = run_file(item, output_root, permutations, bootstraps, seed)
        rows.extend(frame.to_dict("records"))
        if item["cohort"] in ("legacysurvey", "jwst"):
            image_transfer_rows.extend(
                image_update_transfer(
                    item, old_targets[item["cohort"]], bootstraps, seed, output_root
                )
            )
        elif item["cohort"] in ("desi", "sdss"):
            labels = None
            if item["cohort"] == "sdss":
                labels = old_targets["sdss"]["redshift"]
            spectrum_transfer_rows.extend(
                spectrum_update_transfer(item, labels, bootstraps, seed, output_root)
            )
    return rows, image_transfer_rows, spectrum_transfer_rows


def run(context):
    items = context["layerwise_files"]
    workers = min(context.get("workers", 1), len(items))
    groups = [items[number::workers] for number in range(workers)]
    arguments = [
        (
            number if number < context.get("gpu_count", 0) else None,
            group,
            context["output_root"],
            context["old_targets"],
            context["permutations"],
            context["bootstraps"],
            context["seed"],
            context.get("cpu_threads_per_worker", 1),
        )
        for number, group in enumerate(groups)
    ]
    # Spawn creates independent FAISS/CUDA and numerical-library state.
    if workers == 1:
        returned = [_run_group(arguments[0])]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            returned = list(pool.map(_run_group, arguments))
    rows = [row for group_rows, _, _ in returned for row in group_rows]
    image_rows = [row for _, group_rows, _ in returned for row in group_rows]
    spectrum_rows = [row for _, _, group_rows in returned for row in group_rows]
    result = pd.DataFrame(rows)
    if not result.empty:
        result["family"] = result["model"].map(context["model_family"])
        for column in (
            "maximum_update_alignment",
            "update_change_correlation",
            "update_change_p_value",
            "curve_p_value",
        ):
            if column not in result:
                result[column] = np.nan
        tests = result.groupby(["cohort", "model"], as_index=False)[
            "curve_p_value"
        ].first()
        tests["curve_p_value_holm"] = holm_adjust(tests["curve_p_value"])
        result = result.merge(
            tests[["cohort", "model", "curve_p_value_holm"]],
            on=["cohort", "model"],
            how="left",
        )
    result.to_csv(Path(context["output_root"]) / "E5_updates.csv", index=False)
    image_frame = pd.DataFrame(image_rows)
    spectrum_frame = pd.DataFrame(spectrum_rows)
    for frame in (image_frame, spectrum_frame):
        if not frame.empty:
            frame["family"] = frame["model"].map(context["model_family"])
    image_frame.to_csv(
        Path(context["output_root"]) / "E5_image_update_transfer.csv", index=False
    )
    spectrum_frame.to_csv(
        Path(context["output_root"]) / "E5_spectrum_update_transfer.csv", index=False
    )
    # Combine image and spectrum transfer only for cross-family summaries; the two detailed result files remain separate and easier to inspect.
    transfer_frames = [frame for frame in (image_frame, spectrum_frame) if not frame.empty]
    if transfer_frames:
        transfer = pd.concat(transfer_frames, ignore_index=True)
        transfer_family = (
            transfer.dropna(subset=["direction"])
            .groupby(["cohort", "family", "target", "direction"], as_index=False)
            .agg(models=("model", "nunique"), mean_regret=("regret", "mean"))
        )
        transfer_family.to_csv(
            Path(context["output_root"]) / "E5_transfer_family_summary.csv",
            index=False,
        )
        rng = np.random.default_rng(context["seed"])
        inference = []
        for keys, group in transfer_family.groupby(
            ["cohort", "target", "direction"]
        ):
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
            Path(context["output_root"]) / "E5_transfer_cross_family_inference.csv",
            index=False,
        )
    if not result.empty:
        # Each model contributes one value before checkpoints are averaged into families, avoiding accidental overweighting of large model lineages.
        model_summary = (
            result.groupby(["cohort", "model", "family"], as_index=False)
            .agg(
                maximum_update_alignment=("maximum_update_alignment", "first"),
                update_change_correlation=("update_change_correlation", "first"),
                update_change_p_value=("update_change_p_value", "first"),
            )
        )
        family_summary = (
            model_summary.groupby(["cohort", "family"], as_index=False)
            .agg(
                models=("model", "nunique"),
                mean_maximum_update_alignment=("maximum_update_alignment", "mean"),
                mean_update_change_correlation=("update_change_correlation", "mean"),
            )
        )
        family_summary.to_csv(
            Path(context["output_root"]) / "E5_family_summary.csv", index=False
        )
    return result
