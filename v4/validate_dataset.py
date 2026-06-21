#!/usr/bin/env python3
"""Validate the final MMU v3 dataset before uploading to HuggingFace.

Streaming version — reads one shard at a time, ~2 GB RAM max.

Usage:
    python3 validate_dataset.py <YOUR_WORK_DIR>/mmu_v3/shards/final/
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

EXPECTED_POPULATIONS = {"star", "galaxy", "agn"}
EXPECTED_COUNTS = {"star": 300_000, "galaxy": 200_000, "agn": 100_000}

IMAGE_COLS = [
    "legacy_g", "legacy_r", "legacy_z",
    "galex_fuv", "galex_nuv",
    "twomass_j", "twomass_h", "twomass_k",
    "unwise_w1", "unwise_w2",
]
SPECTRUM_COLS = ["apogee_flux", "apogee_flux_err", "galah_flux", "galah_lambda",
                 "sdss_flux", "sdss_loglam", "sdss_ivar"]
LC_COLS = ["ztf_time", "ztf_mag", "ztf_magerr", "ztf_band"]
ARRAY_COLS = IMAGE_COLS + SPECTRUM_COLS + LC_COLS

CROSS_CHECKS = [
    ("galaxy", "apogee_flux", "Galaxies should not have APOGEE"),
    ("galaxy", "ztf_time", "Galaxies should not have ZTF"),
    ("galaxy", "twomass_j", "Galaxies should not have 2MASS"),
]


def _is_data(x):
    return isinstance(x, (list, np.ndarray))


def _image_shape_ok(v):
    if isinstance(v, np.ndarray):
        if v.shape == (64, 64):
            return True
        # pyarrow returns list<list<double>> as an object array of length 64
        # whose elements are length-64 arrays — accept that as a valid 64x64.
        if v.dtype == object and v.shape == (64,):
            try:
                return all(len(r) == 64 for r in v)
            except TypeError:
                return False
        return False
    if isinstance(v, list):
        return len(v) == 64 and len(v[0]) == 64
    return False


def stream_shards(shard_dir):
    """Yield (file_path, DataFrame) for each shard."""
    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not files:
        print(f"ERROR: No parquet files found in {shard_dir}")
        sys.exit(1)
    print(f"Found {len(files)} shards in {shard_dir}\n")
    for f in files:
        try:
            yield f, pd.read_parquet(f)
        except Exception as e:
            print(f"  ERROR reading {os.path.basename(f)}: {e}")


def validate(shard_dir):
    """Single-pass streaming validation."""

    # ── Accumulators (all lightweight: counters, not data) ──
    total_rows = 0
    n_shards = 0
    all_columns = set()
    pop_counts = defaultdict(int)
    object_ids = set()
    n_duplicates = 0

    # Coordinates
    ra_min, ra_max = 999, -999
    dec_min, dec_max = 999, -999
    n_ra_bad = 0
    n_dec_bad = 0
    n_coord_nan = 0

    # Per-population, per-column coverage
    # {pop: {col: count_with_data}}
    coverage = defaultdict(lambda: defaultdict(int))

    # Modality counts per population
    # {pop: {"n_spectra>=1": count, ...}}
    modality_counts = defaultdict(lambda: defaultdict(int))

    # Array shape checks (sample first 200 per column)
    shape_samples = defaultdict(lambda: {"ok": 0, "bad": 0, "checked": 0})
    SHAPE_SAMPLE_LIMIT = 200

    # Value sanity (accumulate from first few shards)
    value_samples = defaultdict(list)
    VALUE_SAMPLE_LIMIT = 50

    # String-encoded arrays
    string_encoded = set()

    # Cross-contamination
    cross_counts = defaultdict(int)  # key = (pop, col)

    # LC length mismatches
    lc_mismatches = 0
    lc_checked = 0

    # ── Stream ──
    for fpath, df in stream_shards(shard_dir):
        n_shards += 1
        total_rows += len(df)
        all_columns.update(df.columns)

        # Population counts
        if "population" in df.columns:
            for pop, cnt in df["population"].value_counts().items():
                pop_counts[pop] += cnt

        # Duplicate object_ids
        if "object_id" in df.columns:
            for oid in df["object_id"]:
                if oid in object_ids:
                    n_duplicates += 1
                object_ids.add(oid)

        # Coordinates
        if "ra" in df.columns and "dec" in df.columns:
            ra = df["ra"].dropna()
            dec = df["dec"].dropna()
            n_coord_nan += df["ra"].isna().sum() + df["dec"].isna().sum()
            if len(ra) > 0:
                ra_min = min(ra_min, ra.min())
                ra_max = max(ra_max, ra.max())
                n_ra_bad += ((ra < 0) | (ra > 360)).sum()
            if len(dec) > 0:
                dec_min = min(dec_min, dec.min())
                dec_max = max(dec_max, dec.max())
                n_dec_bad += ((dec < -90) | (dec > 90)).sum()

        # Per-population coverage and modality counts
        if "population" in df.columns:
            for pop in df["population"].unique():
                pop_mask = df["population"] == pop
                pop_df = df[pop_mask]
                n_pop = len(pop_df)

                for col in df.columns:
                    if col in ["object_id", "ra", "dec", "population",
                               "n_spectra", "n_lightcurves", "n_images"]:
                        continue
                    if df[col].dtype == object:
                        has = pop_df[col].apply(_is_data).sum()
                        coverage[pop][col] += has
                    else:
                        has = pop_df[col].notna().sum()
                        if has < n_pop:  # only track partial coverage
                            coverage[pop][col] += has

                for count_col in ["n_spectra", "n_lightcurves", "n_images"]:
                    if count_col in pop_df.columns:
                        for thresh in [1, 2, 3]:
                            key = f"{count_col}>={thresh}"
                            modality_counts[pop][key] += (pop_df[count_col] >= thresh).sum()

                # Multimodal: >=2 types, all 3 types
                if all(c in pop_df.columns for c in ["n_spectra", "n_lightcurves", "n_images"]):
                    has_s = (pop_df["n_spectra"] > 0).astype(int)
                    has_l = (pop_df["n_lightcurves"] > 0).astype(int)
                    has_i = (pop_df["n_images"] > 0).astype(int)
                    modality_counts[pop][">=2_types"] += ((has_s + has_l + has_i) >= 2).sum()
                    modality_counts[pop]["all_3_types"] += ((has_s + has_l + has_i) == 3).sum()

        # Array shape checks (sampled)
        for col in IMAGE_COLS:
            if col not in df.columns:
                continue
            info = shape_samples[col]
            if info["checked"] >= SHAPE_SAMPLE_LIMIT:
                continue
            vals = df[col].dropna()
            vals = vals[vals.apply(_is_data)]
            remaining = SHAPE_SAMPLE_LIMIT - info["checked"]
            for v in vals.head(remaining):
                info["checked"] += 1
                if _image_shape_ok(v):
                    info["ok"] += 1
                else:
                    info["bad"] += 1

        if "apogee_flux" in df.columns:
            info = shape_samples["apogee_flux"]
            if info["checked"] < SHAPE_SAMPLE_LIMIT:
                vals = df["apogee_flux"].dropna()
                vals = vals[vals.apply(_is_data)]
                remaining = SHAPE_SAMPLE_LIMIT - info["checked"]
                for v in vals.head(remaining):
                    info["checked"] += 1
                    if len(v) == 7514:
                        info["ok"] += 1
                    else:
                        info["bad"] += 1

        # LC length consistency
        if "ztf_time" in df.columns and "ztf_mag" in df.columns:
            for _, row in df[["ztf_time", "ztf_mag"]].dropna().head(50).iterrows():
                t, m = row["ztf_time"], row["ztf_mag"]
                if _is_data(t) and _is_data(m):
                    lc_checked += 1
                    if len(t) != len(m):
                        lc_mismatches += 1

        # Value sanity (sample from early shards)
        for col in ["apogee_flux", "ztf_mag"]:
            if col in df.columns and len(value_samples[col]) < VALUE_SAMPLE_LIMIT:
                vals = df[col].dropna()
                vals = vals[vals.apply(_is_data)]
                remaining = VALUE_SAMPLE_LIMIT - len(value_samples[col])
                for v in vals.head(remaining):
                    try:
                        arr = np.array(v, dtype=np.float32).flatten()
                        value_samples[col].append(arr)
                    except (ValueError, TypeError):
                        pass

        for col in IMAGE_COLS[:3]:
            if col in df.columns and len(value_samples[col]) < 10:
                vals = df[col].dropna()
                vals = vals[vals.apply(_is_data)]
                for v in vals.head(1):
                    try:
                        value_samples[col].append(np.array(v, dtype=np.float32))
                    except (ValueError, TypeError):
                        pass

        # String-encoded arrays
        for col in df.columns:
            if col in string_encoded:
                continue
            if df[col].dtype != object:
                continue
            for val in df[col].dropna().head(5):
                if isinstance(val, str) and val.startswith("["):
                    string_encoded.add(col)
                    break

        # Cross-contamination
        for pop, col, _ in CROSS_CHECKS:
            if col not in df.columns or "population" not in df.columns:
                continue
            pop_df = df[df["population"] == pop]
            if len(pop_df) > 0:
                has = pop_df[col].apply(_is_data).sum()
                cross_counts[(pop, col)] += has

        # Progress
        if n_shards % 20 == 0:
            print(f"  ... processed {n_shards} shards ({total_rows:,} rows)")

    # ── Print Results ──
    results = {}

    print("\n" + "=" * 60)
    print("1. BASIC INTEGRITY")
    print("=" * 60)
    print(f"  Shards: {n_shards}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total columns: {len(all_columns)}")
    ok = True
    for col in ["object_id", "ra", "dec", "population"]:
        if col in all_columns:
            print(f"  OK: '{col}' present")
        else:
            print(f"  FAIL: Missing '{col}'")
            ok = False
    if n_duplicates > 0:
        print(f"  WARN: {n_duplicates} duplicate object_ids")
    else:
        print(f"  OK: No duplicate object_ids")
    results["1. Basic integrity"] = ok

    print("\n" + "=" * 60)
    print("2. POPULATION COUNTS")
    print("=" * 60)
    ok = True
    for pop in sorted(EXPECTED_POPULATIONS):
        actual = pop_counts.get(pop, 0)
        expected = EXPECTED_COUNTS[pop]
        pct = actual / expected * 100 if expected else 0
        status = "OK" if actual == expected else "WARN" if actual > expected * 0.9 else "FAIL"
        print(f"  {status}: {pop}: {actual:,} / {expected:,} ({pct:.1f}%)")
        if status == "FAIL":
            ok = False
    unexpected = set(pop_counts.keys()) - EXPECTED_POPULATIONS
    if unexpected:
        print(f"  FAIL: Unexpected populations: {unexpected}")
        ok = False
    results["2. Population counts"] = ok

    print("\n" + "=" * 60)
    print("3. COORDINATE SANITY")
    print("=" * 60)
    ok = True
    if n_ra_bad > 0:
        print(f"  FAIL: ra: {n_ra_bad} values outside [0, 360]")
        ok = False
    else:
        print(f"  OK: ra in [{ra_min:.4f}, {ra_max:.4f}]")
    if n_dec_bad > 0:
        print(f"  FAIL: dec: {n_dec_bad} values outside [-90, 90]")
        ok = False
    else:
        print(f"  OK: dec in [{dec_min:.4f}, {dec_max:.4f}]")
    if n_coord_nan > 0:
        print(f"  WARN: {n_coord_nan} NaN coordinates")
    results["3. Coordinates"] = ok

    print("\n" + "=" * 60)
    print("4. MODALITY COVERAGE (per population)")
    print("=" * 60)
    for pop in sorted(EXPECTED_POPULATIONS):
        n = pop_counts.get(pop, 0)
        if n == 0:
            continue
        print(f"\n  -- {pop.upper()} ({n:,} objects) --")
        for count_col in ["n_spectra", "n_lightcurves", "n_images"]:
            key1 = f"{count_col}>=1"
            if key1 in modality_counts[pop]:
                has = modality_counts[pop][key1]
                print(f"    {count_col} >= 1: {has:,}/{n:,} ({has/n*100:.1f}%)")
        key2 = ">=2_types"
        if key2 in modality_counts[pop]:
            v = modality_counts[pop][key2]
            print(f"    >= 2 modality types: {v:,}/{n:,} ({v/n*100:.1f}%)")
        key3 = "all_3_types"
        if key3 in modality_counts[pop]:
            v = modality_counts[pop][key3]
            print(f"    all 3 modality types: {v:,}/{n:,} ({v/n*100:.1f}%)")

        print(f"    Per-column:")
        pop_cov = coverage[pop]
        for col in sorted(pop_cov.keys()):
            has = pop_cov[col]
            if has > 0:
                print(f"      {col}: {has:,}/{n:,} ({has/n*100:.1f}%)")
    results["4. Modality coverage"] = True

    print("\n" + "=" * 60)
    print("5. ARRAY SHAPES (sampled)")
    print("=" * 60)
    ok = True
    for col, info in sorted(shape_samples.items()):
        if info["checked"] == 0:
            continue
        if info["bad"] == 0:
            expected = "7514" if col == "apogee_flux" else "64x64"
            print(f"  OK: {col}: {info['ok']}/{info['checked']} sampled = {expected}")
        else:
            print(f"  FAIL: {col}: {info['bad']}/{info['checked']} bad shapes")
            ok = False
    if lc_checked > 0:
        if lc_mismatches == 0:
            print(f"  OK: ztf_time/ztf_mag: {lc_checked} checked, lengths match")
        else:
            print(f"  FAIL: ztf_time/ztf_mag: {lc_mismatches}/{lc_checked} length mismatches")
            ok = False
    results["5. Array shapes"] = ok

    print("\n" + "=" * 60)
    print("6. DATA VALUE SANITY (sampled)")
    print("=" * 60)
    ok = True
    if "apogee_flux" in value_samples and value_samples["apogee_flux"]:
        all_vals = np.concatenate(value_samples["apogee_flux"])
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) > 0:
            med = np.median(finite)
            pct = ((finite > 0) & (finite < 3)).mean() * 100
            print(f"  apogee_flux: median={med:.3f}, {pct:.0f}% in (0,3)")
            if pct < 50:
                print(f"  WARN: apogee_flux looks unusual")
        n_inf = np.isinf(all_vals).sum()
        if n_inf > 0:
            print(f"  WARN: apogee_flux has {n_inf} inf values")

    if "ztf_mag" in value_samples and value_samples["ztf_mag"]:
        all_vals = np.concatenate(value_samples["ztf_mag"])
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) > 0:
            med = np.median(finite)
            pct = ((finite > 5) & (finite < 30)).mean() * 100
            print(f"  ztf_mag: median={med:.2f}, {pct:.0f}% in (5,30)")

    for col in IMAGE_COLS[:3]:
        if col in value_samples and value_samples[col]:
            flat = value_samples[col][0].flatten()
            n_fin = np.isfinite(flat).sum()
            n_zero = (flat == 0).sum()
            print(f"  {col}: sample image {n_fin}/{len(flat)} finite, "
                  f"{n_zero} zero, range=[{np.nanmin(flat):.2f}, {np.nanmax(flat):.2f}]")
    results["6. Data values"] = ok

    print("\n" + "=" * 60)
    print("7. STRING-ENCODED ARRAYS")
    print("=" * 60)
    if string_encoded:
        print(f"  FAIL: String-encoded arrays in: {sorted(string_encoded)}")
        results["7. String-encoded arrays"] = False
    else:
        print(f"  OK: No string-encoded arrays found")
        results["7. String-encoded arrays"] = True

    print("\n" + "=" * 60)
    print("8. CROSS-POPULATION ROUTING")
    print("=" * 60)
    ok = True
    for pop, col, reason in CROSS_CHECKS:
        count = cross_counts.get((pop, col), 0)
        n = pop_counts.get(pop, 0)
        if count > 0 and n > 0:
            print(f"  WARN: {reason} -- but {count} do ({count/n*100:.2f}%)")
        else:
            print(f"  OK: {reason}")
    results["8. Cross-population routing"] = ok

    print("\n" + "=" * 60)
    print("9. MULTIMODAL COVERAGE (key metric)")
    print("=" * 60)
    for pop in sorted(EXPECTED_POPULATIONS):
        n = pop_counts.get(pop, 0)
        if n == 0:
            continue
        print(f"\n  -- {pop.upper()} --")
        for thresh in [1, 2, 3]:
            for mod in ["n_spectra", "n_lightcurves", "n_images"]:
                key = f"{mod}>={thresh}"
                if key in modality_counts[pop]:
                    v = modality_counts[pop][key]
                    print(f"    {mod} >= {thresh}: {v:,}/{n:,} ({v/n*100:.1f}%)")
        v2 = modality_counts[pop].get(">=2_types", 0)
        v3 = modality_counts[pop].get("all_3_types", 0)
        print(f"    >= 2 modality types: {v2:,}/{n:,} ({v2/n*100:.1f}%)")
        print(f"    all 3 modality types: {v3:,}/{n:,} ({v3/n*100:.1f}%)")
    results["9. Multimodal coverage"] = True

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_ok = False
    print()
    if all_ok:
        print("  All checks passed. Safe to upload.")
    else:
        print("  Some checks failed. Review above before uploading.")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-final-shards-dir>")
        sys.exit(1)
    shard_dir = sys.argv[1]
    if not os.path.isdir(shard_dir):
        print(f"ERROR: {shard_dir} is not a directory")
        sys.exit(1)
    validate(shard_dir)
