#!/usr/bin/env python3
"""False match rate estimation via shifted-catalog experiment.

Offsets all catalog positions by 30" in RA and re-runs crossmatches against
modality shard data. The match rate on shifted catalogs = spurious match rate.

Reports per-source and as a function of Galactic latitude.

Usage:
    python3 false_match_test.py /path/to/workdir/

Requires: the work directory with catalogs and modality shards from the pipeline.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

SHIFT_ARCSEC = 30.0  # Offset in RA
MATCH_RADIUS = 3.0   # Same as pipeline

# Galactic latitude bins for analysis
LAT_BINS = [(0, 15, "|b| < 15° (crowded)"),
            (15, 30, "15° < |b| < 30°"),
            (30, 90, "|b| > 30° (sparse)")]


def compute_galactic_lat(ra, dec):
    """Convert equatorial to Galactic latitude."""
    coords = SkyCoord(ra=ra, dec=dec, unit="deg")
    return np.abs(coords.galactic.b.deg)


def load_catalog(work_dir, pop):
    """Load a population catalog."""
    path = os.path.join(work_dir, f"{pop}_catalog.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def load_modality_positions(work_dir, source_name):
    """Load ra/dec from modality shards."""
    shard_dir = os.path.join(work_dir, "shards", source_name)
    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not files:
        return None, None

    all_ra, all_dec = [], []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Look for ra/dec columns (may be prefixed)
            ra_col = next((c for c in df.columns if c.endswith("_ra") or c == "ra"), None)
            dec_col = next((c for c in df.columns if c.endswith("_dec") or c == "dec"), None)
            if ra_col and dec_col:
                ra = pd.to_numeric(df[ra_col], errors="coerce").dropna()
                dec_vals = pd.to_numeric(df[dec_col], errors="coerce")
                dec = dec_vals[ra.index].dropna()
                common = ra.index.intersection(dec.index)
                all_ra.extend(ra[common].values.tolist())
                all_dec.extend(dec_vals[common].values.tolist())
        except Exception:
            pass

    if not all_ra:
        return None, None
    return np.array(all_ra), np.array(all_dec)


def run_shifted_match(cat_ra, cat_dec, target_ra, target_dec, shift_arcsec, radius):
    """Run crossmatch with shifted catalog and return match fraction."""
    # Shift catalog RA by shift_arcsec
    cos_dec = np.cos(np.radians(cat_dec))
    cos_dec = np.where(cos_dec > 0.001, cos_dec, 0.001)
    shifted_ra = cat_ra + (shift_arcsec / 3600.0) / cos_dec

    cat_coords = SkyCoord(ra=shifted_ra, dec=cat_dec, unit="deg")
    tgt_coords = SkyCoord(ra=target_ra, dec=target_dec, unit="deg")

    idx, sep, _ = cat_coords.match_to_catalog_sky(tgt_coords)
    matched = sep < radius * u.arcsec
    return matched.sum() / len(cat_coords)


def run_real_match(cat_ra, cat_dec, target_ra, target_dec, radius):
    """Run crossmatch with real catalog and return separations + match fraction."""
    cat_coords = SkyCoord(ra=cat_ra, dec=cat_dec, unit="deg")
    tgt_coords = SkyCoord(ra=target_ra, dec=target_dec, unit="deg")

    idx, sep, _ = cat_coords.match_to_catalog_sky(tgt_coords)
    sep_arcsec = sep.to(u.arcsec).value
    matched = sep_arcsec < radius
    return matched.sum() / len(cat_coords), sep_arcsec[matched]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <work_dir>")
        sys.exit(1)

    work_dir = sys.argv[1]

    # Load all catalogs
    catalogs = {}
    for pop in ["star", "galaxy", "agn"]:
        cat = load_catalog(work_dir, pop)
        if cat is not None:
            catalogs[pop] = cat
            print(f"  Loaded {pop} catalog: {len(cat)} objects")

    if not catalogs:
        print("ERROR: No catalogs found")
        sys.exit(1)

    combined = pd.concat(catalogs.values(), ignore_index=True)
    cat_ra = combined["ra"].values
    cat_dec = combined["dec"].values
    gal_lat = compute_galactic_lat(cat_ra, cat_dec)

    # Find all modality shard directories
    shard_base = os.path.join(work_dir, "shards")
    if not os.path.isdir(shard_base):
        print(f"ERROR: {shard_base} not found")
        sys.exit(1)

    sources = [d for d in os.listdir(shard_base)
               if os.path.isdir(os.path.join(shard_base, d)) and d != "final"]

    print(f"\nTesting {len(sources)} sources with {SHIFT_ARCSEC}\" offset\n")
    print(f"{'Source':<25} {'Real match%':>12} {'Shifted match%':>15} {'False rate':>12}")
    print("-" * 70)

    results = []

    for source in sorted(sources):
        tgt_ra, tgt_dec = load_modality_positions(work_dir, source)
        if tgt_ra is None or len(tgt_ra) < 10:
            print(f"{source:<25} {'skip (no positions)':>40}")
            continue

        # Real match rate
        real_rate, real_seps = run_real_match(cat_ra, cat_dec, tgt_ra, tgt_dec, MATCH_RADIUS)

        # Shifted match rate (= false match rate)
        false_rate = run_shifted_match(cat_ra, cat_dec, tgt_ra, tgt_dec,
                                       SHIFT_ARCSEC, MATCH_RADIUS)

        print(f"{source:<25} {real_rate*100:>11.2f}% {false_rate*100:>14.2f}% {false_rate*100:>11.2f}%")

        # Per-latitude breakdown
        for lat_lo, lat_hi, lat_label in LAT_BINS:
            lat_mask = (gal_lat >= lat_lo) & (gal_lat < lat_hi)
            if lat_mask.sum() < 100:
                continue
            sub_ra = cat_ra[lat_mask]
            sub_dec = cat_dec[lat_mask]
            sub_false = run_shifted_match(sub_ra, sub_dec, tgt_ra, tgt_dec,
                                          SHIFT_ARCSEC, MATCH_RADIUS)
            print(f"  {lat_label:<23} {'':>12} {sub_false*100:>14.2f}%")

        results.append({
            "source": source,
            "real_match_rate": real_rate,
            "false_match_rate": false_rate,
            "n_target": len(tgt_ra),
            "median_sep_arcsec": np.median(real_seps) if len(real_seps) > 0 else np.nan,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        df = pd.DataFrame(results)
        print(f"\n  Mean false match rate: {df['false_match_rate'].mean()*100:.2f}%")
        print(f"  Max false match rate:  {df['false_match_rate'].max()*100:.2f}% ({df.loc[df['false_match_rate'].idxmax(), 'source']})")
        print(f"  Median match separation: {df['median_sep_arcsec'].median():.2f}\"")

        out_path = os.path.join(work_dir, "false_match_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
