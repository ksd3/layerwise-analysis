"""Star catalog: MAST allStar FITS → SNR>50 → Gaia XMatch."""

import os
import time
import numpy as np
import pandas as pd
from astropy import units as u

from ..config import PipelineConfig, ALLSTAR_URL
from ..utils.download import download_file


def build_star_catalog(config: PipelineConfig) -> pd.DataFrame:
    outpath = os.path.join(config.work_dir, "star_catalog.parquet")
    if os.path.exists(outpath):
        print("  Star catalog exists, loading...")
        return pd.read_parquet(outpath)

    from astropy.io import fits as afits
    from astroquery.xmatch import XMatch
    from astropy.table import Table
    from astropy.coordinates import SkyCoord as SC

    # Step 1: Download allStar from MAST (or use cached)
    allstar_path = os.path.join(config.work_dir, "allStar-dr17-synspec_rev1.fits")
    if not os.path.exists(allstar_path):
        print("    Downloading allStar from MAST (~4 GB)...")
        start = time.time()
        download_file(ALLSTAR_URL, allstar_path)
        elapsed = time.time() - start
        size = os.path.getsize(allstar_path)
        print(f"    Done: {size / 1e9:.1f} GB in {elapsed:.0f}s ({size / elapsed / 1e6:.1f} MB/s)")
    else:
        print("    allStar already cached")

    # Step 2: Read allStar and filter to SNR>50
    print("  Reading allStar FITS and filtering SNR>50...")
    with afits.open(allstar_path, memmap=True) as hdul:
        data = hdul[1].data
        snr = data["SNR"]
        mask = snr > 50

        ra = data["RA"][mask].astype(np.float64)
        dec = data["DEC"][mask].astype(np.float64)
        apogee_id = np.array(data["APOGEE_ID"][mask], dtype=str)
        teff = data["TEFF"][mask].astype(np.float32)
        logg = data["LOGG"][mask].astype(np.float32)
        snr_vals = snr[mask].astype(np.float32)
        field_vals = np.array(data["FIELD"][mask], dtype=str)

        # Store Gaia proper motions for epoch propagation
        col_names = [c.name for c in data.columns]
        if "GAIAEDR3_PMRA" in col_names:
            pmra = data["GAIAEDR3_PMRA"][mask].astype(np.float64)
            pmdec = data["GAIAEDR3_PMDEC"][mask].astype(np.float64)
        elif "GAIA_PMRA" in col_names:
            pmra = data["GAIA_PMRA"][mask].astype(np.float64)
            pmdec = data["GAIA_PMDEC"][mask].astype(np.float64)
        else:
            print("    WARNING: No Gaia proper motions in allStar — epoch propagation disabled")
            pmra = np.full(mask.sum(), np.nan)
            pmdec = np.full(mask.sum(), np.nan)

    apogee_id = np.array([s.strip() for s in apogee_id])
    field_vals = np.array([s.strip() for s in field_vals])

    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "object_id": apogee_id,
        "apogee_teff": teff, "apogee_logg": logg,
        "apogee_snr": snr_vals, "apogee_field": field_vals,
        "pmra": pmra, "pmdec": pmdec,
        "population": "star",
    })
    df = df[(df["ra"] > 0) & (df["ra"] < 360) & (df["dec"] > -90) & (df["dec"] < 90)]
    print(f"    {len(df)} APOGEE rows with SNR>50")

    # Deduplicate: allStar can have multiple rows per star (different fields/telescopes).
    # Keep the highest-SNR observation of each physical star.
    n_before = len(df)
    df = df.sort_values("apogee_snr", ascending=False).drop_duplicates(subset=["object_id"], keep="first")
    print(f"    {n_before - len(df)} duplicates removed (kept highest SNR), {len(df)} unique stars")

    if config.test_mode:
        df = df.head(200)

    # Step 3: XMatch against Gaia DR3
    print("  XMatch vs Gaia DR3 (chunked)...")
    chunk_size = 50000
    all_xm_ras, all_xm_decs = [], []
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end]
        print(f"    Chunk {chunk_start // chunk_size + 1}: rows {chunk_start}-{chunk_end}...")
        for attempt in range(3):
            try:
                xm = XMatch.query(
                    cat1=Table.from_pandas(chunk_df[["ra", "dec"]]),
                    cat2="vizier:I/355/gaiadr3",
                    max_distance=config.match_radius_arcsec * u.arcsec,
                    colRA1="ra", colDec1="dec",
                )
                if len(xm) > 0:
                    all_xm_ras.extend(xm["ra"].data.tolist())
                    all_xm_decs.extend(xm["dec"].data.tolist())
                print(f"      {len(xm)} matches")
                break
            except Exception as e:
                print(f"      Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)

    print(f"    Total Gaia matches: {len(all_xm_ras)}")
    if not all_xm_ras and not config.test_mode:
        raise RuntimeError(
            "CDS XMatch returned 0 Gaia matches. Service may be down. "
            "Cannot build star catalog without Gaia verification."
        )
    if all_xm_ras:
        xm_coords = SC(ra=all_xm_ras, dec=all_xm_decs, unit="deg")
        df_coords = SC(ra=df["ra"].values, dec=df["dec"].values, unit="deg")
        idx, sep, _ = df_coords.match_to_catalog_sky(xm_coords)
        mask = sep < config.match_radius_arcsec * u.arcsec
        df = df[mask]
        print(f"    {len(df)} stars with Gaia match")

    if config.n_stars > 0 and len(df) > config.n_stars:
        df = df.sample(n=config.n_stars, random_state=config.random_seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df.to_parquet(outpath, index=False)
    print(f"    Saved {len(df)} stars")
    return df
