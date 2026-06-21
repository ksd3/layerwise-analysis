"""AGN catalog: SDSS DR16Q FITS (pre-staged via Globus)."""

import os
import time
import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..utils.download import download_file


def _find_dr16q(config):
    """Find the DR16Q FITS file in common locations."""
    candidates = [
        os.path.join(config.work_dir, "DR16Q_v4.fits"),
        os.path.join(config.sdss_spectra_dir, "..", "DR16Q_v4.fits"),
        os.path.join(os.path.dirname(config.work_dir), "DR16Q_v4.fits"),
        os.path.join(config.apogee_dir, "..", "DR16Q_v4.fits"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(config.work_dir, "DR16Q_v4.fits")


def build_agn_catalog(config: PipelineConfig) -> pd.DataFrame:
    outpath = os.path.join(config.work_dir, "agn_catalog.parquet")
    if os.path.exists(outpath):
        print("  AGN catalog exists, loading...")
        return pd.read_parquet(outpath)

    from astropy.io import fits as afits

    dr16q_path = _find_dr16q(config)

    if not os.path.exists(dr16q_path):
        if config.test_mode:
            print("  DR16Q not found — creating minimal test catalog")
            df = pd.DataFrame({
                "ra": [180.0, 150.0, 200.0, 170.0, 190.0],
                "dec": [30.0, 25.0, 35.0, 20.0, 40.0],
                "object_id": [f"TEST_AGN_{i}" for i in range(5)],
                "agn_redshift": [1.0, 0.5, 2.0, 0.8, 1.5],
                "sdss_plate": [np.nan] * 5,
                "sdss_mjd": [np.nan] * 5,
                "sdss_fiberid": [np.nan] * 5,
                "pmra": [0.0] * 5,
                "pmdec": [0.0] * 5,
                "population": "agn",
            })
            df = df.astype({"sdss_plate": "float64", "sdss_mjd": "float64", "sdss_fiberid": "float64"})
            df.to_parquet(outpath, index=False)
            print(f"    Saved {len(df)} test AGN")
            return df
        dr16q_url = "https://data.sdss.org/sas/dr16/eboss/qso/DR16Q/DR16Q_v4.fits"
        print("  Downloading SDSS DR16Q catalog (~500 MB, may be slow from SDSS SAS)...")
        print(f"  TIP: If slow, Ctrl+C and transfer via Globus to: {dr16q_path}")
        start = time.time()
        download_file(dr16q_url, dr16q_path)
        elapsed = time.time() - start
        size = os.path.getsize(dr16q_path)
        print(f"    Done: {size / 1e6:.0f} MB in {elapsed:.0f}s")
    else:
        print(f"  DR16Q catalog found: {dr16q_path}")

    # Validate file isn't truncated
    fsize = os.path.getsize(dr16q_path)
    if fsize < 100_000_000:  # DR16Q should be >500 MB
        print(f"  WARNING: DR16Q file looks truncated ({fsize / 1e6:.0f} MB). Deleting and retrying...")
        os.remove(dr16q_path)
        if config.test_mode:
            return build_agn_catalog(config)  # Will hit test_mode branch above
        raise RuntimeError(f"DR16Q file truncated. Transfer via Globus to: {dr16q_path}")

    print("  Reading DR16Q FITS...")
    with afits.open(dr16q_path, memmap=True) as hdul:
        data = hdul[1].data
        ra = data["RA"].astype(np.float64)
        dec = data["DEC"].astype(np.float64)
        z = data["Z"].astype(np.float32)
        sdss_name = np.array(data["SDSS_NAME"], dtype=str)
        plate = data["PLATE"].astype(np.int32)
        mjd = data["MJD"].astype(np.int32)
        fiberid = data["FIBERID"].astype(np.int32)

    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "object_id": np.array([s.strip() for s in sdss_name]),
        "agn_redshift": z,
        "sdss_plate": plate, "sdss_mjd": mjd, "sdss_fiberid": fiberid,
        "population": "agn",
    })

    df = df[(df["ra"] > 0) & (df["ra"] < 360) &
            (df["dec"] > -90) & (df["dec"] < 90) &
            (df["agn_redshift"] > 0.01)]
    df["pmra"] = 0.0   # Extragalactic — no proper motion
    df["pmdec"] = 0.0
    print(f"    {len(df)} quasars in DR16Q")

    if config.test_mode:
        df = df.head(200)

    if config.n_agn > 0 and len(df) > config.n_agn:
        df = df.sample(n=config.n_agn, random_state=config.random_seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df.to_parquet(outpath, index=False)
    print(f"    Saved {len(df)} AGN (with plate/mjd/fiber for SDSS spectra)")
    return df
