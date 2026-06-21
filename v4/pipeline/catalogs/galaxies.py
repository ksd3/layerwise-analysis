"""Galaxy catalog: Flatiron PROVABGS seed."""

import os
import shutil
import numpy as np
import pandas as pd
import h5py

from ..config import PipelineConfig, FLATIRON_BASE
from ..utils.download import list_healpix_dirs, download_healpix_cell


def build_galaxy_catalog(config: PipelineConfig) -> pd.DataFrame:
    outpath = os.path.join(config.work_dir, "galaxy_catalog.parquet")
    if os.path.exists(outpath):
        print("  Galaxy catalog exists, loading...")
        return pd.read_parquet(outpath)

    print("  Building galaxy catalog from Flatiron PROVABGS...")
    provabgs_url = f"{FLATIRON_BASE}/desi_provabgs/"
    provabgs_dir = os.path.join(config.work_dir, "seeds", "desi_provabgs")

    hp_dirs = list_healpix_dirs(provabgs_url + "datafiles/")
    print(f"    {len(hp_dirs)} PROVABGS healpix cells")

    if config.test_mode:
        hp_dirs = dict(list(hp_dirs.items())[:2])

    all_dfs = []
    for hp, (hp_url, dir_name) in hp_dirs.items():
        local_files, _ = download_healpix_cell(provabgs_url, "datafiles", hp, provabgs_dir, dir_name)
        for lf in local_files:
            try:
                with h5py.File(lf, "r") as f:
                    keys_lower = {k.lower(): k for k in f.keys()}
                    ra_key = keys_lower.get("ra")
                    dec_key = keys_lower.get("dec")
                    oid_key = keys_lower.get("object_id")
                    if ra_key and dec_key and oid_key:
                        chunk = pd.DataFrame({
                            "ra": f[ra_key][:],
                            "dec": f[dec_key][:],
                            "object_id": f[oid_key][:].astype(str),
                        })
                        all_dfs.append(chunk)
            except Exception as e:
                print(f"    WARN: {lf}: {e}")

    if not all_dfs:
        raise RuntimeError("Failed to read any PROVABGS data")

    df = pd.concat(all_dfs, ignore_index=True)
    df["population"] = "galaxy"
    df["pmra"] = 0.0   # Extragalactic — no proper motion
    df["pmdec"] = 0.0
    df = df.dropna(subset=["ra", "dec"])
    if config.n_galaxies > 0 and len(df) > config.n_galaxies:
        df = df.sample(n=config.n_galaxies, random_state=config.random_seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df.to_parquet(outpath, index=False)
    print(f"    Saved {len(df)} galaxies (from PROVABGS)")

    try:
        shutil.rmtree(provabgs_dir, ignore_errors=True)
    except Exception:
        pass

    return df
