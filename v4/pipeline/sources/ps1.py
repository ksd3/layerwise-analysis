"""Legacy Survey images: threaded cutouts from legacysurvey.org (replaces PS1)."""

import os
import time
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import DataSource
from ..config import PipelineConfig, LEGACY_SURVEY_CUTOUT_URL, SURVEY_EPOCHS
from ..utils.crossmatch import propagate_coords


class LegacySurveySource(DataSource):
    name = "legacy_survey"

    def preflight(self) -> tuple[bool, str]:
        try:
            # Test with a well-known position outside galactic plane
            url = (f"{LEGACY_SURVEY_CUTOUT_URL}"
                   f"?ra=180&dec=0&size=10&layer=ls-dr10&bands=grz")
            resp = requests.get(url, timeout=30)
            ok = resp.status_code == 200 and len(resp.content) > 200
            return ok, "Legacy Survey cutout service OK" if ok else "bad response"
        except Exception as e:
            return False, str(e)

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        from astropy.io import fits as afits

        n = len(catalog_df)
        # No shared session — requests.Session is NOT thread-safe
        results = {}
        shard_idx = 0
        n_ok = 0
        n_fail = 0

        # Propagate positions to Legacy Survey epoch
        survey_epoch = SURVEY_EPOCHS.get("legacy_survey", 2017.0)
        propagated = propagate_coords(catalog_df, survey_epoch, self.config.gaia_ref_epoch)
        ras = propagated.ra.deg
        decs = propagated.dec.deg
        oids = catalog_df["object_id"].values.astype(str)

        tasks = list(zip(oids, ras, decs))
        if self.config.test_mode:
            tasks = tasks[:10]

        def _fetch_cutout(task):
            """Download a 3-band (g,r,z) cutout from Legacy Survey."""
            oid, ra, dec = task
            url = (f"{LEGACY_SURVEY_CUTOUT_URL}"
                   f"?ra={ra}&dec={dec}&size=64&layer=ls-dr10&bands=grz")
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code != 200 or len(resp.content) < 200:
                    # Small response = no coverage (galactic plane)
                    return (oid, None)
                with afits.open(BytesIO(resp.content), memmap=False) as hdul:
                    data = hdul[0].data  # shape (3, 64, 64) for g, r, z
                    if data is None or data.shape != (3, 64, 64):
                        return (oid, None)
                    return (oid, {
                        "legacy_g": data[0].astype(np.float32),
                        "legacy_r": data[1].astype(np.float32),
                        "legacy_z": data[2].astype(np.float32),
                    })
            except Exception:
                return (oid, None)

        print(f"    Downloading Legacy Survey cutouts for {len(tasks)} objects "
              f"({self.config.legacy_survey_workers} threads)...")
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.config.legacy_survey_workers) as pool:
            futures = {pool.submit(_fetch_cutout, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures)):
                try:
                    oid, data = future.result()
                    if data is not None:
                        results[oid] = data
                        n_ok += 1
                    else:
                        n_fail += 1
                except Exception:
                    n_fail += 1

                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    remaining = (len(tasks) - i - 1) / rate / 3600 if rate > 0 else 0
                    print(f"    {i + 1}/{len(tasks)} cutouts "
                          f"({rate:.1f}/sec, ~{remaining:.1f}h remaining), "
                          f"{n_ok} OK, {n_fail} no coverage")

                # Flush shard periodically
                if len(results) >= self.config.shard_size:
                    rows = [{"object_id": oid, **d} for oid, d in results.items()]
                    self.save_shard(pd.DataFrame(rows), shard_idx)
                    shard_idx += 1
                    results.clear()

        # Save remaining
        if results:
            rows = [{"object_id": oid, **d} for oid, d in results.items()]
            self.save_shard(pd.DataFrame(rows), shard_idx)

        elapsed = time.time() - start
        print(f"    Total: {n_ok} Legacy Survey cutouts in {elapsed / 3600:.1f}h, "
              f"{n_fail} no coverage")
        return n_ok
