"""GALEX UV images: FUV + NUV cutouts via CDS hips2fits service.

All-sky coverage (~77% of sky) INCLUDING the galactic plane.
Uses CDS hips2fits which has much better coverage than the Legacy Survey viewer.
~64 req/sec, no rate limit.
"""

import os
import time
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import DataSource
from ..config import PipelineConfig, SURVEY_EPOCHS
from ..utils.crossmatch import propagate_coords

HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
GALEX_HIPS = {
    "fuv": "CDS/P/GALEXGR6_7/FUV",
    "nuv": "CDS/P/GALEXGR6_7/NUV",
}
GALEX_FOV = 0.027  # ~97 arcsec for 64 pixels at GALEX pixel scale (~1.5"/pix)


class GALEXImageSource(DataSource):
    name = "galex_images"

    def preflight(self) -> tuple[bool, str]:
        try:
            resp = requests.get(
                f"{HIPS2FITS_URL}?hips=CDS/P/GALEXGR6_7/NUV&ra=180&dec=0"
                f"&width=10&height=10&fov=0.005&projection=TAN",
                timeout=30)
            ok = resp.status_code == 200 and len(resp.content) > 200
            return ok, "CDS hips2fits GALEX OK" if ok else "bad response"
        except Exception as e:
            return False, str(e)

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        from astropy.io import fits as afits

        results = {}
        shard_idx = 0
        n_ok = 0
        n_no_coverage = 0

        # Propagate positions to GALEX epoch for correct cutout centering
        survey_epoch = SURVEY_EPOCHS.get("galex_images", 2007.0)
        propagated = propagate_coords(catalog_df, survey_epoch, self.config.gaia_ref_epoch)
        ras = propagated.ra.deg
        decs = propagated.dec.deg
        oids = catalog_df["object_id"].values.astype(str)
        print(f"    Epoch propagation: {self.config.gaia_ref_epoch} → {survey_epoch}")

        tasks = list(zip(oids, ras, decs))
        if self.config.test_mode:
            tasks = tasks[:10]

        def _fetch_cutout(task):
            oid, ra, dec = task
            cutouts = {}
            for band, hips_id in GALEX_HIPS.items():
                url = (f"{HIPS2FITS_URL}?hips={hips_id}"
                       f"&ra={ra}&dec={dec}&width=64&height=64"
                       f"&fov={GALEX_FOV}&projection=TAN")
                try:
                    resp = requests.get(url, timeout=60)
                    if resp.status_code != 200 or len(resp.content) < 200:
                        continue
                    with afits.open(BytesIO(resp.content), memmap=False) as hdul:
                        img = hdul[0].data
                        if img is not None and img.shape == (64, 64):
                            cutouts[f"galex_{band}"] = img.astype(np.float32)
                except Exception:
                    continue

            if cutouts:
                return (oid, cutouts)
            return (oid, None)

        n_workers = min(100, len(tasks))
        print(f"    Downloading GALEX UV cutouts for {len(tasks)} objects "
              f"via CDS hips2fits ({n_workers} threads)...")
        start = time.time()

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fetch_cutout, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures)):
                try:
                    oid, data = future.result()
                    if data is not None:
                        results[oid] = data
                        n_ok += 1
                    else:
                        n_no_coverage += 1
                except Exception:
                    n_no_coverage += 1

                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    remaining = (len(tasks) - i - 1) / rate / 3600 if rate > 0 else 0
                    print(f"    {i + 1}/{len(tasks)} cutouts "
                          f"({rate:.1f}/sec, ~{remaining:.1f}h remaining), "
                          f"{n_ok} OK, {n_no_coverage} no coverage")

                # Flush shard periodically
                if len(results) >= self.config.shard_size:
                    rows = [{"object_id": oid, **d} for oid, d in results.items()]
                    self.save_shard(pd.DataFrame(rows), shard_idx)
                    shard_idx += 1
                    results.clear()

        if results:
            rows = [{"object_id": oid, **d} for oid, d in results.items()]
            self.save_shard(pd.DataFrame(rows), shard_idx)

        elapsed = time.time() - start
        print(f"    Total: {n_ok} GALEX cutouts in {elapsed / 3600:.1f}h, "
              f"{n_no_coverage} no coverage")
        return n_ok
