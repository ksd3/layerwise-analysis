"""2MASS JHK images: cutouts via CDS hips2fits service.

All-sky coverage including the galactic plane. No rate limit.
~36 req/sec measured. Each request returns a single-band 64x64 FITS cutout.
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
TWOMASS_HIPS = {
    "j": "CDS/P/2MASS/J",
    "h": "CDS/P/2MASS/H",
    "k": "CDS/P/2MASS/K",
}
# FOV in degrees for 64 pixels at 2MASS pixel scale (~1"/pix)
TWOMASS_FOV = 0.018  # ~64 arcsec


class TwoMASSSource(DataSource):
    name = "twomass"

    def preflight(self) -> tuple[bool, str]:
        try:
            resp = requests.get(
                f"{HIPS2FITS_URL}?hips=CDS/P/2MASS/J&ra=180&dec=0"
                f"&width=10&height=10&fov=0.005&projection=TAN",
                timeout=30)
            ok = resp.status_code == 200 and len(resp.content) > 200
            return ok, "CDS hips2fits 2MASS OK" if ok else "bad response"
        except Exception as e:
            return False, str(e)

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        from astropy.io import fits as afits

        results = {}
        shard_idx = 0
        n_ok = 0
        n_fail = 0

        # Propagate positions to 2MASS epoch for correct cutout centering
        survey_epoch = SURVEY_EPOCHS.get("twomass", 1999.5)
        propagated = propagate_coords(catalog_df, survey_epoch, self.config.gaia_ref_epoch)
        ras = propagated.ra.deg
        decs = propagated.dec.deg
        oids = catalog_df["object_id"].values.astype(str)
        print(f"    Epoch propagation: {self.config.gaia_ref_epoch} → {survey_epoch}")

        tasks = list(zip(oids, ras, decs))
        if self.config.test_mode:
            tasks = tasks[:10]

        def _fetch_cutout(task):
            """Download J, H, K cutouts for one object via hips2fits."""
            oid, ra, dec = task
            cutouts = {}
            for band_letter, hips_id in TWOMASS_HIPS.items():
                url = (f"{HIPS2FITS_URL}?hips={hips_id}"
                       f"&ra={ra}&dec={dec}&width=64&height=64"
                       f"&fov={TWOMASS_FOV}&projection=TAN")
                try:
                    resp = requests.get(url, timeout=60)
                    if resp.status_code != 200 or len(resp.content) < 200:
                        continue
                    with afits.open(BytesIO(resp.content), memmap=False) as hdul:
                        img = hdul[0].data
                        if img is not None and img.shape == (64, 64):
                            cutouts[f"twomass_{band_letter}"] = img.astype(np.float32)
                except Exception:
                    continue

            if cutouts:
                return (oid, cutouts)
            return (oid, None)

        n_workers = min(100, len(tasks))
        print(f"    Downloading 2MASS cutouts for {len(tasks)} objects "
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
                        n_fail += 1
                except Exception:
                    n_fail += 1

                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    remaining = (len(tasks) - i - 1) / rate / 3600 if rate > 0 else 0
                    print(f"    {i + 1}/{len(tasks)} objects "
                          f"({rate:.1f}/sec, ~{remaining:.1f}h remaining), "
                          f"{n_ok} OK, {n_fail} failed")

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
        print(f"    Total: {n_ok} 2MASS cutouts in {elapsed / 3600:.1f}h, "
              f"{n_fail} failed")
        return n_ok
