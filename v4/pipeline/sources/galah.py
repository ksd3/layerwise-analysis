"""GALAH spectra: download individual FITS via SSA from Data Central Australia."""

import os
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.coordinates import SkyCoord
from astropy import units as u

from .base import DataSource
from ..config import PipelineConfig, GALAH_CATALOG_URL, GALAH_SSA_URL, SURVEY_EPOCHS
from ..utils.download import download_file
from ..utils.crossmatch import propagate_coords


GALAH_BANDS = ["B", "G", "R", "I"]


class GALAHSource(DataSource):
    name = "galah"

    def preflight(self) -> tuple[bool, str]:
        try:
            resp = requests.head(GALAH_CATALOG_URL, timeout=15)
            return resp.status_code == 200, "GALAH catalog reachable"
        except Exception as e:
            return False, str(e)

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        from astropy.io import fits as afits

        # Step 1: Download GALAH catalog and crossmatch
        galah_cat_path = os.path.join(self.config.work_dir, "galah_dr3_catalog.fits")
        if not os.path.exists(galah_cat_path):
            print("    Downloading GALAH DR3 catalog...")
            download_file(GALAH_CATALOG_URL, galah_cat_path)

        # Validate file isn't truncated (should be ~873 MB)
        if os.path.exists(galah_cat_path):
            fsize = os.path.getsize(galah_cat_path)
            if fsize < 500_000_000:
                print(f"    WARN: GALAH catalog truncated ({fsize / 1e6:.0f} MB). Deleting and retrying...")
                os.remove(galah_cat_path)
                try:
                    download_file(GALAH_CATALOG_URL, galah_cat_path)
                    fsize = os.path.getsize(galah_cat_path)
                    if fsize < 500_000_000:
                        print(f"    GALAH catalog still truncated. Skipping GALAH.")
                        return 0
                except Exception as e:
                    print(f"    GALAH catalog download failed: {e}. Skipping.")
                    return 0
            else:
                print(f"    GALAH catalog cached ({fsize / 1e6:.0f} MB)")

        with afits.open(galah_cat_path) as hdul:
            gdata = hdul[1].data
            galah_ra = gdata["ra_dr2"].astype(np.float64)
            galah_dec = gdata["dec_dr2"].astype(np.float64)
            galah_sobject_ids = np.array(gdata["sobject_id"], dtype=np.int64)

        print(f"    GALAH catalog: {len(galah_sobject_ids)} stars")

        # Propagate catalog positions to GALAH epoch (~2015.5, Gaia DR2 coordinates)
        survey_epoch = SURVEY_EPOCHS.get("galah", 2018.0)
        cat_coords = propagate_coords(catalog_df, survey_epoch, self.config.gaia_ref_epoch)
        galah_coords = SkyCoord(ra=galah_ra, dec=galah_dec, unit="deg")
        print(f"    Epoch propagation: {self.config.gaia_ref_epoch} → {survey_epoch}")

        idx, sep, _ = galah_coords.match_to_catalog_sky(cat_coords)
        match_mask = sep < self.config.match_radius_arcsec * u.arcsec

        sobject_to_catoid = {}
        for gi in range(len(galah_sobject_ids)):
            if match_mask[gi]:
                sid = galah_sobject_ids[gi]
                cat_oid = str(catalog_df.iloc[idx[gi]]["object_id"])
                sobject_to_catoid[sid] = cat_oid

        print(f"    {len(sobject_to_catoid)} GALAH stars matched to catalog")

        if not sobject_to_catoid:
            print("    No GALAH matches -- skipping")
            return 0

        # Step 2: Download individual spectra via SSA service
        # No shared session — requests.Session is NOT thread-safe
        results = {}
        shard_idx = 0
        n_downloaded = 0
        n_failed = 0
        matched_items = list(sobject_to_catoid.items())

        if self.config.test_mode:
            matched_items = matched_items[:5]

        def _fetch_star(sid_oid):
            """Download all 4 bands for one star via SSA (DR4), return assembled spectrum."""
            sid, cat_oid = sid_oid
            all_flux, all_lambda = [], []

            for band in GALAH_BANDS:
                url = (f"{GALAH_SSA_URL}?ID={sid}&DR=galah_dr4"
                       f"&IDX=0&FILT={band}&RESPONSEFORMAT=fits")
                try:
                    resp = requests.get(url, timeout=60)
                    if resp.status_code != 200 or len(resp.content) < 1000:
                        return None
                    with afits.open(BytesIO(resp.content), memmap=False) as hdul:
                        flux = hdul[0].data
                        if flux is None:
                            return None
                        start_wl = hdul[0].header["CRVAL1"]
                        disp = hdul[0].header["CDELT1"]
                        npix = hdul[0].header["NAXIS1"]
                        refpix = hdul[0].header.get("CRPIX1", 1)
                        if refpix == 0:
                            refpix = 1
                        wl = (np.arange(npix) - refpix + 1) * disp + start_wl

                        all_flux.append(flux.astype(np.float32))
                        all_lambda.append(wl.astype(np.float32))
                except Exception:
                    return None

            if len(all_flux) == 4:
                band_names = ["blue", "green", "red", "ir"]
                data = {}
                for bi, bname in enumerate(band_names):
                    data[f"galah_flux_{bname}"] = all_flux[bi]
                    data[f"galah_lambda_{bname}"] = all_lambda[bi]
                return (cat_oid, data)
            return None

        print(f"    Downloading spectra for {len(matched_items)} stars via SSA (20 threads)...")

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_fetch_star, item): item for item in matched_items}
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result is not None:
                        cat_oid, data = result
                        results[cat_oid] = data
                        n_downloaded += 1
                    else:
                        n_failed += 1
                except Exception:
                    n_failed += 1

                if (i + 1) % 5000 == 0:
                    print(f"    {i + 1}/{len(matched_items)} processed, "
                          f"{n_downloaded} OK, {n_failed} failed")

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

        print(f"    Total: {n_downloaded} GALAH spectra downloaded, {n_failed} failed")
        return n_downloaded
