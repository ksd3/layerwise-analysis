"""APOGEE spectra: read pre-staged aspcapStar FITS files from local disk."""

import os
import time
import glob
import numpy as np
import pandas as pd
import requests

from .base import DataSource
from ..config import PipelineConfig, ALLSTAR_URL, APOGEE_GOOD_PIXELS
from ..utils.download import download_file


class APOGEESource(DataSource):
    name = "apogee"

    def preflight(self) -> tuple[bool, str]:
        synspec = os.path.join(self.config.apogee_dir, "synspec_rev1")
        if not os.path.isdir(synspec):
            sample = glob.glob(os.path.join(self.config.apogee_dir, "**", "aspcapStar-*.fits"), recursive=True)
            if sample:
                return True, f"Found {len(sample)} aspcapStar files (non-standard layout)"
            return False, f"No synspec_rev1/ dir at {self.config.apogee_dir}"
        telescopes = [d for d in os.listdir(synspec) if os.path.isdir(os.path.join(synspec, d))]
        return bool(telescopes), f"Telescopes: {telescopes}"

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        from astropy.io import fits as afits

        # Download allStar for FIELD/TELESCOPE lookup
        allstar_path = os.path.join(self.config.work_dir, "allStar-dr17-synspec_rev1.fits")
        if not os.path.exists(allstar_path):
            print(f"    Downloading allStar from MAST (~4 GB)...")
            start = time.time()
            download_file(ALLSTAR_URL, allstar_path)
            elapsed = time.time() - start
            size = os.path.getsize(allstar_path)
            print(f"    Done: {size / 1e9:.1f} GB in {elapsed:.0f}s ({size / elapsed / 1e6:.1f} MB/s)")
        else:
            print(f"    allStar already cached")

        # Build lookup: APOGEE_ID → (FIELD, TELESCOPE)
        # Must prefer highest-SNR observation to match the catalog deduplication
        print(f"    Building APOGEE_ID lookup from allStar...")
        with afits.open(allstar_path, memmap=True) as hdul:
            data = hdul[1].data
            allstar_ids = np.array(data["APOGEE_ID"], dtype=str)
            allstar_fields = np.array(data["FIELD"], dtype=str)
            allstar_telescopes = np.array(data["TELESCOPE"], dtype=str)
            allstar_snr = data["SNR"].astype(np.float32)

        # Sort by SNR descending so first occurrence = highest SNR
        order = np.argsort(allstar_snr)[::-1]
        id_lookup = {}
        for i in order:
            key = allstar_ids[i].strip()
            if key not in id_lookup:
                id_lookup[key] = (allstar_fields[i].strip(), allstar_telescopes[i].strip())
        print(f"    {len(id_lookup)} unique APOGEE IDs in allStar")

        # Read local aspcapStar files — flush to shards incrementally
        results = []
        found = missing = errors = 0
        shard_idx = 0

        for i, row in catalog_df.iterrows():
            apogee_id = str(row["object_id"]).strip()
            lookup = id_lookup.get(apogee_id)
            if lookup is None:
                missing += 1
                continue

            field, telescope = lookup
            local_path = os.path.join(
                self.config.apogee_dir, "synspec_rev1", telescope, field,
                f"aspcapStar-dr17-{apogee_id}.fits")

            if not os.path.exists(local_path):
                missing += 1
                continue

            try:
                with afits.open(local_path, memmap=True) as hdul:
                    flux = hdul[1].data
                    flux_err = hdul[2].data if len(hdul) > 2 else None

                    if flux.ndim > 1:
                        flux = flux[0]
                    if flux_err is not None and flux_err.ndim > 1:
                        flux_err = flux_err[0]

                    cropped_flux = flux[APOGEE_GOOD_PIXELS].astype(np.float32)
                    cropped_err = (flux_err[APOGEE_GOOD_PIXELS].astype(np.float32)
                                   if flux_err is not None else None)

                    results.append({
                        "object_id": apogee_id,
                        "apogee_flux": cropped_flux,
                        "apogee_flux_err": cropped_err,
                    })
                    found += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    WARN: {local_path}: {e}")

            # Flush shard incrementally to avoid OOM
            if len(results) >= self.config.shard_size:
                self.save_shard(pd.DataFrame(results), shard_idx)
                shard_idx += 1
                results.clear()

            if (found + missing + errors) % 50000 == 0:
                print(f"    {found + missing + errors}/{len(catalog_df)} processed, {found} spectra found")

        print(f"    Found: {found}, Missing: {missing}, Errors: {errors}")

        # Save remaining
        if results:
            self.save_shard(pd.DataFrame(results), shard_idx)

        return found
