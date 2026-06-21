"""SDSS spectra: read pre-staged specLite FITS files from local disk.

For AGN (DR16Q): plate/mjd/fiberid are in the catalog → direct file lookup.
For galaxies (PROVABGS): crossmatch against SpecObj to find plate/mjd/fiberid.
"""

import os
import time
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

from .base import DataSource
from ..config import PipelineConfig, SURVEY_EPOCHS
from ..utils.crossmatch import propagate_coords


def _find_specobj(config):
    """Find the SpecObj-dr17.fits file."""
    candidates = [
        os.path.join(config.work_dir, "specObj-dr17.fits"),
        os.path.join(config.sdss_spectra_dir, "..", "specObj-dr17.fits"),
        os.path.join(os.path.dirname(config.work_dir), "specObj-dr17.fits"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _find_spec_file(sdss_spectra_dir, plate, mjd, fiberid):
    """Construct path to a specLite file, checking both SDSS and BOSS redux."""
    plate_str = f"{plate:04d}"
    fiber_str = f"{fiberid:04d}"
    fname = f"spec-{plate_str}-{mjd}-{fiber_str}.fits"

    # Check multiple possible locations
    for subdir in ["boss", "sdss", ""]:
        path = os.path.join(sdss_spectra_dir, subdir, plate_str, fname) if subdir else \
               os.path.join(sdss_spectra_dir, plate_str, fname)
        if os.path.exists(path):
            return path

    # Also check flat layout
    path = os.path.join(sdss_spectra_dir, fname)
    if os.path.exists(path):
        return path

    return None


class SDSSSpectraSource(DataSource):
    name = "sdss_spectra"

    def preflight(self) -> tuple[bool, str]:
        d = self.config.sdss_spectra_dir
        if not os.path.isdir(d):
            return False, f"Directory not found: {d}"
        # Check for any spec-*.fits files
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.startswith("spec-") and f.endswith(".fits"):
                    return True, f"Found specLite files in {d}"
            if dirs:
                break  # Only check first level + one sublevel
        return False, f"No spec-*.fits files found in {d}"

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        from astropy.io import fits as afits

        print(f"    Reading SDSS spectra from {self.config.sdss_spectra_dir}...")

        # Separate objects WITH plate/mjd/fiber (AGN) from those WITHOUT (galaxies)
        has_plate = ("sdss_plate" in catalog_df.columns and
                     catalog_df["sdss_plate"].notna().any())

        if has_plate:
            plate_df = catalog_df[catalog_df["sdss_plate"].notna()].copy()
            no_plate_df = catalog_df[catalog_df["sdss_plate"].isna()].copy()
        else:
            plate_df = pd.DataFrame()
            no_plate_df = catalog_df.copy()

        # For objects without plate info: crossmatch against SpecObj
        if len(no_plate_df) > 0:
            specobj_path = _find_specobj(self.config)
            if specobj_path:
                print(f"    Crossmatching {len(no_plate_df)} objects against SpecObj...")
                no_plate_df = self._add_plate_info(no_plate_df, specobj_path)
                # Move matched ones to plate_df
                matched_mask = no_plate_df["sdss_plate"].notna()
                if matched_mask.any():
                    plate_df = pd.concat([plate_df, no_plate_df[matched_mask]], ignore_index=True)
                    print(f"    {matched_mask.sum()} matched via SpecObj crossmatch")
            else:
                print(f"    SpecObj not found — skipping galaxies without plate info")

        if len(plate_df) == 0:
            print(f"    No objects with SDSS plate info — skipping")
            return 0

        # Read spectra — flush to shards incrementally to avoid OOM
        results = []
        found = missing = errors = 0
        shard_idx = 0

        for _, row in plate_df.iterrows():
            plate = int(row["sdss_plate"])
            mjd = int(row["sdss_mjd"])
            fiberid = int(row["sdss_fiberid"])
            oid = str(row["object_id"])

            spec_path = _find_spec_file(self.config.sdss_spectra_dir, plate, mjd, fiberid)
            if spec_path is None:
                missing += 1
                continue

            try:
                with afits.open(spec_path, memmap=True) as hdul:
                    coadd = hdul[1].data
                    flux = coadd["flux"].astype(np.float32)
                    loglam = coadd["loglam"].astype(np.float32)
                    ivar = coadd["ivar"].astype(np.float32)

                    results.append({
                        "object_id": oid,
                        "sdss_flux": flux,
                        "sdss_loglam": loglam,
                        "sdss_ivar": ivar,
                    })
                    found += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    WARN: {spec_path}: {e}")

            # Flush shard incrementally
            if len(results) >= self.config.shard_size:
                self.save_shard(pd.DataFrame(results), shard_idx)
                shard_idx += 1
                results.clear()

            if (found + missing + errors) % 20000 == 0 and (found + missing + errors) > 0:
                print(f"    {found + missing + errors}/{len(plate_df)} processed, {found} spectra found")

        print(f"    Found: {found}, Missing: {missing}, Errors: {errors}")

        # Save remaining
        if results:
            self.save_shard(pd.DataFrame(results), shard_idx)

        return found

    def _add_plate_info(self, df, specobj_path):
        """Crossmatch against SpecObj to add plate/mjd/fiberid columns."""
        from astropy.io import fits as afits

        with afits.open(specobj_path, memmap=True) as hdul:
            data = hdul[1].data
            # Filter to good spectra
            mask = np.ones(len(data), dtype=bool)
            if "ZWARNING" in data.columns.names:
                mask &= data["ZWARNING"] == 0

            so_ra = data["PLUG_RA"][mask].astype(np.float64)
            so_dec = data["PLUG_DEC"][mask].astype(np.float64)
            so_plate = data["PLATE"][mask].astype(np.int32)
            so_mjd = data["MJD"][mask].astype(np.int32)
            so_fiberid = data["FIBERID"][mask].astype(np.int32)

        # Crossmatch with epoch propagation
        survey_epoch = SURVEY_EPOCHS.get("sdss_spectra", 2005.0)
        cat_coords = propagate_coords(df, survey_epoch, self.config.gaia_ref_epoch)
        so_coords = SkyCoord(ra=so_ra, dec=so_dec, unit="deg")

        idx, sep, _ = cat_coords.match_to_catalog_sky(so_coords)
        match_mask = sep < self.config.match_radius_arcsec * u.arcsec

        df["sdss_plate"] = np.where(match_mask, so_plate[idx], np.nan)
        df["sdss_mjd"] = np.where(match_mask, so_mjd[idx], np.nan)
        df["sdss_fiberid"] = np.where(match_mask, so_fiberid[idx], np.nan)

        return df
