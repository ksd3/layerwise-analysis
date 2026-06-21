"""ZTF light curves: read HATS-partitioned Parquet from AWS S3.

Uses the ZTF DR23 light curve catalog on s3://ipac-irsa-ztf, organized
in HATS (Hierarchical Adaptive Tiling Scheme) healpix partitions.
Each row is one object with its full light curve nested as a struct.

Key advantages over IRSA bulk download:
  - Column projection: read only positions (7 MB) instead of full file (138 MB)
  - Healpix pre-filtering: skip tiles without our objects
  - No rate limits (public S3 bucket)
  - 75 MB/s from Delta AI
"""

import os
import csv
import time
import numpy as np
import pandas as pd
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.coordinates import SkyCoord
from astropy import units as u

from .base import DataSource
from ..config import PipelineConfig, SURVEY_EPOCHS
from ..utils.crossmatch import propagate_coords

S3_BUCKET = "ipac-irsa-ztf"
S3_PREFIX = "contributed/dr23/lc/hats/ztf_dr23_lc-hats"
PARTITION_INFO_KEY = f"{S3_PREFIX}/partition_info.csv"


def _time_sort_lc(data):
    """Sort a light curve dict by time. Ensures monotonic timestamps."""
    if "ztf_time" not in data or len(data["ztf_time"]) == 0:
        return data
    times = np.array(data["ztf_time"])
    order = np.argsort(times)
    return {
        "ztf_time": [data["ztf_time"][i] for i in order],
        "ztf_mag": [data["ztf_mag"][i] for i in order],
        "ztf_magerr": [data["ztf_magerr"][i] for i in order],
        "ztf_band": [data["ztf_band"][i] for i in order],
        "ztf_match_sep_arcsec": data.get("ztf_match_sep_arcsec", np.nan),
    }


class ZTFSource(DataSource):
    name = "ztf"

    def preflight(self) -> tuple[bool, str]:
        try:
            import pyarrow.fs as pafs
            s3 = pafs.S3FileSystem(region="us-east-1", anonymous=True)
            info = s3.get_file_info(f"{S3_BUCKET}/{PARTITION_INFO_KEY}")
            return info.size > 0, f"ZTF HATS on S3 OK ({info.size / 1024:.0f} KB partition index)"
        except Exception as e:
            return False, str(e)

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        import pyarrow.parquet as pq
        import pyarrow.fs as pafs

        s3 = pafs.S3FileSystem(region="us-east-1", anonymous=True)

        # Step 1: Read partition info
        print("    Reading ZTF HATS partition index from S3...")
        with s3.open_input_stream(f"{S3_BUCKET}/{PARTITION_INFO_KEY}") as f:
            part_csv = f.read().decode("utf-8")

        partitions = []
        for row in csv.DictReader(StringIO(part_csv)):
            partitions.append({
                "norder": int(row["Norder"]),
                "dir": int(row["Dir"]),
                "npix": int(row["Npix"]),
                "num_rows": int(row["num_rows"]),
            })
        print(f"    {len(partitions)} HATS partitions available")

        # Step 2: Healpix pre-filter — find partitions containing our objects
        print("    Computing healpix overlap with catalog...")
        cat_ra = catalog_df["ra"].values
        cat_dec = catalog_df["dec"].values
        cat_oids = catalog_df["object_id"].values.astype(str)

        # Build set of healpix pixels our catalog covers at each Norder
        try:
            import healpy as hp
            # Cache cat_pix per norder to avoid recomputing for every partition
            norder_to_catpix = {}
            for part in partitions:
                norder = part["norder"]
                if norder not in norder_to_catpix:
                    nside = 2 ** norder
                    pix = hp.ang2pix(nside, cat_ra, cat_dec, lonlat=True, nest=True)
                    norder_to_catpix[norder] = set(pix)

            needed_partitions = []
            for part in partitions:
                if part["npix"] in norder_to_catpix[part["norder"]]:
                    needed_partitions.append(part)
        except ImportError:
            print("    healpy not available — using all partitions (slower)")
            needed_partitions = partitions

        print(f"    {len(needed_partitions)} partitions overlap with catalog "
              f"(filtered from {len(partitions)})")

        if self.config.test_mode:
            needed_partitions = needed_partitions[:5]

        # Step 3: Process partitions — read positions, crossmatch, extract light curves
        # Propagate catalog positions to ZTF epoch
        survey_epoch = SURVEY_EPOCHS.get("ztf", 2021.0)
        cat_coords = propagate_coords(catalog_df, survey_epoch, self.config.gaia_ref_epoch)
        print(f"    Epoch propagation: {self.config.gaia_ref_epoch} → {survey_epoch}")
        results = {}
        shard_idx = 0
        total_matched = 0
        parts_done = 0

        for part in needed_partitions:
            norder = part["norder"]
            d = part["dir"]
            npix = part["npix"]
            s3_path = (f"{S3_BUCKET}/{S3_PREFIX}/dataset/"
                       f"Norder={norder}/Dir={d}/Npix={npix}/part0.snappy.parquet")

            try:
                # Read only positions first (fast — ~7 MB per partition)
                pos_table = pq.read_table(
                    s3_path, filesystem=s3,
                    columns=["objra", "objdec", "objectid"])

                ztf_ra = pos_table["objra"].to_numpy()
                ztf_dec = pos_table["objdec"].to_numpy()
                ztf_oids = pos_table["objectid"].to_numpy()

                if len(ztf_ra) == 0:
                    continue

                # Crossmatch ZTF objects against our catalog
                ztf_coords = SkyCoord(ra=ztf_ra, dec=ztf_dec, unit="deg")
                idx, sep, _ = ztf_coords.match_to_catalog_sky(cat_coords)
                match_mask = sep < self.config.match_radius_arcsec * u.arcsec
                matched_ztf_indices = np.where(match_mask)[0]

                if len(matched_ztf_indices) == 0:
                    del pos_table
                    parts_done += 1
                    continue

                # Read light curves only for matched rows
                full_table = pq.read_table(
                    s3_path, filesystem=s3,
                    columns=["objectid", "filterid", "nepochs", "lightcurve"])

                sep_arcsec = sep.to(u.arcsec).value

                for zi in matched_ztf_indices:
                    cat_idx = idx[zi]
                    cat_oid = cat_oids[cat_idx]
                    match_sep = float(sep_arcsec[zi])
                    row_lc = full_table["lightcurve"][zi].as_py()
                    filterid = int(full_table["filterid"][zi].as_py())
                    nepochs = int(full_table["nepochs"][zi].as_py())

                    if nepochs < 3:
                        continue

                    # Filter band name from filterid (1=g, 2=r, 3=i)
                    band_map = {1: "zg", 2: "zr", 3: "zi"}
                    band_name = band_map.get(filterid, str(filterid))

                    times = row_lc.get("hmjd", [])
                    mags = row_lc.get("mag", [])
                    magerrs = row_lc.get("magerr", [])

                    if len(times) < 3:
                        continue

                    if cat_oid not in results:
                        results[cat_oid] = {
                            "ztf_time": list(times),
                            "ztf_mag": list(mags),
                            "ztf_magerr": list(magerrs),
                            "ztf_band": [band_name] * len(times),
                            "ztf_match_sep_arcsec": match_sep,
                        }
                    else:
                        results[cat_oid]["ztf_time"].extend(times)
                        results[cat_oid]["ztf_mag"].extend(mags)
                        results[cat_oid]["ztf_magerr"].extend(magerrs)
                        results[cat_oid]["ztf_band"].extend([band_name] * len(times))
                        # Keep minimum separation across all ZTF matches
                        results[cat_oid]["ztf_match_sep_arcsec"] = min(
                            results[cat_oid]["ztf_match_sep_arcsec"], match_sep)

                    total_matched += 1

                del pos_table, full_table

            except Exception as e:
                if parts_done < 3:
                    print(f"    WARN: Norder={norder}/Npix={npix}: {e}")

            parts_done += 1

            # Flush shard periodically
            if len(results) >= self.config.shard_size:
                rows = [{"object_id": oid, **_time_sort_lc(data)}
                        for oid, data in results.items()]
                self.save_shard(pd.DataFrame(rows), shard_idx)
                shard_idx += 1
                results.clear()

            if parts_done % 500 == 0:
                print(f"    {parts_done}/{len(needed_partitions)} partitions, "
                      f"{total_matched} light curves found")

        # Save remaining
        if results:
            rows = [{"object_id": oid, **_time_sort_lc(data)}
                    for oid, data in results.items()]
            self.save_shard(pd.DataFrame(rows), shard_idx)

        print(f"    Total: {total_matched} ZTF light curves from "
              f"{parts_done} partitions")
        return total_matched
