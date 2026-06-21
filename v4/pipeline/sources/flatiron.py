"""Flatiron healpix data: Gaia BP/RP, TESS, DESI.

Key improvements over v2:
  - Healpix pre-filtering (skip cells with no catalog objects)
  - Parallel cell downloads
  - Pre-computed catalog SkyCoord (created once, reused per cell)
  - Column filtering
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import DataSource
from ..config import PipelineConfig
from ..utils.download import list_healpix_dirs, download_healpix_cell, read_hdf5_data
from ..config import SURVEY_EPOCHS
from ..utils.crossmatch import crossmatch_to_catalog, compute_catalog_healpix, filter_healpix_cells, build_skycoord, propagate_coords


class FlatironSource(DataSource):
    """Fetch data from a Flatiron healpix dataset."""

    def __init__(self, config: PipelineConfig, name: str, dataset_url: str,
                 subdirs: list[str], columns_to_keep: list[str] | None = None):
        self.name = name  # Must set before super().__init__ which uses self.name
        self.dataset_url = dataset_url
        self.subdirs = subdirs
        self.columns_to_keep = columns_to_keep
        super().__init__(config)

    def preflight(self) -> tuple[bool, str]:
        import requests
        try:
            url = self.dataset_url + self.subdirs[0] + "/"
            resp = requests.get(url, timeout=15)
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        print(f"  Fetching {self.name} from Flatiron...")

        # Pre-compute catalog SkyCoord with epoch propagation
        survey_epoch = SURVEY_EPOCHS.get(self.name, 2016.0)
        cat_coords = propagate_coords(catalog_df, survey_epoch, self.config.gaia_ref_epoch)
        cat_n = len(catalog_df)
        print(f"    Epoch propagation: {self.config.gaia_ref_epoch} → {survey_epoch}")

        # Try healpix pre-filtering
        cat_healpix = compute_catalog_healpix(
            catalog_df["ra"].values, catalog_df["dec"].values, nside=16)
        if cat_healpix is not None:
            print(f"    Healpix pre-filter: {len(cat_healpix)} unique cells for {cat_n} objects")

        # Process cells and save per-cell matches directly to shards.
        # This avoids accumulating all matches in memory (which OOMs for large spectra).
        shard_idx = 0
        total_matched = 0
        object_ids = catalog_df["object_id"].values

        for subdir in self.subdirs:
            sub_url = self.dataset_url + subdir + "/"
            try:
                available = list_healpix_dirs(sub_url)
            except Exception as e:
                print(f"    {subdir}: failed to list — {e}")
                continue

            filtered = filter_healpix_cells(available, cat_healpix)
            print(f"    {subdir}: {len(available)} total cells, {len(filtered)} with catalog objects")

            if self.config.test_mode:
                filtered = dict(list(filtered.items())[:3])

            cells_processed = 0
            prefix = f"{self.name}_{subdir}" if len(self.subdirs) > 1 else self.name
            pending_rows = []  # accumulate small batches of matched rows

            def _process_cell(hp_info):
                hp, (hp_url, dir_name) = hp_info
                tmp_dir = os.path.join(self.config.work_dir, "tmp_flatiron")
                local_files, nbytes = download_healpix_cell(
                    self.dataset_url, subdir, hp, tmp_dir, dir_name)
                cell_data_list = []
                for lf in local_files:
                    data = read_hdf5_data(lf, columns_to_keep=self.columns_to_keep)
                    if data and "ra" in data and len(data["ra"]) > 0:
                        cell_data_list.append(data)
                    try:
                        os.remove(lf)
                    except OSError:
                        pass
                if local_files:
                    try:
                        cell_dir = os.path.dirname(local_files[0])
                        if cell_dir and os.path.isdir(cell_dir) and not os.listdir(cell_dir):
                            os.rmdir(cell_dir)
                    except OSError:
                        pass
                return cell_data_list

            n_workers = min(self.config.flatiron_workers, len(filtered))
            with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
                futures = {pool.submit(_process_cell, item): item
                           for item in filtered.items()}

                for future in as_completed(futures):
                    try:
                        cell_data_list = future.result()
                        for data in cell_data_list:
                            matched = crossmatch_to_catalog(
                                cat_coords, cat_n, data, prefix,
                                radius=self.config.match_radius_arcsec,
                                columns_to_keep=self.columns_to_keep)
                            if not matched:
                                continue
                            # Extract only matched rows (not full catalog-length arrays)
                            ra_col = next((c for c in matched if c.endswith("_ra")), None)
                            if ra_col is None:
                                continue
                            ra_vals = matched[ra_col]
                            if pd.api.types.is_float_dtype(ra_vals):
                                mask = ~np.isnan(ra_vals)
                            else:
                                mask = ra_vals != ""
                            matched_indices = np.where(mask)[0]
                            if len(matched_indices) == 0:
                                continue
                            # Build rows for matched objects only
                            for idx in matched_indices:
                                row = {"object_id": object_ids[idx]}
                                for col, vals in matched.items():
                                    if vals.dtype == object:
                                        row[col] = vals[idx]
                                    elif np.issubdtype(vals.dtype, np.floating):
                                        v = vals[idx]
                                        row[col] = v if not np.isnan(v) else None
                                    else:
                                        row[col] = vals[idx]
                                pending_rows.append(row)
                            total_matched += len(matched_indices)
                    except Exception as e:
                        print(f"    WARN: cell failed: {e}")

                    cells_processed += 1
                    if cells_processed % self.config.flush_interval == 0:
                        print(f"    {cells_processed}/{len(filtered)} cells processed...")

                    # Flush pending rows to shard when batch is large enough
                    if len(pending_rows) >= self.config.shard_size:
                        self.save_shard(pd.DataFrame(pending_rows), shard_idx)
                        shard_idx += 1
                        pending_rows.clear()

        # Save remaining
        if pending_rows:
            self.save_shard(pd.DataFrame(pending_rows), shard_idx)

        print(f"    Matched: {total_matched}/{cat_n}")
        return total_matched
