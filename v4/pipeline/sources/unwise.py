"""unWISE images: tile download + local Cutout2D."""

import os
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u

from .base import DataSource
from ..config import PipelineConfig, UNWISE_BASE, SURVEY_EPOCHS
from ..utils.download import download_file
from ..utils.crossmatch import propagate_coords


class UnWISESource(DataSource):
    name = "unwise"

    def preflight(self) -> tuple[bool, str]:
        try:
            resp = requests.head(f"{UNWISE_BASE}/allsky-atlas.fits", timeout=15)
            return resp.status_code == 200, "Atlas file reachable"
        except Exception as e:
            return False, str(e)

    def fetch(self, catalog_df: pd.DataFrame) -> int:
        from astropy.io import fits as afits
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D

        stamp_size = self.config.unwise_stamp_size

        # Download atlas
        atlas_path = os.path.join(self.config.work_dir, "allsky-atlas.fits")
        if not os.path.exists(atlas_path):
            print("    Downloading unWISE atlas...")
            download_file(f"{UNWISE_BASE}/allsky-atlas.fits", atlas_path)

        with afits.open(atlas_path) as hdul:
            atlas = hdul[1].data
            tile_ras = atlas["RA"]
            tile_decs = atlas["DEC"]
            tile_names = np.array(atlas["COADD_ID"], dtype=str)

        # Propagate positions to unWISE epoch for correct cutout centering
        survey_epoch = SURVEY_EPOCHS.get("unwise", 2014.0)
        cat_coords = propagate_coords(catalog_df, survey_epoch, self.config.gaia_ref_epoch)
        print(f"    Epoch propagation: {self.config.gaia_ref_epoch} → {survey_epoch}")
        tile_coords = SkyCoord(ra=tile_ras, dec=tile_decs, unit="deg")
        idx, sep, _ = cat_coords.match_to_catalog_sky(tile_coords)

        # Store propagated positions for cutout centering
        prop_ras = cat_coords.ra.deg
        prop_decs = cat_coords.dec.deg

        tile_groups = {}
        for i, (tile_idx, s) in enumerate(zip(idx, sep)):
            if s > 1 * u.deg:
                continue
            tname = tile_names[tile_idx].strip()
            if tname not in tile_groups:
                tile_groups[tname] = []
            tile_groups[tname].append(i)

        print(f"    {len(tile_groups)} unique tiles for {len(catalog_df)} objects")

        results = {}
        tiles_done = 0
        shard_idx = 0

        for tname, obj_indices in tile_groups.items():
            if self.config.test_mode and tiles_done >= 2:
                break

            tdir = tname[:3]
            tile_paths = []

            try:
                for band in ["w1", "w2"]:
                    url = f"{UNWISE_BASE}/{tdir}/{tname}/unwise-{tname}-{band}-img-u.fits"
                    local = os.path.join(self.config.work_dir, f"tmp_unwise_{tname}_{band}.fits")
                    download_file(url, local)
                    tile_paths.append((band, local))

                for band, local in tile_paths:
                    with afits.open(local) as hdul:
                        wcs = WCS(hdul[0].header)
                        img = hdul[0].data
                        if img is None:
                            continue
                        for oi in obj_indices:
                            ra = prop_ras[oi]
                            dec = prop_decs[oi]
                            oid = str(catalog_df.iloc[oi]["object_id"])
                            try:
                                pos = SkyCoord(ra=ra, dec=dec, unit="deg")
                                cutout = Cutout2D(img, pos, stamp_size, wcs=wcs)
                                if oid not in results:
                                    results[oid] = {}
                                results[oid][f"unwise_{band}"] = cutout.data.astype(np.float32)
                            except Exception:
                                pass

                for _, local in tile_paths:
                    try:
                        os.remove(local)
                    except OSError:
                        pass

                tiles_done += 1
                if tiles_done % 100 == 0:
                    print(f"    {tiles_done}/{len(tile_groups)} tiles, "
                          f"{len(results)} objects with stamps")

                # Flush periodically (outside modulo — don't let results grow unbounded)
                if len(results) >= self.config.shard_size:
                    rows = [{"object_id": oid, **data} for oid, data in results.items()]
                    self.save_shard(pd.DataFrame(rows), shard_idx)
                    shard_idx += 1
                    results.clear()

            except Exception as e:
                print(f"    WARN: tile {tname}: {e}")
                for _, local in tile_paths:
                    try:
                        os.remove(local)
                    except OSError:
                        pass

        # Save remaining
        if results:
            rows = [{"object_id": oid, **data} for oid, data in results.items()]
            self.save_shard(pd.DataFrame(rows), shard_idx)

        print(f"    Total: processed {tiles_done} tiles")
        return tiles_done
