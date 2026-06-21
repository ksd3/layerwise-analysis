"""Phase 7: Chunked merge per population with proper modality counting.

Processes each population in chunks of MERGE_CHUNK objects to limit memory.
This prevents OOM when populations are large (e.g., 700k stars with APOGEE
spectra = ~300 GB as Python lists if loaded all at once).
"""

import os
import glob
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils.parquet import save_shard, load_shards


def _load_shards_filtered(shard_dir, object_ids):
    """Load shards, keeping only rows matching the given object_ids set.

    Much more memory-efficient than loading everything and filtering after.
    """
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not shard_files:
        return None
    dfs = []
    for sf in shard_files:
        try:
            df = pd.read_parquet(sf)
            if "object_id" in df.columns:
                df = df[df["object_id"].isin(object_ids)]
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# Which modalities apply to each population
POPULATION_SOURCES = {
    "star": ["flatiron_gaia", "flatiron_tess", "apogee", "sdss_spectra", "galah", "ztf", "legacy_survey", "galex_images", "twomass", "unwise"],
    "galaxy": ["flatiron_desi", "sdss_spectra", "legacy_survey", "galex_images", "unwise"],
    "agn": ["flatiron_desi", "flatiron_tess", "sdss_spectra", "ztf", "legacy_survey", "galex_images", "unwise"],
}


def _has_data(x):
    """Check if a value contains actual data (list/array)."""
    return isinstance(x, (list, np.ndarray))


def _add_split(result: pd.DataFrame, nside: int = 8, seed: int = 42) -> pd.DataFrame:
    """Add a train/val/test split column based on HEALPix sky region.

    Uses NSIDE=8 (768 pixels, ~53 deg² each). Pixels are deterministically
    assigned to splits based on pixel index, ensuring spatial separation
    between train/val/test to avoid information leakage from shared
    dust extinction, stellar populations, and survey depth.

    Split ratio: 70% train, 15% val, 15% test.
    """
    try:
        import healpy as hp
        pix = hp.ang2pix(nside, result["ra"].values, result["dec"].values,
                         lonlat=True, nest=True)
        # Deterministic split: hash pixel index to assign split
        rng = np.random.RandomState(seed)
        n_pix = hp.nside2npix(nside)
        pix_splits = rng.choice(["train", "val", "test"], size=n_pix,
                                p=[0.70, 0.15, 0.15])
        result["split"] = pix_splits[pix]
    except ImportError:
        # Fallback: random split (less ideal but functional)
        rng = np.random.RandomState(seed)
        result["split"] = rng.choice(["train", "val", "test"], size=len(result),
                                     p=[0.70, 0.15, 0.15])
    return result


def _count_modalities(result: pd.DataFrame, bands: str = "grizy"):
    """Count spectra, light curves, and images per object."""
    result["n_spectra"] = 0
    result["n_lightcurves"] = 0
    result["n_images"] = 0

    # ── Spectra ──
    # APOGEE
    if "apogee_flux" in result.columns:
        result["n_spectra"] += result["apogee_flux"].apply(_has_data).astype(int)

    # GALAH (per-band: galah_flux_blue, galah_flux_green, galah_flux_red, galah_flux_ir)
    galah_counted = False
    for col in result.columns:
        if col.startswith("galah_flux_") and not galah_counted:
            if result[col].dtype == object:
                result["n_spectra"] += result[col].apply(_has_data).astype(int)
                galah_counted = True

    # SDSS specLite
    if "sdss_flux" in result.columns:
        result["n_spectra"] += result["sdss_flux"].apply(_has_data).astype(int)

    # DESI (from Flatiron — look for any desi flux column)
    desi_counted = False
    for col in result.columns:
        if ("desi" in col and ("flux" in col or "spectrum" in col)
                and not desi_counted):
            if result[col].dtype == object:
                result["n_spectra"] += result[col].apply(_has_data).astype(int)
                desi_counted = True

    # Gaia BP/RP (from Flatiron — stored as "coeff" or "flux" or "spectrum")
    gaia_counted = False
    for col in result.columns:
        if ("gaia" in col and ("flux" in col or "spectrum" in col or "coeff" in col)
                and col not in ("gaia_ra", "gaia_dec", "flatiron_gaia_ra", "flatiron_gaia_dec")
                and not gaia_counted):
            if result[col].dtype == object:
                result["n_spectra"] += result[col].apply(_has_data).astype(int)
                gaia_counted = True

    # ── Light curves ──
    if "ztf_time" in result.columns:
        result["n_lightcurves"] += result["ztf_time"].apply(_has_data).astype(int)

    tess_counted = False
    for col in result.columns:
        if "tess" in col and "flux" in col and not tess_counted:
            if result[col].dtype == object:
                result["n_lightcurves"] += result[col].apply(_has_data).astype(int)
                tess_counted = True

    # ── Images ──
    # Legacy Survey (g, r, z)
    for band in ["g", "r", "z"]:
        col = f"legacy_{band}"
        if col in result.columns:
            result["n_images"] += result[col].apply(_has_data).astype(int)
    # GALEX (fuv, nuv)
    for band in ["fuv", "nuv"]:
        col = f"galex_{band}"
        if col in result.columns:
            result["n_images"] += result[col].apply(_has_data).astype(int)
    # 2MASS (j, h, k)
    for band in ["j", "h", "k"]:
        col = f"twomass_{band}"
        if col in result.columns:
            result["n_images"] += result[col].apply(_has_data).astype(int)
    # unWISE (w1, w2)
    for band in ["w1", "w2"]:
        col = f"unwise_{band}"
        if col in result.columns:
            result["n_images"] += result[col].apply(_has_data).astype(int)

    return result


def finalize(config: PipelineConfig, catalogs: dict[str, pd.DataFrame]) -> int:
    """Streaming merge: process one population at a time to limit memory."""
    print("\nPHASE FINAL: Merging and writing output...")

    output_dir = os.path.join(config.work_dir, "shards", "final")
    hf_api = None
    if not config.skip_upload:
        from huggingface_hub import HfApi
        hf_api = HfApi()
        try:
            hf_api.create_repo(config.hf_repo, repo_type="dataset",
                               exist_ok=True, token=config.hf_token)
        except Exception:
            pass

    total_objects = 0
    global_shard_idx = 0

    # Chunk size for memory-safe merging. Each chunk of the catalog is merged
    # with modality data independently. Prevents OOM when populations are large
    # (e.g., 700k stars × APOGEE spectra = ~300 GB as Python lists if loaded at once).
    MERGE_CHUNK = 50_000

    for pop_name, catalog_df in catalogs.items():
        n_pop = len(catalog_df)
        print(f"\n  Processing {pop_name} ({n_pop} objects)...")
        modalities = POPULATION_SOURCES.get(pop_name, [])

        # Pre-load and deduplicate modality indexes (object_ids only) to know
        # which modalities have data. The actual data is loaded per-chunk below.
        mod_shard_dirs = {}
        for mod_name in modalities:
            shard_dir = os.path.join(config.work_dir, "shards", mod_name)
            if os.path.isdir(shard_dir) and glob.glob(os.path.join(shard_dir, "*.parquet")):
                mod_shard_dirs[mod_name] = shard_dir
            else:
                print(f"    {mod_name}: no data")

        # Process catalog in chunks to limit memory
        n_chunks = (n_pop + MERGE_CHUNK - 1) // MERGE_CHUNK
        pop_spectra = 0
        pop_lc = 0
        pop_images = 0
        n_cols_reported = False

        for chunk_idx in range(n_chunks):
            start = chunk_idx * MERGE_CHUNK
            end = min(start + MERGE_CHUNK, n_pop)
            result = catalog_df.iloc[start:end].copy()
            chunk_oids = set(result["object_id"].values)

            for mod_name, shard_dir in mod_shard_dirs.items():
                # Load only rows matching this chunk's objects
                mod_df = _load_shards_filtered(shard_dir, chunk_oids)

                if mod_df is None or len(mod_df) == 0:
                    continue

                # Deduplicate: keep closest match
                n_before = len(mod_df)
                sep_col = next((c for c in mod_df.columns if c.endswith("_match_sep_arcsec")), None)
                if sep_col:
                    mod_df = mod_df.sort_values(sep_col, na_position="last")
                mod_df = mod_df.drop_duplicates(subset=["object_id"], keep="first")

                drop_cols = [c for c in mod_df.columns
                             if c != "object_id" and c in result.columns]
                mod_clean = mod_df.drop(columns=drop_cols, errors="ignore")
                result = result.merge(mod_clean, on="object_id", how="left")

                if chunk_idx == 0:
                    n_matched = mod_clean["object_id"].isin(chunk_oids).sum()
                    n_deduped = n_before - len(mod_df.drop_duplicates(subset=["object_id"]))
                    extra = f", deduped {n_deduped}" if n_deduped > 0 else ""
                    print(f"    {mod_name}: +{len(mod_clean.columns) - 1} cols, "
                          f"{n_matched} matched in first chunk{extra}")

                del mod_df, mod_clean

            # Count modalities
            result = _count_modalities(result)
            result["n_modality_types"] = (
                (result["n_spectra"] > 0).astype(int) +
                (result["n_lightcurves"] > 0).astype(int) +
                (result["n_images"] > 0).astype(int)
            )
            result = _add_split(result)

            pop_spectra += (result["n_spectra"] > 0).sum()
            pop_lc += (result["n_lightcurves"] > 0).sum()
            pop_images += (result["n_images"] > 0).sum()

            if not n_cols_reported:
                print(f"    Columns: {len(result.columns)}")
                n_cols_reported = True

            # Write output shards
            for si in range(0, len(result), config.shard_size):
                chunk_slice = result.iloc[si:si + config.shard_size].copy()
                save_shard(chunk_slice, output_dir, global_shard_idx)
                global_shard_idx += 1

            total_objects += len(result)
            del result

            if n_chunks > 1 and (chunk_idx + 1) % 5 == 0:
                print(f"    Chunk {chunk_idx + 1}/{n_chunks} done...")

        print(f"    Spectra coverage:     {pop_spectra}/{n_pop}")
        print(f"    Light curve coverage: {pop_lc}/{n_pop}")
        print(f"    Image coverage:       {pop_images}/{n_pop}")

    print(f"\n  Total: {total_objects} objects in {global_shard_idx} shards")
    return total_objects
