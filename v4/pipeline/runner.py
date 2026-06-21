"""Pipeline orchestrator: runs all phases in order."""

import gc
import time
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import PipelineConfig, FLATIRON_DATASETS
from .catalogs import build_star_catalog, build_galaxy_catalog, build_agn_catalog
from .sources.flatiron import FlatironSource
from .sources.apogee import APOGEESource
from .sources.sdss_spectra import SDSSSpectraSource
from .sources.ztf import ZTFSource
from .sources.ps1 import LegacySurveySource
from .sources.unwise import UnWISESource
from .sources.galah import GALAHSource
from .sources.twomass import TwoMASSSource
from .sources.galex_images import GALEXImageSource
from .preflight import preflight_check
from .finalize import finalize
import os


def run_pipeline(config: PipelineConfig):
    total_start = time.time()

    print("=" * 60)
    print("Multimodal Universe Pipeline v3 (modular)")
    print(f"  Work dir:     {config.work_dir}")
    print(f"  APOGEE dir:   {config.apogee_dir}")
    print(f"  SDSS dir:     {config.sdss_spectra_dir}")
    n_s = config.n_stars if config.n_stars > 0 else "all"
    n_g = config.n_galaxies if config.n_galaxies > 0 else "all"
    n_a = config.n_agn if config.n_agn > 0 else "all"
    print(f"  Populations:  {n_s} stars, {n_g} galaxies, {n_a} AGN")
    print(f"  Test mode:    {config.test_mode}")
    print("=" * 60)

    # ── Build all source instances ──
    flatiron_sources = []
    for ds_name, ds_info in FLATIRON_DATASETS.items():
        flatiron_sources.append(FlatironSource(
            config,
            name=f"flatiron_{ds_name}",
            dataset_url=ds_info["url"],
            subdirs=ds_info["subdirs"],
            columns_to_keep=ds_info["columns_to_keep"],
        ))

    apogee = APOGEESource(config)
    sdss = SDSSSpectraSource(config)
    ztf = ZTFSource(config)
    legacy = LegacySurveySource(config)
    galex_img = GALEXImageSource(config)
    unwise = UnWISESource(config)
    galah = GALAHSource(config)
    twomass = TwoMASSSource(config)

    all_sources = flatiron_sources + [apogee, sdss, ztf, legacy, galex_img, unwise, galah, twomass]

    # ── Phase 0: Preflight ──
    preflight_check(config, all_sources)

    # ── Phase 1: Build catalogs ──
    print("\nPHASE 1: Building catalogs...")
    star_cat = build_star_catalog(config)
    galaxy_cat = build_galaxy_catalog(config)
    agn_cat = build_agn_catalog(config)
    all_cat = pd.concat([star_cat, galaxy_cat, agn_cat], ignore_index=True)
    print(f"  Combined: {len(all_cat)} objects")

    # Build sub-catalogs for routing
    star_agn = pd.concat([star_cat, agn_cat], ignore_index=True)
    gal_agn = pd.concat([galaxy_cat, agn_cat], ignore_index=True)

    # Map populations to their Flatiron datasets
    pop_routing = {
        "star": star_cat,
        "agn": agn_cat,
        "galaxy": galaxy_cat,
    }

    # ── Phase 2: Flatiron data ──
    print("\nPHASE 2: Flatiron data...")
    for source in flatiron_sources:
        ds_name = source.name.replace("flatiron_", "")
        ds_info = FLATIRON_DATASETS[ds_name]
        pops = ds_info["populations"]

        # Build the catalog for this dataset
        cat_parts = [pop_routing[p] for p in pops if p in pop_routing]
        if not cat_parts:
            continue
        cat = pd.concat(cat_parts, ignore_index=True)

        print(f"\n  {source.name} ({len(cat)} objects from {pops})...")
        source.run(cat)

    # ── Phase 3: APOGEE spectra ──
    print("\nPHASE 3: APOGEE local spectra...")
    apogee.run(star_cat)

    # ── Phase 4: SDSS spectra ──
    print("\nPHASE 4: SDSS spectra...")
    sdss.run(gal_agn)

    # Free memory before concurrent phase
    gc.collect()

    # ── Phases 5+: All remaining sources (concurrent) ──
    print("\nPHASES 5+: ZTF + Legacy Survey + GALEX + 2MASS + unWISE + GALAH (concurrent)...")

    def _run_source(source, catalog):
        return source.name, source.run(catalog)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_run_source, ztf, star_agn): "ZTF",
            executor.submit(_run_source, legacy, all_cat): "Legacy Survey",
            executor.submit(_run_source, galex_img, all_cat): "GALEX",
            executor.submit(_run_source, twomass, star_cat): "2MASS",
            executor.submit(_run_source, unwise, all_cat): "unWISE",
            executor.submit(_run_source, galah, star_cat): "GALAH",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                name, count = future.result()
                print(f"  {label} complete: {count}")
            except Exception as e:
                print(f"  {label} FAILED: {e}")
                import traceback
                traceback.print_exc()

    # ── Phase 9: Finalize ──
    catalogs_dict = {
        "star": star_cat,
        "galaxy": galaxy_cat,
        "agn": agn_cat,
    }
    total_objects = finalize(config, catalogs_dict)

    # ── Cleanup ──
    for tmp in ["tmp_flatiron", "seeds"]:
        tmp_path = os.path.join(config.work_dir, tmp)
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 3600:.1f} hours")
    print(f"  Objects: {total_objects}")
    if not config.skip_upload:
        print(f"  Dataset: https://huggingface.co/datasets/{config.hf_repo}")
    else:
        print(f"  Results: {os.path.join(config.work_dir, 'shards', 'final')}")
    print("=" * 60)
