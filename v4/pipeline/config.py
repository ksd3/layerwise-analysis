"""Pipeline configuration — all settings in one place."""

import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # ── Paths ──
    work_dir: str = ""
    apogee_dir: str = ""
    sdss_spectra_dir: str = ""  # Pre-staged SDSS specLite files

    # ── Mode ──
    test_mode: bool = False

    # ── Population sizes (0 = take all that pass quality cuts) ──
    n_stars: int = 0
    n_galaxies: int = 0
    n_agn: int = 0

    # ── Matching ──
    match_radius_arcsec: float = 3.0
    random_seed: int = 42  # For reproducible catalog sampling

    # ── Survey reference epochs (for proper motion correction) ──
    # Gaia DR3 reference epoch is 2016.0; proper motions propagate from here
    gaia_ref_epoch: float = 2016.0

    # ── Concurrency ──
    max_workers: int = 32
    ps1_max_threads: int = 10  # legacy: kept for compat
    legacy_survey_workers: int = 50
    ztf_workers: int = 4
    unwise_workers: int = 4
    flatiron_workers: int = 8

    # ── Image settings ──
    ps1_stamp_size: int = 64
    ps1_bands: str = "grizy"
    unwise_stamp_size: int = 64

    # ── Batch settings ──
    flush_interval: int = 50
    shard_size: int = 5000
    ps1_flush: int = 10_000
    ps1_batch_size: int = 50_000  # Max objects per PS1 filename POST

    # ── HuggingFace ──
    hf_token: str | None = None
    hf_repo: str | None = None
    skip_upload: bool = True

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        test_mode = os.environ.get("TEST_MODE", "0") == "1"
        cfg = cls(
            work_dir=os.environ.get("WORK_DIR", os.path.join(os.getcwd(), "pipeline_v3_workdir")),
            apogee_dir=os.environ.get("APOGEE_DIR", os.path.join(os.getcwd(), "apogee_spectra")),
            sdss_spectra_dir=os.environ.get("SDSS_SPECTRA_DIR", os.path.join(os.getcwd(), "sdss_spectra")),
            test_mode=test_mode,
            n_stars=5 if test_mode else 0,
            n_galaxies=5 if test_mode else 0,
            n_agn=5 if test_mode else 0,
            max_workers=2 if test_mode else int(os.environ.get("MAX_WORKERS", "32")),
            ps1_max_threads=int(os.environ.get("PS1_MAX", "10")),
            legacy_survey_workers=2 if test_mode else int(os.environ.get("LEGACY_WORKERS", "50")),
            ztf_workers=2 if test_mode else int(os.environ.get("ZTF_WORKERS", "4")),
            unwise_workers=2 if test_mode else int(os.environ.get("UNWISE_WORKERS", "4")),
            flatiron_workers=2 if test_mode else int(os.environ.get("FLATIRON_WORKERS", "8")),
            flush_interval=5 if test_mode else int(os.environ.get("FLUSH_INTERVAL", "50")),
            ps1_flush=5 if test_mode else int(os.environ.get("PS1_FLUSH", "10000")),
            hf_token=os.environ.get("HF_TOKEN"),
            hf_repo=os.environ.get("HF_REPO", "YOUR_USERNAME/multimodal-v3"),
        )
        cfg.skip_upload = test_mode or not cfg.hf_token
        os.makedirs(cfg.work_dir, exist_ok=True)
        return cfg


# ── APOGEE detector pixel mask (crop edges) ──
APOGEE_GOOD_PIXELS = np.r_[246:3274, 3585:6080, 6344:8335]  # 7514 pixels

# ── URLs ──
FLATIRON_BASE = "https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1"
PS1_BASE = "https://ps1images.stsci.edu"
UNWISE_BASE = "https://unwise.me/data/allwise/unwise-coadds/fulldepth"
ALLSTAR_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:SDSS/apogee/allStar-dr17-synspec_rev1.fits"
ZTF_BASE = "https://irsa.ipac.caltech.edu/data/ZTF/lc/lc_dr24/"
GALAH_BASE = "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3"
GALAH_CATALOG_URL = f"{GALAH_BASE}/GALAH_DR3_main_allstar_v2.fits"
GALAH_SPECTRA_URL = f"{GALAH_BASE}/spectra/tar_files/"
GALAH_SSA_URL = "https://datacentral.org.au/vo/slink/links"
LEGACY_SURVEY_CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"
TWOMASS_ATLAS_URL = "https://irsa.ipac.caltech.edu/2MASS/download/allsky/"

# ── Survey mean observation epochs (for proper motion propagation) ──
SURVEY_EPOCHS = {
    "twomass": 1999.5,
    "galex_images": 2007.0,
    "sdss_spectra": 2005.0,
    "gaia": 2016.0,
    "flatiron_gaia": 2016.0,
    "flatiron_tess": 2020.0,
    "flatiron_desi": 2021.0,
    "apogee": 2016.0,
    "ztf": 2021.0,
    "unwise": 2014.0,
    "legacy_survey": 2017.0,
    "galah": 2018.0,
}

# ── Flatiron dataset definitions ──
# Each: (url_suffix, subdirs, target_populations, columns_to_keep)
# columns_to_keep=None means keep all; list means keep only those (plus ra/dec)
FLATIRON_DATASETS = {
    "gaia": {
        "url": f"{FLATIRON_BASE}/gaia/",
        "subdirs": ["gaia"],
        "populations": ["star"],
        "columns_to_keep": None,  # Gaia BP/RP — keep all spectral data
    },
    "tess": {
        "url": f"{FLATIRON_BASE}/tess/",
        "subdirs": ["spoc"],
        "populations": ["star", "agn"],
        "columns_to_keep": None,  # TESS light curves — keep all
    },
    "desi": {
        "url": f"{FLATIRON_BASE}/desi/",
        "subdirs": ["edr_sv3"],
        "populations": ["galaxy", "agn"],
        "columns_to_keep": None,  # DESI spectra — keep all
    },
}
