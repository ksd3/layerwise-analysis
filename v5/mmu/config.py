"""Frozen pipeline configuration: sources, radii, paths, shard size, schema version."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    kind: str
    org: str
    dataset: str
    modality: str
    radius_arcsec: float
    columns: tuple[str, ...]
    epoch_jyear: float = 2016.0
    population: str = "galaxy"

    @property
    def is_hats(self) -> bool:
        return self.kind in {"lsdb_mmu", "ztf_s3"}


def env_path(name: str, default: str) -> str:
    return os.environ.get(name, default)


SOURCES: dict[str, Source] = {
    "desi": Source("desi", "lsdb_mmu", "UniverseTBD", "mmu_desi_edr_sv3",
                   "spectrum", 1.0, ("ra", "dec", "flux", "ivar", "lambda"), 2016.0, "galaxy"),
    "hsc": Source("hsc", "lsdb_mmu", "UniverseTBD", "mmu_hsc_pdr3_dud_22.5",
                  "image", 1.0, ("ra", "dec", "image"), 2014.0, "galaxy"),
    "legacy": Source("legacy", "legacy_hdf5", "", env_path("OMNISKY_LEGACY_ROOT", "data/legacy_dr10_south_21"),
                     "image", 1.0, ("ra", "dec", "image_array"), 2015.5, "galaxy"),
    "gaia": Source("gaia", "lsdb_mmu", "UniverseTBD", "mmu_gaia_gaia",
                   "astrometry", 1.0, ("ra", "dec", "pmra", "pmdec", "parallax"), 2016.0, "star"),
    "apogee": Source("apogee", "lsdb_mmu", "hugging-science", "mmu_apogee_dr17",
                     "spectrum", 1.0, ("ra", "dec", "flux", "snr"), 2016.0, "star"),
    "ztf": Source("ztf", "ztf_s3", "", "ztf/enhanced/dr24/lc/hats",
                  "lightcurve", 1.0, ("ra", "dec", "mjd", "mag", "magerr"), 2018.0, "star"),
    "tess": Source("tess", "lsdb_mmu", "UniverseTBD", "mmu_tess_spoc",
                   "lightcurve", 1.0, ("ra", "dec", "time", "flux"), 2019.0, "star"),
    "sdss_dr16q": Source("sdss_dr16q", "local_fits", "", env_path("OMNISKY_SDSS_DR16Q_ROOT", "data/sdss_dr16q"),
                         "spectrum", 1.0, ("ra", "dec", "z", "flux"), 2015.5, "agn"),
}


@dataclass(frozen=True, slots=True)
class Config:
    release_root: Path = Path(env_path("OMNISKY_RELEASE_ROOT", "release/v5"))
    shard_size: int = 50_000
    schema_version: str = "v5.1"
    min_instruments: int = 2
    split_nside: int = 8


DEFAULT = Config()
