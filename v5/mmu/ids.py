"""Canonical global_object_id: HEALPix NESTED order-29 int64 (matches MMU `_healpix_29`)."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy_healpix import HEALPix

ORDER: int = 29
NSIDE: int = 2 ** ORDER
_HP = HEALPix(nside=NSIDE, order="nested")


def assign_global_id(ra_deg, dec_deg) -> np.ndarray:
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=np.float64))
    idx = _HP.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    return np.asarray(idx, dtype=np.int64)
