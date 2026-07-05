"""Canonical global_object_id helpers.

The identifier is the NESTED HEALPix order-29 index of the seed position,
matching the MMU `_healpix_29` convention. Bad seed coordinates fail here so
they do not become silent identity corruption in a cluster run.
"""

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
    if ra.shape != dec.shape:
        raise ValueError(f"ra/dec shape mismatch: {ra.shape} != {dec.shape}")
    if not np.isfinite(ra).all() or not np.isfinite(dec).all():
        raise ValueError("ra/dec must be finite")
    if ((ra < 0.0) | (ra >= 360.0)).any():
        raise ValueError("ra must be in [0, 360) degrees")
    if ((dec < -90.0) | (dec > 90.0)).any():
        raise ValueError("dec must be in [-90, 90] degrees")
    idx = _HP.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    return np.asarray(idx, dtype=np.int64)
