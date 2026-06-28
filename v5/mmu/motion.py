"""Epoch propagation with explicit parallax policy."""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class MissingParallaxPolicy(str, Enum):
    FLAG = "flag"
    DROP = "drop"
    ASSUME_FAR = "assume_far"


def _linear_pm(ra, dec, pmra, pmdec, dt_yr):
    cos_dec = np.cos(np.deg2rad(dec))
    cos_dec = np.where(np.abs(cos_dec) < 1e-12, np.nan, cos_dec)
    return (ra + (pmra / 3.6e6) / cos_dec * dt_yr) % 360.0, dec + (pmdec / 3.6e6) * dt_yr


def propagate_to_epoch(*, ra, dec, pmra, pmdec, parallax_mas, rv_kms,
                       from_epoch_jyear: float, to_epoch_jyear: float,
                       policy: MissingParallaxPolicy = MissingParallaxPolicy.FLAG) -> dict[str, Any]:
    import astropy.units as u
    from astropy.coordinates import Distance, SkyCoord
    from astropy.time import Time

    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    pmra = np.atleast_1d(np.asarray(pmra, dtype=float))
    pmdec = np.atleast_1d(np.asarray(pmdec, dtype=float))
    parallax = np.atleast_1d(np.asarray(parallax_mas, dtype=float))
    rv = np.zeros_like(ra) if rv_kms is None else np.atleast_1d(np.asarray(rv_kms, dtype=float))
    shape = ra.shape
    if not all(x.shape == shape for x in (dec, pmra, pmdec, parallax, rv)):
        raise ValueError("all input arrays must have the same shape")

    out_ra = ra.copy()
    out_dec = dec.copy()
    flags = np.array(["ok"] * len(ra), dtype=object)
    drop = np.zeros(len(ra), dtype=bool)
    good = np.isfinite(parallax) & (parallax > 0)
    bad = ~good
    if bad.any():
        flags[bad] = np.where(np.isfinite(parallax[bad]), "negative_parallax", "missing_parallax")
        if policy == MissingParallaxPolicy.DROP:
            drop[bad] = True
        elif policy == MissingParallaxPolicy.ASSUME_FAR:
            out_ra[bad], out_dec[bad] = _linear_pm(ra[bad], dec[bad], pmra[bad], pmdec[bad],
                                                   to_epoch_jyear - from_epoch_jyear)
            flags[bad] = "assume_far"

    if good.any():
        coord = SkyCoord(ra=ra[good] * u.deg, dec=dec[good] * u.deg,
                         distance=Distance(parallax=parallax[good] * u.mas),
                         pm_ra_cosdec=pmra[good] * u.mas / u.yr,
                         pm_dec=pmdec[good] * u.mas / u.yr,
                         radial_velocity=np.nan_to_num(rv[good], nan=0.0) * u.km / u.s,
                         obstime=Time(from_epoch_jyear, format="jyear", scale="tcb"))
        moved = coord.apply_space_motion(new_obstime=Time(to_epoch_jyear, format="jyear", scale="tcb"))
        if moved.ra is None or moved.dec is None:
            raise RuntimeError("apply_space_motion returned coordinates without ra/dec")
        out_ra[good] = moved.ra.deg
        out_dec[good] = moved.dec.deg
    return {"ra": out_ra, "dec": out_dec, "motion_flag": flags, "drop": drop}
