"""Fail-fast HEALPix helpers for HATS/NESTED indexing."""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy_healpix import HEALPix


def healpix(order: int) -> HEALPix:
    if order < 0:
        raise ValueError("order must be non-negative")
    return HEALPix(nside=2 ** order, order="nested")


def seed_pixel_set(ra_deg, dec_deg, *, order: int) -> set[int]:
    hp = healpix(order)
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=np.float64))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=np.float64))
    if ra.shape != dec.shape:
        raise ValueError("ra/dec shape mismatch")
    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    return {int(p) for p in np.asarray(pix)}


def neighbor_pixels(pixel: int, *, order: int) -> set[int]:
    hp = healpix(order)
    neighbours = np.asarray(hp.neighbours(int(pixel)))
    return {int(pixel)} | {int(p) for p in neighbours if int(p) >= 0}
