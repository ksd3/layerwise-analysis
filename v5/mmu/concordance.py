"""Concordance of our matches against a reference cross-matched catalog."""
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy_healpix import HEALPix
from typing import Any


def match_concordance(our_ra, our_dec, ref_ra, ref_dec, tol_arcsec: float = 1.0) -> dict[str, Any]:
    if tol_arcsec <= 0:
        raise ValueError("tol_arcsec must be positive")
    our = SkyCoord(np.asarray(our_ra) * u.deg, np.asarray(our_dec) * u.deg)
    ref = SkyCoord(np.asarray(ref_ra) * u.deg, np.asarray(ref_dec) * u.deg)
    n_ref = len(ref)
    if len(our) == 0 or n_ref == 0:
        return {"n_ours": len(our), "n_ref": n_ref, "recovered": 0,
                "recall": 0.0, "median_sep_arcsec": float("nan")}
    _, idx_ref, sep, _ = search_around_sky(our, ref, tol_arcsec * u.arcsec)
    recovered = int(np.unique(idx_ref).size)
    n_pairs = int(len(sep))
    return {"n_ours": len(our), "n_ref": n_ref, "recovered": recovered,
            "recall": recovered / n_ref,
            "n_pairs_within_tolerance": n_pairs,
            "duplicate_pairs": max(0, n_pairs - recovered),
            "median_sep_arcsec": float(np.median(np.asarray(sep.arcsec, dtype=np.float64))) if len(sep) else float("nan")}


def filter_reference_to_our_footprint(our_ra, our_dec, ref_ra, ref_dec,
                                      footprint_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a reference table to the small sky patch actually probed.

    Phase 0 compares a one-pixel LSDB run against Smith42. If we computed recall
    against the full Smith42 table, a correct one-pixel result would look like a
    failure. This helper keeps only Smith42 rows near our probed matches.
    """
    if footprint_arcsec <= 0:
        raise ValueError("footprint_arcsec must be positive")
    ref_ra_arr = np.asarray(ref_ra, dtype=np.float64)
    ref_dec_arr = np.asarray(ref_dec, dtype=np.float64)
    if len(ref_ra_arr) == 0:
        return ref_ra_arr, ref_dec_arr
    our = SkyCoord(np.asarray(our_ra, dtype=np.float64) * u.deg,
                   np.asarray(our_dec, dtype=np.float64) * u.deg)
    if len(our) == 0:
        return ref_ra_arr[:0], ref_dec_arr[:0]
    ref = SkyCoord(ref_ra_arr * u.deg, ref_dec_arr * u.deg)
    _, idx_ref, _, _ = search_around_sky(our, ref, footprint_arcsec * u.arcsec)
    keep = np.unique(idx_ref)
    return ref_ra_arr[keep], ref_dec_arr[keep]


def filter_reference_to_healpix_pixel(ref_ra, ref_dec, *, order: int, pixel: int) -> tuple[np.ndarray, np.ndarray]:
    """Restrict reference rows to the exact HEALPix pixel used by a probe.

    Unlike filtering near our returned matches, this denominator is independent
    of our matcher output and therefore cannot make missed references disappear.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    ref_ra_arr = np.asarray(ref_ra, dtype=np.float64)
    ref_dec_arr = np.asarray(ref_dec, dtype=np.float64)
    hp = HEALPix(nside=2 ** order, order="nested")
    ref_pix = np.asarray(hp.lonlat_to_healpix(ref_ra_arr * u.deg, ref_dec_arr * u.deg), dtype=np.int64)
    keep = ref_pix == int(pixel)
    return ref_ra_arr[keep], ref_dec_arr[keep]
