"""Concordance of our matches against a reference (e.g. Smith42) cross-matched catalog."""

from __future__ import annotations

import numpy as np


def match_concordance(our_ra, our_dec, ref_ra, ref_dec, tol_arcsec: float = 1.0) -> dict:
    our_ra = np.asarray(our_ra, dtype=float)
    our_dec = np.asarray(our_dec, dtype=float)
    ref_ra = np.asarray(ref_ra, dtype=float)
    ref_dec = np.asarray(ref_dec, dtype=float)
    n_ref = len(ref_ra)
    n_ours = len(our_ra)
    if n_ours == 0 or n_ref == 0:
        return {
            "n_ours": n_ours,
            "n_ref": n_ref,
            "recovered": 0,
            "recall": 0.0,
            "median_sep_arcsec": float("nan"),
        }

    our_ra_rad = np.deg2rad(our_ra)[:, None]
    our_dec_rad = np.deg2rad(our_dec)[:, None]
    ref_ra_rad = np.deg2rad(ref_ra)[None, :]
    ref_dec_rad = np.deg2rad(ref_dec)[None, :]

    delta_ra = ref_ra_rad - our_ra_rad
    delta_dec = ref_dec_rad - our_dec_rad
    hav = (
        np.sin(delta_dec / 2.0) ** 2
        + np.cos(our_dec_rad) * np.cos(ref_dec_rad) * np.sin(delta_ra / 2.0) ** 2
    )
    sep_arcsec = np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))) * 3600.0
    matched = sep_arcsec <= tol_arcsec
    recovered = int(np.flatnonzero(np.any(matched, axis=0)).size)
    matched_sep = sep_arcsec[matched]
    return {
        "n_ours": n_ours,
        "n_ref": n_ref,
        "recovered": recovered,
        "recall": recovered / n_ref,
        "median_sep_arcsec": float(np.median(matched_sep)) if len(matched_sep) else float("nan"),
    }
