"""Concordance of our matches against a reference (e.g. Smith42) cross-matched catalog."""
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky

def match_concordance(our_ra, our_dec, ref_ra, ref_dec, tol_arcsec: float = 1.0) -> dict:
    our = SkyCoord(np.asarray(our_ra) * u.deg, np.asarray(our_dec) * u.deg)
    ref = SkyCoord(np.asarray(ref_ra) * u.deg, np.asarray(ref_dec) * u.deg)
    n_ref = len(ref)
    if len(our) == 0 or n_ref == 0:
        return {"n_ours": len(our), "n_ref": n_ref, "recovered": 0,
                "recall": 0.0, "median_sep_arcsec": float("nan")}
    _, idx_ref, sep, _ = search_around_sky(our, ref, tol_arcsec * u.arcsec)
    recovered = int(np.unique(idx_ref).size)
    return {"n_ours": len(our), "n_ref": n_ref, "recovered": recovered,
            "recall": recovered / n_ref,
            "median_sep_arcsec": float(np.median(sep.arcsec)) if len(sep) else float("nan")}
