"""Match adjudication and population-aware coordinate preparation."""
from __future__ import annotations

import numpy as np
from typing import Any


def adjudicate(candidates: dict[int, list[tuple[float, int, float]]], *, radius_arcsec: float) -> dict[int, dict[str, Any]]:
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    out: dict[int, dict[str, Any]] = {}
    for seed, cands in candidates.items():
        within = [c for c in cands if c[0] <= radius_arcsec]
        if not within:
            continue
        within.sort(key=lambda c: (c[0], -c[2], c[1]))
        best = within[0]
        out[int(seed)] = {"src_index": int(best[1]), "match_sep_arcsec": float(best[0]),
                          "match_ambiguous": len(within) > 1,
                          "n_candidates_within_radius": len(within)}
    return out


def prepare_stellar(ra, dec, pmra, pmdec, parallax_mas, rv_kms, *,
                    from_epoch_jyear: float = 2016.0, to_epoch_jyear: float) -> dict[str, Any]:
    from mmu.motion import MissingParallaxPolicy, propagate_to_epoch

    return propagate_to_epoch(ra=np.asarray(ra), dec=np.asarray(dec), pmra=np.asarray(pmra),
                              pmdec=np.asarray(pmdec), parallax_mas=np.asarray(parallax_mas),
                              rv_kms=np.asarray(rv_kms), from_epoch_jyear=from_epoch_jyear,
                              to_epoch_jyear=to_epoch_jyear,
                              policy=MissingParallaxPolicy.FLAG)


def match_unit(seed: dict[str, Any], source: dict[str, Any], *, population: str) -> dict[str, Any]:
    if population != "star":
        return {"ra": np.asarray(seed["ra"]), "dec": np.asarray(seed["dec"]),
                "motion_flag": np.array(["not_applicable"] * len(np.atleast_1d(seed["ra"])), dtype=object),
                "drop": np.zeros(len(np.atleast_1d(seed["ra"])), dtype=bool)}
    return prepare_stellar(seed["ra"], seed["dec"], seed["pmra"], seed["pmdec"],
                           seed["parallax"], seed.get("rv", np.zeros(len(seed["ra"]))),
                           to_epoch_jyear=float(source["epoch_jyear"]))
