"""Local FITS fallback readers for APOGEE/SDSS-style sources."""
from __future__ import annotations

import numpy as np
from typing import Any, cast


def dedup_highest_snr(ids, snr) -> np.ndarray:
    ids_arr = np.asarray(ids)
    snr_arr = np.asarray(snr, dtype=float)
    keep: dict[object, int] = {}
    for i, ident in enumerate(ids_arr):
        if ident not in keep or snr_arr[i] > snr_arr[keep[ident]]:
            keep[ident] = i
    return np.array(sorted(keep.values()), dtype=int)


def crop_or_pad_flux(flux, length: int) -> np.ndarray:
    arr = np.asarray(flux, dtype=np.float32).reshape(-1)
    out = np.full(length, np.nan, dtype=np.float32)
    n = min(length, len(arr))
    out[:n] = arr[:n]
    return out


def read_apogee(path, *, snr_min: float = 50.0, flux_len: int = 7514):
    from astropy.io import fits

    data = cast(Any, fits.getdata(path))
    mask = np.asarray(data["snr"], dtype=float) >= snr_min
    data = data[mask]
    keep = dedup_highest_snr(data["apogee_id"], data["snr"])
    data = data[keep]
    return {"ra": np.asarray(data["ra"], dtype=float), "dec": np.asarray(data["dec"], dtype=float),
            "flux": np.stack([crop_or_pad_flux(f, flux_len) for f in data["flux"]])}
