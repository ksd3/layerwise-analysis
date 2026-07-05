"""ZTF public S3/HATS helpers."""
from __future__ import annotations

import numpy as np
from importlib import import_module


def s3_hats_uri(dr: str = "dr24", *, kind: str = "lc") -> str:
    if kind not in {"lc", "objects"}:
        raise ValueError("kind must be lc or objects")
    return f"s3://ipac-irsa-ztf/ztf/enhanced/{dr}/{kind}/hats"


def lightcurve_to_fixed(times, mags, errs, *, max_len: int) -> dict[str, np.ndarray]:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    arrays = [np.asarray(x, dtype=np.float32) for x in (times, mags, errs)]
    n = min(max_len, *(len(a) for a in arrays))
    out = {}
    for name, arr in zip(("time", "mag", "err"), arrays):
        fixed = np.full(max_len, np.nan, dtype=np.float32)
        fixed[:n] = arr[:n]
        out[name] = fixed
    mask = np.zeros(max_len, dtype=bool)
    mask[:n] = True
    out["valid"] = mask
    return out


def read_pixel(order: int, pixel: int, *, dr: str = "dr24"):
    lsdb = import_module("lsdb")

    return lsdb.open_catalog(s3_hats_uri(dr), storage_options={"anon": True},
                             search_filter=lsdb.PixelSearch([(order, pixel)])).compute()
