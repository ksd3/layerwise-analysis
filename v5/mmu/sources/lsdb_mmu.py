"""Generic MMU/HATS reader helpers over lsdb."""
from __future__ import annotations
from importlib import import_module


def collection_uri(org: str, dataset: str) -> str:
    if not org or not dataset:
        raise ValueError("org and dataset are required")
    return f"hf://datasets/{org}/{dataset}"


def candidate_dict_from_crossmatch(seed_idx, src_idx, sep_arcsec, quality):
    out: dict[int, list[tuple[float, int, float]]] = {}
    for s, c, d, q in zip(seed_idx, src_idx, sep_arcsec, quality):
        out.setdefault(int(s), []).append((float(d), int(c), float(q)))
    return out


def read_pixel_crossmatch(seed_cat, org: str, dataset: str, columns, order: int, pixel: int, radius_arcsec: float):
    lsdb = import_module("lsdb")

    src = lsdb.open_catalog(collection_uri(org, dataset),
                            search_filter=lsdb.PixelSearch([(order, pixel)]),
                            columns=list(columns))
    return seed_cat.crossmatch(src, radius_arcsec=radius_arcsec, n_neighbors=5).compute()
