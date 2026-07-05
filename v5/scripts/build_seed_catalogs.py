"""Build population seed catalogs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from mmu.ids import assign_global_id
from mmu.records import Record, seed_path, write_jsonl
from mmu.sources.local_fits import dedup_highest_snr


POPULATION_NAMESPACE = {"galaxy": 0, "star": 1, "agn": 2}


def namespace_ids(ids, population: str) -> np.ndarray:
    """Return a collision-free population-scoped object UID.

    Order-29 HEALPix IDs use roughly 62 bits. Three populations cannot be
    packed injectively into a signed int64 without losing HEALPix fidelity, so
    the namespaced release key is a string while ``global_object_id`` remains
    the raw MMU-compatible int64 HEALPix index.
    """
    if population not in POPULATION_NAMESPACE:
        raise ValueError(f"unknown population: {population}")
    base = np.asarray(ids, dtype=np.int64)
    return np.asarray([f"{population}:{int(value)}" for value in base], dtype=object)


def assign_ids_and_assert_unique(ra, dec, *, population: str = "galaxy") -> np.ndarray:
    ids = namespace_ids(assign_global_id(ra, dec), population)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("global_object_id collision in seed")
    return ids


def build_synthetic_seed(*, population: str, n: int) -> list[Record]:
    if n <= 0:
        raise ValueError("n must be positive")
    offsets = {"galaxy": 10.0, "star": 110.0, "agn": 210.0}
    if population not in offsets:
        raise ValueError(f"unknown population: {population}")
    ra = offsets[population] + np.arange(n, dtype=float) * 0.01
    dec = np.full(n, 2.0 if population != "agn" else -2.0, dtype=float)
    global_ids = assign_global_id(ra, dec)
    object_uids = namespace_ids(global_ids, population)
    rows: list[Record] = []
    for i in range(n):
        row: Record = {
            "object_uid": str(object_uids[i]),
            "global_object_id": int(global_ids[i]),
            "seed_ra_deg": float(ra[i]),
            "seed_dec_deg": float(dec[i]),
            "population": population,
            "native_id": f"{population}-{i:04d}",
        }
        if population == "star":
            row.update({"pmra": 1000.0 if i == 0 else 5.0, "pmdec": 0.0,
                        "parallax": 50.0 if i == 0 else 1.0, "rv": 0.0})
        rows.append(row)
    return rows


def read_seed_csv(path: str | Path, *, population: str) -> list[Record]:
    with Path(path).open(newline="") as f:
        raw = list(csv.DictReader(f))
    ra = np.asarray([float(row["ra"]) for row in raw], dtype=float)
    dec = np.asarray([float(row["dec"]) for row in raw], dtype=float)
    global_ids = assign_global_id(ra, dec)
    object_uids = namespace_ids(global_ids, population)
    rows: list[Record] = []
    for i, row in enumerate(raw):
        out: Record = dict(row)
        out.update({"object_uid": str(object_uids[i]), "global_object_id": int(global_ids[i]),
                    "seed_ra_deg": float(ra[i]), "seed_dec_deg": float(dec[i]),
                    "population": population})
        rows.append(out)
    if len({row["object_uid"] for row in rows}) != len(rows):
        raise ValueError("object_uid collision in seed")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=["galaxy", "star", "agn"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--release-root", default=None)
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--input-csv", default=None)
    args = ap.parse_args()
    if args.test_mode:
        records = build_synthetic_seed(population=args.population, n=args.n)
    elif args.input_csv:
        records = read_seed_csv(args.input_csv, population=args.population)
    else:
        raise SystemExit("provide --test-mode or --input-csv for seed construction")
    out = seed_path(args.release_root, population=args.population) if args.release_root else Path(args.out)
    write_jsonl(records, out)
    print(f"wrote {len(records)} {args.population} seeds to {out}")


if __name__ == "__main__":
    main()
