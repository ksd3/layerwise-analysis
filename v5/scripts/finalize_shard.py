"""Finalize one seed shard by joining source shards and enforcing release criteria."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from mmu.io_atomic import unit_state, write_done_marker
from mmu.records import final_shard_path, read_jsonl, source_shard_path, write_jsonl


def enforce_min_instruments(mask, *, min_instruments: int) -> np.ndarray:
    return np.asarray(mask, dtype=bool).sum(axis=1) >= min_instruments


def assign_split(ra, dec, *, nside: int = 8, seed: int = 42) -> np.ndarray:
    from astropy_healpix import HEALPix
    import astropy.units as u

    hp = HEALPix(nside=nside, order="nested")
    pix = np.asarray(hp.lonlat_to_healpix(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg))
    labels = []
    for p in pix:
        digest = hashlib.sha256(f"{seed}:{int(p)}".encode()).digest()[0] / 255.0
        labels.append("train" if digest < 0.8 else "val" if digest < 0.9 else "test")
    return np.asarray(labels, dtype=object)


def finalize_records(source_rows_by_source: dict[str, list[dict[str, Any]]], *, min_instruments: int = 2) -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for source, rows in source_rows_by_source.items():
        for row in rows:
            uid = str(row["object_uid"])
            current = by_uid.setdefault(uid, {
                "object_uid": uid,
                "global_object_id": int(row["global_object_id"]),
                "population": row["population"],
                "seed_ra_deg": float(row["match_ra_deg"]),
                "seed_dec_deg": float(row["match_dec_deg"]),
                "sources": [],
                "instrument_presence_mask": [],
                "match_sep_arcsec": {},
                "match_ambiguous": {},
                "n_candidates_within_radius": {},
            })
            current["sources"].append(source)
            current["instrument_presence_mask"].append(bool(row.get("instrument_present", True)))
            current["match_sep_arcsec"][source] = float(row.get("match_sep_arcsec", 0.0))
            current["match_ambiguous"][source] = bool(row.get("match_ambiguous", False))
            current["n_candidates_within_radius"][source] = int(row.get("n_candidates_within_radius", 1))
    out: list[dict[str, Any]] = []
    for row in by_uid.values():
        row["n_instruments_present"] = int(sum(row["instrument_presence_mask"]))
        if row["n_instruments_present"] >= min_instruments:
            out.append(row)
    if out:
        split = assign_split(np.asarray([r["seed_ra_deg"] for r in out]),
                             np.asarray([r["seed_dec_deg"] for r in out]))
        for row, label in zip(out, split):
            row["split"] = str(label)
            row["low_confidence"] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--population", default="galaxy")
    ap.add_argument("--sources", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--schema-version", default="v5.1")
    ap.add_argument("--code-sha", default="local")
    ap.add_argument("--min-instruments", type=int, default=2)
    args = ap.parse_args()
    with open(args.manifest) as f:
        manifest = __import__("json").load(f)
    sources = [s for s in args.sources.split(",") if s]
    source_rows: dict[str, list[dict[str, Any]]] = {}
    completed: set[str] = set()
    for source in sources:
        path = source_shard_path(args.release_root, population=args.population, source=source, shard=args.shard)
        state = unit_state(path, manifest_hash=manifest["manifest_hash"], schema_version=args.schema_version, code_sha=args.code_sha)
        if state != "complete":
            raise SystemExit(f"source shard not complete: {path} state={state}")
        source_rows[source] = read_jsonl(path)
        completed.add(source)
    rows = finalize_records(source_rows, min_instruments=args.min_instruments)
    out = final_shard_path(args.release_root, population=args.population, shard=args.shard)
    write_jsonl(rows, out)
    write_done_marker(out, manifest_hash=manifest["manifest_hash"], schema_version=args.schema_version, code_sha=args.code_sha)
    print(f"wrote {len(rows)} final rows to {out}")


if __name__ == "__main__":
    main()
