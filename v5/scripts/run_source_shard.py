"""SLURM array workhorse helpers."""
from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from mmu.config import SOURCES, Source
from mmu.coordination import partition_units
from mmu.io_atomic import unit_state, write_done_marker
from mmu.records import read_jsonl, seed_path, source_shard_path, write_jsonl


def select_unit(units: list[dict[str, Any]], *, task_id: int, partition_id: int, num_partitions: int) -> dict[str, Any]:
    selected = partition_units(units, partition_id=partition_id, num_partitions=num_partitions)
    if task_id < 0 or task_id >= len(selected):
        raise IndexError("task_id outside selected partition")
    return selected[task_id]


def synthetic_source_rows(seed_rows: list[dict[str, Any]], *, source: str, source_epoch: float = 2016.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in seed_rows:
        match_ra = float(row["seed_ra_deg"])
        match_dec = float(row["seed_dec_deg"])
        if row.get("population") == "star" and "pmra" in row:
            from mmu.matching import match_unit
            prepared = match_unit({"ra": [match_ra], "dec": [match_dec], "pmra": [float(row.get("pmra", 0.0))],
                                   "pmdec": [float(row.get("pmdec", 0.0))],
                                   "parallax": [float(row.get("parallax", 1.0))],
                                   "rv": [float(row.get("rv", 0.0))]},
                                  {"epoch_jyear": source_epoch}, population="star")
            match_ra = float(prepared["ra"][0])
            match_dec = float(prepared["dec"][0])
        rows.append({
            "object_uid": row["object_uid"],
            "global_object_id": int(row["global_object_id"]),
            "population": row["population"],
            "source": source,
            "match_ra_deg": match_ra,
            "match_dec_deg": match_dec,
            "match_sep_arcsec": 0.1,
            "match_ambiguous": False,
            "n_candidates_within_radius": 1,
            "instrument_present": True,
            "payload_ref": f"test://{source}/{row['object_uid']}",
        })
    return rows


def _row_value(row: Any, names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def lsdb_source_rows(seed_rows: list[dict[str, Any]], source: Source) -> list[dict[str, Any]]:
    pandas = import_module("pandas")
    lsdb = import_module("lsdb")
    from mmu.sources.lsdb_mmu import collection_uri

    seed_df = pandas.DataFrame([
        {"object_uid": row["object_uid"], "global_object_id": row["global_object_id"],
         "population": row["population"], "ra": row["seed_ra_deg"], "dec": row["seed_dec_deg"]}
        for row in seed_rows
    ])
    seed_cat = lsdb.from_dataframe(seed_df, ra_column="ra", dec_column="dec")
    src = lsdb.open_catalog(collection_uri(source.org, source.dataset), columns=list(source.columns))
    matched = seed_cat.crossmatch(src, radius_arcsec=source.radius_arcsec, suffixes=("_seed", f"_{source.name}")).compute()
    rows: list[dict[str, Any]] = []
    for _, row in matched.iterrows():
        uid = _row_value(row, ["object_uid_seed", "object_uid"])
        if uid is None:
            continue
        rows.append({
            "object_uid": str(uid),
            "global_object_id": int(_row_value(row, ["global_object_id_seed", "global_object_id"])),
            "population": str(_row_value(row, ["population_seed", "population"])),
            "source": source.name,
            "match_ra_deg": float(_row_value(row, [f"ra_{source.name}", "ra"], np.nan)),
            "match_dec_deg": float(_row_value(row, [f"dec_{source.name}", "dec"], np.nan)),
            "match_sep_arcsec": float(_row_value(row, ["_dist_arcsec"], 0.0)),
            "match_ambiguous": False,
            "n_candidates_within_radius": 1,
            "instrument_present": True,
            "payload_ref": f"{source.name}:{uid}",
        })
    return rows


def legacy_hdf5_source_rows(seed_rows: list[dict[str, Any]], source: Source) -> list[dict[str, Any]]:
    import astropy.units as u
    from astropy.coordinates import SkyCoord, search_around_sky
    from mmu.healpix import seed_pixel_set
    from mmu.sources.legacy_hdf5 import pixel_files, read_cutouts

    candidates: list[tuple[float, float, str]] = []
    pixels = seed_pixel_set([row["seed_ra_deg"] for row in seed_rows],
                            [row["seed_dec_deg"] for row in seed_rows], order=4)
    for pixel in pixels:
        for path in pixel_files(source.dataset, pixel):
            ra, dec, _image = read_cutouts(path)
            candidates.extend((float(r), float(d), str(path)) for r, d in zip(ra, dec))
    if not candidates:
        raise RuntimeError(f"no Legacy HDF5 candidates found under {source.dataset}")
    seed_coord = SkyCoord(np.asarray([r["seed_ra_deg"] for r in seed_rows]) * u.deg,
                          np.asarray([r["seed_dec_deg"] for r in seed_rows]) * u.deg)
    cand_coord = SkyCoord(np.asarray([c[0] for c in candidates]) * u.deg,
                          np.asarray([c[1] for c in candidates]) * u.deg)
    idx_seed, idx_cand, sep, _ = search_around_sky(seed_coord, cand_coord, source.radius_arcsec * u.arcsec)
    best: dict[int, tuple[int, float]] = {}
    sep_values = np.asarray(sep.arcsec, dtype=np.float64)
    for s, c, dist in zip(idx_seed, idx_cand, sep_values):
        si = int(s); ci = int(c); dd = float(dist)
        if si not in best or dd < best[si][1]:
            best[si] = (ci, dd)
    rows: list[dict[str, Any]] = []
    for seed_index, (cand_index, sep_arcsec) in best.items():
        seed = seed_rows[seed_index]
        cand = candidates[cand_index]
        rows.append({"object_uid": seed["object_uid"], "global_object_id": int(seed["global_object_id"]),
                     "population": seed["population"], "source": source.name,
                     "match_ra_deg": cand[0], "match_dec_deg": cand[1],
                     "match_sep_arcsec": sep_arcsec, "match_ambiguous": False,
                     "n_candidates_within_radius": 1, "instrument_present": True,
                     "payload_ref": cand[2]})
    return rows


def live_source_rows(seed_rows: list[dict[str, Any]], *, source_name: str) -> list[dict[str, Any]]:
    if source_name not in SOURCES:
        raise ValueError(f"unknown source: {source_name}")
    source = SOURCES[source_name]
    if source.kind == "lsdb_mmu":
        return lsdb_source_rows(seed_rows, source)
    if source.kind == "legacy_hdf5":
        return legacy_hdf5_source_rows(seed_rows, source)
    raise ValueError(f"source kind {source.kind!r} is not supported by run_source_shard yet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--partition-id", type=int, default=0)
    ap.add_argument("--num-partitions", type=int, default=1)
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--schema-version", default="v5.1")
    ap.add_argument("--code-sha", default="local")
    ap.add_argument("--test-mode", action="store_true")
    args = ap.parse_args()
    with open(args.manifest) as f:
        payload = json.load(f)
    unit = select_unit(payload["units"], task_id=args.task_id,
                       partition_id=args.partition_id, num_partitions=args.num_partitions)
    out = source_shard_path(args.release_root, population=unit["population"],
                            source=unit["source"], shard=int(unit["shard"]))
    state = unit_state(out, manifest_hash=payload["manifest_hash"],
                       schema_version=args.schema_version, code_sha=args.code_sha)
    if state == "complete":
        print(f"skip complete {out}")
        return
    seed_rows = read_jsonl(seed_path(args.release_root, population=unit["population"]))
    shard_rows = seed_rows[int(unit["row_start"]):int(unit["row_end"])]
    rows = (synthetic_source_rows(shard_rows, source=unit["source"])
            if args.test_mode else live_source_rows(shard_rows, source_name=unit["source"]))
    write_jsonl(rows, out)
    write_done_marker(out, manifest_hash=payload["manifest_hash"],
                      schema_version=args.schema_version, code_sha=args.code_sha)
    print(f"wrote {len(rows)} source rows to {out}")


if __name__ == "__main__":
    main()
