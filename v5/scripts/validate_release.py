"""Streaming release validators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mmu.records import read_jsonl


def check_uniqueness(ids) -> dict[str, Any]:
    arr = np.asarray(ids)
    return {"ok": len(np.unique(arr)) == len(arr), "n": int(len(arr))}


def check_min_modalities(counts, k: int) -> dict[str, Any]:
    arr = np.asarray(counts)
    return {"ok": bool((arr >= k).all()), "min": int(arr.min()) if len(arr) else 0}


def validate_rows(rows: list[dict[str, Any]], *, min_instruments: int = 2) -> dict[str, Any]:
    ids = [row["object_uid"] for row in rows]
    counts = [int(row.get("n_instruments_present", 0)) for row in rows]
    uniqueness = check_uniqueness(np.asarray(ids, dtype=object))
    modalities = check_min_modalities(np.asarray(counts, dtype=int), min_instruments)
    required = {"object_uid", "global_object_id", "population", "n_instruments_present", "split"}
    missing_rows = [i for i, row in enumerate(rows) if not required <= set(row)]
    ok = uniqueness["ok"] and modalities["ok"] and not missing_rows
    return {"ok": ok, "n_rows": len(rows), "uniqueness": uniqueness,
            "modalities": modalities, "rows_missing_required": missing_rows[:20]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--min-instruments", type=int, default=2)
    args = ap.parse_args()
    data_path = Path(args.release_root) / "release" / "data.jsonl"
    if not data_path.exists():
        raise SystemExit(f"missing release data: {data_path}")
    result = validate_rows(read_jsonl(data_path), min_instruments=args.min_instruments)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
