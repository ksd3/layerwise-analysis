"""Create the authoritative source × shard manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmu.coordination import build_or_attach_manifest
from mmu.records import read_jsonl, seed_path


def enumerate_units(*, sources: list[str], n_objects: int, shard_size: int, population: str = "galaxy") -> list[dict[str, Any]]:
    if n_objects < 0 or shard_size <= 0:
        raise ValueError("n_objects must be >=0 and shard_size >0")
    n_shards = (n_objects + shard_size - 1) // shard_size
    return [{"population": population, "source": source, "shard": shard,
             "row_start": shard * shard_size, "row_end": min((shard + 1) * shard_size, n_objects)}
            for shard in range(n_shards) for source in sources]


def inputs_hash_for_seed(seed_file: Path, *, sources: list[str], shard_size: int) -> str:
    h = hashlib.sha256()
    h.update(seed_file.read_bytes())
    h.update(json.dumps({"sources": sources, "shard_size": shard_size}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--n-objects", type=int, required=True)
    ap.add_argument("--shard-size", type=int, default=50_000)
    ap.add_argument("--population", default="galaxy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--release-root", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--inputs-hash", default="manual")
    args = ap.parse_args()
    seed_file = Path(args.seed) if args.seed else (seed_path(args.release_root, population=args.population) if args.release_root else None)
    n_objects = args.n_objects
    inputs_hash = args.inputs_hash
    sources = [s for s in args.sources.split(",") if s]
    if seed_file is not None and seed_file.exists():
        n_objects = len(read_jsonl(seed_file))
        if args.inputs_hash == "manual":
            inputs_hash = inputs_hash_for_seed(seed_file, sources=sources, shard_size=args.shard_size)
    units = enumerate_units(sources=args.sources.split(","), n_objects=args.n_objects,
                            shard_size=args.shard_size, population=args.population)
    units = enumerate_units(sources=sources, n_objects=n_objects,
                            shard_size=args.shard_size, population=args.population)
    out_dir = Path(args.out)
    mhash, _ = build_or_attach_manifest(out_dir, units, inputs_hash=inputs_hash)
    (out_dir / "work_units.json").write_text(json.dumps({"manifest_hash": mhash, "units": units}, indent=2, sort_keys=True))
    print(f"wrote {len(units)} units to {out_dir / 'work_units.json'}")


if __name__ == "__main__":
    main()
