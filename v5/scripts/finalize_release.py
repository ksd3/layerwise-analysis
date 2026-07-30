"""Aggregate finalized shards into a release manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmu.records import read_jsonl, write_jsonl


def summarize_final_shards(final_paths: list[Path]) -> dict[str, Any]:
    total = 0
    populations: dict[str, int] = {}
    splits: dict[str, int] = {}
    for path in final_paths:
        rows = read_jsonl(path)
        total += len(rows)
        for row in rows:
            populations[row["population"]] = populations.get(row["population"], 0) + 1
            splits[row.get("split", "unknown")] = splits.get(row.get("split", "unknown"), 0) + 1
    return {"n_rows": total, "populations": populations, "splits": splits,
            "shards": [str(p) for p in final_paths]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    args = ap.parse_args()
    root = Path(args.release_root)
    final_paths = sorted((root / "final").glob("population=*/shard=*.jsonl"))
    if not final_paths:
        raise SystemExit(f"no final shards under {root / 'final'}")
    manifest = summarize_final_shards(final_paths)
    release_dir = root / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    combined: list[dict[str, Any]] = []
    for path in final_paths:
        combined.extend(read_jsonl(path))
    write_jsonl(combined, release_dir / "data.jsonl")
    print(f"wrote release manifest with {manifest['n_rows']} rows to {release_dir}")


if __name__ == "__main__":
    main()
