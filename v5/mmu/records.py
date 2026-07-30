"""Small JSONL record helpers used by TEST_MODE and orchestration glue."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmu.io_atomic import atomic_write_bytes


Record = dict[str, Any]


def write_jsonl(records: list[Record], path: str | Path) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    atomic_write_bytes(payload.encode(), Path(path))


def read_jsonl(path: str | Path) -> list[Record]:
    rows: list[Record] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_shard_path(release_root: str | Path, *, population: str, source: str, shard: int) -> Path:
    return Path(release_root) / f"source={source}" / f"population={population}" / f"shard={shard:06d}.jsonl"


def final_shard_path(release_root: str | Path, *, population: str, shard: int) -> Path:
    return Path(release_root) / "final" / f"population={population}" / f"shard={shard:06d}.jsonl"


def seed_path(release_root: str | Path, *, population: str) -> Path:
    return Path(release_root) / "seeds" / f"population={population}" / "seed.jsonl"
