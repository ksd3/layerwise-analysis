"""Manifest creation, partitioning, and finalize barriers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def manifest_hash(units: list[dict[str, Any]], inputs_hash: str) -> str:
    payload = json.dumps({"units": units, "inputs_hash": inputs_hash}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_or_attach_manifest(root: Path, units: list[dict[str, Any]], *, inputs_hash: str) -> tuple[str, bool]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mhash = manifest_hash(units, inputs_hash)
    lock = root / "release.lock"
    manifest = root / "manifest.json"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = json.loads(manifest.read_text())
        if existing["inputs_hash"] != inputs_hash:
            raise ValueError(f"manifest inputs_hash conflict: {existing['inputs_hash']} != {inputs_hash}")
        if existing["manifest_hash"] != mhash:
            raise ValueError("manifest units conflict for the same inputs_hash")
        return existing["manifest_hash"], False
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    manifest.write_text(json.dumps({"manifest_hash": mhash, "inputs_hash": inputs_hash,
                                    "units": units}, sort_keys=True, indent=2))
    return mhash, True


def partition_units(units: list[Any], *, partition_id: int, num_partitions: int) -> list[Any]:
    if num_partitions <= 0 or partition_id < 0 or partition_id >= num_partitions:
        raise ValueError("invalid partition")
    return [u for i, u in enumerate(units) if i % num_partitions == partition_id]


def finalize_ready(shard: int, *, sources: list[str], completed: set[tuple[str, int]]) -> bool:
    return all((source, shard) in completed for source in sources)
