"""Atomic Parquet writes + integrity-checked DONE markers + resume state machine."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from importlib import import_module


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(data: bytes, final_path: Path) -> None:
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final_path)
    dfd = os.open(str(final_path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def atomic_write_parquet(table, final_path: Path) -> None:
    pq = import_module("pyarrow.parquet")

    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + f".tmp.{os.getpid()}")
    pq.write_table(table, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final_path)


def marker_path(final_path: Path) -> Path:
    return Path(str(final_path) + ".done.json")


def write_done_marker(final_path: Path, *, manifest_hash: str, schema_version: str, code_sha: str) -> None:
    final_path = Path(final_path)
    meta = {"manifest_hash": manifest_hash, "schema_version": schema_version,
            "code_sha": code_sha, "byte_size": final_path.stat().st_size,
            "checksum": file_checksum(final_path)}
    atomic_write_bytes(json.dumps(meta, sort_keys=True).encode(), marker_path(final_path))


def read_marker(final_path: Path) -> dict[str, Any] | None:
    path = marker_path(Path(final_path))
    return json.loads(path.read_text()) if path.exists() else None


def unit_state(final_path: Path, *, manifest_hash: str, schema_version: str, code_sha: str) -> str:
    final_path = Path(final_path)
    if not final_path.exists():
        return "pending"
    marker = read_marker(final_path)
    if marker is None:
        return "suspicious"
    if (marker.get("manifest_hash"), marker.get("schema_version"), marker.get("code_sha")) != (manifest_hash, schema_version, code_sha):
        return "stale"
    if marker.get("checksum") != file_checksum(final_path):
        return "corrupt"
    return "complete"
