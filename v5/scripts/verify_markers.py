"""Release marker audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmu.io_atomic import unit_state
from mmu.records import final_shard_path, source_shard_path


def audit_states(states: list[str]) -> dict[str, Any]:
    bad = sorted({state for state in states if state != "complete"})
    return {"passed": not bad, "bad_states": bad}


def collect_states(*, release_root: str | Path, manifest_path: str | Path,
                   schema_version: str, code_sha: str, include_final: bool = True) -> list[str]:
    with Path(manifest_path).open() as f:
        manifest = json.load(f)
    states: list[str] = []
    shards_by_pop: set[tuple[str, int]] = set()
    for unit in manifest["units"]:
        states.append(unit_state(source_shard_path(release_root, population=unit["population"],
                                                   source=unit["source"], shard=int(unit["shard"])),
                                 manifest_hash=manifest["manifest_hash"],
                                 schema_version=schema_version, code_sha=code_sha))
        shards_by_pop.add((unit["population"], int(unit["shard"])))
    if include_final:
        for population, shard in sorted(shards_by_pop):
            states.append(unit_state(final_shard_path(release_root, population=population, shard=shard),
                                     manifest_hash=manifest["manifest_hash"],
                                     schema_version=schema_version, code_sha=code_sha))
    return states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--schema-version", default="v5.1")
    ap.add_argument("--code-sha", default="local")
    args = ap.parse_args()
    states = collect_states(release_root=args.release_root, manifest_path=args.manifest,
                            schema_version=args.schema_version, code_sha=args.code_sha)
    result = audit_states(states)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
