from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mmu.records import read_jsonl
from scripts.validate_release import validate_rows
from scripts.verify_markers import audit_states, collect_states


def run_module(args: list[str], *, cwd: Path) -> None:
    cmd = [sys.executable, "-m", *args]
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_test_mode_end_to_end_galaxy_star_agn(tmp_path):
    root = tmp_path / "release_root"
    manifest_dir = root / "manifest"
    cwd = Path(__file__).resolve().parents[1]
    code_sha = "test"
    for population in ["galaxy", "star", "agn"]:
        run_module(["scripts.build_seed_catalogs", "--population", population,
                    "--release-root", str(root), "--out", "unused", "--test-mode", "--n", "5"], cwd=cwd)
        run_module(["scripts.plan_work_units", "--sources", "desi,hsc", "--n-objects", "0",
                    "--population", population, "--release-root", str(root),
                    "--shard-size", "3", "--out", str(manifest_dir / population)], cwd=cwd)
        manifest = manifest_dir / population / "work_units.json"
        with manifest.open() as f:
            units = json.load(f)["units"]
        for task_id in range(len(units)):
            run_module(["scripts.run_source_shard", "--manifest", str(manifest),
                        "--task-id", str(task_id), "--release-root", str(root),
                        "--code-sha", code_sha, "--test-mode"], cwd=cwd)
        for shard in sorted({int(unit["shard"]) for unit in units}):
            run_module(["scripts.finalize_shard", "--release-root", str(root),
                        "--manifest", str(manifest), "--population", population,
                        "--sources", "desi,hsc", "--shard", str(shard),
                        "--code-sha", code_sha], cwd=cwd)
        states = collect_states(release_root=root, manifest_path=manifest,
                                schema_version="v5.1", code_sha=code_sha)
        assert audit_states(states)["passed"] is True

    run_module(["scripts.finalize_release", "--release-root", str(root)], cwd=cwd)
    result = validate_rows(read_jsonl(root / "release" / "data.jsonl"), min_instruments=2)
    assert result["ok"] is True
    rows = read_jsonl(root / "release" / "data.jsonl")
    assert len(rows) == 15
    assert {row["population"] for row in rows} == {"galaxy", "star", "agn"}
    assert all(row["n_instruments_present"] >= 2 for row in rows)
    assert len({row["object_uid"] for row in rows}) == len(rows)
    run_module(["scripts.validate_release", "--release-root", str(root)], cwd=cwd)
    run_module(["scripts.upload_hf", "--release-root", str(root), "--repo", "UniverseTBD/test", "--dry-run"], cwd=cwd)
