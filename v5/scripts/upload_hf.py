"""Hugging Face upload helpers."""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any


def build_repo_id(name: str, *, org: str = "UniverseTBD") -> str:
    return f"{org}/{name}"


def verify_load_back(repo: str) -> dict[str, Any]:
    """Verify both metadata and at least one JSON row can be read back from HF."""
    hf = import_module("huggingface_hub")
    datasets = import_module("datasets")
    hf.hf_hub_download(repo_id=repo, repo_type="dataset", filename="manifest.json")
    dataset = datasets.load_dataset(
        "json",
        data_files=f"hf://datasets/{repo}/data.jsonl",
        split="train",
        streaming=True,
    )
    first = next(iter(dataset), None)
    if first is None:
        raise RuntimeError("load-back verification found zero rows")
    return {"ok": True, "first_keys": sorted(first.keys())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-load-back", action="store_true")
    args = ap.parse_args()
    release_dir = Path(args.release_root) / "release"
    data = release_dir / "data.jsonl"
    manifest = release_dir / "manifest.json"
    if not data.exists() or not manifest.exists():
        raise SystemExit(f"release is incomplete under {release_dir}")
    if args.dry_run:
        print(f"dry-run upload {release_dir} -> {args.repo}")
        return
    api = import_module("huggingface_hub").HfApi()
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(release_dir))
    if not args.skip_load_back:
        result = verify_load_back(args.repo)
        print(f"load-back verified: {result}")
    print(f"uploaded {release_dir} -> {args.repo}")


if __name__ == "__main__":
    main()
