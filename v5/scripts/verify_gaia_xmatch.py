"""Single-job CDS XMatch helpers for Gaia verification."""
from __future__ import annotations

import argparse


def chunk_for_xmatch(n_rows: int, *, max_rows: int = 2_000_000) -> list[tuple[int, int]]:
    if n_rows < 0 or max_rows <= 0:
        raise ValueError("invalid row counts")
    return [(start, min(start + max_rows, n_rows)) for start in range(0, n_rows, max_rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raise SystemExit(f"CDS XMatch is intentionally single-job/cluster-only; input={args.input}, out={args.out}")


if __name__ == "__main__":
    main()
