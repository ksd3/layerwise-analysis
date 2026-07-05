"""Convert non-HATS sources to HATS or explicitly flag them."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mmu.config import Source


def needs_conversion(source: Source) -> bool:
    return source.kind not in {"lsdb_mmu", "ztf_s3"}


def conversion_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    converted = sorted([r["name"] for r in results if r.get("converted")])
    flagged = sorted([{ "name": r["name"], "reason": r.get("reason", "unknown")}
                      for r in results if not r.get("converted")], key=lambda r: r["name"])
    return {"converted": converted, "flagged_unconvertible": flagged}


def source_to_dict(source: Source) -> dict[str, Any]:
    return asdict(source)
