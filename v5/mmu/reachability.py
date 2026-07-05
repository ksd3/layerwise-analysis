"""Compute-node outbound reachability for HF / S3 / CDS (Phase-0 critical check, C2)."""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import requests

ENDPOINTS = {
    "huggingface": "https://huggingface.co",
    "s3": "https://ipac-irsa-ztf.s3.amazonaws.com",
    "cds": "https://cdsxmatch.u-strasbg.fr",
}


def check_endpoint(url: str, timeout: float = 10.0) -> tuple[bool, float | None]:
    t0 = time.monotonic()
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code in (403, 405):
            r = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
        return (200 <= r.status_code < 500, time.monotonic() - t0)
    except requests.RequestException:
        return (False, None)


def summarize_reachability(results: Mapping[str, tuple[bool, float | None]]) -> dict[str, Any]:
    unreachable = sorted([k for k, (ok, _) in results.items() if not ok])
    return {
        "internet_ok": len(unreachable) == 0,
        "unreachable": unreachable,
        "latencies": {k: lat for k, (_, lat) in results.items()},
    }


def probe_all(timeout: float = 10.0) -> dict[str, Any]:
    results = {name: check_endpoint(url, timeout) for name, url in ENDPOINTS.items()}
    return summarize_reachability(results)
