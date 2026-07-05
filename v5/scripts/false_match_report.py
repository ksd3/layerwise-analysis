"""False-match helpers for galaxy and stellar null tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def random_offsets(*, n: int, r_min_arcsec: float, r_max_arcsec: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 0 or r_min_arcsec < 0 or r_max_arcsec < r_min_arcsec:
        raise ValueError("invalid annulus")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(r_min_arcsec ** 2, r_max_arcsec ** 2, n))
    return r * np.cos(theta), r * np.sin(theta)


def gate_false_match(fmr_by_bin: dict[str, float], *, threshold: float = 0.001) -> dict[str, Any]:
    passed = sorted([k for k, v in fmr_by_bin.items() if v <= threshold])
    low = sorted([k for k, v in fmr_by_bin.items() if v > threshold])
    return {"passed_bins": passed, "low_confidence_bins": low}


def pm_scramble(pmra, pmdec, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pmra = np.asarray(pmra, dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    mag = np.hypot(pmra, pmdec)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, len(mag))
    return mag * np.cos(theta), mag * np.sin(theta)


def confusion_radius(match_radius, pm_mas_yr, dt_yr, parallax_mas, sigma_astro) -> np.ndarray:
    return np.sqrt(np.asarray(match_radius) ** 2 + (np.asarray(pm_mas_yr) * dt_yr / 1000.0) ** 2
                   + (np.asarray(parallax_mas) / 1000.0) ** 2 + np.asarray(sigma_astro) ** 2)


def parse_bin_values(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"bin values must be name=value, got {value!r}")
        name, raw = value.split("=", 1)
        out[name] = float(raw)
    return out


def build_report(fmr_by_bin: dict[str, float], *, threshold: float) -> dict[str, Any]:
    gate = gate_false_match(fmr_by_bin, threshold=threshold)
    return {"threshold": threshold, "fmr_by_bin": fmr_by_bin, **gate,
            "passed": len(gate["low_confidence_bins"]) == 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.001)
    ap.add_argument("--bin", action="append", default=[], help="false-match bin as name=value")
    args = ap.parse_args()
    fmr_by_bin = parse_bin_values(args.bin) if args.bin else {"all": 0.0}
    report = build_report(fmr_by_bin, threshold=args.threshold)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
