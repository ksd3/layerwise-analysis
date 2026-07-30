"""Phase-0 acceptance gate: internet reachable + sources reachable + crossmatch reproduces
Smith42. Exit non-zero if the gate fails (so the sbatch job surfaces failure)."""
from __future__ import annotations
import argparse
import json
import sys
from typing import Any

def evaluate_gate(probe: dict[str, Any], crossmatch: dict[str, Any], min_recall: float = 0.8) -> dict[str, Any]:
    reasons: list[str] = []
    if not probe.get("internet_ok"):
        reasons.append("compute-node internet NOT reachable (switch to Globus pre-stage)")
    if not probe.get("sources"):
        reasons.append("no sources probed")
    if any(not s["reachable"] for s in probe.get("sources", [])):
        bad = [s["name"] for s in probe["sources"] if not s["reachable"]]
        reasons.append(f"unreachable sources: {bad}")
    if any(s.get("n_rows_sampled", 0) <= 0 for s in probe.get("sources", [])):
        empty = [s["name"] for s in probe["sources"] if s.get("n_rows_sampled", 0) <= 0]
        reasons.append(f"sources sampled zero rows: {empty}")
    if crossmatch.get("n_ref_footprint", crossmatch.get("n_ref", 0)) <= 0:
        reasons.append("zero Smith42 reference rows in probed footprint")
    if crossmatch.get("recall", 0.0) < min_recall:
        reasons.append(f"crossmatch recall {crossmatch.get('recall')} < {min_recall}")
    if crossmatch.get("n_matched_pixel", 0) <= 0:
        reasons.append("zero matches in probe pixel")
    return {"passed": len(reasons) == 0, "reasons": reasons}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--crossmatch", required=True)
    ap.add_argument("--min-recall", type=float, default=0.8)
    args = ap.parse_args()
    with open(args.probe) as f:
        probe = json.load(f)
    with open(args.crossmatch) as f:
        crossmatch = json.load(f)
    res = evaluate_gate(probe, crossmatch, args.min_recall)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["passed"] else 1)

if __name__ == "__main__":
    main()
