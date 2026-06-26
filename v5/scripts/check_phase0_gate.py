"""Phase-0 acceptance gate: internet reachable + sources reachable + crossmatch reproduces
Smith42. Exit non-zero if the gate fails (so the sbatch job surfaces failure)."""
from __future__ import annotations
import argparse, json, sys

def evaluate_gate(probe: dict, crossmatch: dict, min_recall: float = 0.8) -> dict:
    reasons: list[str] = []
    if not probe.get("internet_ok"):
        reasons.append("compute-node internet NOT reachable (switch to Globus pre-stage)")
    if not probe.get("sources"):
        reasons.append("no sources probed")
    if any(not s["reachable"] for s in probe.get("sources", [])):
        bad = [s["name"] for s in probe["sources"] if not s["reachable"]]
        reasons.append(f"unreachable sources: {bad}")
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
    res = evaluate_gate(json.load(open(args.probe)), json.load(open(args.crossmatch)),
                        args.min_recall)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["passed"] else 1)

if __name__ == "__main__":
    main()
