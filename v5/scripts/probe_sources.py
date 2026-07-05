"""Phase-0 probe: reachability + cold/warm latency + throughput for MMU HATS.

Run on a compute node, not a login node. The output feeds the Phase-0 gate and
later per-service concurrency defaults.
"""
from __future__ import annotations
import argparse
import time
from importlib import import_module

from mmu.probe_report import ProbeReport, SourceProbe
from mmu.reachability import probe_all

# (catalog, org) — org namespaces differ; see spec sec 3.2
SOURCES = {
    "mmu_desi_edr_sv3": "UniverseTBD",
    "mmu_hsc_pdr3_dud_22.5": "UniverseTBD",
}

def time_open(uri: str, pixel: tuple[int, int], columns: list[str]):
    lsdb = import_module("lsdb")
    t0 = time.monotonic()
    cat = lsdb.open_catalog(uri, search_filter=lsdb.PixelSearch([pixel]), columns=columns)
    df = cat.compute()
    return df, time.monotonic() - t0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--pixel", type=int, default=257)
    ap.add_argument("--out", default="probe_report.json")
    ap.add_argument("--sources", default="all", help="comma-separated source keys, or all")
    args = ap.parse_args()

    reach = probe_all()
    rep = ProbeReport(internet_ok=reach["internet_ok"], notes=f"reachability={reach}")
    px = (args.order, args.pixel)
    selected = SOURCES if args.sources == "all" else {s: SOURCES[s] for s in args.sources.split(",")}
    for name, org in selected.items():
        uri = f"hf://datasets/{org}/{name}"
        try:
            df_cold, cold = time_open(uri, px, ["ra", "dec"])
            _, warm = time_open(uri, px, ["ra", "dec"])
            nbytes = int(df_cold.memory_usage(deep=True).sum())
            tput = (nbytes / 1e6) / cold if cold > 0 else 0.0
            rep.add(SourceProbe(name=name, reachable=True, cold_latency_s=round(cold, 2),
                                warm_latency_s=round(warm, 2), throughput_mb_s=round(tput, 2),
                                rate_limited=False, n_rows_sampled=len(df_cold)))
        except Exception as e:
            rep.add(SourceProbe(name=name, reachable=False, cold_latency_s=-1.0,
                                warm_latency_s=-1.0, throughput_mb_s=0.0,
                                rate_limited=False, n_rows_sampled=0, error=repr(e)))
            print(f"PROBE FAIL {name}: {e}")
    rep.to_json(args.out)
    print(f"wrote {args.out}; internet_ok={rep.internet_ok}; sources={len(rep.sources)}")

if __name__ == "__main__":
    main()
