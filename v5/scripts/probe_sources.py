"""Phase-0 probe: reachability + cold/warm latency + throughput for one MMU HATS source.
Run ON A COMPUTE NODE (not login). Writes probe_report.json."""
from __future__ import annotations
import argparse, time
import lsdb
from mmu.reachability import probe_all
from mmu.probe_report import ProbeReport, SourceProbe

# (catalog, org) — org namespaces differ; see spec sec 3.2
SOURCES = {
    "mmu_desi_edr_sv3": "UniverseTBD",
    "mmu_hsc_pdr3_dud_22.5": "UniverseTBD",
}

def time_open(uri: str, pixel: tuple[int, int], columns: list[str]):
    t0 = time.monotonic()
    cat = lsdb.open_catalog(uri, search_filter=lsdb.PixelSearch([pixel]), columns=columns)
    df = cat.compute()
    return df, time.monotonic() - t0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--pixel", type=int, default=257)
    ap.add_argument("--out", default="probe_report.json")
    args = ap.parse_args()

    reach = probe_all()
    rep = ProbeReport(internet_ok=reach["internet_ok"], notes=f"reachability={reach}")
    px = (args.order, args.pixel)
    for name, org in SOURCES.items():
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
                                rate_limited=False, n_rows_sampled=0))
            print(f"PROBE FAIL {name}: {e}")
    rep.to_json(args.out)
    print(f"wrote {args.out}; internet_ok={rep.internet_ok}; sources={len(rep.sources)}")

if __name__ == "__main__":
    main()
