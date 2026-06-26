from mmu.probe_report import SourceProbe, ProbeReport


def test_roundtrip_json(tmp_path):
    rep = ProbeReport(internet_ok=True, notes="delta cpu node")
    rep.add(SourceProbe(name="mmu_desi_edr_sv3", reachable=True,
                        cold_latency_s=18.2, warm_latency_s=0.6,
                        throughput_mb_s=42.5, rate_limited=False, n_rows_sampled=2000))
    p = tmp_path / "probe_report.json"
    rep.to_json(p)
    back = ProbeReport.from_json(p)
    assert back.internet_ok is True
    assert back.sources[0].name == "mmu_desi_edr_sv3"
    assert back.sources[0].throughput_mb_s == 42.5


def test_concurrency_cap_from_throughput():
    sp = SourceProbe(name="x", reachable=True, cold_latency_s=1.0, warm_latency_s=0.5,
                     throughput_mb_s=10.0, rate_limited=True, n_rows_sampled=100)
    assert sp.suggested_concurrency() == 2
    sp2 = SourceProbe(name="y", reachable=True, cold_latency_s=1.0, warm_latency_s=0.5,
                      throughput_mb_s=10.0, rate_limited=False, n_rows_sampled=100)
    assert sp2.suggested_concurrency() == 16
