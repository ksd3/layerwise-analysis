import json
from scripts.check_phase0_gate import evaluate_gate

def test_gate_passes(tmp_path):
    probe = {"internet_ok": True, "notes": "", "sources": [
        {"name": "mmu_desi_edr_sv3", "reachable": True, "cold_latency_s": 18.0,
         "warm_latency_s": 0.6, "throughput_mb_s": 40.0, "rate_limited": False,
         "n_rows_sampled": 2000}]}
    xm = {"recall": 0.95, "n_matched_pixel": 120, "n_ref_footprint": 100,
          "median_sep_arcsec": 0.2}
    assert evaluate_gate(probe, xm, min_recall=0.8)["passed"] is True

def test_gate_fails_on_no_internet():
    probe = {"internet_ok": False, "notes": "", "sources": []}
    xm = {"recall": 0.99, "n_matched_pixel": 100, "n_ref_footprint": 100,
          "median_sep_arcsec": 0.2}
    res = evaluate_gate(probe, xm, min_recall=0.8)
    assert res["passed"] is False
    assert "internet" in " ".join(res["reasons"]).lower()

def test_gate_fails_on_low_recall():
    probe = {"internet_ok": True, "notes": "", "sources": [
        {"name": "x", "reachable": True, "cold_latency_s": 1.0, "warm_latency_s": 0.5,
         "throughput_mb_s": 10.0, "rate_limited": False, "n_rows_sampled": 10}]}
    xm = {"recall": 0.3, "n_matched_pixel": 100, "n_ref_footprint": 100,
          "median_sep_arcsec": 0.2}
    res = evaluate_gate(probe, xm, min_recall=0.8)
    assert res["passed"] is False


def test_gate_fails_on_empty_probe_sample():
    probe = {"internet_ok": True, "notes": "", "sources": [
        {"name": "x", "reachable": True, "cold_latency_s": 1.0, "warm_latency_s": 0.5,
         "throughput_mb_s": 0.0, "rate_limited": False, "n_rows_sampled": 0}]}
    xm = {"recall": 0.99, "n_matched_pixel": 10, "n_ref_footprint": 10,
          "median_sep_arcsec": 0.2}
    res = evaluate_gate(probe, xm, min_recall=0.8)
    assert res["passed"] is False
    assert "zero rows" in " ".join(res["reasons"])


def test_gate_fails_without_reference_footprint():
    probe = {"internet_ok": True, "notes": "", "sources": [
        {"name": "x", "reachable": True, "cold_latency_s": 1.0, "warm_latency_s": 0.5,
         "throughput_mb_s": 1.0, "rate_limited": False, "n_rows_sampled": 10}]}
    xm = {"recall": 1.0, "n_matched_pixel": 10, "n_ref_footprint": 0,
          "median_sep_arcsec": 0.2}
    assert evaluate_gate(probe, xm, min_recall=0.8)["passed"] is False
