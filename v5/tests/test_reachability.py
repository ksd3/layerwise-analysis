from mmu.reachability import summarize_reachability


def test_all_reachable():
    results = {"huggingface": (True, 0.21), "s3": (True, 0.10), "cds": (True, 0.55)}
    s = summarize_reachability(results)
    assert s["internet_ok"] is True
    assert s["unreachable"] == []


def test_one_unreachable_blocks():
    results = {"huggingface": (True, 0.21), "s3": (False, None), "cds": (True, 0.55)}
    s = summarize_reachability(results)
    assert s["internet_ok"] is False
    assert s["unreachable"] == ["s3"]
