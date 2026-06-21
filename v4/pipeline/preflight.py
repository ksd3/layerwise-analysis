"""Preflight checks: validate all sources before starting."""

import os
import sys
import time

from .config import PipelineConfig


def preflight_check(config: PipelineConfig, sources: list) -> list:
    """Run preflight checks for all sources + disk space.

    Args:
        config: Pipeline configuration
        sources: List of DataSource instances

    Returns:
        List of (name, status, message) tuples
    """
    print("\nPHASE 0: Preflight checks...")
    checks = []

    def check(name, fn):
        try:
            ok, msg = fn()
            status = "PASS" if ok else "FAIL"
            checks.append((name, status, msg))
            sym = "✓" if ok else "✗"
            print(f"  {sym} {name}: {msg}")
            return ok
        except Exception as e:
            checks.append((name, "FAIL", str(e)))
            print(f"  ✗ {name}: {e}")
            return False

    # Disk space
    def check_disk():
        st = os.statvfs(config.work_dir)
        free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        ok = free_gb >= 200
        return ok, f"{free_gb:.0f} GB free {'(need 200)' if not ok else ''}"
    check("Disk space", check_disk)

    # Run each source's preflight
    for source in sources:
        check(source.name, source.preflight)

    # HuggingFace token
    if config.hf_token:
        def check_hf():
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.whoami(token=config.hf_token)
            return True, f"Authenticated as {info.get('name', 'unknown')}"
        check("HuggingFace token", check_hf)

    # Summary
    n_pass = sum(1 for _, s, _ in checks if s == "PASS")
    n_fail = sum(1 for _, s, _ in checks if s == "FAIL")
    print(f"\n  Preflight: {n_pass} passed, {n_fail} failed")

    # Critical: disk space must pass (unless test mode)
    if not config.test_mode:
        critical_fails = [(n, m) for n, s, m in checks if s == "FAIL" and n == "Disk space"]
        if critical_fails:
            print("\n  CRITICAL FAILURE: Not enough disk space. Aborting.")
            sys.exit(1)

    return checks
