#!/usr/bin/env python3
"""
Multimodal Universe Pipeline v3 — entry point.

Usage:
  TEST_MODE=1 python3 run_pipeline.py   # local test, 5 objects per population
  sbatch run_pipeline.sh                 # full run on Delta AI
"""

from pipeline.config import PipelineConfig
from pipeline.runner import run_pipeline

if __name__ == "__main__":
    config = PipelineConfig.from_env()
    if config.test_mode:
        print("╔══════════════════════════════════════════╗")
        print("║  TEST MODE: 5 objects per population     ║")
        print("╚══════════════════════════════════════════╝")
    run_pipeline(config)
