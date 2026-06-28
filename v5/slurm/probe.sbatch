#!/bin/bash
#SBATCH --job-name=omnisky_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=probe_%j.log
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
omnisky_cd_repo_root
omnisky_activate_env
"${OMNISKY_PYTHON_BIN}" -m scripts.probe_sources --order "${PROBE_ORDER:-4}" --pixel "${PROBE_PIXEL:-257}" --out "${PROBE_OUT:-probe_report.json}"
"${OMNISKY_PYTHON_BIN}" -m scripts.probe_crossmatch --order "${PROBE_ORDER:-4}" --pixel "${PROBE_PIXEL:-257}" --smith42-revision "${SMITH42_REVISION:?set immutable HF revision}" --out "${CROSSMATCH_OUT:-crossmatch_probe.json}"
"${OMNISKY_PYTHON_BIN}" -m scripts.check_phase0_gate --probe "${PROBE_OUT:-probe_report.json}" --crossmatch "${CROSSMATCH_OUT:-crossmatch_probe.json}"
