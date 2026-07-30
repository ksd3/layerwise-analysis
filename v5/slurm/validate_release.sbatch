#!/bin/bash
#SBATCH --job-name=omnisky_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=validate_%j.log
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
omnisky_cd_repo_root
omnisky_activate_env
"${OMNISKY_PYTHON_BIN}" -m scripts.verify_markers --release-root "${RELEASE_ROOT:?}" --manifest "${MANIFEST:?}" --code-sha "${CODE_SHA:-local}"
"${OMNISKY_PYTHON_BIN}" -m scripts.finalize_release --release-root "${RELEASE_ROOT:?}"
"${OMNISKY_PYTHON_BIN}" -m scripts.validate_release --release-root "${RELEASE_ROOT:?}" --min-instruments "${MIN_INSTRUMENTS:-2}"
"${OMNISKY_PYTHON_BIN}" -m scripts.false_match_report --out "${FALSE_MATCH_OUT:-false_match_report.json}" --threshold "${FALSE_MATCH_THRESHOLD:-0.001}"
