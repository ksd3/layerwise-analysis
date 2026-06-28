#!/bin/bash
#SBATCH --job-name=omnisky_release
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=release_%j.log
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
omnisky_cd_repo_root
omnisky_activate_env
"${OMNISKY_PYTHON_BIN}" -m scripts.finalize_release --release-root "${RELEASE_ROOT:?}"
"${OMNISKY_PYTHON_BIN}" -m scripts.validate_release --release-root "${RELEASE_ROOT:?}" --min-instruments "${MIN_INSTRUMENTS:-2}"
args=(--release-root "${RELEASE_ROOT:?}" --repo "${HF_REPO:?}")
if [[ -n "${DRY_RUN:-}" ]]; then args+=(--dry-run); fi
"${OMNISKY_PYTHON_BIN}" -m scripts.upload_hf "${args[@]}"
