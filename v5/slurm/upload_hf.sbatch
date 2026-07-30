#!/bin/bash
#SBATCH --job-name=omnisky_upload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=upload_%j.log
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
omnisky_cd_repo_root
omnisky_activate_env
args=(--release-root "${RELEASE_ROOT:?}" --repo "${HF_REPO:?}")
if [[ -n "${DRY_RUN:-}" ]]; then args+=(--dry-run); fi
"${OMNISKY_PYTHON_BIN}" -m scripts.upload_hf "${args[@]}"
