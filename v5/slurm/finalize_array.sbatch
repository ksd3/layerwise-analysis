#!/bin/bash
#SBATCH --job-name=omnisky_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=finalize_%A_%a.log
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
omnisky_cd_repo_root
omnisky_activate_env
"${OMNISKY_PYTHON_BIN}" -m scripts.finalize_shard --release-root "${RELEASE_ROOT:?}" --manifest "${MANIFEST:?}" --population "${POPULATION:?}" --sources "${SOURCES:?comma-separated}" --shard "${SLURM_ARRAY_TASK_ID:?}" --code-sha "${CODE_SHA:-local}" --min-instruments "${MIN_INSTRUMENTS:-2}"
