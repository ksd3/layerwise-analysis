#!/bin/bash
#SBATCH --job-name=omnisky_run_source
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=run_source_%A_%a.log
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
omnisky_cd_repo_root
omnisky_activate_env
args=(--manifest "${MANIFEST:?}" --task-id "${SLURM_ARRAY_TASK_ID:?}" --partition-id "${PARTITION_ID:-0}" --num-partitions "${NUM_PARTITIONS:-1}" --release-root "${RELEASE_ROOT:?}" --code-sha "${CODE_SHA:-local}")
if [[ -n "${TEST_MODE:-}" ]]; then args+=(--test-mode); fi
"${OMNISKY_PYTHON_BIN}" -m scripts.run_source_shard "${args[@]}"
