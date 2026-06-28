#!/bin/bash
#SBATCH --job-name=omnisky_build_seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=build_seeds_%j.log
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
omnisky_cd_repo_root
omnisky_activate_env
args=(--population "${POPULATION:?galaxy|star|agn}" --release-root "${RELEASE_ROOT:?}" --out "${SEED_OUT:-unused}")
if [[ -n "${TEST_MODE:-}" ]]; then args+=(--test-mode); fi
if [[ -n "${N_OBJECTS:-}" ]]; then args+=(--n "${N_OBJECTS}"); fi
if [[ -n "${INPUT_CSV:-}" ]]; then args+=(--input-csv "${INPUT_CSV}"); fi
"${OMNISKY_PYTHON_BIN}" -m scripts.build_seed_catalogs "${args[@]}"
