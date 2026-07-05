#!/bin/bash
# Shared cluster-agnostic setup for OmniSky v5 SLURM wrappers.
#
# Scheduler options are intentionally not hardcoded here. Pass site-specific
# account/partition/QOS values with SBATCH_* variables or sbatch flags, e.g.:
#   SBATCH_ACCOUNT=my_alloc SBATCH_PARTITION=cpu sbatch slurm/probe.sbatch
#   sbatch --account=my_alloc --partition=cpu slurm/probe.sbatch

omnisky_activate_env() {
  if [[ -n "${OMNISKY_ENV_ACTIVATE:-}" ]]; then
    eval "${OMNISKY_ENV_ACTIVATE}"
  elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${OMNISKY_CONDA_ENV:-omnisky}"
  fi

  export OMNISKY_PYTHON_BIN="${OMNISKY_PYTHON_BIN:-python}"
  "${OMNISKY_PYTHON_BIN}" -c 'import sys; print(f"OmniSky Python: {sys.executable}")'
}

omnisky_cd_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.."
}
