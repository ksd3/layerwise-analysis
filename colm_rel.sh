#!/bin/bash
#SBATCH --job-name=colm-rel
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --partition=ghx4
#SBATCH --account=bfir-dtai-gh
#SBATCH --output=colm_rel_%j.log
#SBATCH --error=colm_rel_%j.err
# =============================================================================
#  colm_rel.sh -- scaling-relation / emergence analysis worker (Tiers 0-2).
#
#  Sits on top of colm.py's results: claims any config whose PROBE is done,
#  rebuilds embeddings from the checkpoints already on HF (TEST split only),
#  runs all three tiers, and writes a small relations/<config>/analysis.parquet.
#  No embeddings are stored. Cooperative + idempotent via the same HF locks as
#  colm.py. Drop this next to colm.py at the astropt repo root.
#
#  USAGE:   for i in {1..12}; do sbatch colm_rel.sh; done
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. HF token + shared results repo (must match colm.py's run).
# -----------------------------------------------------------------------------
export HF_TOKEN=mytoken
export COLM_RESULTS_REPO="${COLM_RESULTS_REPO:-HCVYM5w6Gn/colm-results}"
export COLM_NUM_WORKERS="${COLM_NUM_WORKERS:-8}"
# Eval split for the analysis. TEST only, per design (override only if you must).
export COLM_REL_SPLIT="${COLM_REL_SPLIT:-test}"

# -----------------------------------------------------------------------------
# 2. >>> ENV ACTIVATION -- same venv as colm.py (must have astropt + colm.py importable) <<<
# -----------------------------------------------------------------------------
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bfir/kduraphe/colm/.venv/bin/activate

# -----------------------------------------------------------------------------
# 3. Node-local /tmp scratch for HF cache (wiped on exit).
# -----------------------------------------------------------------------------
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME" 2>/dev/null || true' EXIT

# run from the submit dir (colm_rel.py + colm.py sit at the astropt repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec python colm_rel.py "$@"
