#!/bin/bash

# ── SLURM directives ─────────────────────────────────────────────────────
#SBATCH --job-name=mmu_v4
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=408G
#SBATCH --gpus=1
#SBATCH --partition=ghx4
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --output=mmu_v4_%j.log
#SBATCH --error=mmu_v4_%j.err

# ── Environment ──────────────────────────────────────────────────────────
module load python/3.11.9
source <YOUR_VENV_PATH>/bin/activate

echo "Pipeline v4 starting — $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"

# ── Configuration ────────────────────────────────────────────────────────
export WORK_DIR="<YOUR_WORK_DIR>/mmu_v4"
export APOGEE_DIR="<YOUR_WORK_DIR>/apogee_spectra"
export SDSS_SPECTRA_DIR="<YOUR_WORK_DIR>/sdss_spectra"

# Population sizes: 0 = take all that pass quality cuts
# No N_STARS, N_GALAXIES, N_AGN — pipeline defaults to 0 (take all)

# HuggingFace upload (optional — uncomment to enable)
# export HF_TOKEN="hf_..."
# export HF_REPO="<YOUR_USERNAME>/<DATASET_NAME>"

# ── Pipeline settings ────────────────────────────────────────────────────
export MAX_WORKERS=32
export FLATIRON_WORKERS=8
export LEGACY_WORKERS=50
export ZTF_WORKERS=4
export UNWISE_WORKERS=4
export FLUSH_INTERVAL=50

# ── Create work directory ────────────────────────────────────────────────
mkdir -p "$WORK_DIR"
echo "Work dir: $WORK_DIR"
echo "APOGEE dir: $APOGEE_DIR"
echo "SDSS spectra dir: $SDSS_SPECTRA_DIR"
echo "Taking ALL objects that pass quality cuts (no sampling)"

# ── Disk space check ────────────────────────────────────────────────────
AVAIL_GB=$(df --output=avail -BG "$WORK_DIR" | tail -1 | tr -d ' G')
echo "Available disk: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 400 ]; then
    echo "WARNING: Less than 400 GB free. Pipeline needs ~700 GB peak for 1.5M objects."
fi

# ── Verify pre-staged data ──────────────────────────────────────────────
if [ ! -d "$APOGEE_DIR/synspec_rev1" ]; then
    echo "ERROR: APOGEE spectra not found at $APOGEE_DIR/synspec_rev1"
    echo "Transfer via Globus before running."
    exit 1
fi

if [ ! -d "$SDSS_SPECTRA_DIR" ]; then
    echo "WARNING: SDSS spectra dir not found at $SDSS_SPECTRA_DIR"
    echo "SDSS spectra will be skipped for galaxies."
fi

APOGEE_COUNT=$(find "$APOGEE_DIR/synspec_rev1" -name "aspcapStar-*.fits" | head -5 | wc -l)
echo "APOGEE sample files found: $APOGEE_COUNT"

# ── Verify healpy is installed ──────────────────────────────────────────
python3 -c "import healpy; print(f'healpy {healpy.__version__} OK')" 2>&1 || {
    echo "ERROR: healpy not installed. Required for HiPS tile mapping."
    echo "Run: pip install healpy"
    exit 1
}

# ── Run ──────────────────────────────────────────────────────────────────
cd <YOUR_WORK_DIR>
echo ""
echo "=========================================="
echo "Starting pipeline — $(date)"
echo "=========================================="
python3 -u run_pipeline.py 2>&1 | tee "${WORK_DIR}/pipeline_v4.log"
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Pipeline finished — $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS"
    echo ""

    # Count output
    N_SHARDS=$(ls "${WORK_DIR}/shards/final/"*.parquet 2>/dev/null | wc -l)
    echo "Output shards: $N_SHARDS"
    echo "Output size: $(du -sh "${WORK_DIR}/shards/final/" 2>/dev/null | cut -f1)"
    echo ""

    # Run validation
    echo "Running validation..."
    python3 -u validate_dataset.py "${WORK_DIR}/shards/final/" 2>&1 | tee "${WORK_DIR}/validation.log"

    # Run all diagnostic figures (comprehensive — addresses every reviewer objection)
    echo "Generating all diagnostic figures..."
    python3 -u make_all_diagnostics.py "${WORK_DIR}" 2>&1 | tee "${WORK_DIR}/diagnostics.log"

    # Run false match test
    echo "Running false match rate test..."
    python3 -u false_match_test.py "${WORK_DIR}" 2>&1 | tee "${WORK_DIR}/false_match.log"

    echo ""
    echo "All post-pipeline checks complete."
    echo "Logs in: ${WORK_DIR}/"
    echo "Final shards in: ${WORK_DIR}/shards/final/"
else
    echo "FAILED — check ${WORK_DIR}/pipeline_v4.log"
    tail -50 "${WORK_DIR}/pipeline_v4.log"
fi

exit $EXIT_CODE
