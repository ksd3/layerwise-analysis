#!/bin/bash
#SBATCH --job-name=omnisky_probe
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=probe_%j.log
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omnisky
python -m scripts.probe_sources --order 4 --pixel 257 --out probe_report.json
python -m scripts.probe_crossmatch --order 4 --pixel 257 --out crossmatch_probe.json
python -m scripts.check_phase0_gate --probe probe_report.json --crossmatch crossmatch_probe.json
