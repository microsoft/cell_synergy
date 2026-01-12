#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/icb/till.richter/git/data_scaling"
DATASETS_ROOT="/home/icb/till.richter/git/data_scaling/project_folder/data/datasets"
LOG_DIR="$ROOT_DIR/src/data_scaling/spectral_analysis/logs"
mkdir -p "$LOG_DIR"

METHODS=(baseline adversarial barlow_twins byol cca comm concat dcca dim simclr simsiam vicreg)

for M in "${METHODS[@]}"; do
  sbatch "$ROOT_DIR/scripts/spectral_sweep/run_one_model.sbatch" "$M" "$DATASETS_ROOT"
done

echo "Submitted ${#METHODS[@]} jobs. Logs: $LOG_DIR"



