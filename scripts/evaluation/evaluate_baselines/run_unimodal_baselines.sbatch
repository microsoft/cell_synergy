#!/bin/bash
#SBATCH --job-name=scaling_baselines
#SBATCH --output=logs/scaling_baselines_%j.out
#SBATCH --error=logs/scaling_baselines_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1

echo "Starting unimodal baseline evaluations at $(date)"

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

# Run unimodal baseline evaluations
echo "Evaluating UNI-2 IMG and pretrained GEX baselines..."
python src/cell_synergy/evaluation/unimodal_baselines.py

echo "Unimodal baseline evaluations completed at $(date)"

