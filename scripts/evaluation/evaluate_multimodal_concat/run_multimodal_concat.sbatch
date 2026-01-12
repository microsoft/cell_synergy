#!/bin/bash
#SBATCH --job-name=scaling_multimodal_concat
#SBATCH --output=logs/scaling_multimodal_concat_%j.out
#SBATCH --error=logs/scaling_multimodal_concat_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1

echo "Starting multimodal concatenation evaluations at $(date)"

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

# Run multimodal concatenation evaluations
echo "Evaluating multimodal concatenation for all subset/config combinations..."
python src/cell_synergy/evaluation/multimodal_concat.py

echo "Multimodal concatenation evaluations completed at $(date)"

