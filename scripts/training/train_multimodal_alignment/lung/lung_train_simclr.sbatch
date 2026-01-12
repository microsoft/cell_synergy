#!/bin/bash
#SBATCH -J lung_train_simclr
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lung_train_simclr_%j.out
#SBATCH --error=logs/lung_train_simclr_%j.err
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --nice=0
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=8

echo "Starting SimCLR alignment training for lung..."

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../setup_env.sh"

# Create logs directory
mkdir -p logs

# Run training
python -m cell_synergy.finetuning.run_alignment \
    data=lung \
    models=simclr

echo "Training complete!"

