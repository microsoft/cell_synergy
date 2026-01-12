#!/bin/bash
#SBATCH --job-name=comm_1pct
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/scripts/train_comm_scaling/logs/comm_1pct_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/scripts/train_comm_scaling/logs/comm_1pct_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print environment info
echo "Starting CoMM training for 1% subset at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the training script with subset argument
python src/cell_synergy/evaluation/train_single_comm.py 1pct

echo "CoMM training for 1% completed at $(date)"





