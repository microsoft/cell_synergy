#!/bin/bash
#SBATCH --job-name=continued_1pct
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/scripts/finetune/logs/continued_1pct_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/scripts/finetune/logs/continued_1pct_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal


# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting continued training for 1% subset"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"

# Run the continued training script
python src/cell_synergy/finetuning/continued_training.py \
    training.subset_to_train=1pct \
    training.batch_size=1024 \
    training.learning_rate=0.0001 \
    training.max_epochs=100

echo "Job completed at $(date)"
echo "Check logs at: logs/continued_1pct_$SLURM_JOB_ID.out"
