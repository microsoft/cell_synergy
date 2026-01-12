#!/bin/bash
#SBATCH -J breast_train_comm_cfg1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --output=logs/breast_train_comm_cfg1_%j.out
#SBATCH --error=logs/breast_train_comm_cfg1_%j.err
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --nice=0
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=8

echo "Starting CoMM alignment training for breast (cfg1: embed_dim=512, n_layers=1)..."

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../setup_env.sh"

export TORCH_USE_CUDA_DSA=1

# Print some information about the environment
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_GPUS: $SLURM_GPUS"

# Create logs directory
mkdir -p logs

# Run training with large batch size and long training
python -m cell_synergy.finetuning.run_alignment \
    data=breast \
    models=comm_cfg1 \
    training=alignment_large \
    +models.checkpoint_suffix=_cfg1_job${SLURM_JOB_ID}

echo "Training complete!"


