#!/bin/bash
#SBATCH --job-name=breast_add_1pct_cfg1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=logs/add_1pct_cfg1_%j.out
#SBATCH --error=logs/add_1pct_cfg1_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../setup_env.sh"

mkdir -p logs

python -m cell_synergy.data.hf_dataset.add_embeddings_to_hf_dataset \
    --mode unimodal_gex \
    --gex_model_scale finetune_1pct_cfg1 \
    --split train \
    --data_scale S \
    --dataset breast

echo "Completed at $(date)"
