#!/bin/bash
#SBATCH --job-name=breast_comm_ds_3.16pct_cfg1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=logs/create_3.16pct_cfg1_%j.out
#SBATCH --error=logs/create_3.16pct_cfg1_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../../setup_env.sh"

mkdir -p logs

echo "Creating combined dataset for CoMM training: finetune_3.16pct_cfg1 + 200M at $(date)"

python -m cell_synergy.data.hf_dataset.add_embeddings_to_hf_dataset \
    --mode multimodal \
    --gex_model_scale finetune_3.16pct_cfg1 \
    --img_model_scale 200M \
    --split test \
    --data_scale L \
    --dataset breast

echo "Completed at $(date)"
