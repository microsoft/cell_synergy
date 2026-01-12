#!/bin/bash
#SBATCH -J add_finetune_1_pct_embeddings
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH --output=logs/add_finetune_1_pct_embeddings_%j.out
#SBATCH --error=logs/add_finetune_1_pct_embeddings_%j.err
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --nice=0

#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

echo "Starting to add 1% finetune embeddings to HF dataset..."

# Add embeddings for the 1% finetune model
# Use L scale (matches the working example) and pretrain split (though it doesn't matter for naming)
python -m cell_synergy.data.hf_dataset.add_embeddings_to_hf_dataset \
    --mode unimodal_gex \
    --gex_model_scale finetune_1pct_cfg1 \
    --split pretrain \
    --data_scale L \
    --dataset lung

echo "Done adding 1% finetune embeddings to HF dataset!"
