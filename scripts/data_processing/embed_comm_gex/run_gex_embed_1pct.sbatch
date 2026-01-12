#!/bin/bash
#SBATCH --job-name=gex_embed_1pct
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gex_embed_1pct_%j.out
#SBATCH --error=logs/gex_embed_1pct_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

mkdir -p logs

echo "Generating GEX embeddings for CoMM 1pct at $(date)"

python -m cell_synergy.data.hf_dataset.generate_gex_embeddings_with_checkpoints \
    --checkpoint_path project_folder/trained_models/comm_1pct_cfg1.ckpt \
    --model_scale finetune_nicheformer_1_pct \
    --data_scale L \
    --split test \
    --batch_size 64 \
    --force

echo "Completed at $(date)"

