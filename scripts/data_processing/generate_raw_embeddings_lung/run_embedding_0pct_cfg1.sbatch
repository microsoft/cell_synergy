#!/bin/bash
#SBATCH --job-name=lung_emb_0pct_cfg1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=logs/emb_0pct_cfg1_%j.out
#SBATCH --error=logs/emb_0pct_cfg1_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

mkdir -p logs

echo "Generating embeddings for lung 0pct cfg1 at $(date)"

python -m cell_synergy.data.hf_dataset.generate_gex_embeddings_with_checkpoints \
    --checkpoint_path project_folder/trained_models/lung/scaling/0pct/lr0.0001_wd1e-05_bs32/last.ckpt \
    --model_scale finetune_0pct_cfg1 \
    --data_scale L \
    --split test \
    --batch_size 512 \
    --force

echo "Completed at $(date)"

