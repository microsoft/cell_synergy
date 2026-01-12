#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH -o scripts/post_train_scaling/add_embeddings/logs/add_emb_cfg1_3.16pct.out
#SBATCH -e scripts/post_train_scaling/add_embeddings/logs/add_emb_cfg1_3.16pct.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../setup_env.sh"

mkdir -p scripts/post_train_scaling/add_embeddings/logs

echo "Starting add embeddings for 3.16pct cfg1 at $(date)"

python -m cell_synergy.data.hf_dataset.add_embeddings_to_hf_dataset \
    --mode multimodal \
    --gex_model_scale finetune_3.16pct_cfg1 \
    --img_model_scale 200M \
    --split test \
    --data_scale S \
    --dataset lung

echo "Completed at: $(date)"


