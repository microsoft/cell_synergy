#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --output=logs/tokenize_hf_%j.out
#SBATCH --error=logs/tokenize_hf_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../../setup_env.sh"

mkdir -p logs

echo "Tokenizing HFD dataset to merlin format for thymus at $(date)"

# Tokenize the training HFD dataset directly (like lung)
# Note: create_scaling_splits creates img_only_200M_train.L (not nicheformer_full_200M_train.L)
python -m cell_synergy.data.data_curation.tokenize_hf_dataset \
    --hf-dataset-path /lustre/groups/ml01/workspace/till.richter/hf_datasets/thymus/scaling_splits/img_only_200M_train.L \
    --output-dir /home/icb/till.richter/git/cell_synergy/project_folder/thymus/nicheformer_tokens/thymus_train \
    --chunk-size 100000

echo "Completed at: $(date)"

