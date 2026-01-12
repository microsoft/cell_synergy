#!/bin/bash
#SBATCH --job-name=eval_1pct_cfg1
#SBATCH --output=logs/eval_1pct_cfg1_%j.out
#SBATCH --error=logs/eval_1pct_cfg1_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1

echo "Starting evaluation for 1pct-cfg1 at $(date)"
echo "Checkpoint: comm_1pct_cfg1"
echo "HF Dataset: gex_only_comm_1pct_cfg1.S"

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

# Run evaluation
python src/cell_synergy/downstream/eval.py \
    data.dataset_name=gex_only_comm_1pct_cfg1.S \
    evaluation.split=test \
    evaluation.scale=L \
    evaluation.align_method=multimodal_comm \
    hydra.run.dir=project_folder/results/lung/hf_evaluations/1pct/cfg1

echo "Evaluation completed for 1pct-cfg1 at $(date)"
