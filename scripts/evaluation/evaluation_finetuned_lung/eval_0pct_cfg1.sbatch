#!/bin/bash
#SBATCH --job-name=lung_eval_0pct_cfg1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_0pct_cfg1_%j.out
#SBATCH --error=logs/eval_0pct_cfg1_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

mkdir -p logs

echo "Evaluating lung 0pct cfg1 at $(date)"

python -m cell_synergy.downstream.eval \
    --config-name downstream \
    evaluation.split=train \
    evaluation.gex_model=None \
    data.dataset=lung \
    data.max_samples=10000 \
    evaluation.modality=unimodal_gex \
    evaluation.align_method=unimodal_gex \
    hydra.run.dir=project_folder/results/lung/finetune_eval_parallel/0pct/cfg1

echo "Completed at $(date)"

