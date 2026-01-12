#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --output=logs/eval_gex_10pct_cfg1_%j.out
#SBATCH --error=logs/eval_gex_10pct_cfg1_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

mkdir -p logs

echo "Starting evaluation for finetuned Unimodal GEX 10pct cfg1 at $(date)"

timeout 24h python -m cell_synergy.downstream.eval \
    evaluation.gex_model=finetune_10pct_cfg1 \
    evaluation.img_model=200M \
    evaluation.modality=unimodal_gex \
    evaluation.align_method=unimodal_gex \
    evaluation.cross_validation=false \
    evaluation.tasks.classify=true \
    evaluation.tasks.regress=true \
    evaluation.tasks.spatial=false \
    evaluation.seed=42 \
    data.dataset=lung \
    data.dataset_name=gex_only_finetune_10pct_cfg1_test.S

echo "Evaluation completed at: $(date)"

