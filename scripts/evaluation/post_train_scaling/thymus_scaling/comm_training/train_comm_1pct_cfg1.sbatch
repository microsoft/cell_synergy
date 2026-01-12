#!/bin/bash
#SBATCH --job-name=thymus_comm_1pct_cfg1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_comm_1pct_cfg1_%j.out
#SBATCH --error=logs/train_comm_1pct_cfg1_%j.err

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../../../setup_env.sh"

mkdir -p logs

echo "Training CoMM on test set with pretrained IMG + finetuned 1pct cfg1 GEX embeddings for thymus at $(date)"

python -m cell_synergy.evaluation.train_single_comm \
    testset_aligned_1pct_cfg1 \
    cfg1 \
    --dataset_path /lustre/groups/ml01/workspace/till.richter/hf_datasets/thymus/finetune_1pct_cfg1_200M_test.L \
    --results_dir project_folder/results/thymus/scaling_experiment/comm_models_testset_aligned \
    --max_epochs 1000

echo "Training completed at: $(date)"
