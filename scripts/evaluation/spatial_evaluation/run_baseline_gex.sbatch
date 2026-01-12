#!/bin/bash
#SBATCH --job-name=baseline_gex
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/logs/baseline_gex_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/logs/baseline_gex_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

echo "=========================================="
echo "BASELINE EVALUATION: Unimodal GEX (pretrained)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

python src/cell_synergy/downstream/eval.py \
    evaluation.modality=unimodal_gex \
    evaluation.align_method=unimodal_gex \
    evaluation.gex_model=full \
    evaluation.img_model=200M \
    data.dataset_name=gex_only_full_train.L \
    evaluation.cross_validation=true \
    evaluation.n_cv_folds=5 \
    evaluation.tasks.classify=true \
    evaluation.tasks.regress=true \
    evaluation.seed=42

echo "=========================================="
echo "Results saved to: project_folder/results/lung/embedding_baseline_200M/"
echo "Expected file: cv_eval_unimodal_gex_nicheformer_results.json"
echo "=========================================="

