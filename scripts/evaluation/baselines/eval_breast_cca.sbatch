#!/bin/bash
#SBATCH --job-name=eval_breast_cca
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/logs/eval_breast_cca_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/logs/eval_breast_cca_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -e

echo "=========================================="
echo "Evaluating Breast CCA Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

# Run evaluation with model-specific parameters
python src/cell_synergy/downstream/eval.py \
    data.dataset=breast \
    data.train_split=L \
    data.img_embed_key=img_uni_pool \
    evaluation.modality=multimodal_cca \
    evaluation.align_method=multimodal_cca \
    models.method=cca \
    models.projection_dim=256 \
    models.img_embed_dim=1536 \
    models.gex_embed_dim=512 \
    evaluation.gex_model=full \
    evaluation.img_model=200M \
    evaluation.cross_validation=true \
    evaluation.n_cv_folds=5 \
    evaluation.tasks.classify=true \
    evaluation.tasks.regress=false \
    evaluation.tasks.spatial=false \
    evaluation.seed=42

echo "=========================================="
echo "Evaluation Complete"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="




