#!/bin/bash
#SBATCH --job-name=eval_thymus_dcca_v1
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/logs/eval_thymus_dcca_v1_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/logs/eval_thymus_dcca_v1_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

set -e

echo "=========================================="
echo "Evaluating THYMUS DCCA Model - Variant 1 (batch_size=1024)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../setup_env.sh"

# Checkpoint paths
CKPT_DIR="project_folder/trained_models/thymus/full_ds"
V1_CKPT="${CKPT_DIR}/dcca-v1.ckpt"
DEFAULT_CKPT="${CKPT_DIR}/dcca.ckpt"
BACKUP_CKPT="${CKPT_DIR}/dcca.ckpt.backup"

# Verify v1 checkpoint exists
if [ ! -f "$V1_CKPT" ]; then
    echo "ERROR: Checkpoint not found: $V1_CKPT"
    exit 1
fi
echo "Using checkpoint: $V1_CKPT"
ls -lh "$V1_CKPT"

# Backup default checkpoint if it exists and rename v1 to default
if [ -f "$DEFAULT_CKPT" ]; then
    echo "Backing up default checkpoint..."
    mv "$DEFAULT_CKPT" "$BACKUP_CKPT"
fi

# Temporarily rename v1 checkpoint to default name
cp "$V1_CKPT" "$DEFAULT_CKPT"
echo "Temporarily using v1 checkpoint as default for evaluation"

# Run evaluation
python src/cell_synergy/downstream/eval.py \
    data.dataset=thymus \
    data.train_split=L \
    data.img_embed_key=img_uni_pool \
    evaluation.modality=multimodal_dcca \
    evaluation.align_method=multimodal_dcca \
    models.method=dcca \
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

# Restore original checkpoint
if [ -f "$BACKUP_CKPT" ]; then
    echo "Restoring original default checkpoint..."
    rm "$DEFAULT_CKPT"
    mv "$BACKUP_CKPT" "$DEFAULT_CKPT"
else
    echo "Removing temporary checkpoint..."
    rm "$DEFAULT_CKPT"
fi

echo "=========================================="
echo "Evaluation Complete"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

