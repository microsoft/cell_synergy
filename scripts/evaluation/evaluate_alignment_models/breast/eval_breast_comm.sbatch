#!/bin/bash
#SBATCH --job-name=eval_breast_comm_cfg1
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/logs/eval_breast_comm_cfg1_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/logs/eval_breast_comm_cfg1_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -e

echo "=========================================="
echo "Evaluating BREAST CoMM Model (cfg1 - job 32029272)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../setup_env.sh"

# Verify checkpoint exists
CKPT_PATH="project_folder/trained_models/breast/full_ds/comm_cfg1_job32029272.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi
echo "Using checkpoint: $CKPT_PATH"
ls -lh "$CKPT_PATH"

# Run evaluation
python src/cell_synergy/downstream/eval.py \
    data.dataset=breast \
    data.train_split=L \
    data.img_embed_key=img_uni_pool \
    evaluation.modality=multimodal_comm \
    evaluation.align_method=multimodal_comm \
    models.method=comm \
    models.embed_dim=512 \
    models.fusion=x-attn \
    models.pool=mean \
    models.n_heads=8 \
    models.n_layers=1 \
    models.temperature=0.07 \
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
    evaluation.seed=42 \
    +evaluation.checkpoint_config=cfg1_job32029272

echo "=========================================="
echo "Evaluation Complete"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="



