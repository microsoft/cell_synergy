#!/bin/bash
#SBATCH --job-name=spectral_lung
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/scripts/spectral_analysis/logs/spectral_lung_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/scripts/spectral_analysis/logs/spectral_lung_%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -e

echo "=========================================="
echo "Spectral Analysis: Lung"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

# Run spectral analysis for all methods
python -m cell_synergy.spectral_analysis.evaluate_spectral_convergence \
    data.dataset=lung \
    data.train_split=L \
    data.img_embed_key=img_uni_pool \
    evaluation.img_model=200M \
    evaluation.gex_model=full \
    "+spectral_analysis.methods='comm,simclr,byol,barlow_twins,vicreg,cca,dcca,dim,simsiam'" \
    "+spectral_analysis.comm_checkpoint=cfg2"

echo "=========================================="
echo "Spectral Analysis Complete"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

