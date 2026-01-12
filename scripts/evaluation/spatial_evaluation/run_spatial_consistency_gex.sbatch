#!/bin/bash
#SBATCH --job-name=spatial_cons_gex
#SBATCH --output=/home/icb/till.richter/git/cell_synergy/logs/spatial_cons_gex_%j.out
#SBATCH --error=/home/icb/till.richter/git/cell_synergy/logs/spatial_cons_gex_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nice=10000

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"

python src/cell_synergy/downstream/spatial_consistency_evaluation.py \
  evaluation.modality=unimodal_gex \
  training.spatial_neighbor.enable_spatial_consistency=true \
  training.spatial_neighbor.enable_niche_consistency=true

echo "Spatial consistency evaluation (GEX only) completed"

