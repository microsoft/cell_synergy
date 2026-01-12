#!/bin/bash
#SBATCH --job-name=run_all_embeddings_evals
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_all_embeddings_and_evals_%j.out
#SBATCH --error=logs/run_all_embeddings_and_evals_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../setup_env.sh"

echo "Running all embeddings and evaluations..."
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"

# Run the orchestrator
python run_all_embeddings_and_evals.py

echo "All embeddings and evaluations completed at: $(date)"









