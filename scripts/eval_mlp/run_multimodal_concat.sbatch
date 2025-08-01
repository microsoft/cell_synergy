#!/bin/bash 
#SBATCH -J eval_mlp_concat
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --output=eval_mlp_concat_out_%j.txt
#SBATCH --error=eval_mlp_concat_err_%j.txt
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --nice=0
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=8

source ~/.bashrc
conda activate nicheformer_env

export TORCH_USE_CUDA_DSA=1

# Print some information about the environment
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_GPUS: $SLURM_GPUS"

# Change to the correct directory
cd /home/icb/till.richter/git/data_scaling/src/data_scaling/downstream

# Run multimodal concatenation eval_mlp evaluation
CUDA_LAUNCH_BLOCKING=1 python -u eval_mlp.py evaluation.modality=multimodal evaluation.align_method=concat 