#!/bin/bash 
#SBATCH -J train_fm_4000
#SBATCH -p gpu_p
#SBATCH --qos gpu_long
#SBATCH --gres=gpu:4
#SBATCH --output=train_fm_4000_out_%j.txt
#SBATCH --error=train_fm_4000_err_%j.txt
#SBATCH --mem=500G
#SBATCH --time=72:00:00
#SBATCH --nice=0
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=16


source ~/.bashrc
conda activate nicheformer_env

export TORCH_USE_CUDA_DSA=1

# Set MASTER_ADDR and MASTER_PORT for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

# Print some information about the environment
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_GPUS: $SLURM_GPUS"

# Run with nf_4000donors subset
CUDA_LAUNCH_BLOCKING=1 python -u train_nicheformer.py models.subset=nf_4000donors wandb.name=nf_4000donors_run 