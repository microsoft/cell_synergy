#!/bin/bash
# This script submits two separate SLURM jobs for lung and breast datasets.

# Submit lung job
sbatch <<EOF
#!/bin/bash
#SBATCH -o /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/data/niche_scGPT/lung/log_%j.txt
#SBATCH -e /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/data/niche_scGPT/lung/error_%j.txt
#SBATCH --job-name=lung_gex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --partition=gpu_p
#SBATCH --qos gpu_power_platter
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

echo "Starting pcascviniche for lung dataset..."
source $HOME/.bashrc
conda deactivate
conda activate scgpt_merel

python /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/cameo_data/scripts/pcascvi_niche_scgpt_hf.py \
    --dataset_name theislab-multimodal-ssl/lung_GAT_scvi_PCA_UNI_CONCH_CTtranspath \
    --scvi_model_path /lustre/groups/shared/users/multimodal-ssl/model_weights/scVI/lung/model_dir \
    --scvi_train_adata_path /lustre/groups/shared/users/multimodal-ssl/model_weights/scVI/lung/lung_train_median55.h5ad \
    --output_dir /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/data/niche_scGPT/lung
EOF


# Submit breast job
sbatch <<EOF
#!/bin/bash
#SBATCH -o /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/data/niche_scGPT/breast/log_%j.txt
#SBATCH -e /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/data/niche_scGPT/breast/error_%j.txt
#SBATCH --job-name=breast_gex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --partition=gpu_p
#SBATCH --qos gpu_power_platter
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

echo "Starting pcascviniche for breast dataset..."
source $HOME/.bashrc
conda deactivate
conda activate scgpt_merel

python /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/cameo_data/scripts/pcascvi_niche_scgpt_hf.py \
    --dataset_name theislab-multimodal-ssl/breast_7samples_GAT_scvi_PCA_UNI_CONCH_CTtranspath \
    --scvi_model_path /lustre/groups/shared/users/multimodal-ssl/model_weights/scVI/breast/model_dir \
    --scvi_train_adata_path /lustre/groups/shared/users/multimodal-ssl/model_weights/scVI/breast/breast_train_median98.h5ad \
    --output_dir /lustre/groups/epigenereg01/workspace/users/korbinian.traeuble/projects/multimodal-ssl/data/niche_scGPT/breast
EOF