#!/bin/bash
# Create all testset-aligned CoMM scripts for all percentages

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Percentages and their configs
declare -A PERCENTAGES=(
    ["1pct"]="cfg2"
    ["3.16pct"]="cfg1"
    ["10pct"]="cfg1"
    ["31.6pct"]="cfg2"
    ["100pct"]="cfg1"
)

for pct in "${!PERCENTAGES[@]}"; do
    cfg="${PERCENTAGES[$pct]}"
    
    # Generate GEX embeddings script
    cat > "generate_gex_for_testset_${pct}_${cfg}.sbatch" << EOF
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --output=logs/gen_gex_testset_${pct}_${cfg}_%j.out
#SBATCH --error=logs/gen_gex_testset_${pct}_${cfg}_%j.err

# Source common environment setup
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/../../../../setup_env.sh"

mkdir -p logs

echo "Generating GEX embeddings for TEST SET using finetuned ${pct} ${cfg} checkpoint at \$(date)"

# Find the Nicheformer checkpoint
CHECKPOINT_DIR="/home/icb/till.richter/git/cell_synergy/project_folder/trained_models/continued_training/${pct}"
BEST_CKPT=\$(find "\$CHECKPOINT_DIR" -name "merlin_${pct}_epoch=epoch=*_val_loss=val_loss=0.000*.ckpt" 2>/dev/null | sort -V | head -1)

if [ -z "\$BEST_CKPT" ]; then
    BEST_CKPT=\$(find "\$CHECKPOINT_DIR" -name "last.ckpt" 2>/dev/null | head -1)
fi

if [ -z "\$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint found for ${pct} ${cfg}"
    exit 1
fi

echo "Using checkpoint: \$BEST_CKPT"

# Generate embeddings for TEST SET (data_scale L, split test)
python -m cell_synergy.data.hf_dataset.generate_gex_embeddings_with_checkpoints \\
    --checkpoint_path "\$BEST_CKPT" \\
    --model_scale finetune_${pct}_${cfg} \\
    --data_scale L \\
    --split test \\
    --force

echo "Completed at: \$(date)"
EOF

    # Add embeddings script
    cat > "add_gex_to_testset_${pct}_${cfg}.sbatch" << EOF
#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --output=logs/add_gex_testset_${pct}_${cfg}_%j.out
#SBATCH --error=logs/add_gex_testset_${pct}_${cfg}_%j.err

# Source common environment setup
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/../../../../setup_env.sh"

mkdir -p logs

echo "Adding finetuned ${pct} ${cfg} GEX embeddings to full test dataset at \$(date)"

python -m cell_synergy.data.hf_dataset.add_embeddings_to_hf_dataset \\
    --mode multimodal \\
    --gex_model_scale finetune_${pct}_${cfg} \\
    --img_model_scale 200M \\
    --split test \\
    --data_scale L \\
    --dataset lung \\
    --output_suffix "_gex_finetune_${pct}_${cfg}"

echo "Completed at: \$(date)"
EOF

    # Train CoMM script
    cat > "train_comm_testset_${pct}_${cfg}.sbatch" << EOF
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --output=logs/train_comm_testset_${pct}_${cfg}_%j.out
#SBATCH --error=logs/train_comm_testset_${pct}_${cfg}_%j.err

# Source common environment setup
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/../../../../setup_env.sh"

mkdir -p logs

echo "Training CoMM on test set with pretrained IMG + finetuned ${pct} ${cfg} GEX embeddings at \$(date)"

python -m cell_synergy.evaluation.train_single_comm \\
    testset_aligned_${pct}_${cfg} \\
    ${cfg} \\
    --dataset_path /lustre/groups/ml01/workspace/till.richter/hf_datasets/lung/full_200M_test_gex_finetune_${pct}_${cfg}.L \\
    --results_dir project_folder/results/lung/scaling_experiment/comm_models_testset_aligned \\
    --max_epochs 1000

echo "Training completed at: \$(date)"
EOF

    # Evaluate script
    cat > "eval_comm_testset_${pct}_${cfg}.sbatch" << EOF
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --output=logs/eval_comm_testset_${pct}_${cfg}_%j.out
#SBATCH --error=logs/eval_comm_testset_${pct}_${cfg}_%j.err

# Source common environment setup
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/../../../../setup_env.sh"

mkdir -p logs

echo "Evaluating CoMM trained on test set (pretrained IMG + finetuned ${pct} ${cfg} GEX) at \$(date)"

timeout 24h python -m cell_synergy.downstream.eval \\
    evaluation.gex_model=testset_aligned_${pct}_${cfg} \\
    evaluation.img_model=200M \\
    evaluation.modality=multimodal \\
    evaluation.align_method=multimodal_comm \\
    evaluation.cross_validation=true \\
    evaluation.n_cv_folds=5 \\
    evaluation.tasks.classify=true \\
    evaluation.tasks.regress=true \\
    evaluation.tasks.spatial=false \\
    evaluation.seed=42 \\
    data.dataset=lung \\
    data.dataset_name=full_200M_test_gex_finetune_${pct}_${cfg}.L

echo "Evaluation completed at: \$(date)"
EOF

    chmod +x "generate_gex_for_testset_${pct}_${cfg}.sbatch"
    chmod +x "add_gex_to_testset_${pct}_${cfg}.sbatch"
    chmod +x "train_comm_testset_${pct}_${cfg}.sbatch"
    chmod +x "eval_comm_testset_${pct}_${cfg}.sbatch"
done

echo "Created scripts for all percentages!"


