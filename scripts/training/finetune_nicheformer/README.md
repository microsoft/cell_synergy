# Nicheformer Finetuning Sweep Scripts

This directory contains scripts for running hyperparameter sweeps to finetune Nicheformer on different subset sizes of the Merlin dataset.

This directory contains scripts for running hyperparameter sweeps on different subset sizes of the Merlin dataset.

## Files

- `sweep_configs.py` - Contains 3 optimized configurations for each subset size
- `run_sweep.py` - Script to run the sweeps
- `README.md` - This file

## Subset Sizes

Each subset gets 3 configurations optimized for its characteristics:

### 1pct (Very Small Dataset: 10K-100K samples)
- **Strategy**: Higher learning rates + more regularization to prevent overfitting
- **Configs**: 3 variations with LR 2e-4 to 1e-3, weight decay 1e-3 to 1e-2
- **Epochs**: 30 (more epochs for small dataset)

### 3.16pct (Small Dataset: 100K-300K samples)
- **Strategy**: Moderate learning rates with good regularization
- **Configs**: 3 variations with LR 1e-4 to 5e-4, weight decay 1e-4 to 1e-3
- **Epochs**: 25

### 10pct (Medium Dataset: 300K-1M samples)
- **Strategy**: Balanced approach with standard hyperparameters
- **Configs**: 3 variations with LR 5e-5 to 2e-4, weight decay 1e-5 to 5e-4
- **Epochs**: 20

### 31.6pct (Large Dataset: 1M-3M samples)
- **Strategy**: Lower learning rates, moderate regularization
- **Configs**: 3 variations with LR 2e-5 to 1e-4, weight decay 5e-5 to 1e-3
- **Epochs**: 20

### 100pct (Full Dataset: 3M+ samples)
- **Strategy**: Lowest learning rates, minimal regularization needed
- **Configs**: 3 variations with LR 1e-5 to 5e-5, weight decay 1e-5 to 1e-4
- **Epochs**: 20

## Usage

### View All Configurations
```bash
cd git/cell_synergy/scripts/training/finetune_nicheformer
python sweep_configs.py
```

### Run All Sweeps for a Subset
```bash
python run_sweep.py 100pct
python run_sweep.py 31.6pct
python run_sweep.py 10pct
python run_sweep.py 3.16pct
python run_sweep.py 1pct
```

### Run Specific Configuration
```bash
# Run config 0 (first config) for 100pct subset
python run_sweep.py 100pct 0

# Run config 1 (second config) for 10pct subset
python run_sweep.py 10pct 1
```

## Key Hyperparameters

### Learning Rate Strategy
- **Small datasets**: Higher LRs (1e-3 to 2e-4) for faster convergence
- **Large datasets**: Lower LRs (1e-5 to 1e-4) for stability

### Weight Decay Strategy
- **Small datasets**: Higher weight decay (1e-3 to 1e-2) for regularization
- **Large datasets**: Lower weight decay (1e-5 to 1e-4) as less needed

### Batch Size Strategy
- **Small datasets**: Smaller batch sizes (64-96) for better generalization
- **Large datasets**: Standard batch size (128) for efficiency

### Epochs Strategy
- **Small datasets**: More epochs (25-30) to learn from limited data
- **Large datasets**: Fewer epochs (20) as convergence is faster

## Rationale

The configurations are designed based on the principle that:
1. **Smaller datasets** need higher learning rates to converge quickly and more regularization to prevent overfitting
2. **Larger datasets** can use lower learning rates for stability and need less regularization
3. **20 epochs** should be sufficient for 1M+ datapoints based on the rapid convergence observed in training

## Notes

- All configurations use AdamW optimizer with cosine annealing scheduler
- The training script automatically handles weight decay configuration
- Wandb logging is enabled for all runs
- Checkpoints are saved in `project_folder/trained_models/continued_training/{subset_name}/`


# Posttrain

This directory contains SLURM batch scripts for training models.

## Main Script

The main script is `run_1pct.sbatch` which runs the primary task for this directory.

### Variants

- `run_100pct.sbatch`
- `run_10pct.sbatch`
- `run_3.16pct.sbatch`
- `run_31.6pct.sbatch`
- `run_all_lora_subsets.sbatch`
- `run_all_merlin_subsets.sbatch`
- `run_all_subsets.sbatch`
- `run_lora_100pct.sbatch`
- `run_lora_10pct.sbatch`
- `run_lora_1pct.sbatch`
- `run_lora_3.16pct.sbatch`
- `run_lora_31.6pct.sbatch`
- `run_merlin_100pct.sbatch`
- `run_merlin_100pct_config1.sbatch`
- `run_merlin_100pct_config2.sbatch`
- `run_merlin_100pct_config3.sbatch`
- `run_merlin_10pct.sbatch`
- `run_merlin_10pct_config1.sbatch`
- `run_merlin_10pct_config2.sbatch`
- `run_merlin_10pct_config3.sbatch`
- `run_merlin_1pct.sbatch`
- `run_merlin_1pct_config1.sbatch`
- `run_merlin_1pct_config2.sbatch`
- `run_merlin_1pct_config3.sbatch`
- `run_merlin_3.16pct.sbatch`
- `run_merlin_3.16pct_config1.sbatch`
- `run_merlin_3.16pct_config2.sbatch`
- `run_merlin_3.16pct_config3.sbatch`
- `run_merlin_31.6pct.sbatch`
- `run_merlin_31.6pct_config1.sbatch`
- `run_merlin_31.6pct_config2.sbatch`
- `run_merlin_31.6pct_config3.sbatch`
- `run_merlin_multi_gpu.sbatch`
- `submit_all_sweeps.sbatch`


## Usage

```bash
sbatch run_1pct.sbatch
```
