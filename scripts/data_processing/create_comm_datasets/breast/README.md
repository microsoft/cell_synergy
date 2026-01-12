# Breast CoMM Dataset Creation

## Current Status
- Only 0pct (pretrained) results available
- Finetune datasets (1pct, 3.16pct, 10pct, 31.6pct, 100pct) need to be created first

## Steps Required
1. First, create finetune gex datasets using add_embeddings scripts
2. Then create combined datasets using the scripts in this directory

## Scripts
Scripts will be created once finetune datasets are available.


# Breast

This directory contains SLURM batch scripts for running computational tasks.

## Main Script

The main script is `create_3.16pct_cfg1.sbatch` which runs the primary task for this directory.

### Configuration cfg2

- `create_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `create_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `create_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `create_31.6pct_cfg2.sbatch` - Runs with configuration cfg2


## Usage

```bash
sbatch create_3.16pct_cfg1.sbatch
```
