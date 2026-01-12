# Add Embeddings to HF Datasets - Breast

This directory contains SLURM batch scripts for adding generated embeddings to existing HuggingFace datasets.

## Purpose
Adds pre-computed embeddings (from `generate_raw_embeddings_*` directories) to existing HuggingFace datasets for downstream evaluation.

## Main Script

The main script is `add_1pct_cfg1.sbatch` which runs the primary task for this directory.

### Breast dataset

- `submit_all_breast.sbatch` - For breast dataset

### Configuration cfg1

- `add_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `add_100pct_cfg1_train.sbatch` - Runs with configuration cfg1
- `add_10pct_cfg1_train.sbatch` - Runs with configuration cfg1
- `add_1pct_cfg1_train.sbatch` - Runs with configuration cfg1
- `add_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `add_3.16pct_cfg1_train.sbatch` - Runs with configuration cfg1
- `add_31.6pct_cfg1.sbatch` - Runs with configuration cfg1
- `add_31.6pct_cfg1_train.sbatch` - Runs with configuration cfg1

### Configuration cfg2

- `add_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `add_100pct_cfg2_train.sbatch` - Runs with configuration cfg2
- `add_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `add_10pct_cfg2_train.sbatch` - Runs with configuration cfg2
- `add_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `add_1pct_cfg2_train.sbatch` - Runs with configuration cfg2
- `add_3.16pct_cfg2_train.sbatch` - Runs with configuration cfg2
- `add_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `add_31.6pct_cfg2_train.sbatch` - Runs with configuration cfg2

### Configuration cfg3

- `add_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `add_100pct_cfg3_train.sbatch` - Runs with configuration cfg3
- `add_10pct_cfg3_train.sbatch` - Runs with configuration cfg3
- `add_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `add_1pct_cfg3_train.sbatch` - Runs with configuration cfg3
- `add_3.16pct_cfg3_train.sbatch` - Runs with configuration cfg3
- `add_31.6pct_cfg3.sbatch` - Runs with configuration cfg3
- `add_31.6pct_cfg3_train.sbatch` - Runs with configuration cfg3


## Usage

```bash
sbatch add_1pct_cfg1.sbatch
```
