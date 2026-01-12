# Thymus Scaling

This directory contains SLURM batch scripts for running computational tasks.

## Main Script

The main script is `tokenize_h5ad.sbatch` which runs the primary task for this directory.

### Configuration cfg1

- `train_nicheformer_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_10pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_31.6pct_cfg1.sbatch` - Runs with configuration cfg1

### Configuration cfg2

- `train_nicheformer_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_3.16pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_31.6pct_cfg2.sbatch` - Runs with configuration cfg2

### Configuration cfg3

- `train_nicheformer_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_3.16pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_31.6pct_cfg3.sbatch` - Runs with configuration cfg3

### Variants

- `convert_hf_to_h5ad.sbatch`
- `create_merlin_subsets.sbatch`


## Usage

```bash
sbatch tokenize_h5ad.sbatch
```
