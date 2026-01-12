# Breast Embedding Generation Scripts

This directory contains scripts for generating raw H5 embedding files from finetuned breast checkpoints.

## Purpose
Generate raw H5 embedding files from post-training checkpoints for downstream evaluation.

## Scripts

### Generation Scripts
- `run_embedding_{scale}_{cfg}.sbatch` - Generate embeddings for each scale/config combination
  - Scales: 0pct, 1pct, 3.16pct, 10pct, 31.6pct (100pct skipped - still training)
  - Configs: cfg1, cfg2, cfg3
  - Total: 15 scripts (5 scales × 3 configs)

### Submission Script
- `submit_all_breast.sh` - Submit all embedding generation jobs

## How It Works

1. **Input**: Finetuned checkpoints from `project_folder/trained_models/breast/scaling/{scale}/{lr_dir}/last.ckpt`
2. **Process**: Uses `cell_synergy.data.hf_dataset.generate_gex_embeddings_with_checkpoints` module
3. **Output**: Raw H5 files in `project_folder/breast/unimodal_embeddings/gex/` (or similar)

## Usage

```bash
# Submit all jobs
bash submit_all_breast.sh

# Or submit individual jobs
sbatch run_embedding_1pct_cfg1.sbatch
```

## Checkpoint Mapping

- cfg1: `lr0.0001_wd1e-05_bs32`
- cfg2: `lr2e-05_wd0.0005_bs32`
- cfg3: `lr5e-05_wd0.0001_bs32`

Note: 100% configs are excluded as they are still training.


# Generate Raw Embeddings Breast

This directory contains SLURM batch scripts for generating embeddings or figures.

## Main Script

The main script is `submit_all_breast.sbatch` which runs the primary task for this directory.

### Configuration cfg1

- `run_embedding_0pct_cfg1.sbatch` - Runs with configuration cfg1
- `run_embedding_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `run_embedding_10pct_cfg1.sbatch` - Runs with configuration cfg1
- `run_embedding_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `run_embedding_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `run_embedding_31.6pct_cfg1.sbatch` - Runs with configuration cfg1

### Configuration cfg2

- `run_embedding_0pct_cfg2.sbatch` - Runs with configuration cfg2
- `run_embedding_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `run_embedding_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `run_embedding_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `run_embedding_3.16pct_cfg2.sbatch` - Runs with configuration cfg2
- `run_embedding_31.6pct_cfg2.sbatch` - Runs with configuration cfg2

### Configuration cfg3

- `run_embedding_0pct_cfg3.sbatch` - Runs with configuration cfg3
- `run_embedding_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `run_embedding_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `run_embedding_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `run_embedding_3.16pct_cfg3.sbatch` - Runs with configuration cfg3
- `run_embedding_31.6pct_cfg3.sbatch` - Runs with configuration cfg3


## Usage

```bash
sbatch submit_all_breast.sbatch
```
