# Generate CoMM GEX Embeddings

This directory contains SLURM batch scripts for generating GEX (gene expression) embeddings from CoMM (multimodal) model checkpoints.

## Purpose
Generates GEX embeddings from trained CoMM model checkpoints for downstream evaluation and analysis.

## Dataset
The scripts in this directory are **dataset-agnostic** - they work with any dataset (lung, breast, thymus) by specifying the appropriate checkpoint path.

## Main Script

The main script is `submit_all_gex.sbatch` which runs the primary task for this directory.

### Variants

- `run_gex_embed_100pct.sbatch`
- `run_gex_embed_10pct.sbatch`
- `run_gex_embed_1pct.sbatch`
- `run_gex_embed_3.16pct.sbatch`
- `run_gex_embed_31.6pct.sbatch`


## Usage

```bash
sbatch submit_all_gex.sbatch
```
