# Generate Raw Embeddings - Lung

This directory contains scripts for generating raw H5 embedding files from finetuned lung checkpoints.

## Purpose
Generate raw H5 embedding files from post-training checkpoints for downstream evaluation.

## Main Script
The main script is `run_embedding_0pct_cfg1.sbatch` which generates embeddings for 0% cfg1 checkpoint.

## Usage
```bash
sbatch run_embedding_0pct_cfg1.sbatch
```

## Pipeline Context
This is the first step in the embedding pipeline:
1. Generate raw embeddings (this directory)
2. Convert to HF format (`create_hf_datasets`)
3. Add to existing datasets (`add_embeddings_*`)

