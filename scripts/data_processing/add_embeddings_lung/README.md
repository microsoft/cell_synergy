# Add Embeddings to HF Datasets - Lung

This directory contains SLURM batch scripts for adding generated embeddings to existing HuggingFace datasets.

## Purpose
Adds pre-computed embeddings (from `generate_raw_embeddings_*` directories) to existing HuggingFace datasets for downstream evaluation.

## Main Script
The main script is `add_1pct_cfg1.sbatch` which adds 1% cfg1 embeddings to the lung HF dataset.

## Usage
```bash
sbatch add_1pct_cfg1.sbatch
```

## Pipeline Context
This step comes after:
1. Generating raw embeddings (`generate_raw_embeddings_*`)
2. Converting to HF format (`create_hf_datasets`)

This step adds additional embeddings to existing HF datasets.

