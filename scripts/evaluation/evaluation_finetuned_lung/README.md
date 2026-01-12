# Finetuned Model Evaluations - Lung

This directory contains SLURM batch scripts for evaluating finetuned models on the lung dataset.

## Purpose
Evaluates models that have been finetuned on different percentages of the training data (1%, 3.16%, 10%, 31.6%, 100%) for the lung dataset.

## Note on Data Processing
The `add_embeddings_to_hf_dataset` functionality is handled by scripts in `data_processing/add_embeddings_lung/`. This directory is specifically for **evaluation** scripts, not data processing.

## Related Directories
- `evaluation_finetuned_breast/` - Breast-specific finetuned model evaluations
- `evaluation_finetuned_thymus/` - Thymus-specific finetuned model evaluations
- `data_processing/add_embeddings_lung/` - Data processing scripts for adding embeddings to lung datasets

## Usage
Evaluation scripts will be added here as needed. For data processing (adding embeddings), use scripts in `data_processing/add_embeddings_lung/`.
