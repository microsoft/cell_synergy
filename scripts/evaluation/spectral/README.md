# Spectral Analysis

This directory contains SLURM batch scripts for spectral analysis of model embeddings.

## Purpose
Performs spectral analysis to understand the structure and properties of learned embeddings across different models.

## Scripts

### Individual Analysis
- `spectral_analysis_breast.sbatch` - Spectral analysis for breast dataset
- `spectral_analysis_lung.sbatch` - Spectral analysis for lung dataset
- `spectral_analysis_thymus.sbatch` - Spectral analysis for thymus dataset

### Sweep Analysis
- `submit_all.sbatch` - Orchestrates spectral analysis across all models (baseline, adversarial, barlow_twins, byol, cca, comm, concat, dcca, dim, simclr, simsiam, vicreg)

## Utility Scripts
Utility Python scripts for creating focused jobs and running sweeps have been moved to `src/cell_synergy/visualization/spectral/` for better organization.

## Usage
```bash
# Run individual analysis
sbatch spectral_analysis_breast.sbatch

# Run sweep across all models
sbatch submit_all.sbatch
```
