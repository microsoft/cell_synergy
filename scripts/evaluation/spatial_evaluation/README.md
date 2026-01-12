# Spatial Evaluation

This directory contains SLURM batch scripts for spatial-related evaluations, including spatial consistency, distance bin analysis, and spatial model evaluations.

## Purpose
Evaluates how well models capture spatial relationships and biological contexts in tissue data.

## Scripts

### Spatial Consistency
- `run_spatial_consistency_gex.sbatch` - Evaluates spatial consistency for GEX embeddings

### Model-Specific Spatial Evaluations
- `spatial_eval_breast_cca.sbatch` - Spatial evaluation for breast CCA model
- `run_baseline_gex.sbatch` - Baseline GEX spatial evaluation

## Visualization Scripts
Visualization and analysis Python scripts have been moved to `src/cell_synergy/visualization/spatial/` for better organization.

## Usage
```bash
sbatch run_spatial_consistency_gex.sbatch
```

