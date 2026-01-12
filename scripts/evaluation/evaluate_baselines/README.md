# Unimodal Baseline Evaluations

This directory contains SLURM batch scripts for evaluating unimodal baseline models (UNI-2 IMG and pretrained GEX) on downstream tasks.

## Purpose
Evaluates unimodal baselines (image-only and GEX-only) to compare against multimodal methods.

## Main Script
The main script is `run_unimodal_baselines.sbatch` which evaluates both UNI-2 IMG and pretrained GEX baselines.

## Difference from `baselines/`
- `baselines/` - Evaluates **multimodal** baseline methods (CCA, CoMM, DCCA, etc.)
- `evaluate_baselines/` - Evaluates **unimodal** baseline methods (image-only, GEX-only)

## Usage
```bash
sbatch run_unimodal_baselines.sbatch
```

