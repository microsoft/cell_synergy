# Evaluate Alignment Models

This directory contains SLURM batch scripts for evaluating trained multimodal alignment models (CoMM, CCA, DCCA, etc.) on downstream tasks.

## Purpose
Evaluates alignment models that have been trained (see `training/train_multimodal_alignment/`) on downstream classification and regression tasks.

## Main Script
The main script is `eval_alignment_breast_completed.sbatch` which evaluates alignment models on the breast dataset.

## Related Directories
- `training/train_multimodal_alignment/` - Training scripts for alignment models
- `baselines/` - Baseline model evaluations (for comparison)

## Usage
```bash
sbatch eval_alignment_breast_completed.sbatch
```
