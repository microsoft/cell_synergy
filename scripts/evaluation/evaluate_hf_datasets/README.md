# HF Dataset Evaluation Scripts

This directory contains scripts for evaluating the 15 Hugging Face datasets created from finetuned model embeddings.

## Overview

After generating embeddings from 15 finetuned checkpoints (5 sizes × 3 configs) and creating HF datasets, this directory provides scripts to evaluate each dataset using the regular `eval.py` script.

## Files

- `eval_*.sbatch`: Individual SLURM scripts for each size-config combination
- `logs/`: Directory containing job output and error logs

## Usage

**Submit individual jobs:**
```bash
sbatch eval_1pct_cfg1.sbatch
```

To create additional evaluation scripts, copy an existing `.sbatch` file and modify the parameters.

## Dataset Configurations

- **Sizes**: 1pct, 3.16pct, 10pct, 31.6pct, 100pct
- **Configs**: cfg1, cfg2, cfg3
- **Total**: 15 datasets (5 × 3)

## Output

Results will be saved to:
```
project_folder/results/lung/hf_evaluations/{size}/{config}/
```

Each evaluation includes:
- Classification metrics (F1-macro, accuracy, etc.)
- Regression metrics (R², MSE, etc.)
- Detailed logs and plots

## Job Configuration

- **Time limit**: 8 hours
- **Memory**: 64GB
- **CPUs**: 8
- **GPU**: 1 (gpu_p partition)
- **QOS**: gpu_normal
