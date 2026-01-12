# Post-Training Scaling Evaluation Scripts

This directory contains scripts for evaluating post-training data scaling effects on downstream performance.

## Purpose
Evaluate how different amounts of post-training data (1%, 3.16%, 10%, 31.6%, 100%) affect downstream model performance on classification and regression tasks.

## Scripts

### Main Evaluation Scripts
- `run_posttraining_scaling_1_pct.sbatch` - Evaluate 1% subset
- `run_posttraining_scaling_3_16_pct.sbatch` - Evaluate 3.16% subset
- `run_posttraining_scaling_10_pct.sbatch` - Evaluate 10% subset
- `run_posttraining_scaling_31_6_pct.sbatch` - Evaluate 31.6% subset
- `run_posttraining_scaling_100_pct.sbatch` - Evaluate 100% subset

### Submission Script
- `submit_posttraining_scaling_evals.sh` - Submit all evaluation jobs

## How It Works

1. **Input**: HuggingFace datasets with finetuned embeddings from `project_folder/hf_datasets/`
2. **Process**: Uses `cell_synergy.downstream.fast_embedding_eval` module
3. **Evaluation**: 
   - Classification: F1-macro score
   - Regression: R² score
4. **Output**: Results saved to `outputs/posttraining_scaling_eval_{subset}/`

## Evaluation Protocol (Option 1)
- **Same test set**: All models evaluated on identical holdout test set
- **Fair comparison**: Only training data amount varies, test data is constant
- **Tasks**: Both classification and regression

## Usage

```bash
# Submit all evaluations
bash submit_posttraining_scaling_evals.sh

# Or submit individual jobs
sbatch run_posttraining_scaling_1_pct.sbatch
```

## Dependencies
- `cell_synergy.downstream.fast_embedding_eval` module
- Finetuned HF datasets from embedding generation
- scikit-learn for linear probes

## Expected Results
Compare performance across data scaling:
- 1% → 3.16% → 10% → 31.6% → 100% post-training data
- Metrics: Accuracy, F1-macro, R², MSE
- Analysis: How does more post-training data improve downstream performance?




# Evaluation

This directory contains SLURM batch scripts for evaluating models.

## Main Script

The main script is `run_all_embeddings_and_evals.sbatch` which runs the primary task for this directory.

### Variants

- `evaluate_zero_shot_test_splits.sbatch`
- `run_posttraining_scaling_100_pct.sbatch`
- `run_posttraining_scaling_10_pct.sbatch`
- `run_posttraining_scaling_1_pct.sbatch`
- `run_posttraining_scaling_31_6_pct.sbatch`
- `run_posttraining_scaling_3_16_pct.sbatch`
- `run_posttraining_scaling_orchestrator.sbatch`
- `submit_posttraining_scaling_evals.sbatch`


## Usage

```bash
sbatch run_all_embeddings_and_evals.sbatch
```
