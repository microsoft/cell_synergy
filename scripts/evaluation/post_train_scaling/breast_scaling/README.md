# Breast Scaling Experiments

Complete pipeline for post-training scaling experiments on breast dataset.

## Pipeline Overview

1. **Convert HF → h5ad** (one job)
2. **Tokenize h5ad → merlin format** (one job)  
3. **Create subsets** (one job)
4. **Train Nicheformer** (5 jobs - one per fraction)
5. **Generate GEX embeddings** (5 jobs)
6. **Add embeddings to testset** (5 jobs)
7. **Train CoMM** (5 jobs)
8. **Evaluate** (5 jobs)

## Scripts

### Step 1: Data Preparation
- `convert_hf_to_h5ad.sbatch` - Convert HF dataset to h5ad format
- `create_merlin_subsets.sbatch` - Create subset directories from full dataset
- Note: Tokenization scripts are in `data_processing/` directory

### Step 2-5: Training and Evaluation
- `train_nicheformer_{fraction}.sbatch` - Train Nicheformer on each fraction
- `generate_gex_for_testset_{fraction}.sbatch` - Generate GEX embeddings
- `add_gex_to_testset_{fraction}.sbatch` - Add embeddings to test dataset
- `train_comm_testset_{fraction}.sbatch` - Train CoMM on test set
- `eval_comm_testset_{fraction}.sbatch` - Evaluate with 5-fold CV

Fractions: 1pct, 3.16pct, 10pct, 31.6pct, 100pct

## Usage

```bash
# Step 1: Data preparation (sequential)
sbatch convert_hf_to_h5ad.sbatch
# Wait for completion, then:
# (Tokenization handled by data_processing scripts)
# Wait for completion, then:
sbatch create_merlin_subsets.sbatch

# Step 2: Train Nicheformer (can run in parallel)
for frac in 1pct 3.16pct 10pct 31.6pct 100pct; do
    sbatch train_nicheformer_${frac}.sbatch
done

# Step 3-5: Generate embeddings, train CoMM, evaluate (sequential per fraction)
# For each fraction, wait for previous step to complete
```

## Notes

- Reuses existing infrastructure (tokenize_h5ad_data.py, continued_training_merlin_correct.py)
- Converts full dataset once, then creates subsets by sampling
- All scripts adapted for breast dataset paths



# Breast Scaling

This directory contains SLURM batch scripts for running computational tasks.

## Main Script

The main script is `train_comm_1pct_cfg3.sbatch` which trains CoMM models for post-training scaling evaluation.

### Breast dataset

- `submit_all_breast_scaling.sbatch` - For breast dataset

### Configuration cfg1

- `train_nicheformer_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_100pct_cfg1_8gpu.sbatch` - Runs with configuration cfg1
- `train_nicheformer_10pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `train_nicheformer_31.6pct_cfg1.sbatch` - Runs with configuration cfg1

### Configuration cfg2

- `train_nicheformer_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_100pct_cfg2_8gpu.sbatch` - Runs with configuration cfg2
- `train_nicheformer_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_3.16pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_nicheformer_31.6pct_cfg2.sbatch` - Runs with configuration cfg2

### Configuration cfg3

- `train_nicheformer_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_100pct_cfg3_8gpu.sbatch` - Runs with configuration cfg3
- `train_nicheformer_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_3.16pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_nicheformer_31.6pct_cfg3.sbatch` - Runs with configuration cfg3

### Variants

- `add_gex_to_testset_100pct.sbatch`
- `add_gex_to_testset_10pct.sbatch`
- `add_gex_to_testset_1pct.sbatch`
- `add_gex_to_testset_3.16pct.sbatch`
- `add_gex_to_testset_31.6pct.sbatch`
- `convert_hf_to_h5ad.sbatch`
- `create_merlin_subsets.sbatch`
- `eval_comm_testset_100pct.sbatch`
- `eval_comm_testset_10pct.sbatch`
- `eval_comm_testset_1pct.sbatch`
- `eval_comm_testset_3.16pct.sbatch`
- `eval_comm_testset_31.6pct.sbatch`
- `generate_gex_for_testset_100pct.sbatch`
- `generate_gex_for_testset_10pct.sbatch`
- `generate_gex_for_testset_1pct.sbatch`
- `generate_gex_for_testset_3.16pct.sbatch`
- `generate_gex_for_testset_31.6pct.sbatch`
- `train_comm_testset_100pct.sbatch`
- `train_comm_testset_10pct.sbatch`
- `train_comm_testset_1pct.sbatch`
- `train_comm_testset_3.16pct.sbatch`
- `train_comm_testset_31.6pct.sbatch`
- `train_nicheformer_100pct.sbatch`
- `train_nicheformer_10pct.sbatch`
- `train_nicheformer_1pct.sbatch`
- `train_nicheformer_3.16pct.sbatch`
- `train_nicheformer_31.6pct.sbatch`


## Usage

```bash
sbatch train_comm_1pct_cfg3.sbatch
```
