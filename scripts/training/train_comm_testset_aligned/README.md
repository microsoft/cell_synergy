# Train CoMM on Test Set with Aligned Embeddings

## Overview

This pipeline trains CoMM on the test set (unsupervised) for each finetuned percentage, aligning:
- **Pretrained IMG embeddings** (constant, from `full_200M_test.L`)
- **Finetuned GEX embeddings** (from 1pct, 3.16pct, 10pct, 31.6pct, 100pct cfg1)

This approach maintains consistency with the main experiment methodology.

## Pipeline Steps

For each percentage (1pct cfg2, 3.16pct cfg1, 10pct cfg1, 31.6pct cfg2, 100pct cfg1):

1. **Generate GEX embeddings** for test set using finetuned Nicheformer checkpoint
   - Script: `generate_gex_for_testset_{pct}_cfg{X}.sbatch`
   - Output: `.h5` file with embeddings

2. **Add embeddings to test dataset**
   - Script: `add_gex_to_testset_{pct}_cfg{X}.sbatch`
   - Creates: `full_200M_test_gex_finetune_{pct}_cfg{X}.L`
   - Contains: Pretrained IMG + Finetuned GEX embeddings

3. **Train CoMM on test set**
   - Script: `train_comm_testset_{pct}_cfg{X}.sbatch`
   - Trains CoMM on the mixed dataset (unsupervised)
   - Saves checkpoint

4. **Evaluate with 5-fold CV**
   - Script: `eval_comm_testset_{pct}_cfg{X}.sbatch`
   - Uses trained CoMM checkpoint
   - Evaluates on test set with CV

## Benefits

- Consistent with main experiment (CoMM trained on test set)
- Direct comparison across percentages (all use same test set)
- Isolates effect of GEX quality on alignment
- Maintains unsupervised learning paradigm

## Usage

```bash
sbatch create_all_scripts.sbatch
```
