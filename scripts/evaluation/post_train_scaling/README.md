# Post Train Scaling

This directory contains SLURM batch scripts for training models.

## Main Script

The main script is `eval_gex_10pct_cfg1.sbatch` which runs the primary task for this directory.

### Breast dataset

- `create_scaling_splits_breast.sbatch` - For breast dataset

### Configuration cfg1

- `eval_comm_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_comm_10pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_comm_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_10pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_gex_3.16pct_cfg1.sbatch` - Runs with configuration cfg1

### Configuration cfg2

- `eval_100pct_cfg2_alt.sbatch` - Runs with configuration cfg2
- `eval_100pct_cfg2_correct.sbatch` - Runs with configuration cfg2
- `eval_10pct_cfg2_correct.sbatch` - Runs with configuration cfg2
- `eval_1pct_cfg2_correct.sbatch` - Runs with configuration cfg2
- `eval_3.16pct_cfg2_correct.sbatch` - Runs with configuration cfg2
- `eval_31.6pct_cfg2_correct.sbatch` - Runs with configuration cfg2
- `eval_comm_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_testset_aligned_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `submit_all_eval_cfg2.sbatch` - Runs with configuration cfg2

### Configuration cfg3

- `eval_10pct_cfg3_correct.sbatch` - Runs with configuration cfg3
- `eval_1pct_cfg3_correct.sbatch` - Runs with configuration cfg3
- `eval_3.16pct_cfg3_correct.sbatch` - Runs with configuration cfg3
- `eval_31.6pct_cfg3_correct.sbatch` - Runs with configuration cfg3
- `eval_comm_100pct_cfg3.sbatch` - Runs with configuration cfg3

### Configuration cfg4

- `eval_comm_100pct_cfg4.sbatch` - Runs with configuration cfg4

### Configuration cfg5

- `eval_comm_100pct_cfg5.sbatch` - Runs with configuration cfg5

### Thymus dataset

- `create_scaling_splits_thymus.sbatch` - For thymus dataset

### Variants

- `eval_pretrained_comm_test_cv.sbatch`
- `eval_pretrained_concat_test_cv.sbatch`
- `eval_pretrained_gex_test_cv.sbatch`
- `submit_all_evaluations.sbatch`


## Usage

```bash
sbatch eval_gex_10pct_cfg1.sbatch
```
