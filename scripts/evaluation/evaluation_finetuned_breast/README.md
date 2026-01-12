# Breast Finetune Evaluation Scripts

This directory contains evaluation scripts for breast finetune scaling results.

## Structure

### Unimodal GEX Evaluations
- Scales: 0%, 1%, 3.16%, 10%, 31.6% (100% configs still running, skipped)
- Configs: cfg1, cfg2, cfg3
- Total: 15 scripts (5 scales × 3 configs)

### Unimodal IMG Evaluation
- Unscaled (0%) image-only evaluation
- Total: 1 script

### Multimodal Concat Evaluations
- Scales: 0%, 1%, 3.16%, 10%, 31.6% (100% skipped)
- Concatenates unimodal GEX scales with unscaled IMG
- Total: 5 scripts

## Submission

To submit all jobs:
```bash
bash submit_all_breast.sh
```

Note: 100% configs are excluded as they are still training.

## Results

Results are saved to:
- Unimodal GEX: `project_folder/results/breast/finetune_eval_parallel/{scale}/{cfg}/`
- Unimodal IMG: `project_folder/results/breast/finetune_eval_parallel/unimodal_img/`
- Multimodal Concat: `project_folder/results/breast/finetune_eval_parallel/multimodal_concat/{scale}/`

Each evaluation produces a JSON file with niche classification results.


# Evaluation Finetuned Breast

This directory contains SLURM batch scripts for evaluating models.

## Main Script

The main script is `eval_0pct_cfg1.sbatch` which runs the primary task for this directory.

### Breast dataset

- `submit_all_breast.sbatch` - For breast dataset

### Configuration cfg1

- `eval_10pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_31.6pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_31.6pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_31_6pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_concat_3_16pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_gex_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_gex_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_gex_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_gex_31.6pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_gex_31_6pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_gex_3_16pct_cfg1.sbatch` - Runs with configuration cfg1
- `generate_gex_finetune_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `generate_gex_finetune_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `generate_gex_finetune_31_6pct_cfg1.sbatch` - Runs with configuration cfg1
- `generate_gex_finetune_3_16pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_31.6pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_finetune_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_finetune_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_finetune_31_6pct_cfg1.sbatch` - Runs with configuration cfg1
- `readd_finetune_3_16pct_cfg1.sbatch` - Runs with configuration cfg1

### Configuration cfg2

- `eval_0pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_3.16pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_31_6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_fixed_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_fixed_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_fixed_31_6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_new_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_comm_new_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_31_6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_concat_3_16pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_31_6pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_gex_3_16pct_cfg2.sbatch` - Runs with configuration cfg2
- `generate_gex_finetune_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `generate_gex_finetune_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `generate_gex_finetune_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `generate_gex_finetune_31_6pct_cfg2.sbatch` - Runs with configuration cfg2
- `generate_gex_finetune_3_16pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_31.6pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_3_16pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_finetune_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_finetune_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_finetune_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_finetune_31_6pct_cfg2.sbatch` - Runs with configuration cfg2
- `readd_finetune_3_16pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_comm_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_comm_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `train_comm_31_6pct_cfg2.sbatch` - Runs with configuration cfg2

### Configuration cfg3

- `eval_0pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_3.16pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_31.6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_3_16pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_fixed_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_fixed_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_fixed_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_fixed_3_16pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_new_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_new_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_new_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_comm_new_3_16pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_concat_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_concat_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_concat_31.6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_concat_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_gex_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_gex_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_gex_31.6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_gex_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_gex_3_16pct_cfg3.sbatch` - Runs with configuration cfg3
- `generate_gex_finetune_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `generate_gex_finetune_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `generate_gex_finetune_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `generate_gex_finetune_3_16pct_cfg3.sbatch` - Runs with configuration cfg3
- `readd_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `readd_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `readd_31.6pct_cfg3.sbatch` - Runs with configuration cfg3
- `readd_finetune_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `readd_finetune_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `readd_finetune_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `readd_finetune_3_16pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_comm_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_comm_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_comm_31_6pct_cfg3.sbatch` - Runs with configuration cfg3
- `train_comm_3_16pct_cfg3.sbatch` - Runs with configuration cfg3

### Configuration cfg5

- `eval_comm_100pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_10pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_1pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_31_6pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_3_16pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_fixed_100pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_fixed_10pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_fixed_1pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_fixed_31_6pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_fixed_3_16pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_new_100pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_new_10pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_new_1pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_new_31_6pct_cfg5.sbatch` - Runs with configuration cfg5
- `eval_comm_new_3_16pct_cfg5.sbatch` - Runs with configuration cfg5
- `train_comm_100pct_cfg5.sbatch` - Runs with configuration cfg5
- `train_comm_10pct_cfg5.sbatch` - Runs with configuration cfg5
- `train_comm_1pct_cfg5.sbatch` - Runs with configuration cfg5
- `train_comm_31_6pct_cfg5.sbatch` - Runs with configuration cfg5
- `train_comm_3_16pct_cfg5.sbatch` - Runs with configuration cfg5

### Variants

- `eval_concat_0pct.sbatch`
- `eval_concat_10pct.sbatch`
- `eval_concat_1pct.sbatch`
- `eval_concat_3.16pct.sbatch`
- `eval_concat_31.6pct.sbatch`
- `eval_concat_baseline.sbatch`
- `eval_gex_baseline.sbatch`
- `eval_img_baseline.sbatch`
- `eval_img_pretrained.sbatch`
- `eval_unimodal_img.sbatch`


## Usage

```bash
sbatch eval_0pct_cfg1.sbatch
```
