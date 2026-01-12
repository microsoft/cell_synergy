# Thymus Finetune Evaluation Scripts

This directory contains evaluation scripts for thymus finetune scaling results.

## Structure

### Unimodal GEX Evaluations
- Scales: 0%, 1%, 3.16%, 10%, 31.6%, 100%
- Configs: cfg1, cfg2, cfg3
- Total: 18 scripts (6 scales × 3 configs)

### Unimodal IMG Evaluation
- Unscaled (0%) image-only evaluation
- Total: 1 script

### Multimodal Concat Evaluations
- Scales: 0%, 1%, 3.16%, 10%, 31.6%, 100%
- Concatenates unimodal GEX scales with unscaled IMG
- Total: 6 scripts

## Submission

All jobs have been submitted. To resubmit:
```bash
bash submit_all_thymus.sh
```

## Results

Results are saved to:
- Unimodal GEX: `project_folder/results/thymus/finetune_eval_parallel/{scale}/{cfg}/`
- Unimodal IMG: `project_folder/results/thymus/finetune_eval_parallel/unimodal_img/`
- Multimodal Concat: `project_folder/results/thymus/finetune_eval_parallel/multimodal_concat/{scale}/`

Each evaluation produces a JSON file with niche classification results.


# Evaluation Finetuned Thymus

This directory contains SLURM batch scripts for evaluating models.

## Main Script

The main script is `eval_0pct_cfg1.sbatch` which runs the primary task for this directory.

### Configuration cfg1

- `eval_100pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_10pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_1pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_3.16pct_cfg1.sbatch` - Runs with configuration cfg1
- `eval_31.6pct_cfg1.sbatch` - Runs with configuration cfg1

### Configuration cfg2

- `eval_0pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_100pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_10pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_1pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_3.16pct_cfg2.sbatch` - Runs with configuration cfg2
- `eval_31.6pct_cfg2.sbatch` - Runs with configuration cfg2

### Configuration cfg3

- `eval_0pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_100pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_10pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_1pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_3.16pct_cfg3.sbatch` - Runs with configuration cfg3
- `eval_31.6pct_cfg3.sbatch` - Runs with configuration cfg3

### Thymus dataset

- `submit_all_thymus.sbatch` - For thymus dataset

### Variants

- `eval_concat_0pct.sbatch`
- `eval_concat_100pct.sbatch`
- `eval_concat_10pct.sbatch`
- `eval_concat_1pct.sbatch`
- `eval_concat_3.16pct.sbatch`
- `eval_concat_31.6pct.sbatch`
- `eval_unimodal_img.sbatch`


## Usage

```bash
sbatch eval_0pct_cfg1.sbatch
```
