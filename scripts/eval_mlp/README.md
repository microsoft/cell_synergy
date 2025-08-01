# Eval MLP Evaluation Scripts

This directory contains SLURM batch scripts for running eval_mlp evaluation on the HPC cluster.

## Overview

The eval_mlp evaluation performs standard classification and regression tasks using MLP heads instead of linear probes, using the full train/test split approach.

## Scripts

### Individual Evaluation Scripts

1. **`run_unimodal_gex.sbatch`** - Unimodal gene expression evaluation
2. **`run_unimodal_img.sbatch`** - Unimodal image evaluation  
3. **`run_random.sbatch`** - Random baseline evaluation
4. **`run_multimodal_concat.sbatch`** - Multimodal concatenation evaluation
5. **`run_multimodal_comm.sbatch`** - Multimodal CoMM evaluation

### Master Script

- **`submit_all.sbatch`** - Submits all 5 evaluation jobs in sequence

## Usage

### Submit All Jobs at Once

```bash
cd /home/icb/till.richter/git/data_scaling/scripts/eval_mlp
chmod +x submit_all.sbatch
./submit_all.sbatch
```

### Submit Individual Jobs

```bash
cd /home/icb/till.richter/git/data_scaling/scripts/eval_mlp

# Submit individual jobs
sbatch run_unimodal_gex.sbatch
sbatch run_unimodal_img.sbatch
sbatch run_random.sbatch
sbatch run_multimodal_concat.sbatch
sbatch run_multimodal_comm.sbatch
```

## Job Configuration

Each job is configured with:
- **GPU**: 1 H100 GPU
- **Memory**: 64GB RAM
- **Time Limit**: 2 days
- **CPUs**: 8 cores
- **Queue**: gpu_normal

## Output

- **Log files**: `eval_mlp_*_out_<job_id>.txt`
- **Error files**: `eval_mlp_*_err_<job_id>.txt`
- **Results**: Saved to `project_folder/results/dataset_name/embedding_baseline_mlp_img_model/`

## Monitoring

```bash
# Check job status
./check_status.sh

# Check specific job details
scontrol show job <job_id>

# Check job logs
tail -f eval_mlp_*_out_<job_id>.txt
```

## Requirements

- CoMM checkpoint (`comm_train.ckpt`) must exist for multimodal CoMM evaluation
- Conda environment `nicheformer_env` must be activated

## Expected Results

Each evaluation will produce:
- Classification metrics (accuracy, F1-macro)
- Regression metrics (R², MSE)
- JSON results file with comprehensive metrics

## Troubleshooting

1. **Job fails with "checkpoint not found"**: Ensure `comm_train.ckpt` exists in the trained models directory
2. **Out of memory**: Increase `--mem` parameter in the sbatch script
3. **Timeout**: Increase `--time` parameter if jobs are taking longer than expected 