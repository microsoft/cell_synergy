# Multimodal Baseline Evaluations

This directory contains SLURM batch scripts for evaluating multimodal baseline models (CCA, CoMM, DCCA, etc.) on downstream tasks.

## Main Script

The main script is `eval_breast_cca.sbatch` which runs the primary task for this directory.

### Breast dataset

- `eval_breast_barlow_twins.sbatch` - For breast dataset
- `eval_breast_byol.sbatch` - For breast dataset
- `eval_breast_comm.sbatch` - For breast dataset
- `eval_breast_concat.sbatch` - For breast dataset
- `eval_breast_dcca.sbatch` - For breast dataset
- `eval_breast_dim.sbatch` - For breast dataset
- `eval_breast_simclr.sbatch` - For breast dataset
- `eval_breast_simsiam.sbatch` - For breast dataset
- `eval_breast_vicreg.sbatch` - For breast dataset

### Configuration cfg1

- `eval_thymus_comm_cfg1_job31975900.sbatch` - Runs with configuration cfg1 - For thymus dataset - Uses specific checkpoint from job job31975900

### Configuration cfg2

- `eval_breast_comm_cfg2.sbatch` - Runs with configuration cfg2 - For breast dataset
- `eval_thymus_comm_cfg2.sbatch` - Runs with configuration cfg2 - For thymus dataset
- `eval_thymus_comm_cfg2_job31975901.sbatch` - Runs with configuration cfg2 - For thymus dataset - Uses specific checkpoint from job job31975901

### Configuration cfg3

- `eval_breast_comm_cfg3_job31975907.sbatch` - Runs with configuration cfg3 - For breast dataset - Uses specific checkpoint from job job31975907
- `eval_thymus_comm_cfg3.sbatch` - Runs with configuration cfg3 - For thymus dataset
- `eval_thymus_comm_cfg3_job31975902.sbatch` - Runs with configuration cfg3 - For thymus dataset - Uses specific checkpoint from job job31975902

### Configuration cfg4

- `eval_breast_comm_cfg4.sbatch` - Runs with configuration cfg4 - For breast dataset
- `eval_thymus_comm_cfg4.sbatch` - Runs with configuration cfg4 - For thymus dataset
- `eval_thymus_comm_cfg4_job31975903.sbatch` - Runs with configuration cfg4 - For thymus dataset - Uses specific checkpoint from job job31975903

### Configuration cfg5

- `eval_breast_comm_cfg5.sbatch` - Runs with configuration cfg5 - For breast dataset
- `eval_breast_comm_cfg5_job31975909.sbatch` - Runs with configuration cfg5 - For breast dataset - Uses specific checkpoint from job job31975909
- `eval_thymus_comm_cfg5.sbatch` - Runs with configuration cfg5 - For thymus dataset
- `eval_thymus_comm_cfg5_job31975904.sbatch` - Runs with configuration cfg5 - For thymus dataset - Uses specific checkpoint from job job31975904

### Configuration cfg6

- `eval_breast_comm_cfg6.sbatch` - Runs with configuration cfg6 - For breast dataset
- `eval_thymus_comm_cfg6.sbatch` - Runs with configuration cfg6 - For thymus dataset

### Configuration v1

- `eval_breast_comm_v1.sbatch` - Runs with configuration v1 - For breast dataset
- `eval_lung_comm_v1.sbatch` - Runs with configuration v1 - For lung dataset

### Configuration v2

- `eval_breast_comm_v2.sbatch` - Runs with configuration v2 - For breast dataset

### Lung dataset

- `eval_lung_comm_1gpu.sbatch` - For lung dataset

### Thymus dataset

- `eval_thymus_barlow_twins.sbatch` - For thymus dataset
- `eval_thymus_byol.sbatch` - For thymus dataset
- `eval_thymus_cca.sbatch` - For thymus dataset
- `eval_thymus_comm.sbatch` - For thymus dataset
- `eval_thymus_concat.sbatch` - For thymus dataset
- `eval_thymus_dcca.sbatch` - For thymus dataset
- `eval_thymus_dim.sbatch` - For thymus dataset
- `eval_thymus_simclr.sbatch` - For thymus dataset
- `eval_thymus_simsiam.sbatch` - For thymus dataset
- `eval_thymus_vicreg.sbatch` - For thymus dataset


## Usage

```bash
sbatch eval_breast_cca.sbatch
```
