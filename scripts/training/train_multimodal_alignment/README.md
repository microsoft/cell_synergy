# Train Multimodal Alignment Models

This directory contains SLURM batch scripts for training models.

## Main Script

The main script is `submit_all.sbatch` which runs the primary task for this directory.

### Breast dataset

- `breast_train_adversarial.sbatch` - For breast dataset
- `breast_train_barlow_twins.sbatch` - For breast dataset
- `breast_train_byol.sbatch` - For breast dataset
- `breast_train_cca.sbatch` - For breast dataset
- `breast_train_comm.sbatch` - For breast dataset
- `breast_train_dcca.sbatch` - For breast dataset
- `breast_train_dim.sbatch` - For breast dataset
- `breast_train_simclr.sbatch` - For breast dataset
- `breast_train_simsiam.sbatch` - For breast dataset
- `breast_train_vicreg.sbatch` - For breast dataset
- `submit_all_breast.sbatch` - For breast dataset

### Configuration v1

- `thymus_train_dcca_v1.sbatch` - Runs with configuration v1 - For thymus dataset

### Configuration v2

- `thymus_train_dcca_v2.sbatch` - Runs with configuration v2 - For thymus dataset

### Configuration v3

- `thymus_train_dcca_v3.sbatch` - Runs with configuration v3 - For thymus dataset

### Lung dataset

- `lung_train_adversarial.sbatch` - For lung dataset
- `lung_train_barlow_twins.sbatch` - For lung dataset
- `lung_train_byol.sbatch` - For lung dataset
- `lung_train_cca.sbatch` - For lung dataset
- `lung_train_comm.sbatch` - For lung dataset
- `lung_train_comm_4gpu.sbatch` - For lung dataset
- `lung_train_comm_4gpu_alt.sbatch` - For lung dataset
- `lung_train_dcca.sbatch` - For lung dataset
- `lung_train_dim.sbatch` - For lung dataset
- `lung_train_simclr.sbatch` - For lung dataset
- `lung_train_simsiam.sbatch` - For lung dataset
- `lung_train_vicreg.sbatch` - For lung dataset

### Thymus dataset

- `fix_thymus_scripts.sbatch` - For thymus dataset
- `submit_all_thymus.sbatch` - For thymus dataset
- `thymus_train_adversarial.sbatch` - For thymus dataset
- `thymus_train_barlow_twins.sbatch` - For thymus dataset
- `thymus_train_byol.sbatch` - For thymus dataset
- `thymus_train_cca.sbatch` - For thymus dataset
- `thymus_train_comm.sbatch` - For thymus dataset
- `thymus_train_dcca.sbatch` - For thymus dataset
- `thymus_train_dim.sbatch` - For thymus dataset
- `thymus_train_simclr.sbatch` - For thymus dataset
- `thymus_train_simsiam.sbatch` - For thymus dataset
- `thymus_train_vicreg.sbatch` - For thymus dataset


## Usage

```bash
sbatch submit_all.sbatch
```
