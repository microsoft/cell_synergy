# Training Scripts

This directory contains SLURM batch scripts for training models.

## Subdirectories

### `train_multimodal_alignment/`
Training scripts for multimodal alignment models (CoMM, CCA, DCCA, SimCLR, BYOL, etc.) on different datasets (breast, lung, thymus).

### `train_comm_data_scaling/`
Training scripts for CoMM scaling experiments with different data percentages (1pct, 3.16pct, 10pct, 31.6pct, 100pct).

### `train_comm_testset_aligned/`
Training scripts for CoMM models trained on test sets with aligned embeddings. This is used for post-training scaling analysis where CoMM is trained on test sets with finetuned GEX embeddings.

### `finetune_nicheformer/`
Nicheformer finetuning sweep scripts for different subset sizes (1pct, 3.16pct, 10pct, 31.6pct, 100pct) with optimized hyperparameter configurations.

## Usage
Navigate to the specific subdirectory and submit the relevant `.sbatch` file.
