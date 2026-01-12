# Scripts Directory

This directory contains SLURM batch scripts (`.sbatch` files) for running computational tasks, organized into logical categories.

## Setup

Before using the scripts, you need to configure the environment setup. All scripts use a common setup script (`setup_env.sh`) that handles environment configuration automatically.

### How It Works

All `.sbatch` files use a standardized pattern to source the common setup script:

```bash
# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"
```

This pattern:
1. Uses `${BASH_SOURCE[0]}` to get the absolute path of the current script
2. Calculates the script's directory
3. Navigates to the `scripts/` directory (using `../` based on depth)
4. Sources `setup_env.sh` which handles all environment setup

### Initial Configuration

You can customize the setup by setting environment variables **before** submitting jobs. These are optional - the setup script has sensible defaults:

1. **Set repository root** (optional - auto-detected from script location):
   ```bash
   export CELL_SYNERGY_ROOT="/path/to/cell_synergy"
   ```
   If not set, the setup script automatically detects it from the script's location.

2. **Set conda environment name** (optional - defaults to `cell_synergy_env`):
   ```bash
   export CONDA_ENV="your_env_name"
   ```

3. **Set conda setup script path** (optional - defaults to `~/miniforge3/etc/profile.d/conda.sh`):
   ```bash
   export CONDA_SETUP_SCRIPT="~/anaconda3/etc/profile.d/conda.sh"
   ```
   Common locations:
   - `~/miniforge3/etc/profile.d/conda.sh`
   - `~/anaconda3/etc/profile.d/conda.sh`
   - Or leave unset if conda is initialized in `~/.bashrc`

4. **Set temporary directory** (optional - for large files on fast storage):
   ```bash
   export TMPDIR_BASE="/path/to/fast/storage/tmp"
   ```

**Recommended**: Add these to your `~/.bashrc` or `~/.bash_profile` for persistence:
```bash
# Cell Synergy Configuration
export CONDA_ENV="cell_synergy_env"
export CONDA_SETUP_SCRIPT="$HOME/miniforge3/etc/profile.d/conda.sh"
export TMPDIR_BASE="/lustre/groups/ml01/workspace/$USER/tmp"  # Example for HPC systems
```

### Environment Variables

The setup script (`setup_env.sh`) automatically configures:
- `CELL_SYNERGY_ROOT` - Repository root directory (auto-detected)
- `PYTHONPATH` - Includes repository root for Python imports
- `TMPDIR`, `TEMP`, `TMP` - Temporary directory (if `TMPDIR_BASE` is set)
- Conda environment activation
- Working directory change to repository root

### Verification

To verify your setup is working, you can enable verbose output:
```bash
export CELL_SYNERGY_VERBOSE=1
sbatch scripts/data_processing/create_hf_datasets/run_finetune_1_pct.sbatch
```

This will print environment information in the job output.

## Directory Structure

### `data_processing/`
- **Purpose**: Scripts for data preprocessing, dataset creation, and embedding generation
- **Subdirectories**: 
  - `datasets/` - Creating and processing datasets (HF datasets, embeddings, etc.)
- **Usage**: Each subdirectory contains `.sbatch` files that can be submitted with `sbatch <script.sbatch>`

### `evaluation/`
- **Purpose**: Scripts for evaluating model performance across various tasks
- **Subdirectories**:
  - `baselines/` - Baseline model evaluations
  - `evaluate_baselines/` - Unimodal baseline evaluations
  - `evaluate_hf_datasets/` - HuggingFace dataset evaluations
  - `evaluate_multimodal_concat/` - Multimodal concatenation evaluations
  - `evaluation_finetuned_*/` - Finetuned model evaluations (breast, lung, thymus)
  - `post_train_scaling/` - Post-training scaling experiments
  - `spatial_evaluation/` - Spatial evaluation tasks
  - `spectral/` - Spectral analysis evaluations
- **Usage**: Navigate to the specific evaluation subdirectory and submit the relevant `.sbatch` file

### `training/`
- **Purpose**: Scripts for training models
- **Subdirectories**:
  - `finetune_nicheformer/` - Nicheformer finetuning scripts
  - `train_comm_data_scaling/` - CoMM training with data scaling
  - `train_comm_testset_aligned/` - CoMM training on test sets
  - `train_multimodal_alignment/` - Multimodal alignment training (breast, lung, thymus)
- **Usage**: Navigate to the specific training subdirectory and submit the relevant `.sbatch` file

## Usage

Each subdirectory contains:
- Individual `.sbatch` files for specific jobs
- `README.md` files explaining the purpose of each directory and its scripts
- Some directories may contain `logs/` subdirectories for SLURM output files

To submit a job:
```bash
sbatch <script.sbatch>
```

## Notes

- All Python source code is in `src/`, not here
- All scripts use the common `setup_env.sh` for consistent environment setup
- Each subdirectory is self-contained with its own README explaining its purpose
- Scripts are designed to work from any directory - they automatically change to the repository root
