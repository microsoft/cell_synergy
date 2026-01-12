# Updating .sbatch Files to Use Common Setup

This document explains how to update existing `.sbatch` files to use the common `setup_env.sh` script for consistent environment configuration.

## Pattern to Replace

**Old pattern** (inconsistent across files):
```bash
# Various inconsistent patterns:
source ~/.bashrc
conda activate cell_synergy_env

# OR
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cell_synergy_env

cd /home/icb/till.richter/git/cell_synergy
export PYTHONPATH="/home/icb/till.richter/git/cell_synergy:$PYTHONPATH"

# Sometimes TMPDIR is set, sometimes not
export TMPDIR=/lustre/groups/ml01/workspace/till.richter/git/cell_synergy/tmp
```

**New pattern** (consistent):
```bash
# Source common environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setup_env.sh"  # Adjust depth based on subdirectory level

mkdir -p logs  # If logs directory is needed
```

## Path Calculation

The `SCRIPT_DIR` calculation finds the script's directory, then you source `setup_env.sh` with the appropriate relative path:

- **1 level deep** (e.g., `scripts/training/script.sbatch`): `source "$SCRIPT_DIR/../setup_env.sh"`
- **2 levels deep** (e.g., `scripts/training/train_multimodal_alignment/script.sbatch`): `source "$SCRIPT_DIR/../../setup_env.sh"`
- **3 levels deep** (e.g., `scripts/training/train_multimodal_alignment/lung/script.sbatch`): `source "$SCRIPT_DIR/../../../setup_env.sh"`

## What setup_env.sh Handles

The common setup script automatically:
- Detects repository root (or uses `$CELL_SYNERGY_ROOT` if set)
- Activates conda environment (configurable via `$CONDA_ENV`)
- Sets `PYTHONPATH` to include repository root
- Sets `TMPDIR`, `TEMP`, `TMP` if `$TMPDIR_BASE` is configured
- Changes to repository root directory

## Customization

Users can customize behavior via environment variables (set in `~/.bashrc` or before running scripts):
- `CELL_SYNERGY_ROOT` - Repository root (auto-detected if not set)
- `CONDA_ENV` - Conda environment name (default: `cell_synergy_env`)
- `CONDA_SETUP_SCRIPT` - Path to conda.sh (default: `~/miniforge3/etc/profile.d/conda.sh`)
- `TMPDIR_BASE` - Temporary directory for large files (optional)

## Benefits

1. **Consistency**: All scripts use the same environment setup
2. **Portability**: Easy to customize for different systems
3. **Maintainability**: Update paths in one place (`setup_env.sh`)
4. **No hardcoded paths**: Works on any system with proper configuration

