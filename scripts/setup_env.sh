#!/bin/bash
# Common environment setup for all SLURM batch scripts
# 
# This script sets up the environment variables and paths needed for running
# cell_synergy scripts. Users should customize the paths below to match their
# system configuration.
#
# Usage: Source this file at the beginning of your .sbatch scripts:
#   source "$(dirname "$0")/../setup_env.sh"
#   or if in a subdirectory:
#   source "$(dirname "$0")/../../setup_env.sh"

# =============================================================================
# USER CONFIGURATION - Customize these paths for your system
# =============================================================================

# Repository root directory (absolute path)
# Default: assumes script is in scripts/ subdirectory
if [ -z "$CELL_SYNERGY_ROOT" ]; then
    # Try to auto-detect from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CELL_SYNERGY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Conda environment name
CONDA_ENV="${CONDA_ENV:-cell_synergy_env}"

# Conda installation path (for sourcing conda.sh)
# Common locations:
#   - ~/miniforge3/etc/profile.d/conda.sh
#   - ~/anaconda3/etc/profile.d/conda.sh
#   - ~/.bashrc (if conda is initialized there)
CONDA_SETUP_SCRIPT="${CONDA_SETUP_SCRIPT:-$HOME/miniforge3/etc/profile.d/conda.sh}"

# Temporary directory for large files (optional)
# Set to a directory on fast storage (e.g., lustre, scratch)
# Leave empty to use system default
TMPDIR_BASE="${TMPDIR_BASE:-}"

# =============================================================================
# Environment Setup
# =============================================================================

# Set repository root
export CELL_SYNERGY_ROOT

# Activate conda environment
if [ -f "$CONDA_SETUP_SCRIPT" ]; then
    source "$CONDA_SETUP_SCRIPT"
elif [ -f "$HOME/.bashrc" ]; then
    # Fallback: try sourcing .bashrc if conda is initialized there
    source "$HOME/.bashrc"
fi

# Activate conda environment
if command -v conda &> /dev/null; then
    conda activate "$CONDA_ENV" || {
        echo "Warning: Failed to activate conda environment '$CONDA_ENV'"
        echo "Please ensure the environment exists and conda is properly configured"
    }
else
    echo "Warning: conda command not found. Please ensure conda is installed and in PATH"
fi

# Set PYTHONPATH to include repository root
export PYTHONPATH="$CELL_SYNERGY_ROOT:$PYTHONPATH"

# Set temporary directory if specified
if [ -n "$TMPDIR_BASE" ]; then
    export TMPDIR="$TMPDIR_BASE"
    export TEMP="$TMPDIR_BASE"
    export TMP="$TMPDIR_BASE"
    mkdir -p "$TMPDIR"
fi

# Change to repository root directory
cd "$CELL_SYNERGY_ROOT" || {
    echo "Error: Failed to change to repository root: $CELL_SYNERGY_ROOT"
    exit 1
}

# =============================================================================
# Optional: Print environment info (useful for debugging)
# =============================================================================
if [ "${CELL_SYNERGY_VERBOSE:-0}" = "1" ]; then
    echo "=== Cell Synergy Environment Setup ==="
    echo "Repository root: $CELL_SYNERGY_ROOT"
    echo "Conda environment: $CONDA_ENV"
    echo "Python path: $(which python)"
    echo "PYTHONPATH: $PYTHONPATH"
    if [ -n "$TMPDIR" ]; then
        echo "TMPDIR: $TMPDIR"
    fi
    echo "======================================"
fi

