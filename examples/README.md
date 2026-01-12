# Examples and Tutorials

This directory contains simple, tutorial-style examples demonstrating how to use the Cell Synergy codebase for common tasks.

## Overview

These examples are designed to be educational and use minimal synthetic data to demonstrate key concepts. For production use with real data, refer to the scripts in `scripts/` directory.

## Available Examples

1. **`example_compute_sis.py`** - Computing the Synergistic Information Score (SIS)
2. **`example_finetune_nicheformer.py`** - Finetuning Nicheformer on new data
3. **`example_align_comm.py`** - Training a CoMM alignment model
4. **`example_evaluate_f1_r2.py`** - Evaluating F1 and R² scores on downstream tasks
5. **`example_spatial_consistency.py`** - Evaluating spatial consistency
6. **`example_spatial_neighbors.py`** - Evaluating spatial neighbor prediction

## Running Examples

All examples can be run directly with Python:

```bash
python examples/example_compute_sis.py
```

Note: These examples use synthetic data and are designed for demonstration purposes. For real experiments, use the SLURM batch scripts in `scripts/`.

