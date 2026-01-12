#!/usr/bin/env python3
"""
Create train/test splits and data fractions for post-training scaling experiments.

This script:
1. Loads the full train dataset (e.g., full_200M_train.L or nicheformer_full_200M.L)
2. Creates a 80/20 train/test split (ensuring test set is large enough for 5-fold CV)
3. Creates data fractions on the train set: 1%, 3.16%, 10%, 31.6%, 100%
4. Saves all datasets with appropriate naming

Usage:
    python -m cell_synergy.data.hf_dataset.create_scaling_splits \
        --dataset thymus \
        --input_dataset nicheformer_full_200M.L \
        --test_fraction 0.2 \
        --seed 42
"""

import os
import argparse
import numpy as np
from pathlib import Path
from datasets import load_from_disk
import logging
from sklearn.model_selection import train_test_split
from cell_synergy.paths import PROJECT_DIR
from cell_synergy.config import load_data_splits

# Set environment variables to use lustre for all temporary files
os.environ['TMPDIR'] = '/lustre/groups/ml01/workspace/till.richter/tmp'
os.environ['TEMP'] = '/lustre/groups/ml01/workspace/till.richter/tmp'
os.environ['TMP'] = '/lustre/groups/ml01/workspace/till.richter/tmp'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data fractions for scaling experiments
FRACTIONS = [0.01, 0.0316, 0.10, 0.316, 1.0]  # 1%, 3.16%, 10%, 31.6%, 100%
FRACTION_NAMES = ["1pct", "3.16pct", "10pct", "31.6pct", "100pct"]


def create_scaling_splits(
    dataset_name: str,
    input_dataset_name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    min_test_samples: int = 1000  # Minimum test samples for 5-fold CV (200 per fold)
):
    """
    Create train/test splits and data fractions for post-training scaling.

    Args:
        dataset_name: Dataset name (thymus, breast)
        input_dataset_name: Name of input dataset (e.g., "nicheformer_full_200M.L")
        test_fraction: Fraction of data to use for test set (default: 0.2 = 20%)
        seed: Random seed for reproducibility
        min_test_samples: Minimum number of test samples required (for 5-fold CV)
    """
    # Set random seed
    np.random.seed(seed)

    # Load config
    cfg = load_data_splits(dataset_name=dataset_name)

    # Define paths
    hf_dir = Path("/lustre/groups/ml01/workspace/till.richter/hf_datasets") / dataset_name
    input_dataset_path = hf_dir / input_dataset_name

    if not input_dataset_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dataset_path}")

    logger.info("Loading input dataset: %s", input_dataset_path)
    dataset = load_from_disk(str(input_dataset_path))
    total_samples = len(dataset)
    logger.info("Total samples: %s", total_samples:,)

    # Calculate test size
    test_size = int(total_samples * test_fraction)
    train_size = total_samples - test_size

    # Ensure test set is large enough for 5-fold CV
    if test_size < min_test_samples:
        logger.warning(
            f"Test set size ({test_size}) is smaller than minimum required ({min_test_samples}). "
            "Adjusting test fraction to ensure minimum test samples."
        )
        test_size = min_test_samples
        train_size = total_samples - test_size
        actual_test_fraction = test_size / total_samples
        logger.info("Adjusted test fraction: %s (test=%s, train=%s)", actual_test_fraction:.2%, test_size, train_size)
    else:
        actual_test_fraction = test_fraction
        logger.info("Test fraction: %s (test=%s, train=%s)", actual_test_fraction:.2%, test_size, train_size)

    # Create train/test split
    logger.info("Creating train/test split...")
    indices = np.arange(total_samples)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    logger.info("Train set: %s samples", len(train_dataset):,)
    logger.info("Test set: %s samples", len(test_dataset):,)

    # Save test dataset
    test_output_path = hf_dir / f"{input_dataset_name.replace('.L', '_test.L')}"
    logger.info("Saving test dataset to: %s", test_output_path)
    logger.info("This may take a while for large datasets with embeddings...")
    test_dataset.save_to_disk(
        str(test_output_path),
        num_proc=16,  # Use multiple processes for faster saving
        max_shard_size="10GB"  # Larger shards for better performance
    )

    # Create data fractions on train set
    logger.info("\nCreating data fractions on train set...")
    scaling_dir = hf_dir / "scaling_splits"
    scaling_dir.mkdir(exist_ok=True)

    for fraction, fraction_name in zip(FRACTIONS, FRACTION_NAMES):
        if fraction >= 1.0:
            # 100% = use full train set
            fraction_dataset = train_dataset
            fraction_size = len(fraction_dataset)
        else:
            # Sample fraction from train set
            fraction_size = int(len(train_dataset) * fraction)
            fraction_indices = np.random.choice(
                len(train_dataset),
                size=fraction_size,
                replace=False
            )
            fraction_dataset = train_dataset.select(fraction_indices)

        logger.info("  %s: %s samples (%s%)", fraction_name, fraction_size:,, 100*fraction:.2f)

        # Save fraction dataset
        # Format: {input_dataset_name}_train_{fraction_name}.L
        # Example: nicheformer_full_200M_train_1pct.L
        base_name = input_dataset_name.replace('.L', '')
        fraction_output_path = scaling_dir / f"{base_name}_train_{fraction_name}.L"
        logger.info("    Saving to: %s", fraction_output_path)
        fraction_dataset.save_to_disk(
            str(fraction_output_path),
            num_proc=16,  # Use multiple processes for faster saving
            max_shard_size="10GB"  # Larger shards for better performance
        )

    # Also save full train set
    train_output_path = scaling_dir / f"{input_dataset_name.replace('.L', '_train.L')}"
    logger.info("\nSaving full train dataset to: %s", train_output_path)
    train_dataset.save_to_disk(
        str(train_output_path),
        num_proc=16,  # Use multiple processes for faster saving
        max_shard_size="10GB"  # Larger shards for better performance
    )

    logger.info("\nScaling splits created successfully!")
    logger.info("  Test dataset: %s", test_output_path)
    logger.info("  Train dataset: %s", train_output_path)
    logger.info("  Fraction datasets: %s", scaling_dir)

    return {
        'test_path': test_output_path,
        'train_path': train_output_path,
        'fraction_dir': scaling_dir,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'fractions': {name: int(len(train_dataset) * frac) for name, frac in zip(FRACTION_NAMES, FRACTIONS)}
    }


def main():
    parser = argparse.ArgumentParser(description="Create train/test splits and data fractions for post-training scaling")
    parser.add_argument("--dataset", type=str, required=True, choices=["thymus", "breast"],
                       help="Dataset name")
    parser.add_argument("--input_dataset", type=str, required=True,
                       help="Input dataset name (e.g., 'nicheformer_full_200M.L')")
    parser.add_argument("--test_fraction", type=float, default=0.2,
                       help="Fraction of data to use for test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--min_test_samples", type=int, default=1000,
                       help="Minimum test samples required for 5-fold CV (default: 1000)")

    args = parser.parse_args()

    create_scaling_splits(
        dataset_name=args.dataset,
        input_dataset_name=args.input_dataset,
        test_fraction=args.test_fraction,
        seed=args.seed,
        min_test_samples=args.min_test_samples
    )


if __name__ == "__main__":
    main()
