"""Configuration utilities for cell_synergy package.

This module provides functions to load and manage configuration files
using Hydra and OmegaConf.
"""
import os
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Get the project root directory
from cell_synergy.paths import CONFIG_DIR


def get_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration from yaml files.

    Args:
        config_path: Path to config file or directory relative to the configs directory.
                    If None, loads all configs in the configs directory.

    Returns:
        DictConfig: The loaded configuration
    """
    # Set the default config directory
    if config_path is None:
        config_path = str(CONFIG_DIR)
    elif not os.path.isabs(config_path):
        # If it's a relative path, make it relative to the configs directory
        config_path = str(CONFIG_DIR / config_path)

    # Initialize hydra
    hydra.initialize(version_base=None, config_path=str(config_path))

    # Load config
    cfg = hydra.compose(config_name="config")

    return cfg


def load_data_splits(dataset_name: str):
    """
    Load data splits configuration for a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'lung', 'breast', 'thymus')

    Returns:
        DictConfig: Configuration containing train/test/validation splits
    """
    # Load the splits.yaml config specifically
    splits_path = CONFIG_DIR / "data" / f"{dataset_name}.yaml"
    return OmegaConf.load(splits_path)
