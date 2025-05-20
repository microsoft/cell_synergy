#!/usr/bin/env python
"""
Script to train Nicheformer models on different data subsets.
"""
import os
import sys
import argparse
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.pretraining.nicheformer import train_nicheformer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Nicheformer on data subsets")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to Hydra config file"
    )
    return parser.parse_args()


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function."""
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Train Nicheformer
    checkpoint_path = train_nicheformer(cfg)
    
    print(f"Best checkpoint saved at: {checkpoint_path}")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set Hydra configuration path
    config_path = Path(args.config)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    
    # Configure Hydra
    hydra.initialize(version_base=None, config_path=str(config_path.parent))
    
    # Run main with the specified config
    main_cfg = OmegaConf.load(config_path)
    main(main_cfg) 