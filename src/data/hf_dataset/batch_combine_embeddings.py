#!/usr/bin/env python
"""
Batch combine embeddings for multiple combinations of models and scales.

This script automates the process of combining embeddings for multiple combinations of:
- Image model scales (500K, 3M, 100M, etc.)
- Gene expression model scales (1000_donors, 2000_donors, etc.)
- Data scales (S, M, L)
- Splits (pretrain, finetune, test)

Usage:
    python batch_combine_embeddings.py --config configs/data/splits.yaml
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import itertools
import subprocess
from tqdm import tqdm

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch combine embeddings for multiple combinations")
    parser.add_argument(
        "--config", type=str, default="configs/data/splits.yaml",
        help="Path to splits YAML config file"
    )
    parser.add_argument(
        "--img_model_scales", type=str, nargs="+",
        help="Image model scales to process (e.g., '500K', '3M')"
    )
    parser.add_argument(
        "--gex_model_scales", type=str, nargs="+",
        help="Gene expression model scales to process (e.g., '1000_donors', '2000_donors')"
    )
    parser.add_argument(
        "--data_scales", type=str, nargs="+", choices=["S", "M", "L"], default=["S"],
        help="Data scales to process (S, M, L)"
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", choices=["pretrain", "finetune", "test"], default=["pretrain"],
        help="Dataset splits to process"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Local directory to save the datasets"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--input_prefix", type=str, default="theislab-multimodal-ssl/lung",
        help="Prefix for input dataset names"
    )
    parser.add_argument(
        "--output_prefix", type=str, default="theislab-multimodal-ssl/lung-clip",
        help="Prefix for output dataset names"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate all combinations
    combinations = []
    
    # Get available model scales from config if not specified
    if args.img_model_scales is None:
        img_model_scales = list(config["img_pretrain"].keys())
    else:
        img_model_scales = args.img_model_scales
    
    if args.gex_model_scales is None:
        gex_model_scales = list(config["gex_pretrain"].keys())
    else:
        gex_model_scales = args.gex_model_scales
    
    # Generate combinations
    for img_scale, gex_scale, data_scale, split in itertools.product(
        img_model_scales, gex_model_scales, args.data_scales, args.splits
    ):
        # Skip if model scales don't exist in config
        if img_scale not in config["img_pretrain"]:
            print(f"Warning: {img_scale} not found in img_pretrain config. Skipping.")
            continue
        if gex_scale not in config["gex_pretrain"]:
            print(f"Warning: {gex_scale} not found in gex_pretrain config. Skipping.")
            continue
        
        # Skip if data scale doesn't exist in config
        if data_scale not in config["multimodal"][split]:
            print(f"Warning: {data_scale} not found in config for {split}. Skipping.")
            continue
        
        combinations.append((img_scale, gex_scale, data_scale, split))
    
    print(f"Generated {len(combinations)} combinations to process")
    
    # Process each combination
    for img_scale, gex_scale, data_scale, split in tqdm(combinations, desc="Processing combinations"):
        # Generate input dataset names
        img_dataset = f"{args.input_prefix}-img-{img_scale}-{data_scale}-{split}"
        gex_dataset = f"{args.input_prefix}-gex-{gex_scale}-{data_scale}-{split}"
        
        # Generate output dataset name
        output_dataset = f"{args.output_prefix}-{img_scale}-{gex_scale}-{data_scale}-{split}"
        
        # Construct command
        cmd = [
            "python", "scripts/combine_embeddings.py",
            "--img_dataset", img_dataset,
            "--gex_dataset", gex_dataset,
            "--output_dataset", output_dataset,
        ]
        
        # Add optional arguments
        if args.cache_dir:
            cmd.extend(["--cache_dir", args.cache_dir])
        
        if args.push_to_hub:
            cmd.append("--push_to_hub")
        else:
            cmd.extend(["--output_dir", args.output_dir])
        
        # Execute or print command
        cmd_str = " ".join(cmd)
        print(f"Executing: {cmd_str}")
        
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
                print(f"Successfully processed {output_dataset}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {output_dataset}: {e}")
        
    print("Batch processing complete!")

if __name__ == "__main__":
    main() 