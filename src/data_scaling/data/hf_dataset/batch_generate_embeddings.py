#!/usr/bin/env python
"""
Batch process multiple combinations of models and data scales for embedding generation.

This script automates the process of generating embeddings for multiple combinations of:
- Modalities (img, gex)
- Model scales (500K, 3M, 100M, etc. for images; 1000_donors, etc. for gex)
- Data scales (S, M, L)
- Splits (pretrain, finetune, test)

Usage:
    python batch_generate_embeddings.py --config configs/data/splits.yaml
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
    parser = argparse.ArgumentParser(description="Batch generate embeddings for multiple combinations")
    parser.add_argument(
        "--config", type=str, default="configs/data/splits.yaml",
        help="Path to splits YAML config file"
    )
    parser.add_argument(
        "--modalities", type=str, nargs="+", choices=["img", "gex"], default=["img"],
        help="Modalities to process (img, gex)"
    )
    parser.add_argument(
        "--model_scales", type=str, nargs="+",
        help="Model scales to process (e.g., '500K', '3M' for images; '1000_donors' for gex)"
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
        help="Cache directory for models and datasets"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for encoding"
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
        "--output_prefix", type=str, default="theislab-multimodal-ssl/lung",
        help="Prefix for dataset names"
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
    
    for modality in args.modalities:
        # Get available model scales from config if not specified
        if args.model_scales is None:
            if modality == "img":
                model_scales = list(config["img_pretrain"].keys())
            else:  # gex
                model_scales = list(config["gex_pretrain"].keys())
        else:
            model_scales = args.model_scales
        
        # Generate combinations
        for model_scale, data_scale, split in itertools.product(
            model_scales, args.data_scales, args.splits
        ):
            # Skip if model scale doesn't exist in config
            if (modality == "img" and model_scale not in config["img_pretrain"]) or \
               (modality == "gex" and model_scale not in config["gex_pretrain"]):
                print(f"Warning: {model_scale} not found in config for {modality}. Skipping.")
                continue
            
            # Skip if data scale doesn't exist in config
            if data_scale not in config["multimodal"][split]:
                print(f"Warning: {data_scale} not found in config for {split}. Skipping.")
                continue
            
            combinations.append((modality, model_scale, data_scale, split))
    
    print(f"Generated {len(combinations)} combinations to process")
    
    # Process each combination
    for modality, model_scale, data_scale, split in tqdm(combinations, desc="Processing combinations"):
        # Generate output dataset name
        output_dataset = f"{args.output_prefix}-{modality}-{model_scale}-{data_scale}-{split}"
        
        # Construct command
        cmd = [
            "python", "scripts/generate_embeddings.py",
            "--config", args.config,
            "--modality", modality,
            "--model_scale", model_scale,
            "--data_scale", data_scale,
            "--split", split,
            "--output_dataset", output_dataset,
            "--batch_size", str(args.batch_size),
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