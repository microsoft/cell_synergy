#!/usr/bin/env python
"""
Combine image and gene expression embeddings for CLIP training.

This script:
1. Loads datasets with image embeddings and gene expression embeddings
2. Combines them into a single dataset for CLIP training
3. Saves the combined dataset locally or pushes it to HuggingFace Hub

Usage:
    python combine_embeddings.py \
        --img_dataset theislab-multimodal-ssl/lung-img-500K-S-pretrain \
        --gex_dataset theislab-multimodal-ssl/lung-gex-1000_donors-S-pretrain \
        --output_dataset theislab-multimodal-ssl/lung-clip-500K-1000_donors-S-pretrain
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Combine embeddings for CLIP training")
    parser.add_argument(
        "--img_dataset", type=str, required=True,
        help="Dataset with image embeddings (HF path or local path)"
    )
    parser.add_argument(
        "--gex_dataset", type=str, required=True,
        help="Dataset with gene expression embeddings (HF path or local path)"
    )
    parser.add_argument(
        "--output_dataset", type=str, required=True,
        help="Name for the output combined dataset"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Local directory to save the dataset (if not pushing to Hub)"
    )
    return parser.parse_args()

def load_dataset_from_path(dataset_path, cache_dir=None):
    """
    Load a dataset from HuggingFace Hub or local path.
    
    Args:
        dataset_path: Path to the dataset (HF path or local path)
        cache_dir: Cache directory for HF datasets
    
    Returns:
        The loaded dataset
    """
    # Check if dataset_path is a local path
    local_path = Path(dataset_path)
    if local_path.exists() and local_path.is_dir():
        # Load from local disk
        print(f"Loading dataset from local path: {dataset_path}")
        from datasets import load_from_disk
        return load_from_disk(dataset_path)
    
    # Otherwise, load from HuggingFace Hub
    print(f"Loading dataset from HuggingFace Hub: {dataset_path}")
    
    # Set HF_DATASETS_TOKEN if available
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    if hf_token is None:
        print("Warning: HF_DATASETS_TOKEN environment variable not set. Some datasets may not be accessible.")
    
    return load_dataset(
        dataset_path,
        split="train",  # Assume the main split is called "train"
        cache_dir=cache_dir,
        use_auth_token=hf_token,
    )

def combine_embeddings(img_dataset, gex_dataset):
    """
    Combine image and gene expression embeddings into a single dataset.
    
    Args:
        img_dataset: Dataset with image embeddings
        gex_dataset: Dataset with gene expression embeddings
    
    Returns:
        Combined dataset
    """
    print("Combining image and gene expression embeddings...")
    
    # Check that both datasets have the same samples in the same order
    if len(img_dataset) != len(gex_dataset):
        raise ValueError(f"Datasets have different lengths: {len(img_dataset)} vs {len(gex_dataset)}")
    
    # Check sample names
    img_names = img_dataset["name"]
    gex_names = gex_dataset["name"]
    
    if img_names != gex_names:
        # If names don't match, try to align the datasets
        print("Warning: Dataset sample names don't match. Attempting to align datasets...")
        
        # Create a mapping from name to index for both datasets
        img_name_to_idx = {name: i for i, name in enumerate(img_names)}
        gex_name_to_idx = {name: i for i, name in enumerate(gex_names)}
        
        # Find common names
        common_names = set(img_name_to_idx.keys()).intersection(set(gex_name_to_idx.keys()))
        
        if not common_names:
            raise ValueError("No common samples found between datasets")
        
        print(f"Found {len(common_names)} common samples")
        
        # Create aligned datasets
        img_dataset = img_dataset.select([img_name_to_idx[name] for name in common_names])
        gex_dataset = gex_dataset.select([gex_name_to_idx[name] for name in common_names])
    
    # Now combine the datasets by taking each feature from the appropriate source
    # Start with all columns from img_dataset
    combined_dataset = img_dataset
    
    # Update gexp_embed from gex_dataset
    def update_gex_embed(example, idx):
        example["gexp_embed"] = gex_dataset[idx]["gexp_embed"]
        return example
    
    combined_dataset = combined_dataset.map(
        update_gex_embed,
        with_indices=True,
        desc="Updating gexp_embed"
    )
    
    return combined_dataset

def main():
    """Main function."""
    args = parse_args()
    
    # Load the datasets
    img_dataset = load_dataset_from_path(args.img_dataset, args.cache_dir)
    gex_dataset = load_dataset_from_path(args.gex_dataset, args.cache_dir)
    
    # Combine the embeddings
    combined_dataset = combine_embeddings(img_dataset, gex_dataset)
    
    # Save or push to HuggingFace Hub
    if args.push_to_hub:
        print(f"Pushing combined dataset to HuggingFace Hub: {args.output_dataset}")
        
        # Set HF_DATASETS_TOKEN if available
        hf_token = os.environ.get("HF_DATASETS_TOKEN")
        if hf_token is None:
            print("Warning: HF_DATASETS_TOKEN environment variable not set. Some datasets may not be accessible.")
        
        combined_dataset.push_to_hub(
            args.output_dataset,
            private=True,
            token=hf_token,
        )
    else:
        # Save locally
        output_dir = Path(args.output_dir) / args.output_dataset.replace("/", "_")
        print(f"Saving combined dataset locally to: {output_dir}")
        combined_dataset.save_to_disk(output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main() 