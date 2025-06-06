#!/usr/bin/env python
"""
Generate embeddings for HuggingFace datasets with different model scales.

This script:
1. Loads the original paired multimodal dataset from HuggingFace
2. Filters it based on the specified split (S, M, L from splits.yaml)
3. Loads the specified image model (500K, 3M, 100M, etc.) 
4. Generates embeddings for each image or gene expression data
5. Creates a new HuggingFace dataset with the updated embeddings

Usage:
    python generate_embeddings.py --config configs/data/splits.yaml \
        --modality img --model_scale 500K --data_scale S \
        --output_dataset theislab-multimodal-ssl/lung-500K-S
"""
import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor

# Add parent directory to the path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings for HuggingFace datasets")
    parser.add_argument(
        "--config", type=str, default="configs/data/splits.yaml",
        help="Path to splits YAML config file"
    )
    parser.add_argument(
        "--modality", type=str, required=True, choices=["img", "gex"],
        help="Modality to generate embeddings for (img or gex)"
    )
    parser.add_argument(
        "--model_scale", type=str, required=True,
        help="Model scale (e.g., '500K', '3M', '100M' for images; '1000_donors', etc. for gex)",
        choices=["500K", "3M", "100M", "200M", "1B"]
    )
    parser.add_argument(
        "--data_scale", type=str, required=True, choices=["S", "M", "L"],
        help="Data scale (S, M, L) for multimodal dataset"
    )
    parser.add_argument(
        "--split", type=str, default="pretrain",
        choices=["pretrain", "finetune", "test"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Cache directory for models and datasets"
    )
    parser.add_argument(
        "--output_dataset", type=str, required=True,
        help="Name of the output HuggingFace dataset (e.g., 'theislab-multimodal-ssl/lung-500K-S')"
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

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_vision_model(model_id, cache_dir=None, device=None):
    """Load a vision model from HuggingFace."""
    # Set HF_MODELS_TOKEN if available
    hf_token = os.environ.get("HF_MODELS_TOKEN")
    if hf_token is None:
        print("Warning: HF_MODELS_TOKEN environment variable not set. Some models may not be accessible.")
    
    print(f"Loading vision model: {model_id}")
    
    # Load model and feature extractor
    model = AutoModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_auth_token=hf_token,
    )
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_auth_token=hf_token,
    )
    
    # Move model to device if specified
    if device is not None:
        model = model.to(device)
    
    model.eval()  # Set to evaluation mode
    
    return model, feature_extractor

def load_gex_model(model_id, cache_dir=None, device=None):
    """
    Load a gene expression model (Nicheformer).
    
    Assumes the model is installed via pip.
    """
    try:
        from nicheformer import Nicheformer
    except ImportError:
        raise ImportError(
            "Nicheformer package not found. Please install it with: "
            "pip install git+https://github.com/theislab/nicheformer.git"
        )
    
    print(f"Loading gene expression model: {model_id}")
    
    # The model_id here is actually a path to a subset in the config
    # We need to load the corresponding checkpoint
    
    # This would be your implementation to load the Nicheformer model
    # For now, it's a placeholder
    model = Nicheformer.load_from_checkpoint(model_id)
    
    # Move model to device if specified
    if device is not None:
        model = model.to(device)
    
    model.eval()  # Set to evaluation mode
    
    return model

def filter_dataset_by_names(dataset, names_list):
    """Filter a dataset to include only samples with names in the provided list."""
    return dataset.filter(lambda x: x["name"] in names_list)

def generate_img_embeddings(model, feature_extractor, dataset, batch_size=16, device=None):
    """Generate image embeddings using the vision model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating image embeddings"):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        
        # Process images with feature extractor
        if isinstance(batch["image"][0], dict):  # HF Datasets Image format
            # Convert to numpy arrays for feature extractor
            images = [img["bytes"] for img in batch["image"]]
        else:
            images = batch["image"]
        
        inputs = feature_extractor(images=images, return_tensors="pt").to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract embeddings (this may vary by model architecture)
        if hasattr(outputs, "pooler_output"):
            batch_embeddings = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            batch_embeddings = outputs[0].mean(dim=1)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    
    return all_embeddings

def generate_gex_embeddings(model, dataset, batch_size=16, device=None):
    """Generate gene expression embeddings using the Nicheformer model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating GEX embeddings"):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        
        # Get gene expression data
        gex_data = torch.tensor(np.array(batch["gexp"])).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = model.encode(gex_data)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    
    return all_embeddings

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get sample names from config
    if args.modality == "img":
        model_id = config["img_pretrain"][args.model_scale]
        sample_names = config["multimodal"][args.split][args.data_scale]
    else:  # gex
        model_id = config["gex_pretrain"][args.model_scale]
        sample_names = config["multimodal"][args.split][args.data_scale]
    
    print(f"Using {args.modality} model: {model_id}")
    print(f"Processing {len(sample_names)} samples from {args.split} {args.data_scale} split")
    
    # Set HF_DATASETS_TOKEN if available
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    if hf_token is None:
        print("Warning: HF_DATASETS_TOKEN environment variable not set. Some datasets may not be accessible.")
    
    # Load source dataset
    print("Loading source dataset...")
    dataset = load_dataset(
        "theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6",
        split="train",
        cache_dir=args.cache_dir,
        use_auth_token=hf_token,
    )
    
    # Filter dataset by sample names
    print(f"Filtering dataset to include only {args.split} {args.data_scale} samples...")
    filtered_dataset = filter_dataset_by_names(dataset, sample_names)
    
    print(f"Filtered dataset has {len(filtered_dataset)} samples")
    
    # Load appropriate model
    if args.modality == "img":
        model, feature_extractor = load_vision_model(model_id, args.cache_dir, device)
        
        # Generate embeddings
        embeddings = generate_img_embeddings(
            model, feature_extractor, filtered_dataset, args.batch_size, device
        )
        
        # Update dataset with new embeddings
        def update_img_embeddings(example, idx):
            example["img_embed"] = embeddings[idx]
            return example
        
        updated_dataset = filtered_dataset.map(
            update_img_embeddings,
            with_indices=True,
            desc="Updating dataset with image embeddings"
        )
        
    else:  # gex
        model = load_gex_model(model_id, args.cache_dir, device)
        
        # Generate embeddings
        embeddings = generate_gex_embeddings(
            model, filtered_dataset, args.batch_size, device
        )
        
        # Update dataset with new embeddings
        def update_gex_embeddings(example, idx):
            example["gexp_embed"] = embeddings[idx]
            return example
        
        updated_dataset = filtered_dataset.map(
            update_gex_embeddings,
            with_indices=True,
            desc="Updating dataset with GEX embeddings"
        )
    
    # Save or push to HuggingFace Hub
    if args.push_to_hub:
        print(f"Pushing dataset to HuggingFace Hub: {args.output_dataset}")
        updated_dataset.push_to_hub(
            args.output_dataset,
            private=True,
            token=hf_token,
        )
    else:
        # Save locally
        output_dir = Path(args.output_dir) / args.output_dataset.replace("/", "_")
        print(f"Saving dataset locally to: {output_dir}")
        updated_dataset.save_to_disk(output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main() 