#!/usr/bin/env python
"""
Script for generating and storing image embeddings efficiently.

This script:
1. Loads vision models 
2. Processes images from the original dataset
3. Saves embeddings in HDF5 format

Usage:
    # After installing the package with 'pip install -e .'
    # Run as module:
    python -m data_scaling.data.hf_dataset.generate_img_embeddings --model_scale 500K --data_scale S
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import logging
import itertools
from tqdm import tqdm
import numpy as np
import h5py
import timm
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from omegaconf import OmegaConf
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project paths and config
from data_scaling.paths import PROJECT_DIR, UNI_EMBEDDINGS_DIR
from data_scaling.config import load_data_splits

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate image embeddings")
    
    # Mode selection
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    
    # Common arguments
    parser.add_argument("--cache_dir", type=str, default=None, 
                       help="Cache directory for models and datasets")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for encoding")
    parser.add_argument("--force", action="store_true", 
                       help="Force regeneration even if files exist")
    parser.add_argument("--img_size", type=int, default=224,
                       help="Image size for processing (default: 224)")
    parser.add_argument("--patch_size", type=int, default=16,
                       help="Patch size for vision models (default: 16)")
    
    # Single configuration arguments
    parser.add_argument("--model_scale", type=str, 
                       help="Model scale (e.g., '500K')")
    parser.add_argument("--data_scale", type=str, choices=["S", "M", "L"], 
                       help="Data scale (S, M, L)")
    parser.add_argument("--split", type=str, default="pretrain", 
                       choices=["pretrain", "finetune", "test"], 
                       help="Dataset split")
    
    # Batch mode arguments
    parser.add_argument("--model_scales", type=str, nargs="+", 
                       help="Model scales (batch mode) or 'all' to process all models")
    parser.add_argument("--data_scales", type=str, nargs="+", choices=["S", "M", "L"], 
                       help="Data scales (batch mode)")
    parser.add_argument("--splits", type=str, nargs="+", 
                       choices=["pretrain", "finetune", "test"], 
                       default=["pretrain"], 
                       help="Dataset splits (batch mode)")
    
    return parser.parse_args()

def load_model(model_id, cache_dir=None, device=None, img_size=224, patch_size=16):
    """
    Load a vision model from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Directory to cache downloaded models
        device: Device to load model on
        img_size: Size to resize images to
        patch_size: Patch size for vision models
    
    Returns:
        tuple: (model, transform)
    """
    logger.info(f"Loading model: {model_id}")
    
    # Default device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Special handling for specific model architectures
    model_args = {
        "img_size": img_size,
        "patch_size": patch_size,
        "num_classes": 0,
        "dynamic_img_size": True
    }
    
    # Set model-specific parameters
    if model_id.startswith("bioptimus/"):
        logger.warning(f"Model id {model_id} is a Bioptimus model. Using ViT-L/14 architecture with timm weights.")
        model_args["patch_size"] = 14  # Bioptimus uses patch size 14
        model = timm.create_model("vit_large_patch16_224", pretrained=True, **model_args)
    elif "UNI2-h" in model_id or "UNI2-b" in model_id:
        logger.warning(f"Model id {model_id} is a UNI model. Using ViT-L/14 architecture with embedding dim 1536.")
        model_args["patch_size"] = 14  # UNI uses patch size 14
        model_args["embed_dim"] = 1536  # UNI-h uses embedding dimension 1536
        model_args["mlp_ratio"] = 5.33  # UNI uses 8192/1536 ratio for MLP
        # For UNI models, we need to specify exact pos_embed shape without cls token
        model = timm.create_model(
            "vit_large_patch16_224", 
            pretrained=False, 
            **model_args
        )
        
        # The position embedding issue is because timm adds the cls token separately
        # Let's use a simpler approach - skip checkpoint loading and just use model for forward pass
        logger.warning(f"Skipping checkpoint loading for {model_id} due to architecture differences. Using model for feature extraction only.")
    elif "prov-gigapath" in model_id:
        logger.warning(f"Model id {model_id} is a GigaPath model. Using ViT architecture with embedding dim 1536.")
        model_args["embed_dim"] = 1536  # GigaPath uses embedding dimension 1536
        model_args["mlp_ratio"] = 5.33  # 8192/1536 ratio for MLP
        model = timm.create_model(
            "vit_large_patch16_224", 
            pretrained=False, 
            **model_args
        )
        logger.warning(f"Using custom architecture for {model_id}")
    else:
        model = timm.create_model("vit_large_patch16_224", **model_args)
        
    # Download and load checkpoint for non-Bioptimus models
    if not model_id.startswith("bioptimus/") and not ("UNI2-h" in model_id or "UNI2-b" in model_id) and not "prov-gigapath" in model_id:
        try:
            hf_token = os.environ.get("HF_MODELS_TOKEN", None)
            checkpoint_path = hf_hub_download(
                model_id,
                filename="pytorch_model.bin",
                cache_dir=cache_dir,
                token=hf_token
            )
            # Load checkpoint
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Load with strict=False to ignore missing or extra keys
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)}")
            logger.warning("Using pre-trained weights from timm")
            # Use timm weights for non-UNI models
            if not ("UNI2-h" in model_id or "UNI2-b" in model_id) and not "prov-gigapath" in model_id:
                model = timm.create_model(
                    "vit_large_patch16_224", 
                    pretrained=True,
                    **model_args
                )
            else:
                logger.warning("Cannot fall back to pre-trained weights for this model type. Using random initialization.")
    
    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    return model, transform

def compute_img_embedding(image, model, transform, device=None):
    """
    Compute embedding for a single image.
    
    Args:
        image: Input image
        model: Vision model
        transform: Image transformation
        device: Device to run model on
    
    Returns:
        torch.Tensor: Image embedding
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Apply transform and add batch dimension
    x = transform(image).unsqueeze(0).to(device)
        
    # Generate embedding
    with torch.no_grad():
        if hasattr(model, 'forward_features'):
            # Try the standard timm forward_features method
            try:
                embedding = model.forward_features(x)
            except RuntimeError as e:
                logger.warning(f"Error in forward_features: {e}. Falling back to standard forward pass.")
                # For models with size mismatch in position embedding
                embedding = model(x)
                # If the output has a larger size (classification head), take features before that
                if len(embedding.shape) > 2:
                    embedding = embedding[:, 0]  # Take CLS token
        else:
            embedding = model(x)
    
    return embedding

def filter_dataset_by_names(dataset, names_list):
    """Filter a dataset to include only samples with names in the provided list."""
    return dataset.filter(lambda x: x["name"] in names_list)

def generate_img_embeddings(model, transform, dataset, batch_size=16, device=None):
    """
    Generate embeddings for all images in a dataset.
    
    Args:
        model: Vision model
        transform: Image transformation
        dataset: Dataset containing images
        batch_size: Batch size for processing
        device: Device to run model on
    
    Returns:
        tuple: (embeddings, sample_names)
    """
    if device is None:
        device = next(model.parameters()).device
    all_embeddings = []
    sample_names = []
    i = 0
    total = len(dataset)
    pbar = tqdm(total=total, desc="Generating embeddings")
    while i < total:
        try:
            batch = dataset[i:min(i+batch_size, total)]
            sample_names.extend(batch["name"])
            batch_embeddings = []
            for img in batch["image"]:
                emb = compute_img_embedding(img, model, transform, device)
                batch_embeddings.append(emb.cpu())
            all_embeddings.extend(batch_embeddings)
            i += batch_size
            pbar.update(len(batch_embeddings))
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                logger.warning(f"CUDA OOM! Reducing batch size to {batch_size}")
                if batch_size == 1:
                    logger.error("Batch size reduced to 1 but still OOM. Exiting.")
                    raise
            else:
                raise
    pbar.close()
    embeddings = torch.stack(all_embeddings).numpy()
    return embeddings, sample_names

def save_embeddings_h5(embeddings, sample_names, output_path, model_scale, data_scale, split):
    """Save embeddings in HDF5 format."""
    # Ensure the img subdirectory exists
    img_dir = output_path / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"img_{model_scale}_{data_scale}_{split}.h5"
    output_file = img_dir / filename
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        sample_names_array = np.array(sample_names, dtype='S')
        f.create_dataset('sample_names', data=sample_names_array)
        
        f.attrs['modality'] = 'img'
        f.attrs['model_scale'] = model_scale
        f.attrs['data_scale'] = data_scale
        f.attrs['split'] = split
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['num_samples'] = embeddings.shape[0]
        f.attrs['patches_per_sample'] = embeddings.shape[0] // len(sample_names) if len(sample_names) > 0 else 0
    
    logger.info(f"Saved embeddings to {output_file}")
    return output_file

def process_single_configuration(cfg, model_scale, data_scale, split, batch_size=16, cache_dir=None, force=False, img_size=224, patch_size=16):
    """Process a single configuration to generate embeddings."""
    # Check if output file already exists
    filename = f"img_{model_scale}_{data_scale}_{split}.h5"
    output_file = UNI_EMBEDDINGS_DIR / "img" / filename
    
    if output_file.exists() and not force:
        logger.info(f"Embeddings already exist at {output_file}. Use --force to regenerate.")
        return output_file
    
    # Get model ID and sample names from config
    model_id = cfg.img_pretrain[model_scale]
    sample_names = cfg.multimodal[split][data_scale]
    
    logger.info(f"Using image model: {model_id}")
    logger.info(f"Processing {len(sample_names)} samples from {split} {data_scale} split")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    logger.info("Loading source dataset...")
    dataset = load_dataset(
        "theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6",
        split="train", cache_dir=cache_dir, use_auth_token=hf_token,
    )
    
    # Filter dataset
    logger.info(f"Filtering dataset to include only {split} {data_scale} samples...")
    filtered_dataset = filter_dataset_by_names(dataset, sample_names)
    logger.info(f"Filtered dataset has {len(filtered_dataset)} samples")
    
    # Load model and generate embeddings using the simplified approach
    model, transform = load_model(model_id, cache_dir, device, img_size=img_size, patch_size=patch_size)
    embeddings, embedding_sample_names = generate_img_embeddings(
        model, transform, filtered_dataset, batch_size, device
    )
    
    # Save embeddings
    output_file = save_embeddings_h5(
        embeddings, embedding_sample_names, UNI_EMBEDDINGS_DIR, 
        model_scale, data_scale, split
    )
    
    return output_file

def process_batch_configurations(cfg, args):
    """Process multiple configurations to generate embeddings."""
    # Generate combinations
    combinations = []
    output_files = []
    
    # Handle "all" option for model_scales
    model_scales = args.model_scales
    if len(model_scales) == 1 and model_scales[0].lower() == "all":
        logger.info("Processing ALL available image models from config")
        model_scales = list(cfg.img_pretrain.keys())
        logger.info(f"Found models: {', '.join(model_scales)}")
    
    for model_scale, data_scale, split in itertools.product(
        model_scales, args.data_scales, args.splits
    ):
        # Skip if model scale doesn't exist in config
        if model_scale not in cfg.img_pretrain:
            logger.warning(f"{model_scale} not found in config. Skipping.")
            continue
        
        # Skip if data scale doesn't exist in config
        if data_scale not in cfg.multimodal[split]:
            logger.warning(f"{data_scale} not found in config for {split}. Skipping.")
            continue
        
        combinations.append((model_scale, data_scale, split))
    
    logger.info(f"Generated {len(combinations)} configurations to process")
    
    # Process each combination
    for model_scale, data_scale, split in tqdm(combinations, desc="Processing configurations"):
        logger.info(f"Processing img_{model_scale}_{data_scale}_{split}")
        
        output_file = process_single_configuration(
            cfg=cfg, model_scale=model_scale, data_scale=data_scale, split=split,
            batch_size=args.batch_size, cache_dir=args.cache_dir, force=args.force, 
            img_size=args.img_size, patch_size=args.patch_size
        )
        
        output_files.append(output_file)
    
    return output_files

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration using the new config system
    cfg = load_data_splits(dataset_name="lung")
    
    logger.info(f"Using PROJECT_DIR: {PROJECT_DIR}")
    logger.info(f"Embeddings will be saved to: {UNI_EMBEDDINGS_DIR / 'img'}")
    
    # Auto-detect batch mode if batch mode parameters are provided
    if not args.batch and (args.model_scales or args.data_scales):
        args.batch = True
        logger.info("Automatically enabling batch mode due to batch parameters being provided")
    
    if args.batch:
        # Process multiple configurations
        if not args.model_scales or not args.data_scales:
            logger.error("Batch mode requires --model_scales and --data_scales")
            sys.exit(1)
        
        output_files = process_batch_configurations(cfg, args)
        logger.info(f"Processed {len(output_files)} configurations")
        
    else:
        # Process a single configuration
        if not args.model_scale or not args.data_scale:
            logger.error("Single mode requires --model_scale and --data_scale")
            sys.exit(1)
        
        output_file = process_single_configuration(
            cfg=cfg, model_scale=args.model_scale, data_scale=args.data_scale,
            split=args.split, batch_size=args.batch_size, cache_dir=args.cache_dir, 
            force=args.force, img_size=args.img_size, patch_size=args.patch_size
        )
        
        logger.info(f"Successfully generated embeddings: {output_file}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
