#!/usr/bin/env python
"""
Script for generating and storing image embeddings efficiently.

This script:
1. Loads UNI models
2. Processes image data from the original dataset
3. Saves embeddings in H5 format

Usage:
    python -m cell_synergy.data.hf_dataset.generate_img_embeddings --model_scales 1M 22M 100M --splits test
    python -m cell_synergy.data.hf_dataset.generate_img_embeddings --batch --model_scales 1M 22M 100M --data_scales S M L --splits pretrain finetune
"""

import os
import h5py
import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
import sys
import time
import argparse
from pathlib import Path
import logging
from cell_synergy.paths import PROJECT_DIR, ROOT
from cell_synergy.config import load_data_splits
import itertools
import torch
import timm
from huggingface_hub import hf_hub_download
from torchvision import transforms
from timm.layers import SwiGLUPacked

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root directory

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate image embeddings")

    # Mode selection
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")

    # Common arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration even if files exist")
    parser.add_argument("--img_size", type=int, default=224,
                       help="Image size for processing (default: 224)")
    parser.add_argument("--patch_size", type=int, default=16,
                       help="Patch size for vision models (default: 16)")

    # Single configuration arguments
    parser.add_argument("--model_scale", type=str,
                       help="Model scale (e.g., '1M', '22M', '100M')")
    parser.add_argument("--data_scale", type=str, choices=["S", "M", "L"],
                       help="Data scale (S, M, L)")
    parser.add_argument("--split", type=str, default="pretrain",
                       choices=["pretrain", "finetune", "test"],
                       help="Dataset split")

    # Batch mode arguments
    parser.add_argument("--model_scales", type=str, nargs="+",
                       help="Model scales (batch mode)")
    parser.add_argument("--data_scales", type=str, nargs="+", choices=["S", "M", "L"],
                       help="Data scales (batch mode)")
    parser.add_argument("--splits", type=str, nargs="+",
                       choices=["pretrain", "finetune", "test"],
                       default=["pretrain"],
                       help="Dataset splits (batch mode)")

    # Dataset name
    parser.add_argument("--dataset", type=str, default="lung",
                       help="Dataset name (default: lung)")

    return parser.parse_args()

def load_model(model_id, device=None, img_size=224, patch_size=16):
    """Load a vision model from HuggingFace or local files."""
    logger.info("Loading model: %s", model_id)

    # Default device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model arguments
    if "UNI2-h" in model_id:
        model_args = {
            'img_size': img_size,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
    else:
        model_args = {
            "img_size": img_size,
            "patch_size": patch_size,
            "num_classes": 0,  # No classification head needed
            "global_pool": ""  # No pooling needed
        }

    # Handle local model files
    if model_id.startswith('local/'):
        local_path = model_id.replace('local/', '')
        model_dir = PROJECT_DIR / "trained_models" / "uni_img" / local_path
        if not model_dir.exists():
            raise ValueError(f"Local model directory not found: {model_dir}")

        # Create base model
        if "UNI2-h" in model_id:
            model = timm.create_model("vit_giant_patch14_224", pretrained=False, **model_args)
        else:
            model = timm.create_model("vit_large_patch16_224", pretrained=False, **model_args)

        # Load checkpoint
        checkpoint_path = model_dir / "pytorch_model.bin"
        if not checkpoint_path.exists():
            raise ValueError(f"Model checkpoint not found: {checkpoint_path}")

        logger.info("Loading local checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning("Missing keys when loading checkpoint: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected_keys)

    # Handle HuggingFace models
    else:
        if "UNI2-h" in model_id:
            model = timm.create_model("vit_giant_patch14_224", pretrained=False, **model_args)
        else:
            model = timm.create_model("vit_large_patch16_224", pretrained=False, **model_args)
        try:
            hf_token = os.environ.get("HF_MODELS_TOKEN")
            checkpoint_path = hf_hub_download(
                model_id,
                filename="pytorch_model.bin",
                token=hf_token
            )
            logger.info("Loading checkpoint from %s", checkpoint_path)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning("Missing keys when loading checkpoint: %s", missing_keys)
            if unexpected_keys:
                logger.warning("Unexpected keys when loading checkpoint: %s", unexpected_keys)
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", str(e))
            raise

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


def process_single_configuration(cfg, model_scale, data_scale, split, batch_size, num_workers, dataset_name="lung", force=False):
    """Process a single configuration to generate embeddings."""
    # Determine output path
    if split == "test":
        filename = f"img_{model_scale}_test.h5"
    else:
        filename = f"img_{model_scale}_{split}_{data_scale}.h5"

    output_file = PROJECT_DIR / "unimodal_embeddings" / dataset_name / "img" / filename

    if output_file.exists() and not force:
        try:
            # Quick validation of existing file
            with h5py.File(str(output_file), 'r') as f:
                if len(f['embeddings']) > 0:
                    logger.info("Valid embeddings already exist at %s. Use --force to regenerate.", output_file)
                    return output_file
        except Exception as e:
            logger.warning("Found invalid embeddings at %s, will regenerate: %s", output_file, str(e))
    else:
        logger.info("No embeddings found at %s. Will regenerate.", output_file)

    # Get model ID and sample names from config
    model_id = cfg.img_pretrain[model_scale]

    # For alignment training, we might want ALL samples from the already-loaded dataset
    # Check if we should use the already-loaded dataset instead
    already_loaded_dataset_path = PROJECT_DIR / dataset_name / "hf_datasets_full" / "full_200M.L"
    use_already_loaded = already_loaded_dataset_path.exists() and split == "pretrain" and data_scale == "L"

    if use_already_loaded:
        # Load from already-loaded dataset to get ALL samples
        from datasets import load_from_disk
        logger.info("Using already-loaded dataset for ALL samples: %s", already_loaded_dataset_path)
        loaded_dataset = load_from_disk(str(already_loaded_dataset_path))
        sample_names = list(set(loaded_dataset['name']))
        logger.info("Found %s unique samples in already-loaded dataset", len(sample_names))
        hf_dataset_name = cfg.hf_datasets.get('original', 'theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6')
    else:
        sample_names = cfg.multimodal[split][data_scale] if split != "test" else cfg.multimodal[split]
        hf_dataset_name = cfg.hf_datasets.get('original', 'theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6')

    logger.info("Using image model: %s", model_id)
    logger.info("Processing %s samples from %s %s split", len(sample_names), split, data_scale)
    logger.info("Using HF dataset: %s", hf_dataset_name)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA is not available. Using CPU. This will be slow!")

    # Load dataset
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    logger.info("Loading source dataset...")
    dataset = load_dataset(
        hf_dataset_name,
        split="train", cache_dir=str(PROJECT_DIR / "hf_cache"), use_auth_token=hf_token,
    )

    # Filter dataset
    logger.info("Filtering dataset for %s samples...", len(sample_names))
    filtered = dataset.filter(
        lambda batch: [name in sample_names for name in batch['name']],
        batched=True,
        batch_size=1000
    )
    logger.info("Filtered dataset has %s samples", len(filtered))

    # Load model
    model, transform = load_model(model_id, device)

    # Create PyTorch dataset and dataloader for efficient parallel loading
    from torch.utils.data import Dataset, DataLoader

    class HFImageDataset(Dataset):
        def __init__(self, hf_dataset, transform):
            self.hf_dataset = hf_dataset
            self.transform = transform
            self.failed_indices = []

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            try:
                sample = self.hf_dataset[idx]
                image = self.transform(sample["image"])
                # Create unique name
                name = sample["name"]
                coords = sample["cell_coords"]
                coord_str = f"{coords[0][0]}_{coords[0][1]}"
                unique_name = f"{name}_{coord_str}"
                return image, unique_name
            except Exception as e:
                self.failed_indices.append(idx)
                logger.warning("Failed to load sample %s: %s", idx, e)
                # Return None to signal failure - will be handled by custom collate_fn
                return None, None

    dataset_wrapped = HFImageDataset(filtered, transform)

    # Custom collate function to handle failed samples
    def collate_fn(batch):
        # Filter out None values (failed samples)
        batch = [(img, name) for img, name in batch if img is not None]
        if len(batch) == 0:
            return None, None
        images, names = zip(*batch)
        return torch.stack(images), list(names)

    dataloader = DataLoader(
        dataset_wrapped,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,  # Don't drop incomplete batches
        collate_fn=collate_fn
    )

    # Process in batches
    all_embeddings = []
    all_names = []

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Process in batches - no checkpointing for speed
    pbar = tqdm(total=len(filtered), desc=f"Generating embeddings for {output_file.name}")

    try:
        for batch_images, batch_names in dataloader:
            # Skip empty batches (all samples failed)
            if batch_images is None:
                continue

            # Move to device
            batch_images = batch_images.to(device, non_blocking=True)

            # Generate embeddings
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    if hasattr(model, 'forward_features'):
                        patch_embeddings = model.forward_features(batch_images)
                        if patch_embeddings.ndim == 3:
                            pooled = patch_embeddings[:, 0]  # CLS token
                        else:
                            pooled = patch_embeddings
                    else:
                        pooled = model(batch_images)
                        if pooled.ndim == 3:
                            pooled = pooled[:, 0]

            # Store results
            embeddings = pooled.cpu().numpy()
            all_embeddings.extend(embeddings)
            all_names.extend(batch_names)

            # Update progress - use actual batch size
            pbar.update(len(batch_images))

    except Exception as e:
        logger.error("Error during embedding generation: %s", str(e))
        if output_file.exists():
            logger.info("Saving partial progress...")
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('embeddings', data=np.array(all_embeddings))
                f.create_dataset('sample_names', data=np.array(all_names, dtype='S'))
        raise
    finally:
        pbar.close()

    # Report any failed samples
    if dataset_wrapped.failed_indices:
        logger.warning("Failed to process %s samples", len(dataset_wrapped.failed_indices))
        logger.warning("Failed indices: %s%s", dataset_wrapped.failed_indices[:10], '...' if len(dataset_wrapped.failed_indices) > 10 else '')

    # Report statistics
    logger.info("Successfully processed %s / %s samples (%s%)", len(all_embeddings):,, len(filtered):,, 100*len(all_embeddings)/len(filtered):.1f)

    # Save final embeddings if not already saved
    if not output_file.exists():
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('embeddings', data=np.array(all_embeddings))
            f.create_dataset('sample_names', data=np.array(all_names, dtype='S'))

    logger.info("Saved embeddings to %s", output_file)
    return output_file

def process_batch_configurations(cfg, args, dataset_name="lung"):
    """Process multiple configurations to generate embeddings."""
    combinations = []
    output_files = []

    # For test split, we don't need data scales
    if len(args.splits) == 1 and args.splits[0] == "test":
        logger.info("Processing test split - ignoring data scales")
        for model_scale in args.model_scales:
            # Skip if model scale doesn't exist in config
            if model_scale not in cfg.img_pretrain:
                logger.warning("%s not found in config. Skipping.", model_scale)
                continue
            combinations.append((model_scale, None, "test"))
    else:
        # For non-test splits, we need both model scales and data scales
        if not args.data_scales:
            logger.error("Non-test splits require --data_scales")
            sys.exit(1)

        # Process all combinations for non-test splits
        for model_scale, data_scale, split in itertools.product(
                args.model_scales, args.data_scales, args.splits
        ):
            # Skip if model scale doesn't exist in config
            if model_scale not in cfg.img_pretrain:
                logger.warning("%s not found in config. Skipping.", model_scale)
                continue

            # Skip if data scale doesn't exist in config for non-test splits
            if split != "test" and data_scale not in cfg.multimodal[split]:
                logger.warning("%s not found in config for %s. Skipping.", data_scale, split)
                continue

            combinations.append((model_scale, data_scale, split))

    logger.info("Generated %s configurations to process", len(combinations))

    # Process each combination
    for model_scale, data_scale, split in tqdm(combinations, desc="Processing configurations"):
        logger.info(f"Processing img_{model_scale}_{split}" + (f"_{data_scale}" if data_scale else ""))

        output_file = process_single_configuration(
            cfg=cfg, model_scale=model_scale, data_scale=data_scale, split=split,
            batch_size=args.batch_size, num_workers=args.num_workers, dataset_name=dataset_name, force=args.force
        )

        output_files.append(output_file)

    return output_files

def main():
    args = parse_args()

    # Get dataset name from args
    dataset_name = args.dataset if hasattr(args, 'dataset') else "lung"

    # Load configuration
    logger.info("Loading splits config for dataset: %s", dataset_name)
    cfg = load_data_splits(dataset_name=dataset_name)

    logger.info("Using PROJECT_DIR: %s", PROJECT_DIR)
    logger.info("Embeddings will be saved to: %s/unimodal_embeddings/%s/img", PROJECT_DIR, dataset_name)

    # Auto-detect batch mode if batch mode parameters are provided
    if not args.batch and (args.model_scales or args.data_scales):
        args.batch = True
        logger.info("Automatically enabling batch mode due to batch parameters being provided")

    if args.batch:
        # Process multiple configurations
        if not args.model_scales:
            logger.error("Batch mode requires --model_scales")
            sys.exit(1)

        output_files = process_batch_configurations(cfg, args, dataset_name=dataset_name)
        logger.info("Processed %s configurations", len(output_files))

    else:
        # Process a single configuration
        if not args.model_scale or not args.data_scale:
            logger.error("Single mode requires --model_scale and --data_scale")
            sys.exit(1)

        output_file = process_single_configuration(
            cfg=cfg, model_scale=args.model_scale, data_scale=args.data_scale,
            split=args.split, batch_size=args.batch_size, num_workers=args.num_workers,
            dataset_name=dataset_name, force=args.force
        )

        logger.info("Successfully generated embeddings: %s", output_file)

    logger.info("Done!")

if __name__ == "__main__":
    main()
