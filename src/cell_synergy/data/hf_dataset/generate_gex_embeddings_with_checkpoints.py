#!/usr/bin/env python
"""
Script for generating and storing GEX embeddings efficiently with support for different checkpoints.

This script:
1. Loads Nicheformer models from specified checkpoints
2. Processes GEX data from the original dataset
3. Saves embeddings in both NPY and H5 formats (H5 includes sample names)
4. Supports both single checkpoint and batch checkpoint processing

Usage:
    # Single checkpoint
    python -m cell_synergy.data.hf_dataset.generate_gex_embeddings_with_checkpoints \
        --checkpoint_path /path/to/checkpoint.ckpt \
        --model_scale full --data_scale S --split pretrain

    # Batch mode with multiple checkpoints
    python -m cell_synergy.data.hf_dataset.generate_gex_embeddings_with_checkpoints \
        --batch \
        --checkpoint_paths /path/to/checkpoint1.ckpt /path/to/checkpoint2.ckpt \
        --model_scales full 4000_donors \
        --data_scales S M L \
        --splits pretrain finetune test
"""

import os

# Set PyTorch memory allocator BEFORE importing torch
# This must be done before any torch imports to be effective
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import h5py
import yaml
from datasets import load_dataset
from tqdm import tqdm
import sys
import time
import argparse
from pathlib import Path
import logging
import gc
from cell_synergy.paths import PROJECT_DIR, ROOT
from cell_synergy.config import load_data_splits
from cell_synergy.data.hf_dataset.create_hf_dataset.process.nicheformer.embedder import NicheformerEmbedder
import itertools
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check GPU availability and capabilities."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available. Using CPU.")
        return 'cpu'

    try:
        # Test basic CUDA operations
        test_tensor = torch.zeros(1).cuda()
        del test_tensor

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info("Found %s CUDA device(s)", device_count)
        logger.info("Using GPU: %s", device_name)

        # Test memory allocation
        try:
            # Try allocating and operating on a small tensor
            x = torch.randn(1000, 1000).cuda()
            y = x @ x.t()
            del x, y
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("GPU memory test failed. Falling back to CPU")
                return 'cpu'
            raise

        return 'cuda'
    except Exception as e:
        logger.warning("GPU initialization failed with error: %s", str(e))
        logger.info("Falling back to CPU")
        return 'cpu'

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate GEX embeddings with checkpoint support")

    # Mode selection
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")

    # Common arguments
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for encoding (number of samples to process in parallel)")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration even if files exist")

    # Single configuration arguments
    parser.add_argument("--model_scale", type=str,
                       help="Model scale (e.g., 'full', '4000_donors')")
    parser.add_argument("--data_scale", type=str, choices=["S", "M", "L"],
                       help="Data scale (S, M, L)")
    parser.add_argument("--split", type=str, default="pretrain",
                       choices=["pretrain", "finetune", "test"],
                       help="Dataset split")
    parser.add_argument("--checkpoint_path", type=str,
                       help="Path to the Nicheformer checkpoint file")
    parser.add_argument("--dataset", type=str, choices=["lung", "thymus", "breast"],
                       help="Dataset name (lung, thymus, or breast). If not provided, will be inferred from checkpoint_path")

    # Batch mode arguments
    parser.add_argument("--model_scales", type=str, nargs="+",
                       help="Model scales (batch mode) or 'all' to process all models")
    parser.add_argument("--data_scales", type=str, nargs="+", choices=["S", "M", "L"],
                       help="Data scales (batch mode)")
    parser.add_argument("--splits", type=str, nargs="+",
                       choices=["pretrain", "finetune", "test"],
                       help="Dataset splits (batch mode)")
    parser.add_argument("--checkpoint_paths", type=str, nargs="+",
                       help="Paths to Nicheformer checkpoint files (batch mode)")

    return parser.parse_args()

def get_sample_ids_for_scale(scale, split, config):
    """Get sample IDs for a specific scale and split."""
    if split == "test":
        return config.multimodal[split]
    else:
        return config.multimodal[split][scale]

def load_nicheformer_model(device, checkpoint_path=None):
    """Load Nicheformer model (like multimodal-ssl)."""
    logger.info("Loading Nicheformer model from checkpoint: %s", checkpoint_path)

    # Create embedder exactly like multimodal-ssl
    # batch_size here is for internal cell processing, not sample-level batching
    embedder = NicheformerEmbedder(
        batch_size=1024,
        device=device,
        technology_mean="xenium_mean_script.npy",
        checkpoint_path=checkpoint_path
    )

    return embedder

def compute_nicheformer_embeddings(embedder, device, out_path, batch_size=128, dataset_name="lung"):
    """
    Compute Nicheformer embeddings for the full HF dataset with batched processing.

    Args:
        embedder: Nicheformer embedder instance
        device: Device to run on
        out_path: Output path for embeddings
        batch_size: Number of samples to process in each batch (uses padding for variable-length sequences)
        dataset_name: Dataset name (lung, thymus, or breast)
    """
    logger.info("Loading source dataset for %s...", dataset_name)

    # Map dataset names to HF dataset identifiers
    dataset_map = {
        "lung": "theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6",
        "thymus": "theislab-multimodal-ssl/thymus_ts2",
        "breast": "theislab-multimodal-ssl/breast_embeddings",
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of: {list(dataset_map.keys())}")

    # Load the full dataset
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    dataset = load_dataset(
        dataset_map[dataset_name],
        split="train",
        use_auth_token=hf_token,
    )

    logger.info("Processing full dataset with %s samples...", len(dataset))

    # Prepare data for the embedder
    all_embeddings = []
    all_sample_names = []

    logger.info("Starting embedding extraction for %s (batching %s samples at a time)...", out_path, batch_size)

    # Track statistics
    total_samples = len(dataset)
    zero_expr_samples = 0

    # Process samples in batches for efficiency
    def process_batch(batch_samples, batch_indices):
        """Process a batch of samples together with padding."""
        batch_gexp = []
        batch_names = []
        batch_num_cells = []
        valid_samples = []
        valid_indices = []

        # Prepare batch data
        for idx, sample in zip(batch_indices, batch_samples):
            try:
                gexp = sample["gexp"]
                gexp_tensor = torch.tensor(gexp, dtype=torch.float32)

                # Check dimensions
                if gexp_tensor.ndim != 2:
                    continue

                # Check for zero expression
                if gexp_tensor.sum() == 0:
                    continue

                batch_gexp.append(gexp_tensor)
                batch_num_cells.append(gexp_tensor.size(0))
                valid_samples.append(sample)
                valid_indices.append(idx)

                # Create sample name
                name = sample["name"]
                coords = sample["cell_coords"]
                coord_str = f"{coords[0][0]}_{coords[0][1]}"
                unique_name = f"{name}_{coord_str}"
                batch_names.append(unique_name)
            except Exception:
                continue

        if len(batch_gexp) == 0:
            return []

        # Pad to same length (max number of cells in batch)
        max_cells = max(batch_num_cells)
        num_genes = batch_gexp[0].size(1)

        # Create padded batch tensor
        padded_batch = torch.zeros(len(batch_gexp), max_cells, num_genes, dtype=torch.float32, device=device)
        mask = torch.zeros(len(batch_gexp), max_cells, dtype=torch.bool, device=device)

        for i, (gexp_tensor, num_cells) in enumerate(zip(batch_gexp, batch_num_cells)):
            padded_batch[i, :num_cells, :] = gexp_tensor.to(device)
            mask[i, :num_cells] = True

        # Get embeddings using the embedder
        with torch.no_grad():
            embeddings, batch_indices_embed = embedder((padded_batch, mask))

        # embeddings shape: (num_real_cells, embedding_dim)
        # batch_indices_embed shape: (num_real_cells,) - indicates which sample each cell belongs to

        # Average embeddings across cells for each sample
        batch_embeddings = []
        for sample_idx in range(len(batch_gexp)):
            # Find all cells belonging to this sample
            cell_mask = (batch_indices_embed == sample_idx)
            if cell_mask.sum() > 0:
                sample_embedding = embeddings[cell_mask].mean(dim=0).cpu().numpy()
            else:
                # No valid cells for this sample
                sample_embedding = np.full(512, np.nan, dtype=np.float32)
            batch_embeddings.append(sample_embedding)

        # Clean up
        del padded_batch, mask, embeddings, batch_indices_embed
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

        return list(zip(valid_indices, batch_embeddings, batch_names))

    # Process samples in batches
    i = 0
    while i < total_samples:
        batch_end = min(i + batch_size, total_samples)
        batch_samples = [dataset[j] for j in range(i, batch_end)]
        batch_indices = list(range(i, batch_end))

        try:
            batch_results = process_batch(batch_samples, batch_indices)

            # Store results, handling samples that weren't processed in batch
            processed_indices = {idx for idx, _, _ in batch_results}
            for j, sample in enumerate(batch_samples):
                idx = batch_indices[j]
                if idx in processed_indices:
                    # Find the result for this index
                    for result_idx, embedding, name in batch_results:
                        if result_idx == idx:
                            all_embeddings.append(embedding)
                            all_sample_names.append(name)
                            break
                else:
                    # Sample wasn't processed (error case), add NaN embedding
                    sample_embedding = np.full(512, np.nan, dtype=np.float32)
                    all_embeddings.append(sample_embedding)

                    name = sample.get("name", f"error_{idx}")
                    coords = sample.get("cell_coords", [[0, 0]])
                    coord_str = f"{coords[0][0]}_{coords[0][1]}"
                    unique_name = f"{name}_{coord_str}"
                    all_sample_names.append(unique_name)

            i = batch_end
        except Exception as e:
            logger.error("Error processing batch starting at sample %s: %s", i, str(e))
            # Add NaN embeddings for all samples in failed batch
            for j, sample in enumerate(batch_samples):
                idx = batch_indices[j]
                sample_embedding = np.full(512, np.nan, dtype=np.float32)
                all_embeddings.append(sample_embedding)

                name = sample.get("name", f"error_{idx}")
                coords = sample.get("cell_coords", [[0, 0]])
                coord_str = f"{coords[0][0]}_{coords[0][1]}"
                unique_name = f"{name}_{coord_str}"
                all_sample_names.append(unique_name)

        i = batch_end

        # Update progress
        if i % 100 == 0 or i == total_samples:
            logger.info("Processed %s/%s samples...", i, total_samples)

        # Periodic garbage collection
        if i % (batch_size * 10) == 0:
            gc.collect()

    # Add progress bar for final summary
    logger.info("Completed processing all %s samples", total_samples)

    # Convert to numpy arrays
    embeddings_array = np.array(all_embeddings)
    sample_names_array = np.array(all_sample_names, dtype='S')

    # Count NaN embeddings
    nan_count = np.sum(np.any(np.isnan(embeddings_array), axis=1))
    logger.info("Processed %s samples total", len(embeddings_array))
    if zero_expr_samples > 0:
        logger.info("  - %s samples had no gene expression (used NaN embeddings)", zero_expr_samples)
    if nan_count > 0:
        logger.info("  - %s samples total had NaN embeddings", nan_count)
        logger.info("  - %s samples had valid gene expression", len(embeddings_array) - nan_count)

    # Save in NPY format (backward compatibility)
    np.save(str(out_path), embeddings_array)
    logger.info("Saved Nicheformer pooled embeddings to %s", out_path)

    # Save in H5 format with sample names
    h5_path = out_path.with_suffix('.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings_array)
        f.create_dataset('sample_names', data=sample_names_array)
    logger.info("Saved Nicheformer embeddings with names to %s", h5_path)

def process_single_configuration(cfg, model_scale, data_scale, split, batch_size, force=False, checkpoint_path=None, dataset_name="lung"):
    """Process a single configuration to generate embeddings."""
    # Determine output path
    if checkpoint_path:
        # Use model_scale if provided (contains pct and cfg info), otherwise use checkpoint name
        if model_scale:
            filename = f"nicheformer_{model_scale}_full_dataset.npy"
        else:
            # Extract checkpoint name for filename
            checkpoint_name = Path(checkpoint_path).stem
            filename = f"nicheformer_{checkpoint_name}_full_dataset.npy"
    else:
        # Use original naming convention (for backward compatibility)
        if model_scale and data_scale and split:
            if split == "test":
                filename = f"nicheformer_{model_scale}_test.npy"
            else:
                filename = f"nicheformer_{model_scale}_{split}_{data_scale}.npy"
        else:
            filename = "nicheformer_full_dataset.npy"

    output_file = PROJECT_DIR / "unimodal_embeddings" / dataset_name / "gex" / filename

    if output_file.exists() and not force:
        try:
            # Quick validation of existing file
            embeddings = np.load(str(output_file), allow_pickle=True)
            if len(embeddings) > 0:
                logger.info("Valid embeddings already exist at %s. Use --force to regenerate.", output_file)
                return output_file
        except Exception as e:
            logger.warning("Found invalid embeddings at %s, will regenerate: %s", output_file, str(e))

    if checkpoint_path:
        logger.info("Processing full HF dataset with checkpoint: %s", Path(checkpoint_path).stem)
        print(f"Processing full HF dataset with checkpoint: {Path(checkpoint_path).stem}", flush=True)
    else:
        logger.info("Processing full HF dataset")
        print("Processing full HF dataset", flush=True)

    import sys
    sys.stdout.flush()

    # Force CUDA if available, no CPU fallback
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")

    print("CUDA available, creating device...", flush=True)
    sys.stdout.flush()
    device = torch.device('cuda')

    # Load model with checkpoint
    print(f"Loading Nicheformer model from checkpoint: {checkpoint_path}", flush=True)
    sys.stdout.flush()
    embedder = load_nicheformer_model(device, checkpoint_path)
    print("Model loaded successfully", flush=True)
    sys.stdout.flush()

    # Compute embeddings for full dataset with batching
    compute_nicheformer_embeddings(
        embedder=embedder,
        device=device,
        out_path=output_file,
        batch_size=batch_size,
        dataset_name=dataset_name
    )

    return output_file

def process_batch_configurations(cfg, args, dataset_name="lung"):
    """Process multiple configurations in batch mode."""
    logger.info("Processing configurations:")

    # Generate all configurations
    configs = []

    # If checkpoint_paths are provided, create configurations for each checkpoint
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths:
        for checkpoint_path in args.checkpoint_paths:
            for model_scale in args.model_scales:
                for split in args.splits:
                    if split == "test":
                        configs.append((model_scale, split, "S", checkpoint_path))  # Scale doesn't matter for test
                    else:
                        for data_scale in args.data_scales:
                            configs.append((model_scale, split, data_scale, checkpoint_path))
    else:
        # Original behavior without checkpoints
        for model_scale in args.model_scales:
            for split in args.splits:
                if split == "test":
                    configs.append((model_scale, split, "S", None))  # Scale doesn't matter for test
                else:
                    for data_scale in args.data_scales:
                        configs.append((model_scale, split, data_scale, None))

    logger.info("Generated %s configurations to process", len(configs))

    # Process each configuration
    for i, (model_scale, split, data_scale, checkpoint_path) in enumerate(tqdm(configs, desc="Processing configurations")):
        try:
            if checkpoint_path:
                checkpoint_name = Path(checkpoint_path).stem
                logger.info("Processing nicheformer_%s_%s_%s", checkpoint_name, split, data_scale)
            else:
                logger.info("Processing nicheformer_%s_%s_%s", model_scale, split, data_scale)

            process_single_configuration(
                cfg=cfg,
                model_scale=model_scale,
                data_scale=data_scale,
                split=split,
                batch_size=args.batch_size,
                force=args.force,
                checkpoint_path=checkpoint_path,
                dataset_name=dataset_name
            )
        except Exception as e:
            if checkpoint_path:
                checkpoint_name = Path(checkpoint_path).stem
                logger.error("Error processing %s_%s_%s: %s", checkpoint_name, split, data_scale, str(e))
            else:
                logger.error("Error processing %s_%s_%s: %s", model_scale, split, data_scale, str(e))
            continue

def infer_dataset_from_checkpoint(checkpoint_path):
    """Infer dataset name from checkpoint path."""
    checkpoint_path_str = str(checkpoint_path)
    if "thymus" in checkpoint_path_str.lower():
        return "thymus"
    elif "breast" in checkpoint_path_str.lower():
        return "breast"
    elif "lung" in checkpoint_path_str.lower():
        return "lung"
    else:
        # Default to lung for backward compatibility
        return "lung"

def main():
    """Main function."""
    import sys
    print("Python script started", flush=True)
    sys.stdout.flush()

    args = parse_args()
    print(f"Arguments parsed: checkpoint_path={args.checkpoint_path}, batch_size={args.batch_size}", flush=True)
    sys.stdout.flush()

    # Determine dataset
    if args.dataset:
        dataset_name = args.dataset
    elif args.checkpoint_path:
        dataset_name = infer_dataset_from_checkpoint(args.checkpoint_path)
        logger.info("Inferred dataset '%s' from checkpoint path", dataset_name)
    else:
        # Default to lung for backward compatibility
        dataset_name = "lung"
        logger.info("No dataset specified and no checkpoint path, defaulting to 'lung'")

    # Validate checkpoint paths if provided
    if args.checkpoint_path and not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths:
        for checkpoint_path in args.checkpoint_paths:
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print("Loading configuration...", flush=True)
    sys.stdout.flush()
    # Load configuration
    logger.info("Loading splits config via load_data_splits for dataset: %s", dataset_name)
    cfg = load_data_splits(dataset_name=dataset_name)
    print("Configuration loaded", flush=True)
    sys.stdout.flush()

    # Set up output directory
    output_dir = PROJECT_DIR / "unimodal_embeddings" / dataset_name / "gex"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Embeddings will be saved to: %s", output_dir)
    print(f"Output directory: {output_dir}", flush=True)
    sys.stdout.flush()

    if args.batch:
        # Batch mode
        process_batch_configurations(cfg, args, dataset_name=dataset_name)
    else:
        # Single checkpoint mode - just process the full dataset
        if not args.checkpoint_path:
            raise ValueError("Single mode requires --checkpoint_path")

        print("Starting process_single_configuration...", flush=True)
        sys.stdout.flush()
        process_single_configuration(
            cfg=cfg,
            model_scale=args.model_scale,  # Use model_scale for unique filenames
            dataset_name=dataset_name,
            data_scale=None,    # Not used for full dataset
            split=None,         # Not used for full dataset
            batch_size=args.batch_size,
            force=args.force,
            checkpoint_path=args.checkpoint_path
        )

    logger.info("Done!")
    print("Script completed successfully", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
