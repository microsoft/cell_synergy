#!/usr/bin/env python
"""
Script to train CLIP models with different combinations of Nicheformer and vision models.
"""
import os
import sys
import argparse
from pathlib import Path
import itertools

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.dataset.loader import load_paired_data, create_dataloaders
from src.clip.train import train_clip
from src.paths import NICHEFORMER_SUBSET_PATHS, VISION_MODEL_IDS, get_checkpoint_dir


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train CLIP models with different combinations")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to Hydra config file"
    )
    parser.add_argument(
        "--nf_model", type=str, help="Path to Nicheformer checkpoint (optional)"
    )
    parser.add_argument(
        "--vision_model", type=str, help="Vision model key (optional)"
    )
    parser.add_argument(
        "--subset", type=str, help="Dataset subset for training (optional)"
    )
    parser.add_argument(
        "--matrix", action="store_true", help="Run matrix of all combinations"
    )
    return parser.parse_args()


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function."""
    args = parse_args()
    
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Load data
    print("Loading dataset...")
    datasets = load_paired_data(
        dataset_name=cfg.dataset.name,
        cache_dir=cfg.experiment.cache_dir,
        split_by_name=cfg.dataset.split_by_name,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
        seed=cfg.experiment.seed,
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        datasets=datasets,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    
    # Determine Nicheformer models to use
    if args.nf_model:
        nf_models = [args.nf_model]
    elif args.matrix:
        # Get all available Nicheformer checkpoints
        nf_models = []
        for subset_key in NICHEFORMER_SUBSET_PATHS.keys():
            checkpoint_dir = get_checkpoint_dir("nicheformer", subset_key)
            if os.path.exists(checkpoint_dir):
                # Find the best checkpoint (lowest train_loss)
                checkpoints = list(Path(checkpoint_dir).glob("*.ckpt"))
                if checkpoints:
                    # Sort by loss (assuming filename contains loss)
                    checkpoints.sort(key=lambda x: float(str(x).split("_")[-1].replace(".ckpt", "")))
                    nf_models.append(str(checkpoints[0]))
    else:
        # Default to the full dataset model
        checkpoint_dir = get_checkpoint_dir("nicheformer", "nf_100pct_donor")
        checkpoints = list(Path(checkpoint_dir).glob("*.ckpt"))
        if not checkpoints:
            raise ValueError("No Nicheformer checkpoints found. Please train Nicheformer first.")
        # Sort by loss
        checkpoints.sort(key=lambda x: float(str(x).split("_")[-1].replace(".ckpt", "")))
        nf_models = [str(checkpoints[0])]
    
    # Determine vision models to use
    if args.vision_model:
        vision_models = [args.vision_model]
    elif args.matrix:
        vision_models = list(VISION_MODEL_IDS.keys())
    else:
        # Default to UNI
        vision_models = ["uni"]
    
    # Determine dataset subsets to use
    if args.subset:
        subsets = [args.subset]
    elif args.matrix:
        # Use different sample subsets for training
        subsets = ["full", "50pct", "25pct", "10pct"]
    else:
        # Default to full dataset
        subsets = ["full"]
    
    # Train CLIP models for each combination
    results = {}
    for nf_model, vision_model, subset in itertools.product(nf_models, vision_models, subsets):
        print(f"\nTraining CLIP model with:")
        print(f"  Nicheformer: {nf_model}")
        print(f"  Vision model: {vision_model}")
        print(f"  Dataset subset: {subset}")
        
        # Create a unique name for this combination
        nf_subset = Path(nf_model).parent.name if isinstance(nf_model, str) else "unknown"
        run_name = f"clip_{nf_subset}_{vision_model}_{subset}"
        
        # Train the model
        checkpoint_path = train_clip(
            nicheformer_checkpoint=nf_model,
            vision_model_key=vision_model,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            embedding_dim=cfg.models.clip.embedding_dim,
            projection_dim=cfg.models.clip.projection_dim,
            temperature=cfg.models.clip.temperature,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            max_epochs=cfg.training.max_epochs,
            batch_size=cfg.training.batch_size,
            cache_dir=cfg.experiment.cache_dir,
            wandb_project=f"{cfg.wandb.project}-clip",
            wandb_entity=cfg.wandb.entity,
            wandb_group=f"{nf_subset}_{vision_model}",
            wandb_name=run_name,
        )
        
        # Store result
        results[run_name] = checkpoint_path
        print(f"Best checkpoint saved at: {checkpoint_path}")
    
    # Print summary of all trained models
    print("\nTraining complete. Summary of trained models:")
    for name, path in results.items():
        print(f"  {name}: {path}")


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