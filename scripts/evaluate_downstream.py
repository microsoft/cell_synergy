#!/usr/bin/env python
"""
Script to evaluate models on downstream tasks using linear probing.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.dataset.loader import load_paired_data
from src.evaluation.linear_probe import train_linear_probe
from src.clip.train import CLIPModel
from src.pretraining.vision import VisionEncoder


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate models on downstream tasks")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to Hydra config file"
    )
    parser.add_argument(
        "--model_type", type=str, required=True, 
        choices=["nicheformer", "vision", "clip"], 
        help="Type of model to evaluate"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--task", type=str, default="classification",
        choices=["classification", "regression"],
        help="Type of downstream task"
    )
    return parser.parse_args()


def extract_embeddings(model_type, checkpoint_path, dataset, cache_dir=None):
    """
    Extract embeddings from a model.
    
    Args:
        model_type: Type of model ('nicheformer', 'vision', or 'clip')
        checkpoint_path: Path to model checkpoint
        dataset: Dataset to extract embeddings from
        cache_dir: Cache directory for models
    
    Returns:
        Dictionary of embeddings for each split
    """
    embeddings = {}
    
    if model_type == "nicheformer":
        # Import here to avoid circular import
        try:
            from models._nicheformer import Nicheformer
        except ImportError:
            raise ImportError(
                "Nicheformer package not found. Please install it with: "
                "pip install git+https://github.com/theislab/nicheformer.git"
            )
        
        # Load Nicheformer model
        model = Nicheformer.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        # Extract embeddings for each split
        for split, data in dataset.items():
            print(f"Extracting Nicheformer embeddings for {split} split...")
            split_embeddings = []
            
            # Process batches
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
                with torch.no_grad():
                    batch_embeddings = model.encode(batch["gexp"])
                split_embeddings.append(batch_embeddings)
            
            # Concatenate embeddings
            embeddings[split] = torch.cat(split_embeddings)
    
    elif model_type == "vision":
        # Load vision model
        vision_model_key = Path(checkpoint_path).name.split("_")[1]
        model = VisionEncoder(model_key=vision_model_key, cache_dir=cache_dir)
        model.eval()
        
        # Extract embeddings for each split
        for split, data in dataset.items():
            print(f"Extracting vision embeddings for {split} split...")
            split_embeddings = []
            
            # Process batches
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
                with torch.no_grad():
                    batch_embeddings = model(batch["image"])
                split_embeddings.append(batch_embeddings)
            
            # Concatenate embeddings
            embeddings[split] = torch.cat(split_embeddings)
    
    elif model_type == "clip":
        # Load CLIP model
        model = CLIPModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        # Extract embeddings for each split
        for split, data in dataset.items():
            print(f"Extracting CLIP embeddings for {split} split...")
            split_embeddings = []
            
            # Process batches
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
                with torch.no_grad():
                    # Get vision and gexp embeddings
                    vision_proj, gexp_proj = model(batch["image"], batch["gexp"])
                    # Concatenate embeddings
                    batch_embeddings = torch.cat([vision_proj, gexp_proj], dim=1)
                split_embeddings.append(batch_embeddings)
            
            # Concatenate embeddings
            embeddings[split] = torch.cat(split_embeddings)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return embeddings


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
    
    # Extract embeddings
    embeddings = extract_embeddings(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        dataset=datasets,
        cache_dir=cfg.experiment.cache_dir,
    )
    
    # Create model name for logging
    model_name = Path(args.checkpoint).stem
    
    # Get labels
    if args.task == "classification":
        # Use niche type as labels
        labels = {}
        for split, data in datasets.items():
            labels[split] = torch.tensor([
                data[i]["annotation"]["niche_type"] for i in range(len(data))
            ], dtype=torch.long)
    else:
        # Use cell composition as labels (assuming it's available)
        labels = {}
        for split, data in datasets.items():
            labels[split] = torch.tensor([
                data[i]["annotation"]["cell_composition"] for i in range(len(data))
            ], dtype=torch.float32)
    
    # Train linear probe
    print(f"Training linear probe for {args.task} task...")
    metrics = train_linear_probe(
        train_embeddings=embeddings["train"],
        train_labels=labels["train"],
        val_embeddings=embeddings["val"],
        val_labels=labels["val"],
        test_embeddings=embeddings["test"],
        test_labels=labels["test"],
        task_type=args.task,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.early_stopping_patience,
        embedding_source=f"{args.model_type}_{model_name}",
        wandb_project=f"{cfg.wandb.project}-evaluation",
        wandb_entity=cfg.wandb.entity,
        wandb_group=args.model_type,
        wandb_name=f"{model_name}_{args.task}",
    )
    
    # Print metrics
    print(f"Evaluation metrics for {args.model_type} model {model_name} on {args.task} task:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


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