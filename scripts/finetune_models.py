#!/usr/bin/env python
"""
Script for fine-tuning models on different data subsets.
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

from src.dataset.loader import load_paired_data, create_dataloaders
from src.finetuning.continued_training import finetune_clip, finetune_nicheformer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune models on data subsets")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to Hydra config file"
    )
    parser.add_argument(
        "--model_type", type=str, required=True, 
        choices=["nicheformer", "clip"], 
        help="Type of model to fine-tune"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--subset", type=str, default="finetune",
        help="Dataset subset for fine-tuning"
    )
    parser.add_argument(
        "--freeze_vision", action="store_true",
        help="Freeze vision encoder (only for CLIP)"
    )
    return parser.parse_args()


def filter_dataset_by_names(dataset, names_list):
    """
    Filter a dataset to include only samples with names in the provided list.
    
    Args:
        dataset: Dataset to filter
        names_list: List of sample names to include
    
    Returns:
        Filtered dataset
    """
    # Check if dataset has a filter method
    if hasattr(dataset, "filter"):
        return dataset.filter(lambda x: x["name"] in names_list)
    
    # Otherwise filter manually
    filtered_indices = [i for i, sample in enumerate(dataset) if sample["name"] in names_list]
    from torch.utils.data import Subset
    return Subset(dataset, filtered_indices)


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
    
    # Filter dataset for fine-tuning if subset is specified
    if args.subset != "finetune":
        # Here we'd implement logic to filter the dataset
        # to only include a subset of samples for fine-tuning
        ratio = float(args.subset.replace("pct", "")) / 100.0
        
        # Get unique sample names
        all_names = [sample["name"] for sample in datasets["train"]]
        unique_names = list(set(all_names))
        
        # Take a subset of the unique names
        import random
        random.seed(cfg.experiment.seed)
        random.shuffle(unique_names)
        subset_names = unique_names[:int(len(unique_names) * ratio)]
        
        # Filter datasets
        datasets["train"] = filter_dataset_by_names(datasets["train"], subset_names)
        
        print(f"Fine-tuning on {args.subset} subset: {len(datasets['train'])} samples")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        datasets=datasets,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    
    # Create a unique name for this fine-tuning run
    model_name = Path(args.checkpoint).stem
    run_name = f"ft_{model_name}_{args.subset}"
    if args.freeze_vision and args.model_type == "clip":
        run_name += "_frozen_vision"
    
    # Fine-tune model
    if args.model_type == "clip":
        print(f"Fine-tuning CLIP model: {args.checkpoint}")
        checkpoint_path = finetune_clip(
            clip_checkpoint=args.checkpoint,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            learning_rate=cfg.training.learning_rate * 0.1,  # Lower learning rate for fine-tuning
            weight_decay=cfg.training.weight_decay,
            max_epochs=int(cfg.training.max_epochs * 0.5),  # Fewer epochs for fine-tuning
            freeze_vision=args.freeze_vision,
            cache_dir=cfg.experiment.cache_dir,
            wandb_project=f"{cfg.wandb.project}-finetune",
            wandb_entity=cfg.wandb.entity,
            wandb_group=args.model_type,
            wandb_name=run_name,
        )
    elif args.model_type == "nicheformer":
        print(f"Fine-tuning Nicheformer model: {args.checkpoint}")
        checkpoint_path = finetune_nicheformer(
            nicheformer_checkpoint=args.checkpoint,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            learning_rate=cfg.training.learning_rate * 0.1,
            weight_decay=cfg.training.weight_decay,
            max_epochs=int(cfg.training.max_epochs * 0.5),
            wandb_project=f"{cfg.wandb.project}-finetune",
            wandb_entity=cfg.wandb.entity,
            wandb_group=args.model_type,
            wandb_name=run_name,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Fine-tuning complete. Best checkpoint saved at: {checkpoint_path}")


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