"""
Dataset loading utilities for the data scaling experiments.
"""
from typing import Dict, List, Optional, Tuple, Union
import os

import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def load_paired_data(
    dataset_name: str,
    cache_dir: Optional[str] = None,
    split_by_name: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dataset]:
    """
    Load the paired multimodal dataset and split it into train/val/test.
    
    Args:
        dataset_name: HuggingFace dataset name or path
        cache_dir: Directory to cache dataset
        split_by_name: Whether to split by sample name (rather than randomly)
        train_ratio: Ratio of data for training (if random split)
        val_ratio: Ratio of data for validation (if random split)
        test_ratio: Ratio of data for testing (if random split)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing train, val, and test datasets
    """
    # Set HF_DATASETS_TOKEN if available
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    
    # Load dataset
    dataset = load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        use_auth_token=hf_token,
    )
    
    # Check if we have a standard split or need to create one
    if "train" in dataset and "validation" in dataset and "test" in dataset:
        return {
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        }
    
    # Use the first (and likely only) split
    dataset = dataset["train"] if isinstance(dataset, dict) else dataset
    
    # Split data by name if requested
    if split_by_name:
        assert "name" in dataset.features, "Dataset must contain 'name' feature for name-based splitting"
        
        # Get unique names
        names = np.array(dataset["name"])
        unique_names = np.unique(names)
        np.random.seed(seed)
        np.random.shuffle(unique_names)
        
        # Split names
        num_samples = len(unique_names)
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        
        train_names = unique_names[:train_size]
        val_names = unique_names[train_size:train_size+val_size]
        test_names = unique_names[train_size+val_size:]
        
        # Create masks
        train_mask = np.isin(names, train_names)
        val_mask = np.isin(names, val_names)
        test_mask = np.isin(names, test_names)
        
        # Create indices
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]
    else:
        # Random split
        indices = np.arange(len(dataset))
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
    
    # Create subsets
    return {
        "train": Subset(dataset, train_indices),
        "val": Subset(dataset, val_indices),
        "test": Subset(dataset, test_indices),
    }


def create_dataloaders(
    datasets: Dict[str, Dataset],
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from datasets.
    
    Args:
        datasets: Dictionary of datasets (train, val, test)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return dataloaders 