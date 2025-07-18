"""
Simple HuggingFace Dataset PyTorch Lightning DataModule for Nicheformer.
This module extracts individual cells from HF dataset patches (removing padding).
"""
import os
import logging
import torch
from typing import List, Optional
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from datasets import load_dataset
from huggingface_hub import login

logger = logging.getLogger(__name__)


class SimpleHFDataModule(pl.LightningDataModule):
    """
    Simple DataModule that extracts individual cells from HF patches.
    """
    
    def __init__(
        self,
        hf_dataset_name: str,
        sample_names: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        seed: int = 42,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.hf_dataset_name = hf_dataset_name
        self.sample_names = sample_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.pad_token_id = pad_token_id
        
        self.all_cells = None
        self.train_dataset = None
        self.val_dataset = None
    
    def _extract_cells_from_patches(self, hf_dataset):
        """
        Extract individual cells from HF dataset patches, removing padding.
        
        Args:
            hf_dataset: HuggingFace dataset with patches
            
        Returns:
            List of cell token tensors
        """
        all_cells = []
        
        logger.info(f"Extracting cells from {len(hf_dataset)} patches...")
        
        for idx, data in enumerate(hf_dataset):
            if idx % 1000 == 0:
                logger.info(f"Processed {idx}/{len(hf_dataset)} patches, extracted {len(all_cells)} cells")
            
            # Get the gene expression data for this patch
            gexp_raw = data['gexp']
            gexp_tensor = torch.as_tensor(gexp_raw)
            
            # Handle different tensor shapes - get to [num_cells, num_genes]
            if gexp_tensor.dim() == 3 and gexp_tensor.shape[0] == 1:
                gexp_tensor = gexp_tensor.squeeze(0)  # [1, 200, genes] -> [200, genes]
            elif gexp_tensor.dim() == 3:
                gexp_tensor = gexp_tensor[0]  # Take first sample if multiple
            
            # Convert to token indices (argmax = most expressed gene per cell)
            cell_tokens = torch.argmax(gexp_tensor, dim=-1).long()  # [200]
            
            # Remove padding (cells with token_id = 0)
            valid_mask = (cell_tokens != self.pad_token_id)
            valid_cells = cell_tokens[valid_mask]
            
            # Add each valid cell individually
            for cell_token in valid_cells:
                all_cells.append(cell_token.unsqueeze(0))  # Make it [1] for consistency
        
        logger.info(f"Extracted {len(all_cells)} total cells from {len(hf_dataset)} patches")
        return all_cells
    
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets by extracting cells from HF data."""
        if self.all_cells is None:
            # Login to HuggingFace if token is available
            token = os.getenv("HF_DATASETS_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            if token:
                try:
                    login(token=token)
                    logger.info("Successfully logged into HuggingFace")
                except Exception as e:
                    logger.warning(f"Failed to login to HuggingFace: {e}")
            
            # Load the dataset
            logger.info(f"Loading HuggingFace dataset: {self.hf_dataset_name}")
            
            dataset_name = self.hf_dataset_name
            load_args = {"split": "train"}
            
            if "@" in dataset_name:
                dataset_name, revision = dataset_name.split("@", 1)
                load_args["revision"] = revision
            
            if token:
                load_args["token"] = token
            
            try:
                hf_dataset = load_dataset(dataset_name, **load_args)
                logger.info(f"Loaded dataset with {len(hf_dataset)} patches")
            except Exception as e:
                logger.warning(f"Failed with full args: {e}. Trying simplified loading...")
                hf_dataset = load_dataset(dataset_name, split="train")
                logger.info(f"Loaded dataset with simplified approach: {len(hf_dataset)} patches")
            
            # Filter by sample names if provided
            if self.sample_names is not None:
                def filter_fn(example):
                    return example.get('name', '') in self.sample_names or example.get('donor_id', '') in self.sample_names
                
                original_size = len(hf_dataset)
                hf_dataset = hf_dataset.filter(filter_fn)
                logger.info(f"Filtered dataset from {original_size} to {len(hf_dataset)} patches")
            
            # Extract all cells
            self.all_cells = self._extract_cells_from_patches(hf_dataset)
        
        if stage == "fit" or stage is None:
            # Create train/val split
            total_cells = len(self.all_cells)
            
            if self.val_ratio > 0:
                val_size = int(total_cells * self.val_ratio)
                train_size = total_cells - val_size
                
                # Random split
                generator = torch.Generator().manual_seed(self.seed)
                indices = torch.randperm(total_cells, generator=generator)
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                train_cells = [self.all_cells[i] for i in train_indices]
                val_cells = [self.all_cells[i] for i in val_indices]
            else:
                train_cells = self.all_cells
                val_cells = self.all_cells
            
            # Create simple TensorDatasets 
            self.train_dataset = TensorDataset(torch.stack(train_cells))
            self.val_dataset = TensorDataset(torch.stack(val_cells))
            
            logger.info(f"Training dataset: {len(self.train_dataset)} cells")
            logger.info(f"Validation dataset: {len(self.val_dataset)} cells")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """
        Simple collate function for individual cells.
        """
        # batch is a list of tuples: [(cell_tensor,), ...]
        cells = torch.stack([item[0] for item in batch])  # [batch_size, 1]
        cells = cells.squeeze(-1)  # [batch_size]
        
        return {
            'X': cells,
            'attention_mask': torch.ones_like(cells, dtype=torch.bool)  # All cells are valid
        }
