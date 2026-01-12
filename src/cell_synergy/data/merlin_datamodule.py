"""
Merlin datamodule for properly tokenized lung data.
Follows the original Nicheformer datamodule structure.
"""

import logging
from pathlib import Path
from typing import Optional
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class TokenizedCellDataset(Dataset):
    """Dataset for tokenized individual cells."""

    def __init__(self, data_dir: Path, split: str = "train"):
        self.data_dir = data_dir
        self.split = split

        # Load all parquet files for this split
        self.data_files = list(data_dir.glob("*.parquet"))
        self.data_files.sort()  # Ensure consistent ordering

        if not self.data_files:
            raise ValueError(f"No parquet files found in {data_dir}")

        logger.info("Found %d data files for %s split", len(self.data_files), split)

        # Load first file to get column structure
        sample_data = pd.read_parquet(self.data_files[0])
        self.columns = list(sample_data.columns)
        logger.info("Data columns: %s", self.columns)

        # Count total cells across all files
        self.total_cells = 0
        self.file_cell_counts = []

        for file_path in self.data_files:
            file_data = pd.read_parquet(file_path)
            cell_count = len(file_data)
            self.file_cell_counts.append(cell_count)
            self.total_cells += cell_count

        logger.info("Total cells in %s split: %s", split, self.total_cells)

        # Create cumulative index for efficient access
        self.cumulative_counts = np.cumsum([0] + self.file_cell_counts[:-1])

    def __len__(self):
        return self.total_cells

    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = np.searchsorted(self.cumulative_counts, idx, side='right') - 1
        file_idx = min(file_idx, len(self.data_files) - 1)

        # Load the file
        file_data = pd.read_parquet(self.data_files[file_idx])

        # Get the cell at the relative index
        relative_idx = idx - self.cumulative_counts[file_idx]
        cell_data = file_data.iloc[relative_idx]

        # Extract token and metadata
        token = torch.tensor(cell_data['token'], dtype=torch.long)
        cell_id = cell_data['cell_id']
        sample_id = cell_data['sample_id']

        return {
            'token': token,
            'cell_id': cell_id,
            'sample_id': sample_id
        }


class MerlinDataModule(pl.LightningDataModule):
    """Merlin datamodule for tokenized lung data."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: str = "val",
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        # Set up datasets
        self.train_dataset = None
        self.val_dataset = None

        # Validate data directory structure
        self._validate_data_dir()

    def _validate_data_dir(self):
        """Validate that the data directory has the expected structure."""
        required_splits = ["train", self.val_split]

        for split in required_splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                raise ValueError(f"Required split directory not found: {split_dir}")

            parquet_files = list(split_dir.glob("*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in split directory: {split_dir}")

            logger.info("Split %s: %s parquet files", split, len(parquet_files))

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets."""
        if stage == "fit" or stage is None:
            train_dir = self.data_dir / "train"
            val_dir = self.data_dir / self.val_split

            self.train_dataset = TokenizedCellDataset(train_dir, "train")
            self.val_dataset = TokenizedCellDataset(val_dir, self.val_split)

            logger.info("Train dataset: %s cells", len(self.train_dataset))
            logger.info("Validation dataset: %s cells", len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Collate function for tokenized cells."""
        # Extract tokens from batch
        tokens = torch.stack([item['token'] for item in batch])
        cell_ids = [item['cell_id'] for item in batch]
        sample_ids = [item['sample_id'] for item in batch]

        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones_like(tokens, dtype=torch.bool)

        return {
            'X': tokens,
            'attention_mask': attention_mask,
            'cell_ids': cell_ids,
            'sample_ids': sample_ids
        }


class SubsetMerlinDataModule(MerlinDataModule):
    """Merlin datamodule for log-scale subsets."""

    def __init__(
        self,
        data_dir: Path,
        subset_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: str = "val",
        seed: int = 42
    ):
        # Set subset_name BEFORE calling parent constructor
        self.subset_name = subset_name

        # Use the subset directory
        subset_data_dir = data_dir / "subsets" / subset_name
        if not subset_data_dir.exists():
            raise ValueError(f"Subset directory not found: {subset_data_dir}")

        super().__init__(
            data_dir=subset_data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            seed=seed
        )

        logger.info("Initialized subset datamodule for: %s", subset_name)

    def _validate_data_dir(self):
        """Override validation for subset directories."""
        # Subset directories should contain parquet files directly
        parquet_files = list(self.data_dir.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in subset directory: {self.data_dir}")

        logger.info("Subset %s: %s parquet files", self.subset_name, len(parquet_files))

    def setup(self, stage: Optional[str] = None):
        """Set up the subset datasets."""
        if stage == "fit" or stage is None:
            # For subsets, we only have training data
            self.train_dataset = TokenizedCellDataset(self.data_dir, self.subset_name)
            self.val_dataset = TokenizedCellDataset(self.data_dir, self.subset_name)  # Use same data for validation

            logger.info("Subset %s - Train dataset: %s cells", self.subset_name, len(self.train_dataset))
            logger.info("Subset %s - Validation dataset: %s cells", self.subset_name, len(self.val_dataset))
