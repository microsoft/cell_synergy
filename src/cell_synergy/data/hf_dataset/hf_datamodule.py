"""
Simple HuggingFace Dataset PyTorch Lightning DataModule for Nicheformer.
Based on the multimodal-ssl project workflow.
"""
import os
import logging
import torch
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from datasets import load_dataset
from huggingface_hub import login

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    Custom collate function for Nicheformer that handles variable-length sequences.
    Ensures tensors are properly shaped and padded.
    """
    if not batch:
        return {}

    # Extract sequences and masks
    sequences = [item['X'] for item in batch]
    masks = [item['attention_mask'] for item in batch]

    # Find max sequence length
    max_len = max(len(seq) for seq in sequences)

    # Pad sequences and masks
    padded_sequences = []
    padded_masks = []

    for seq, mask in zip(sequences, masks):
        seq_len = len(seq)
        if seq_len < max_len:
            # Pad with zeros (assuming 0 is a valid padding token)
            padding = torch.zeros(max_len - seq_len, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding])

            # Pad mask with False
            mask_padding = torch.zeros(max_len - seq_len, dtype=torch.bool)
            padded_mask = torch.cat([mask, mask_padding])
        else:
            padded_seq = seq
            padded_mask = mask

        padded_sequences.append(padded_seq)
        padded_masks.append(padded_mask)

    # Stack into tensors
    X = torch.stack(padded_sequences)  # [batch_size, max_seq_len]
    attention_mask = torch.stack(padded_masks)  # [batch_size, max_seq_len]

    return {
        'X': X,
        'attention_mask': attention_mask
    }


class SimpleHFDataset(Dataset):
    """Simple PyTorch Dataset wrapper for HuggingFace datasets."""

    def __init__(self, hf_dataset, sample_names: Optional[List[str]] = None):
        self.dataset = hf_dataset

        # Filter by sample names if provided
        if sample_names is not None:
            def filter_fn(example):
                return example.get('name', '') in sample_names or example.get('donor_id', '') in sample_names

            original_size = len(self.dataset)
            self.dataset = self.dataset.filter(filter_fn)
            logger.info("Filtered dataset from %s to %s samples", original_size, len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Extract gene expression data
        gexp_raw = data['gexp']

        # Debug logging for the first few samples to understand data format
        if not hasattr(self, '_debug_logged'):
            logger.info("Raw gexp data type: %s", type(gexp_raw))
            logger.info("Raw gexp shape: %s", gexp_raw.shape if hasattr(gexp_raw, 'shape') else 'N/A')
            logger.info("Raw gexp dtype: %s", gexp_raw.dtype if hasattr(gexp_raw, 'dtype') else 'N/A')
            if hasattr(gexp_raw, 'shape') and len(gexp_raw.shape) > 0:
                logger.info("Raw gexp value range: min=%s, max=%s", gexp_raw.min(), gexp_raw.max())
            self._debug_logged = True

        # Convert to tensor first
        gexp_tensor = torch.as_tensor(gexp_raw)

        # Handle different data formats
        if gexp_tensor.dim() == 3:
            # If 3D [batch_size, seq_len, gene_dim], we need to tokenize this
            logger.info("Processing 3D tensor with shape: %s", gexp_tensor.shape)
            if gexp_tensor.shape[0] == 1:
                gexp_tensor = gexp_tensor.squeeze(0)  # Remove batch dim -> [seq_len, gene_dim]
            else:
                # Take the first sample from the batch
                gexp_tensor = gexp_tensor[0]  # [seq_len, gene_dim]

            # Convert gene expression values to token indices
            if gexp_tensor.dim() == 2:
                # Method 1: Use argmax to get the most expressed gene per cell
                gexp_tokens = torch.argmax(gexp_tensor, dim=-1).long()  # [seq_len]

                # Alternative Method 2: Bin continuous values into discrete tokens
                # Uncomment if argmax doesn't work well:
                # num_bins = 1000  # Adjust based on your vocabulary size
                # gexp_flat = gexp_tensor.flatten()
                # bins = torch.linspace(gexp_flat.min(), gexp_flat.max(), num_bins)
                # gexp_tokens = torch.bucketize(gexp_tensor.mean(dim=-1), bins).long()

            else:
                gexp_tokens = gexp_tensor.flatten().long()

        elif gexp_tensor.dim() == 2:
            # If 2D [seq_len, gene_dim], convert to token indices
            logger.info("Processing 2D tensor with shape: %s", gexp_tensor.shape)
            gexp_tokens = torch.argmax(gexp_tensor, dim=-1).long()  # [seq_len]

        elif gexp_tensor.dim() == 1:
            # Already token indices, just ensure correct dtype
            logger.info("Processing 1D tensor with shape: %s", gexp_tensor.shape)
            gexp_tokens = gexp_tensor.long()

        else:
            raise ValueError(f"Unexpected gexp tensor shape: {gexp_tensor.shape}")

        # Ensure we have a 1D tensor of token indices
        if gexp_tokens.dim() != 1:
            raise ValueError(f"Expected 1D token indices, got shape: {gexp_tokens.shape}")

        # Validate token range (should be non-negative integers)
        if gexp_tokens.min() < 0:
            logger.warning("Found negative token values: min=%s. Converting to absolute values.", gexp_tokens.min())
            gexp_tokens = torch.abs(gexp_tokens)

        # Extract or create mask
        if 'mask' in data:
            mask = torch.as_tensor(data['mask'], dtype=torch.bool)
            # Ensure mask matches sequence length
            if mask.shape[0] != gexp_tokens.shape[0]:
                logger.warning("Mask length %s doesn't match sequence length %s. Creating new mask.", mask.shape[0], gexp_tokens.shape[0])
                mask = torch.ones(gexp_tokens.shape[0], dtype=torch.bool)
        else:
            mask = torch.ones(gexp_tokens.shape[0], dtype=torch.bool)

        # Debug final shapes
        if not hasattr(self, '_final_debug_logged'):
            logger.info("Final gexp_tokens shape: %s, dtype: %s", gexp_tokens.shape, gexp_tokens.dtype)
            logger.info("Final mask shape: %s, dtype: %s", mask.shape, mask.dtype)
            logger.info("Token value range: min=%s, max=%s", gexp_tokens.min(), gexp_tokens.max())
            logger.info("Sample token values: %s", gexp_tokens[:10] if len(gexp_tokens) > 0 else 'Empty')
            self._final_debug_logged = True

        # Return what Nicheformer expects: X and attention_mask
        return {
            'X': gexp_tokens,
            'attention_mask': mask
        }


class SimpleHFDataModule(pl.LightningDataModule):
    """Simple PyTorch Lightning DataModule for HuggingFace datasets."""

    def __init__(
        self,
        hf_dataset_name: str,
        sample_names: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        super().__init__()
        self.hf_dataset_name = hf_dataset_name
        self.sample_names = sample_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets."""
        if self.dataset is None:
            # Login to HuggingFace if token is available
            token = os.getenv("HF_DATASETS_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            if token:
                try:
                    login(token=token)
                    logger.info("Successfully logged into HuggingFace")
                except Exception as e:
                    logger.warning("Failed to login to HuggingFace: %s", e)

            # Load the dataset
            logger.info("Loading HuggingFace dataset: %s", self.hf_dataset_name)

            # Parse dataset name and revision if present
            dataset_name = self.hf_dataset_name
            load_args = {"split": "train"}

            if "@" in dataset_name:
                dataset_name, revision = dataset_name.split("@", 1)
                load_args["revision"] = revision

            if token:
                load_args["token"] = token

            try:
                self.dataset = load_dataset(dataset_name, **load_args)
                logger.info("Loaded dataset with %s samples", len(self.dataset))
            except Exception as e:
                # Fallback: try without extra parameters
                logger.warning("Failed with full args: %s. Trying simplified loading...", e)
                self.dataset = load_dataset(dataset_name, split="train")
                logger.info("Loaded dataset with simplified approach: %s samples", len(self.dataset))

        if stage == "fit" or stage is None:
            # Split the dataset
            if self.val_ratio > 0:
                split_data = self.dataset.train_test_split(test_size=self.val_ratio, seed=self.seed)
                train_data = split_data['train']
                val_data = split_data['test']
            else:
                train_data = self.dataset
                val_data = self.dataset

            self.train_dataset = SimpleHFDataset(train_data, self.sample_names)
            self.val_dataset = SimpleHFDataset(val_data, self.sample_names)

            logger.info("Training dataset size: %s", len(self.train_dataset))
            logger.info("Validation dataset size: %s", len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )
