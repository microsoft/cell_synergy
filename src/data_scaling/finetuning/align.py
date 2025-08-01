import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, load_dataset
import pytorch_lightning as pl
import os
import tempfile
from pathlib import Path
from torch.utils.data import IterableDataset
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm


def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/Projects/till_richter/")
    resolved = os.path.expandvars(raw)
    return Path(resolved)
PROJECT_DIR = get_project_dir()


class ProcessedHFDataset(Dataset):
    """Dataset for aligned image-GEX pairs with minimal memory footprint."""
    
    def __init__(self, cfg, val_split=False, val_fraction=0.2, random_seed=42):
        """
        Initialize dataset with configuration.
        Uses dataset config for splits, models, and paths.
        
        Args:
            cfg: Configuration object
            val_split: If True, create validation split. If False, create training split.
            val_fraction: Fraction of data to use for validation (0.0-1.0)
            random_seed: Random seed for reproducible train/val splits
        """
        self.cfg = cfg
        self.val_split = val_split
        self.val_fraction = val_fraction
        self.random_seed = random_seed
        
        # Get dataset name and validate
        if not hasattr(cfg.data, 'dataset'):
            raise ValueError("Config must specify data.dataset")
        self.dataset = cfg.data.dataset
        
        # Get split and scale
        if not hasattr(cfg.data, 'split'):
            raise ValueError("Config must specify data.split")
        self.split = cfg.data.split
        
        # Validate split
        if self.split not in ['train', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be one of: train, test")
        
        # Get the appropriate scale based on the split
        if self.split == 'train':
            if not hasattr(cfg.data, 'train_split'):
                raise ValueError("Config must specify data.train_split when split is 'train'")
            self.scale = cfg.data.train_split  # This will be S, M, or L
        else:  # test split
            self.scale = 'test'  # Test split doesn't have a scale
        
        # Get model choices from config
        self.gex_model = cfg.data.gex_train_choice
        self.img_model = cfg.data.img_train_choice
        
        print(f"Initializing {self.dataset} dataset with:")
        print(f"Split: {self.split}")
        print(f"Scale: {self.scale}")
        print(f"GEX model: {self.gex_model}")
        print(f"Image model: {self.img_model}")
        
        # Pre-compute and store only what we need
        self._precompute_filtered_data()

    def _precompute_filtered_data(self):
        """Pre-compute and store only the needed embeddings."""
        try:
            # Load from the standardized location in project_folder
            subdir = f"{self.gex_model}_{self.img_model}_{self.split}.{self.scale}"
            dataset_path = PROJECT_DIR / f"{self.dataset}" / "hf_datasets" / subdir

            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {dataset_path}. "
                    f"Check that the dataset has been properly generated."
                )
            
            print("📦 Loading dataset...")
            self.dataset = load_from_disk(str(dataset_path), keep_in_memory=False)
            print(f"🔍 Filtering for samples from {self.split}-{self.scale}...")
            
            # Create indices list for samples matching our split and scale
            print("🧱 Building sample index...")
            print(f"🧾 Dataset columns: {self.dataset.column_names}")
            
            # Simple datapoint-level validation split for better signal
            print("Datapoint-level validation split for stable validation metrics")
            
            total_samples = len(self.dataset)
            print(f"Total samples in dataset: {total_samples}")

            # Create reproducible random indices
            rng = np.random.RandomState(self.random_seed)
            all_indices = np.arange(total_samples)
            rng.shuffle(all_indices)

            # Split indices based on validation fraction
            val_count = int(total_samples * self.val_fraction)
            if self.val_split:
                self.indices = all_indices[:val_count].tolist()
                split_type = "validation"
            else:
                self.indices = all_indices[val_count:].tolist()
                split_type = "training"
            
            print(f"Selected {len(self.indices)} samples for {split_type} split.")

            if not self.indices:
                raise ValueError(f"No samples found for {split_type} split")

            # Stats and dimension validation
            print(f"\n📈 Dataset statistics:")
            print(f"  ✅ Found {len(self.indices)} total samples")
            
            # Check for NaN embeddings in the dataset
            nan_count = 0
            valid_count = 0
            for idx in self.indices[:100]:  # Sample first 100 to estimate
                sample = self.dataset[idx]
                img_embed = torch.tensor(sample['img_uni_pool'], dtype=torch.float32)
                gex_embed = torch.tensor(sample['nicheformer_pool'], dtype=torch.float32)
                if torch.isnan(img_embed).any() or torch.isnan(gex_embed).any():
                    nan_count += 1
                else:
                    valid_count += 1
            
            # Estimate total NaN samples
            if len(self.indices) > 100:
                nan_ratio = nan_count / 100
                estimated_nan = int(nan_ratio * len(self.indices))
                estimated_valid = len(self.indices) - estimated_nan
                print(f"  ⚠️  Estimated {estimated_nan} samples with NaN embeddings (will be filtered)")
                print(f"  ✅ Estimated {estimated_valid} valid samples for training")
            else:
                print(f"  ⚠️  {nan_count} samples with NaN embeddings (will be filtered)")
                print(f"  ✅ {valid_count} valid samples for training")
            
            sample = self.dataset[self.indices[0]]
            self.img_dim = len(sample['img_uni_pool'])
            self.gex_dim = len(sample['nicheformer_pool'])
            print(f"  🧬 Image embedding dimension: {self.img_dim}")
            print(f"  🔬 GEX embedding dimension: {self.gex_dim}")

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise


    def __len__(self):
        """Return total number of samples."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Get item by index."""
        if idx >= len(self.indices):
            raise IndexError("Index out of bounds")
            
        # Get embeddings directly from dataset
        dataset_idx = self.indices[idx]
        item = self.dataset[dataset_idx]
        
        # Ensure embeddings have correct dimensions
        img_embed = torch.tensor(item['img_uni_pool'], dtype=torch.float32)
        gex_embed = torch.tensor(item['nicheformer_pool'], dtype=torch.float32)
        
        if img_embed.shape[0] != self.img_dim or gex_embed.shape[0] != self.gex_dim:
            raise ValueError(
                f"Embedding dimension mismatch at index {idx}:\n"
                f"Expected img_dim={self.img_dim}, got {img_embed.shape[0]}\n"
                f"Expected gex_dim={self.gex_dim}, got {gex_embed.shape[0]}"
            )
        
        # Filter out samples with NaN embeddings
        if torch.isnan(img_embed).any() or torch.isnan(gex_embed).any():
            # Return None to indicate this sample should be skipped
            return None
        
        return img_embed, gex_embed


def nan_filtering_collate(batch):
    """Custom collate function that filters out None values (NaN embeddings)."""
    # Filter out None values
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        # If all items are None, return empty tensors
        return torch.empty(0, 1024), torch.empty(0, 512)  # Default dimensions
    
    # Stack valid items
    img_embeds = torch.stack([item[0] for item in valid_batch])
    gex_embeds = torch.stack([item[1] for item in valid_batch])
    
    return img_embeds, gex_embeds

def get_paired_dataloader(cfg, batch_size=None, shuffle=True, num_workers=None, val_split=False, val_fraction=0.2):
    """Get dataloader with memory-efficient settings."""
    dataset = ProcessedHFDataset(cfg, val_split=val_split, val_fraction=val_fraction)
    
    # Calculate memory-safe batch size if not provided
    if batch_size is None:
        # Assuming 40GB RAM limit and rough estimates of embedding sizes
        img_size = 394752  # UNI-200M embedding size
        gex_size = 512   # Nicheformer embedding size
        total_size = (img_size + gex_size) * 4  # 4 bytes per float32
        safe_batch_size = min(
            cfg.training.batch_size,
            int((20 * 1024 * 1024 * 1024) / total_size)  # Use half of RAM
        )
        batch_size = safe_batch_size
    
    # Adaptive num_workers: use config value, or default based on dataset size
    if num_workers is None:
        config_workers = cfg.training.get('num_workers', 8)
        dataset_size = len(dataset)
        
        # For small datasets, use fewer workers to avoid overhead
        if dataset_size < 1000:
            num_workers = min(config_workers, 4)  # Max 4 workers for small datasets
            print(f"Small dataset ({dataset_size} samples), using {num_workers} workers")
        else:
            num_workers = config_workers
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=num_workers > 0,  # Keep workers alive between batches
        collate_fn=nan_filtering_collate,  # Use custom collate function
    )


def pool_patches(x):
    # If shape is [batch, 1, patches, embed_dim], squeeze and mean over patches
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)  # [batch, patches, embed_dim]
    if x.ndim == 3:
        return x.mean(dim=1)  # mean over patches
    return x


class AlignmentTrainer(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = True
        
        # Add moving average for validation loss smoothing
        self.val_losses = []
        self.val_loss_smoothing = config.training.get('val_loss_smoothing', 0.9)

    def training_step(self, batch, batch_idx):
        img_embed, gex_embed = batch
        
        # Handle empty batches (all samples had NaN embeddings)
        if img_embed.size(0) == 0 or gex_embed.size(0) == 0:
            # Return a dummy loss that won't affect training
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log('train_loss_step', dummy_loss, 
                    on_step=True, 
                    on_epoch=False, 
                    prog_bar=False, 
                    logger=True,
                    sync_dist=True)
            self.log('train_loss', dummy_loss, 
                    on_step=False, 
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True,
                    sync_dist=True)
            return dummy_loss
        
        # No need for pooling anymore as data is already flattened
        
        if hasattr(self.model, 'compute_loss'):
            loss = self.model.compute_loss(img_embed, gex_embed)
        else:
            loss = self.model(img_embed, gex_embed)
        
        # Log per step for detailed monitoring, and per epoch for aggregation
        self.log('train_loss_step', loss, 
                on_step=True, 
                on_epoch=False, 
                prog_bar=False, 
                logger=True,
                sync_dist=True)
        
        self.log('train_loss', loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True, 
                logger=True,
                sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - special handling for adversarial training."""
        img_embed, gex_embed = batch

        # Handle empty batches (e.g., all samples had NaN embeddings)
        if img_embed.size(0) == 0 or gex_embed.size(0) == 0:
            print(f"Skipping empty batch at idx {batch_idx}")
            return None

        try:
            # For adversarial training, keep model in train mode
            if hasattr(self.model, 'compute_loss') and 'adversarial' in str(type(self.model)).lower():
                self.model.train()
                with torch.no_grad():
                    loss = self.model.compute_loss(img_embed, gex_embed)
            else:
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(img_embed, gex_embed)
                else:
                    loss = self.model(img_embed, gex_embed)

            # Skip invalid loss values
            if torch.isnan(loss) or torch.isinf(loss) or loss > 1e6:
                print(f"Skipping batch {batch_idx} due to invalid loss: {loss}")
                return None

        except Exception as e:
            print(f"Exception in validation_step at batch {batch_idx}: {e}")
            return None

        # Store loss for epoch-level metrics
        self.val_losses.append(loss.detach().cpu().item())

        # Log per-step and per-epoch loss
        self.log('val_loss', loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)

        self.log('val_loss_raw', loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True)

        return loss


    def on_validation_epoch_end(self):
        """Compute smoothed validation metrics at epoch end."""
        if len(self.val_losses) > 0:
            val_mean = np.mean(self.val_losses)

            # Update EMA of validation loss
            if hasattr(self, '_val_loss_ema'):
                self._val_loss_ema = (self.val_loss_smoothing * self._val_loss_ema +
                                    (1 - self.val_loss_smoothing) * val_mean)
            else:
                self._val_loss_ema = val_mean

            # Log EMA-smoothed loss
            self.log('val_loss_smooth', self._val_loss_ema,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True)

        else:
            print("Warning: No valid validation batches in this epoch.")

        self.val_losses.clear()


    def configure_optimizers(self):

        lr = getattr(self.config.training, 'learning_rate', 1e-3)
        wd = getattr(self.config.training, 'weight_decay', 1e-5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.training.get('max_epochs', 100)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss, not training loss
                "frequency": 1,
                "interval": "epoch"  # Update scheduler every epoch
            }
        }
