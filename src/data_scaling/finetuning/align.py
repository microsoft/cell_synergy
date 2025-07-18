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
        if self.split not in ['pretrain', 'finetune', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Must be one of: pretrain, finetune, test")
        
        # Get the appropriate scale based on the split
        if self.split == 'pretrain':
            if not hasattr(cfg.data, 'pretrain_split'):
                raise ValueError("Config must specify data.pretrain_split when split is 'pretrain'")
            self.scale = cfg.data.pretrain_split  # This will be S, M, or L
        elif self.split == 'finetune':
            if not hasattr(cfg.data, 'finetune_split'):
                raise ValueError("Config must specify data.finetune_split when split is 'finetune'")
            self.scale = cfg.data.finetune_split  # This will be S, M, or L
        else:  # test split
            self.scale = 'test'  # Test split doesn't have a scale
        
        # Get model choices from config
        self.gex_model = cfg.data.gex_pretrain_choice
        self.img_model = cfg.data.img_pretrain_choice
        
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
            if self.cfg.data.dataset == "lung":
                dataset_path = PROJECT_DIR / "lung_hf"
            else:
                dataset_path = PROJECT_DIR / f"{self.cfg.data.dataset}_hf"
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
            self.indices = []

            # Check if dataset has split/scale columns (e.g. alignment case)
            has_split_columns = 'split' in self.dataset.column_names and 'scale' in self.dataset.column_names

            print(f"🧾 Dataset columns: {self.dataset.column_names}")
            
            if has_split_columns:
                print(f"🔧 Has split/scale columns: {has_split_columns}")
                if self.split == 'test':
                    for i, split_val in enumerate(tqdm(self.dataset['split'])):
                        if split_val == 'test':
                            self.indices.append(i)
                else:
                    for i, (split_val, scale_val) in enumerate(tqdm(zip(self.dataset['split'], self.dataset['scale']))):
                        if split_val == self.split and scale_val == self.scale:
                            self.indices.append(i)
            else:
                # 🧪 Fallback for datasets without split/scale columns (e.g. combined HF dataset)
                print("⚠️  Falling back to filtering based on name matching")
                
                donor_list = self.cfg.data.multimodal.test

                for i, name in enumerate(tqdm(self.dataset['name'])):
                    # Use exact name match rather than partial donor ID (more robust)
                    if name in donor_list:
                        self.indices.append(i)

            if not self.indices:
                raise ValueError(
                    f"No samples found for combination:\n"
                    f"  Split: {self.split}\n"
                    f"  Scale: {self.scale}"
                )

            # Optional: train/val split
            if self.val_fraction > 0:
                np.random.seed(self.random_seed)
                total_samples = len(self.indices)
                val_size = int(total_samples * self.val_fraction)
                shuffled_indices = np.array(self.indices.copy())
                np.random.shuffle(shuffled_indices)

                if self.val_split:
                    self.indices = shuffled_indices[:val_size].tolist()
                    split_type = "validation"
                else:
                    self.indices = shuffled_indices[val_size:].tolist()
                    split_type = "training"
                print(f"📊 Created {split_type} split with {len(self.indices)} samples")

            # Stats and dimension validation
            print(f"\n📈 Dataset statistics:")
            print(f"  ✅ Found {len(self.indices)} total samples")
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
        
        return img_embed, gex_embed


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

    def training_step(self, batch, batch_idx):
        img_embed, gex_embed = batch
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
        
        # For adversarial training, ensure model stays in training mode for stability
        if hasattr(self.model, 'compute_loss') and 'adversarial' in str(type(self.model)).lower():
            # Keep the model in training mode for adversarial networks during validation
            # This prevents BatchNorm and other issues that cause instability in eval mode
            self.model.train()
            
            with torch.no_grad():  # Still disable gradients for validation
                loss = self.model.compute_loss(img_embed, gex_embed)
        else:
            # Standard validation for other models
            if hasattr(self.model, 'compute_loss'):
                loss = self.model.compute_loss(img_embed, gex_embed)
            else:
                loss = self.model(img_embed, gex_embed)
        
        # Add validation loss clipping for numerical stability
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e6:
            print(f"Warning: Invalid validation loss {loss} at batch {batch_idx}, skipping...")
            return None
        
        # Only log per epoch for validation
        self.log('val_loss', loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True, 
                logger=True,
                sync_dist=True)
        
        return loss

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

