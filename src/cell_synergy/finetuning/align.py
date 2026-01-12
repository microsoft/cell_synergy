import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from cell_synergy.paths import PROJECT_DIR


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

        # Get model choices from config
        self.gex_model = cfg.data.gex_train_choice
        self.img_model = cfg.data.img_train_choice

        # Get embedding keys from config with fallbacks
        # Image embedding: prefer img_embed_key from config, fallback to img_uni_pool, then img_embed
        if hasattr(cfg.data, 'img_embed_key') and cfg.data.img_embed_key:
            self.img_embed_key = cfg.data.img_embed_key
        else:
            # Default fallback order: img_uni_pool (for alignment-ready datasets) -> img_embed
            self.img_embed_key = 'img_uni_pool'

        # GEX embedding: prefer gex_embed_key from config, fallback to nicheformer_pool
        if hasattr(cfg.data, 'gex_embed_key') and cfg.data.gex_embed_key:
            self.gex_embed_key = cfg.data.gex_embed_key
        else:
            self.gex_embed_key = 'nicheformer_pool'

        print(f"Initializing {self.dataset} dataset with:")
        print(f"GEX model: {self.gex_model}")
        print(f"Image model: {self.img_model}")
        print(f"Image embedding key: {self.img_embed_key}")
        print(f"GEX embedding key: {self.gex_embed_key}")

        # Pre-compute and store only what we need
        self._precompute_filtered_data()

    def _precompute_filtered_data(self):
        """Pre-compute and store only the needed embeddings."""
        try:
            # Load from the full dataset location (should have img_uni_pool with 1536D embeddings)
            dataset_path = PROJECT_DIR / f"{self.dataset}" / "hf_datasets_full" / f"full_{self.img_model}.L"

            # Fallback to lustre location if not in project folder
            if not dataset_path.exists():
                dataset_path = Path("/lustre/groups/ml01/workspace/till.richter/hf_datasets") / \
                    f"{self.dataset}" / f"full_{self.img_model}.L"

            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at {dataset_path}. "
                    "Check that the full dataset has been properly generated with img_uni_pool (1536D) embeddings."
                )

            print(f"📦 Loading dataset from: {dataset_path}")
            self.dataset = load_from_disk(str(dataset_path), keep_in_memory=False)
            print("Loading full dataset...")
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
            print("\n📈 Dataset statistics:")
            print(f"  Found {len(self.indices)} total samples")

            # Validate that required embedding keys exist in dataset
            sample = self.dataset[0]
            available_keys = list(sample.keys())

            # Check for image embedding key (try img_embed_key, then img_uni_pool, then img_embed)
            img_key_candidates = [self.img_embed_key, 'img_uni_pool', 'img_embed']
            self.img_embed_key = None
            for key in img_key_candidates:
                if key in available_keys:
                    self.img_embed_key = key
                    break

            if self.img_embed_key is None:
                raise KeyError(
                    f"Could not find image embedding key. Tried: {img_key_candidates}. Available keys: {available_keys}")

            # Check for GEX embedding key (try gex_embed_key, then nicheformer_pool)
            gex_key_candidates = [self.gex_embed_key, 'nicheformer_pool']
            self.gex_embed_key = None
            for key in gex_key_candidates:
                if key in available_keys:
                    self.gex_embed_key = key
                    break

            if self.gex_embed_key is None:
                raise KeyError(
                    f"Could not find GEX embedding key. Tried: {gex_key_candidates}. Available keys: {available_keys}")

            print(f"  Using image embedding key: {self.img_embed_key}")
            print(f"  Using GEX embedding key: {self.gex_embed_key}")

            # Check for NaN embeddings in the dataset
            nan_count = 0
            valid_count = 0
            for idx in self.indices[:100]:  # Sample first 100 to estimate
                sample = self.dataset[idx]
                img_embed = torch.tensor(sample[self.img_embed_key], dtype=torch.float32)
                gex_embed = torch.tensor(sample[self.gex_embed_key], dtype=torch.float32)
                if torch.isnan(img_embed).any() or torch.isnan(gex_embed).any():
                    nan_count += 1
                else:
                    valid_count += 1

            # Estimate total NaN samples
            if len(self.indices) > 100:
                nan_ratio = nan_count / 100
                estimated_nan = int(nan_ratio * len(self.indices))
                estimated_valid = len(self.indices) - estimated_nan
                print(f"  Estimated {estimated_nan} samples with NaN embeddings (will be filtered)")
                print(f"  Estimated {estimated_valid} valid samples for training")
            else:
                print(f"  {nan_count} samples with NaN embeddings (will be filtered)")
                print(f"  {valid_count} valid samples for training")

            sample = self.dataset[self.indices[0]]
            self.img_dim = len(sample[self.img_embed_key])
            self.gex_dim = len(sample[self.gex_embed_key])
            print(f"  🧬 Image embedding dimension: {self.img_dim}")
            print(f"  🔬 GEX embedding dimension: {self.gex_dim}")

        except Exception as e:
            print(f"Failed to load dataset: {e}")
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
        img_embed = torch.tensor(item[self.img_embed_key], dtype=torch.float32)
        gex_embed = torch.tensor(item[self.gex_embed_key], dtype=torch.float32)

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
        drop_last=True,  # Ensure all ranks have same number of batches (prevents DDP deadlock)
    )


def get_full_alignment_dataloader(cfg, batch_size=None, shuffle=True, num_workers=None):
    """Get dataloader for full dataset alignment (no train/val split).
    This should only be used for unsupervised alignment methods like CoMM."""

    dataset = ProcessedHFDataset(
        cfg,
        val_split=False,  # Don't split for validation
        val_fraction=0.0  # Use entire dataset
    )

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
            num_workers = min(config_workers, 4)
            print(f"Small dataset ({dataset_size} samples), using {num_workers} workers")
        else:
            num_workers = config_workers

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=nan_filtering_collate,
        drop_last=True,  # Ensure all ranks have same number of batches (prevents DDP deadlock)
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
        # Disable automatic optimization for adversarial models (they need manual backward)
        # For other models, we can use automatic optimization
        is_adversarial = hasattr(model, 'optimizer_G') and hasattr(model, 'optimizer_D')
        self.automatic_optimization = not is_adversarial

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

        # Check if this is an adversarial model
        is_adversarial = hasattr(self.model, 'optimizer_G') and hasattr(self.model, 'optimizer_D')

        if is_adversarial:
            # Adversarial training: alternate between generator and discriminator
            return self._adversarial_training_step(img_embed, gex_embed, batch_idx)
        else:
            # Standard training for other models
            return self._standard_training_step(img_embed, gex_embed, batch_idx)

    def _adversarial_training_step(self, img_embed, gex_embed, batch_idx):
        """
        Custom training step for adversarial models that alternates between
        generator and discriminator updates, EXACTLY like CAMEO.

        Match CAMEO pattern exactly:
        1. Compute generator outputs ONCE
        2. Use them in generator loss (with gradients)
        3. Backward generator loss (graph is freed)
        4. Use SAME generator outputs (detached) in discriminator loss
        5. Backward discriminator loss

        The key: We use the SAME tensor objects, just detached. This works because
        .detach() creates a new tensor that doesn't reference the freed graph.
        """
        # Project embeddings to aligned latent dimension (CAMEO lines 245-246)
        img_embed_proj = self.model.img_proj(img_embed)
        gex_embed_proj = self.model.gex_proj(gex_embed)

        # Compute generator outputs ONCE (CAMEO lines 249-252)
        # These will be used in both generator and discriminator steps
        gex2img_recon, _ = self.model.gex2img(gex_embed_proj)
        gex_latent_recon, _ = self.model.img2gex(gex2img_recon)
        img2gex_recon, _ = self.model.img2gex(img_embed_proj)
        img_latent_recon, _ = self.model.gex2img(img2gex_recon)

        # Cycle consistency loss (CAMEO lines 255-259)
        img_cycle_loss = F.l1_loss(img_embed_proj, img_latent_recon)
        gex_cycle_loss = F.l1_loss(gex_embed_proj, gex_latent_recon)
        recon_loss = (img_cycle_loss + gex_cycle_loss) * self.model.config.get('lambda1', 1.0)

        # Latent alignment loss (CAMEO lines 272-276) - treat all as paired for alignment training
        img_latent = self.model.img2gex.encoder(img_embed_proj)
        gex_latent = self.model.gex2img.encoder(gex_embed_proj)
        latent_alignment_loss = F.mse_loss(img_latent, gex_latent)

        # Adversarial training logic (CAMEO lines 279-285)
        img_batch_size = img_embed.size(0)
        gex_batch_size = gex_embed.size(0)
        img_real = torch.ones(img_batch_size, 1, device=img_embed.device)
        img_fake = torch.zeros(img_batch_size, 1, device=img_embed.device)
        gex_real = torch.ones(gex_batch_size, 1, device=gex_embed.device)
        gex_fake = torch.zeros(gex_batch_size, 1, device=gex_embed.device)

        # Train Generator (CAMEO lines 288-307)
        self.model.optimizer_G.zero_grad()
        if self.model.config.get('gan_type', 'wasserstein') == 'wasserstein':
            d_loss = (
                -self.model.D_img(gex2img_recon).mean()
                - self.model.D_gex(img2gex_recon).mean()
            )
        else:
            d_loss = (
                F.binary_cross_entropy_with_logits(self.model.D_img(gex2img_recon), gex_real)
                + F.binary_cross_entropy_with_logits(self.model.D_gex(img2gex_recon), img_real)
            )

        G_loss = (
            recon_loss
            + self.model.config.get('lambda3', 0.5) * d_loss
            + self.model.config.get('lambda_align', 0.1) * latent_alignment_loss
        )
        G_loss.backward()  # CAMEO line 306 - no retain_graph
        self.model.optimizer_G.step()

        # Train Discriminator (CAMEO lines 310-341)
        # Use the same gex2img_recon and img2gex_recon tensors, just detached
        # This matches CAMEO exactly - they use .detach() on the same tensor objects
        # The graph is freed after generator backward, but .detach() creates a new tensor
        # that doesn't reference the freed graph, so this is safe.
        self.model.optimizer_D.zero_grad()
        if self.model.config.get('gan_type', 'wasserstein') == 'wasserstein':
            img_D_loss = (
                self.model.D_img(gex2img_recon.detach()).mean()
                - self.model.D_img(img_embed_proj.detach()).mean()
            )
            gex_D_loss = (
                self.model.D_gex(img2gex_recon.detach()).mean()
                - self.model.D_gex(gex_embed_proj.detach()).mean()
            )
            D_loss = (img_D_loss + gex_D_loss) * self.model.config.get('lambda3', 0.5)
        else:
            img_D_loss = (
                F.binary_cross_entropy_with_logits(
                    self.model.D_img(img_embed_proj.detach()), img_real
                )
                + F.binary_cross_entropy_with_logits(
                    self.model.D_img(gex2img_recon.detach()), img_fake
                )
            ) / 2
            gex_D_loss = (
                F.binary_cross_entropy_with_logits(
                    self.model.D_gex(gex_embed_proj.detach()), gex_real
                )
                + F.binary_cross_entropy_with_logits(
                    self.model.D_gex(img2gex_recon.detach()), gex_fake
                )
            ) / 2
            D_loss = (img_D_loss + gex_D_loss) * self.model.config.get('lambda3', 0.5)

        D_loss.backward()  # CAMEO line 340
        self.model.optimizer_D.step()

        # Apply weight clipping for Wasserstein GAN (CAMEO lines 344-354)
        self.model.apply_weight_clipping()

        # For logging, compute loss components with no_grad to avoid graph issues
        # Use the same tensors we already computed (detached for safety)
        with torch.no_grad():
            loss_components = {
                'G_loss': G_loss.detach(),
                'D_loss': D_loss.detach(),
                'recon_loss': recon_loss.detach(),
                'img_cycle_loss': img_cycle_loss.detach(),
                'gex_cycle_loss': gex_cycle_loss.detach(),
                'latent_alignment_loss': latent_alignment_loss.detach(),
                'img_D_loss': img_D_loss.detach(),
                'gex_D_loss': gex_D_loss.detach()
            }

        # Log all metrics (use detached values for safety)
        self.log('train_loss_step', loss_components['G_loss'],
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 logger=True,
                 sync_dist=True)

        self.log('train_loss', loss_components['G_loss'],
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)

        # Log adversarial-specific metrics
        self.log(
            'G_loss',
            loss_components['G_loss'],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True)
        self.log(
            'D_loss',
            loss_components['D_loss'],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True)
        self.log(
            'recon_loss',
            loss_components['recon_loss'],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True)
        self.log(
            'img_cycle_loss',
            loss_components['img_cycle_loss'],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True)
        self.log(
            'gex_cycle_loss',
            loss_components['gex_cycle_loss'],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True)
        self.log(
            'latent_alignment_loss',
            loss_components['latent_alignment_loss'],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True)

        return G_loss

    def _standard_training_step(self, img_embed, gex_embed, batch_idx):
        """
        Standard training step for non-adversarial models.
        """
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

        # Store training loss for aggregation
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
        self.train_losses.append(loss.detach().cpu().item())

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - special handling for adversarial training."""
        img_embed, gex_embed = batch

        # Handle empty batches (all samples had NaN embeddings)
        if img_embed.size(0) == 0 or gex_embed.size(0) == 0:
            # Return None to skip this validation step
            return None

        # For adversarial training, ensure model stays in training mode for stability
        is_adversarial = hasattr(self.model, 'optimizer_G') and hasattr(self.model, 'optimizer_D')

        if is_adversarial:
            # Keep the model in training mode for adversarial networks during validation
            # This prevents BatchNorm and other issues that cause instability in eval mode
            self.model.train()

            with torch.no_grad():  # Still disable gradients for validation
                # For adversarial models, use generator loss as validation metric
                loss = self.model.compute_generator_loss(img_embed, gex_embed)
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

        # Store raw validation loss for smoothing
        self.val_losses.append(loss.detach().cpu().item())

        # Only log per epoch for validation
        self.log('val_loss', loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True)

        # Also log raw validation loss per step
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
            # Compute exponential moving average for smooth validation loss
            current_mean = np.mean(self.val_losses)

            if hasattr(self, '_val_loss_ema'):
                self._val_loss_ema = (self.val_loss_smoothing * self._val_loss_ema
                                      + (1 - self.val_loss_smoothing) * current_mean)
            else:
                self._val_loss_ema = current_mean

            # Log smoothed validation loss
            self.log('val_loss_smooth', self._val_loss_ema,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True,
                     sync_dist=True)

            # Also log the current epoch mean for comparison
            self.log('val_loss_epoch_mean', current_mean,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True,
                     sync_dist=True)

            # Clear losses for next epoch
            self.val_losses.clear()

    def on_train_epoch_end(self):
        """Compute training metrics at epoch end."""
        if hasattr(self, 'train_losses') and len(self.train_losses) > 0:
            # Log the actual epoch mean for comparison with PyTorch Lightning's aggregation
            train_epoch_mean = np.mean(self.train_losses)
            self.log('train_loss_epoch_mean', train_epoch_mean,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,
                     logger=True,
                     sync_dist=True)

            # Clear training losses for next epoch
            self.train_losses.clear()

    def configure_optimizers(self):
        # Check if this is an adversarial model with its own optimizers
        if hasattr(self.model, 'optimizer_G') and hasattr(self.model, 'optimizer_D'):
            # For adversarial models, return the model's own optimizers
            # PyTorch Lightning will use these optimizers but won't manage them
            # We handle the optimization manually in the training step
            return {
                "optimizer": self.model.optimizer_G,  # Use generator optimizer as primary
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.model.optimizer_G,
                        T_max=self.config.training.get('max_epochs', 100)
                    ),
                    "monitor": "val_loss_smooth",
                    "frequency": 1,
                    "interval": "epoch"
                }
            }
        else:
            # Standard configuration for non-adversarial models
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
                    "monitor": "val_loss_smooth",  # Monitor smoothed validation loss for better stability
                    "frequency": 1,
                    "interval": "epoch"  # Update scheduler every epoch
                }
            }
