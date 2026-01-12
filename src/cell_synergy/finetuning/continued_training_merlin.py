#!/usr/bin/env python3
"""
Continued training script for Nicheformer using the new Merlin dataset.
This script is specifically designed to work with the properly tokenized lung data.
"""

import os
import random
import string
import logging
import numpy as np
from torch import optim
import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from cell_synergy.pretraining.adapted_nicheformer import AdaptedNicheformer
from cell_synergy.data.merlin_datamodule import SubsetMerlinDataModule
from cell_synergy.paths import PROJECT_DIR


class LearningRateMonitorCallback(Callback):
    """Custom callback to monitor learning rate changes."""

    def on_train_start(self, trainer, pl_module):
        """Log learning rate at the start of training."""
        logger.info("=== TRAINING STARTED ===")
        logger.info("Model hparams:")
        logger.info("  - lr: %s", getattr(pl_module.hparams, 'lr', 'NOT SET'))
        logger.info("  - weight_decay: %s", getattr(pl_module.hparams, 'weight_decay', 'NOT SET'))

        if hasattr(pl_module, 'trainer') and pl_module.trainer is not None:
            optimizer = pl_module.trainer.optimizers[0] if pl_module.trainer.optimizers else None
            if optimizer:
                current_lr = optimizer.param_groups[0]['lr']
                current_wd = optimizer.param_groups[0]['weight_decay']
                logger.info("Optimizer parameters:")
                logger.info("  - LR: %s", f'{current_lr:.2e}')
                logger.info("  - WD: %s", f'{current_wd:.2e}')

    def on_train_epoch_start(self, trainer, pl_module):
        """Log learning rate at the start of each epoch."""
        if hasattr(pl_module, 'trainer') and pl_module.trainer is not None:
            optimizer = pl_module.trainer.optimizers[0] if pl_module.trainer.optimizers else None
            if optimizer:
                current_lr = optimizer.param_groups[0]['lr']
                current_wd = optimizer.param_groups[0]['weight_decay']
                logger.info("Epoch %s: LR = %s, WD = %s", trainer.current_epoch, f'{current_lr:.2e}', f'{current_wd:.2e}')
                # Log to wandb if available
                if hasattr(pl_module, 'log'):
                    pl_module.log('epoch_lr', current_lr, on_step=False, on_epoch=True)
                    pl_module.log('epoch_wd', current_wd, on_step=False, on_epoch=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Log learning rate at the start of each batch (but only occasionally to avoid spam)."""
        if batch_idx % 500 == 0:  # Log every 500 batches to reduce spam
            if hasattr(pl_module, 'trainer') and pl_module.trainer is not None:
                optimizer = pl_module.trainer.optimizers[0] if pl_module.trainer.optimizers else None
                if optimizer:
                    current_lr = optimizer.param_groups[0]['lr']
                    current_wd = optimizer.param_groups[0]['weight_decay']
                    logger.info("Batch %s: LR = %s, WD = %s", batch_idx, f'{current_lr:.2e}', f'{current_wd:.2e}')


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with cosine annealing and warmup."""

    def __init__(self, optimizer: optim.Optimizer, warmup: int, max_epochs: int):
        self.warmup = warmup
        self.max_num_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get learning rates for all parameter groups."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        new_lrs = [max(1e-5, base_lr * lr_factor) for base_lr in self.base_lrs]

        # Log learning rate changes (but only occasionally to avoid spam)
        if hasattr(self, '_last_logged_epoch') and self._last_logged_epoch != self.last_epoch:
            if self.last_epoch % 5 == 0:  # Log every 5 epochs
                logger.info(
                    f"Epoch {self.last_epoch}: LR factor = {f'{lr_factor:.6f}'}, \
                    Base LRs = {self.base_lrs}, \
                    New LRs = {[f'{lr:.2e}' for lr in new_lrs]}")
            self._last_logged_epoch = self.last_epoch
        elif not hasattr(self, '_last_logged_epoch'):
            self._last_logged_epoch = self.last_epoch
            logger.info("Initial LR: Base LRs = %s, Current LRs = %s' for lr in new_lrs]}", self.base_lrs, [f'{lr:.2e}' for lr in new_lrs])

        return new_lrs

    def get_lr_factor(self, epoch: int) -> float:
        """Calculate learning rate factor based on epoch."""
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs))
        if self.warmup > 0 and epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup

        # Log the LR factor calculation (but only occasionally to avoid spam)
        if epoch % 5 == 0:  # Log every 5 epochs
            warmup_factor = epoch * 1.0 / self.warmup if self.warmup > 0 and epoch <= self.warmup else 1.0
            logger.info(
                f"LR Factor calculation for epoch {epoch}: \
                base_factor={f'{0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs)):.6f}'}, \
                warmup_factor={f'{warmup_factor:.6f}'}, \
                final_factor={f'{lr_factor:.6f}'}")

        return lr_factor


# Get logger for this module
logger = logging.getLogger(__name__)


class MerlinNicheformer(AdaptedNicheformer):
    """Nicheformer model specifically designed for Merlin dataset training."""

    def __init__(self, *args, **kwargs):
        # Store parameters that the parent class doesn't expect but we need
        if 'weight_decay' in kwargs:
            self._config_weight_decay = kwargs['weight_decay']
            logger.info("Stored config weight decay: %s", self._config_weight_decay)
        else:
            self._config_weight_decay = 0.0001  # Default weight decay

        # Filter out parameters that the parent class doesn't expect
        unexpected_params = ['autoregressive', 'pool', 'karpathy', 'weight_decay']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unexpected_params}

        # Log which parameters were filtered out
        for param in unexpected_params:
            if param in kwargs:
                logger.info("Filtered out %s parameter: %s", param, kwargs[param])

        # Log all remaining parameters
        logger.info("Remaining parameters for parent constructor: %s", list(filtered_kwargs.keys()))

        # Call parent constructor with filtered kwargs
        super().__init__(*args, **filtered_kwargs)

        # Ensure the learning rate is properly set in hparams
        if hasattr(self, 'hparams'):
            if 'lr' not in self.hparams or self.hparams.lr is None:
                # Use lr parameter if available
                if 'lr' in kwargs:
                    self.hparams.lr = kwargs['lr']
                    logger.info("Set hparams.lr to %s", self.hparams.lr)

            # Also ensure weight_decay is set in hparams for consistency
            if 'weight_decay' not in self.hparams or self.hparams.weight_decay is None:
                if 'weight_decay' in kwargs:
                    self.hparams.weight_decay = kwargs['weight_decay']
                    logger.info("Set hparams.weight_decay to %s", self.hparams.weight_decay)

        # Log the final hparams after parent constructor
        logger.info("Final hparams after parent constructor:")
        logger.info("  - lr: %s", getattr(self.hparams, 'lr', 'NOT SET'))
        logger.info("  - weight_decay: %s", getattr(self.hparams, 'weight_decay', 'NOT SET'))

    def training_step(self, batch, batch_idx):
        """Training step for individual cell token prediction."""
        x = batch['X']
        attention_mask = batch['attention_mask']

        # Forward pass
        outputs = self.forward(x, attention_mask)

        # Compute loss for masked language modeling on individual cells
        if 'mlm_prediction' in outputs:
            predictions = outputs['mlm_prediction']  # [batch_size, seq_len, vocab_size]
            targets = x  # [batch_size, seq_len] - the actual token indices

            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                predictions.view(-1, predictions.size(-1)),  # [batch_size * seq_len, vocab_size]
                targets.view(-1),  # [batch_size * seq_len]
                ignore_index=-100  # Ignore padding tokens
            )

            # Log the loss
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            # Log learning rate from hparams
            current_lr = self.hparams.lr
            self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            # Also log the actual optimizer learning rate to verify it's correct
            if hasattr(self, 'trainer') and self.trainer is not None:
                optimizer = self.trainer.optimizers[0] if self.trainer.optimizers else None
                if optimizer:
                    actual_lr = optimizer.param_groups[0]['lr']
                    self.log('lr-AdamW', actual_lr, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            return loss
        else:
            logger.error("No mlm_prediction in model outputs!")
            raise ValueError("Model forward pass did not return expected outputs")

    def validation_step(self, batch, batch_idx):
        """Validation step for individual cell token prediction."""
        x = batch['X']
        attention_mask = batch['attention_mask']

        # Forward pass
        outputs = self.forward(x, attention_mask)

        # Compute validation loss
        if 'mlm_prediction' in outputs:
            predictions = outputs['mlm_prediction']  # [batch_size, seq_len, vocab_size]
            targets = x  # [batch_size, seq_len] - the actual token indices

            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                predictions.view(-1, predictions.size(-1)),  # [batch_size * seq_len, vocab_size]
                targets.view(-1),  # [batch_size * seq_len]
                ignore_index=-100  # Ignore padding tokens
            )

            # Log the validation loss
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

            # Also log the current learning rate and weight decay from optimizer
            if hasattr(self, 'trainer') and self.trainer is not None:
                optimizer = self.trainer.optimizers[0] if self.trainer.optimizers else None
                if optimizer:
                    current_lr = optimizer.param_groups[0]['lr']
                    current_wd = optimizer.param_groups[0]['weight_decay']
                    self.log('val_lr', current_lr, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))
                    self.log('val_wd', current_wd, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))

            return loss
        else:
            logger.error("No mlm_prediction in model outputs during validation!")
            raise ValueError("Model forward pass did not return expected outputs during validation")

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Handle individual cell format from Merlin dataset."""
        if 'X' in batch:
            x = batch['X']
            if x.dim() == 1:
                # Convert [batch_size] to [batch_size, 1] for individual cells
                batch['X'] = x.unsqueeze(1)

            # Update attention mask to match
            if 'attention_mask' in batch:
                mask = batch['attention_mask']
                if mask.dim() == 1:
                    batch['attention_mask'] = mask.unsqueeze(1)

        return batch

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Forward pass expecting properly preprocessed token indices."""
        # Data should be 2D tensor with token indices (dtype=torch.long)
        if x.dtype != torch.long:
            raise ValueError(f"Expected token indices with dtype=torch.long, got {x.dtype}")

        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor [batch_size, seq_len], got shape {x.shape}")

        # Use embedding layer to convert token indices to embeddings
        token_embedding = self.embeddings(x)

        # Continue with normal Nicheformer forward pass
        if self.hparams.learnable_pe:
            # For individual cells (seq_len=1), we only need the first positional embedding
            seq_len = x.shape[1]
            pos_indices = self.pos[:seq_len].to(token_embedding.device)
            pos_embedding = self.positional_embedding(pos_indices)
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        # Modified to not use autoregressive parameter
        transformer_output = self.encoder(
            embeddings,
            is_causal=False,  # Just hardcode to False
            src_key_padding_mask=attention_mask
        )

        prediction = self.classifier_head(transformer_output)

        return {
            'mlm_prediction': prediction,
            'transformer_output': transformer_output
        }

    def configure_optimizers(self) -> tuple:
        """Configure optimizers with the correct learning rate from Hydra config."""
        # Use the learning rate from hparams (set by Hydra)
        lr_to_use = self.hparams.lr
        logger.info("Using learning rate from config: %s", lr_to_use)

        # Get weight decay from stored config value
        weight_decay = self._config_weight_decay
        logger.info("Using weight decay from config: %s", weight_decay)

        # Log all hparams to see what's actually being passed
        logger.info("All hparams: %s", self.hparams)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr_to_use, weight_decay=weight_decay)

        # Log optimizer parameters to verify they're correct
        logger.info("Optimizer created with:")
        logger.info("  - Learning rate: %s", optimizer.param_groups[0]['lr'])
        logger.info("  - Weight decay: %s", optimizer.param_groups[0]['weight_decay'])

        # Check if warmup is defined in hparams
        warmup_epochs = getattr(self.hparams, 'warmup', 0)
        logger.info("Using warmup epochs: %s", warmup_epochs)

        # Ensure warmup is not too large
        if warmup_epochs >= self.hparams.max_epochs:
            logger.warning(
                f"Warmup epochs ({warmup_epochs}) >= max_epochs ({self.hparams.max_epochs}), setting warmup to 0")
            warmup_epochs = 0

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=warmup_epochs,
            max_epochs=self.hparams.max_epochs
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]


@rank_zero_only
def get_wandb_logger(config, wandb_name, subset_name):
    """Initialize wandb logger for training."""
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if not is_rank_zero:
        return None

    return WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=f"{config.wandb.group}_{subset_name}",
        name=wandb_name,
        mode=config.wandb.mode,
        settings=wandb.Settings(init_timeout=600)
    )


def continue_training_merlin(config: DictConfig) -> str:
    """Continue training a Nicheformer model using the Merlin dataset."""
    try:
        # Setup logging and environment
        pl.seed_everything(config.training.seed)

        logger.info("=== MERLIN DATASET CONTINUED TRAINING ===")
        logger.info("Configuration:")
        logger.info("  - Subset to train: %s", config.training.subset_to_train)
        logger.info("  - Learning rate: %s", config.training.learning_rate)
        logger.info("  - Batch size: %s", config.training.batch_size)
        logger.info("  - Max epochs: %s", config.training.max_epochs)
        logger.info("  - Weight decay: %s", config.training.weight_decay)

        # Validate subset name
        valid_subsets = ["100pct", "31.6pct", "10pct", "3.16pct", "1pct"]
        if config.training.subset_to_train not in valid_subsets:
            raise ValueError(f"Invalid subset '{config.training.subset_to_train}'. Must be one of: {valid_subsets}")

        subset_to_train = config.training.subset_to_train

        # Path to the processed Merlin data - use the new location
        merlin_data_dir = PROJECT_DIR / "lung" / "merlin_datamodule"

        if not merlin_data_dir.exists():
            # Fallback to old location for compatibility
            merlin_data_dir = PROJECT_DIR / "data" / "xenium_human_lung" / "processed" / "merlin_datamodule"

            if not merlin_data_dir.exists():
                raise FileNotFoundError(
                    "Merlin datamodule not found at either location. "
                    "Please run the preprocessing script first: python scripts/preprocess_lung_data.py"
                )

        logger.info("Using Merlin datamodule at: %s", merlin_data_dir)

        # Create subset datamodule
        # Pass the parent directory so SubsetMerlinDataModule can find the subsets
        if "lung" in str(merlin_data_dir):
            # New location structure
            processed_data_dir = PROJECT_DIR / "lung"
        else:
            # Old location structure
            processed_data_dir = PROJECT_DIR / "data" / "xenium_human_lung" / "processed"

        logger.info("Using Batch Size: %s", config.training.batch_size)

        datamodule = SubsetMerlinDataModule(
            data_dir=processed_data_dir,  # Pass the parent directory, not the merlin_datamodule subdirectory
            subset_name=subset_to_train,
            batch_size=config.training.batch_size,
            num_workers=min(8, os.cpu_count() // 2),
            seed=config.training.seed
        )

        logger.info("Created SubsetMerlinDataModule for subset: %s", subset_to_train)

        # Setup datamodule
        datamodule.setup(stage="fit")

        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        logger.info("Datamodule setup complete:")
        logger.info("  - Training batches: %s", len(train_dataloader))
        logger.info("  - Validation batches: %s", len(val_dataloader))
        logger.info("  - Training samples: %s", len(datamodule.train_dataset))
        logger.info("  - Validation samples: %s", len(datamodule.val_dataset))

        # Load checkpoint
        checkpoint_path = PROJECT_DIR / "trained_models" / "uni_gex" / "nicheformer.ckpt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        logger.info("Loading model from checkpoint: %s", checkpoint_path)

        # Load model with Hydra config parameters
        logger.info("Loading model with parameters:")
        logger.info("  - Learning rate: %s", config.training.learning_rate)
        logger.info("  - Weight decay: %s", config.training.weight_decay)
        logger.info("  - Batch size: %s", config.training.batch_size)
        logger.info("  - Max epochs: %s", config.training.max_epochs)

        model = MerlinNicheformer.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            batch_size=config.training.batch_size,
            lr=config.training.learning_rate,  # Use learning_rate from config as lr parameter
            weight_decay=config.training.weight_decay,
            max_epochs=config.training.max_epochs,
            map_location="cpu"
        )

        logger.info("Model loaded successfully. Model type: %s", type(model))

        # Test model forward pass
        logger.info("Testing model forward pass...")
        test_batch = next(iter(train_dataloader))

        # Ensure data types are correct
        if test_batch['X'].dtype != torch.long:
            test_batch['X'] = test_batch['X'].long()
        if test_batch['attention_mask'].dtype != torch.bool:
            test_batch['attention_mask'] = test_batch['attention_mask'].bool()

        # Apply reshaping for individual cells
        if test_batch['X'].dim() == 1:
            test_batch['X'] = test_batch['X'].unsqueeze(1)
        if test_batch['attention_mask'].dim() == 1:
            test_batch['attention_mask'] = test_batch['attention_mask'].unsqueeze(1)

        with torch.no_grad():
            test_output = model(test_batch['X'], test_batch['attention_mask'])

        logger.info("Model forward pass successful. Output keys: %s", list(test_output.keys()))

        # Setup wandb logging
        wandb_name = f"merlin_continued_{subset_to_train}_{random.choices(string.ascii_lowercase + string.digits, k=6)[0]}"
        wandb_logger = get_wandb_logger(config, wandb_name, subset_to_train)

        if wandb_logger is not None:
            logger.info("Wandb logger initialized: %s", wandb_logger)
            wandb_logger.log_hyperparams({
                "subset": subset_to_train,
                "subset_samples": len(datamodule.train_dataset),
                "lr": config.training.learning_rate,  # Use lr for consistency with model hparams
                "batch_size": config.training.batch_size,
                "max_epochs": config.training.max_epochs,
                "weight_decay": config.training.weight_decay,
                "checkpoint_path": str(checkpoint_path),
            })
        else:
            logger.warning("Wandb logger is None - check wandb configuration")

        # Setup callbacks
        # Include configuration info in checkpoint directory to avoid conflicts
        config_id = f"lr{config.training.learning_rate}_wd{config.training.weight_decay}_bs{config.training.batch_size}"
        checkpoint_dir = PROJECT_DIR / "trained_models" / "continued_training" / subset_to_train / config_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Checkpoint directory: %s", checkpoint_dir)
        logger.info("Configuration ID: %s", config_id)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"merlin_{subset_to_train}_epoch={{epoch:02d}}_val_loss={{val_loss:.4f}}",
            monitor="val_loss",
            save_top_k=2,
            mode="min",
            every_n_train_steps=500,
            save_last=True,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=50,
            mode="min",
            verbose=True,
        )

        # Setup trainer
        if torch.cuda.is_available():
            accelerator = "gpu"
            if torch.cuda.device_count() > 1:
                strategy = "ddp_find_unused_parameters_true"
            else:
                strategy = "auto"
        else:
            accelerator = "cpu"
            strategy = "auto"

        trainer = pl.Trainer(
            logger=wandb_logger,
            accelerator=accelerator,
            strategy=strategy,
            max_epochs=config.training.max_epochs,
            log_every_n_steps=100,  # Log more frequently for wandb
            check_val_every_n_epoch=2,
            default_root_dir=str(checkpoint_dir),
            callbacks=[
                checkpoint_callback,
                early_stopping,
                LearningRateMonitor(logging_interval="epoch"),
                LearningRateMonitorCallback(),  # Add our custom callback
            ],
            precision=config.training.get("precision", "bf16-mixed"),
            gradient_clip_val=config.training.gradient_clip_val,
            accumulate_grad_batches=config.training.accumulate_grad_batches,
            enable_progress_bar=True,  # Keep progress bar but reduce updates
            enable_checkpointing=True,
        )

        # Start training
        logger.info("Starting training on subset %s...", subset_to_train)
        trainer.fit(model=model, datamodule=datamodule)

        logger.info("Training completed successfully on subset %s!", subset_to_train)

        # Return the best checkpoint
        return checkpoint_callback.best_model_path

    except Exception as e:
        logger.error("Error in continue_training_merlin: %s", e)
        raise


@hydra.main(config_path="../../../configs", config_name="continued_training", version_base=None)
def main(config: DictConfig) -> None:
    """Main function for continued training with Merlin dataset."""
    continue_training_merlin(config)

    subset_trained = config.training.subset_to_train
    logger.info("Model trained on subset: %s", subset_trained)
    logger.info("To train on other subsets, set config.training.subset_to_train to one of: 100pct, 31.6pct, 10pct, 3.16pct, 1pct")


if __name__ == "__main__":
    main()
