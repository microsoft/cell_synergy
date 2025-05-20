"""
Module for continued pretraining (fine-tuning) of models.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..clip.train import CLIPModel


def finetune_clip(
    clip_checkpoint: str,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    learning_rate: float = 1e-5,  # Lower learning rate for fine-tuning
    weight_decay: float = 1e-6,
    max_epochs: int = 50,
    freeze_vision: bool = False,
    cache_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_name: Optional[str] = None,
) -> str:
    """
    Fine-tune a CLIP model with continued pretraining.
    
    Args:
        clip_checkpoint: Path to CLIP checkpoint
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum number of epochs
        freeze_vision: Whether to freeze the vision encoder
        cache_dir: Cache directory for models
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        wandb_group: W&B group name
        wandb_name: W&B run name
    
    Returns:
        Path to best checkpoint
    """
    # Load CLIP model
    model = CLIPModel.load_from_checkpoint(clip_checkpoint)
    
    # Optionally freeze vision encoder
    if freeze_vision:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    
    # Set learning rate and other parameters
    model.learning_rate = learning_rate
    model.weight_decay = weight_decay
    model.max_epochs = max_epochs
    
    # Create logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=wandb_name,
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch:02d}-{val_loss:.4f}",
    )
    
    # Create learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="bf16-mixed",
    )
    
    # Fine-tune model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    return checkpoint_callback.best_model_path


def finetune_nicheformer(
    nicheformer_checkpoint: str,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-6,
    max_epochs: int = 50,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_name: Optional[str] = None,
) -> str:
    """
    Fine-tune a Nicheformer model.
    
    Args:
        nicheformer_checkpoint: Path to Nicheformer checkpoint
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum number of epochs
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        wandb_group: W&B group name
        wandb_name: W&B run name
    
    Returns:
        Path to best checkpoint
    """
    # Import here to avoid circular import
    try:
        from models._nicheformer import Nicheformer
    except ImportError:
        raise ImportError(
            "Nicheformer package not found. Please install it with: "
            "pip install git+https://github.com/theislab/nicheformer.git"
        )
    
    # Load Nicheformer model
    model = Nicheformer.load_from_checkpoint(nicheformer_checkpoint)
    
    # Update learning rate and other parameters
    model.lr = learning_rate
    model.weight_decay = weight_decay
    model.max_epochs = max_epochs
    
    # Create logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=wandb_name,
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch:02d}-{val_loss:.4f}",
    )
    
    # Create learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="bf16-mixed",
    )
    
    # Fine-tune model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    return checkpoint_callback.best_model_path 