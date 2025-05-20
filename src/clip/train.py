"""
CLIP training module for data scaling experiments.
"""
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..pretraining.vision import VisionEncoder


class CLIPModel(pl.LightningModule):
    """
    CLIP model for contrastive learning between Nicheformer and vision embeddings.
    """
    
    def __init__(
        self,
        nicheformer_checkpoint: str,
        vision_model_key: str,
        embedding_dim: int = 512,
        projection_dim: int = 256,
        temperature: float = 0.07,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the CLIP model.
        
        Args:
            nicheformer_checkpoint: Path to Nicheformer checkpoint
            vision_model_key: Key for vision model
            embedding_dim: Dimension of embeddings
            projection_dim: Dimension of projection space
            temperature: Temperature parameter for contrastive loss
            learning_rate: Learning rate
            weight_decay: Weight decay
            max_epochs: Maximum number of epochs
            cache_dir: Cache directory for models
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load Nicheformer model
        self.load_nicheformer(nicheformer_checkpoint)
        
        # Load vision encoder
        self.vision_encoder = VisionEncoder(
            model_key=vision_model_key,
            cache_dir=cache_dir,
            embedding_dim=embedding_dim,
        )
        
        # Projection layers
        self.gexp_projection = nn.Linear(embedding_dim, projection_dim)
        self.vision_projection = nn.Linear(embedding_dim, projection_dim)
        
        # Temperature parameter
        self.temperature = temperature
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
    
    def load_nicheformer(self, checkpoint_path: str) -> None:
        """
        Load Nicheformer model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        # Import here to avoid circular import
        try:
            from models._nicheformer import Nicheformer
        except ImportError:
            raise ImportError(
                "Nicheformer package not found. Please install it with: "
                "pip install git+https://github.com/theislab/nicheformer.git"
            )
        
        # Load checkpoint
        self.nicheformer = Nicheformer.load_from_checkpoint(checkpoint_path)
        
        # Freeze Nicheformer
        for param in self.nicheformer.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        images: torch.Tensor, 
        gexp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Image tensor
            gexp: Gene expression tensor
        
        Returns:
            Tuple of projected image and gene expression embeddings
        """
        # Get vision embeddings
        vision_embeddings = self.vision_encoder(images)
        
        # Get gene expression embeddings
        with torch.no_grad():
            gexp_embeddings = self.nicheformer.encode(gexp)
        
        # Project embeddings
        vision_proj = self.vision_projection(vision_embeddings)
        gexp_proj = self.gexp_projection(gexp_embeddings)
        
        # Normalize projections
        vision_proj = F.normalize(vision_proj, p=2, dim=1)
        gexp_proj = F.normalize(gexp_proj, p=2, dim=1)
        
        return vision_proj, gexp_proj
    
    def contrastive_loss(
        self, 
        vision_proj: torch.Tensor, 
        gexp_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            vision_proj: Projected vision embeddings
            gexp_proj: Projected gene expression embeddings
        
        Returns:
            Contrastive loss
        """
        # Compute cosine similarity
        logits = torch.matmul(vision_proj, gexp_proj.t()) / self.temperature
        
        # Compute loss
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        
        return (loss_i + loss_t) / 2.0
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get data
        images, gexp = batch["image"], batch["gexp"]
        
        # Forward pass
        vision_proj, gexp_proj = self(images, gexp)
        
        # Compute loss
        loss = self.contrastive_loss(vision_proj, gexp_proj)
        
        # Log loss
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Get data
        images, gexp = batch["image"], batch["gexp"]
        
        # Forward pass
        vision_proj, gexp_proj = self(images, gexp)
        
        # Compute loss
        loss = self.contrastive_loss(vision_proj, gexp_proj)
        
        # Log loss
        self.log("val_loss", loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers."""
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Create scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def train_clip(
    nicheformer_checkpoint: str,
    vision_model_key: str,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    embedding_dim: int = 512,
    projection_dim: int = 256,
    temperature: float = 0.07,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    batch_size: int = 32,
    cache_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_name: Optional[str] = None,
) -> str:
    """
    Train a CLIP model.
    
    Args:
        nicheformer_checkpoint: Path to Nicheformer checkpoint
        vision_model_key: Key for vision model
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        embedding_dim: Dimension of embeddings
        projection_dim: Dimension of projection space
        temperature: Temperature parameter for contrastive loss
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum number of epochs
        batch_size: Batch size
        cache_dir: Cache directory for models
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        wandb_group: W&B group name
        wandb_name: W&B run name
    
    Returns:
        Path to best checkpoint
    """
    # Create model
    model = CLIPModel(
        nicheformer_checkpoint=nicheformer_checkpoint,
        vision_model_key=vision_model_key,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        temperature=temperature,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        cache_dir=cache_dir,
    )
    
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
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    return checkpoint_callback.best_model_path 