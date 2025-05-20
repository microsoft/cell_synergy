"""
Linear probing module for evaluating embeddings on downstream tasks.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, r2_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


class LinearProbe(pl.LightningModule):
    """
    Linear probe for evaluating embeddings on downstream tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str = "classification",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
    ):
        """
        Initialize the linear probe.
        
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output predictions
            task_type: Type of task ('classification' or 'regression')
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Task type
        self.task_type = task_type
        
        # Linear layer
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Loss function
        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
            self.activation = nn.Identity()  # No activation for classification
        elif task_type == "regression":
            self.loss_fn = nn.MSELoss()
            self.activation = nn.Identity()  # No activation for regression
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings
        
        Returns:
            Output predictions
        """
        return self.activation(self.linear(x))
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get data
        x, y = batch
        
        # Forward pass
        logits = self(x)
        
        # Compute loss
        loss = self.loss_fn(logits, y)
        
        # Log loss
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Get data
        x, y = batch
        
        # Forward pass
        logits = self(x)
        
        # Compute loss
        loss = self.loss_fn(logits, y)
        
        # Log loss
        self.log("val_loss", loss, prog_bar=True)
        
        # Compute metrics
        if self.task_type == "classification":
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        elif self.task_type == "regression":
            self.log("val_mse", loss, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        # Get data
        x, y = batch
        
        # Forward pass
        logits = self(x)
        
        # Compute loss
        loss = self.loss_fn(logits, y)
        
        # Log loss
        self.log("test_loss", loss)
        
        # Store predictions for computing metrics
        if self.task_type == "classification":
            preds = torch.argmax(logits, dim=1)
            return {"preds": preds, "targets": y}
        elif self.task_type == "regression":
            return {"preds": logits, "targets": y}
    
    def test_epoch_end(self, outputs):
        """Compute metrics at the end of the test epoch."""
        # Concatenate predictions and targets
        preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        targets = torch.cat([x["targets"] for x in outputs]).cpu().numpy()
        
        # Compute metrics
        if self.task_type == "classification":
            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average="macro")
            self.log("test_acc", acc)
            self.log("test_f1", f1)
            print(f"Test accuracy: {acc:.4f}, F1 score: {f1:.4f}")
        elif self.task_type == "regression":
            r2 = r2_score(targets, preds)
            mse = np.mean((targets - preds) ** 2)
            self.log("test_r2", r2)
            self.log("test_mse", mse)
            print(f"Test R2 score: {r2:.4f}, MSE: {mse:.4f}")
    
    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    test_embeddings: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    task_type: str = "classification",
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    max_epochs: int = 100,
    patience: int = 10,
    embedding_source: str = "unknown",
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train a linear probe on embeddings.
    
    Args:
        train_embeddings: Training embeddings
        train_labels: Training labels
        val_embeddings: Validation embeddings
        val_labels: Validation labels
        test_embeddings: Test embeddings
        test_labels: Test labels
        task_type: Type of task ('classification' or 'regression')
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum number of epochs
        patience: Patience for early stopping
        embedding_source: Source of embeddings
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        wandb_group: W&B group name
        wandb_name: W&B run name
    
    Returns:
        Dictionary of metrics
    """
    # Determine input and output dimensions
    input_dim = train_embeddings.shape[1]
    if task_type == "classification":
        output_dim = int(train_labels.max().item()) + 1
    else:
        output_dim = train_labels.shape[1] if len(train_labels.shape) > 1 else 1
    
    # Create model
    model = LinearProbe(
        input_dim=input_dim,
        output_dim=output_dim,
        task_type=task_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Create datasets
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Create test dataloader if test data is provided
    test_dataloader = None
    if test_embeddings is not None and test_labels is not None:
        test_dataset = TensorDataset(test_embeddings, test_labels)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    
    # Create logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        name=wandb_name,
    )
    
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        callbacks=[early_stopping],
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )
    
    # Test model if test data is provided
    metrics = {}
    if test_dataloader is not None:
        trainer.test(model, test_dataloader)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            test_preds = []
            for batch in test_dataloader:
                x, _ = batch
                logits = model(x)
                if task_type == "classification":
                    preds = torch.argmax(logits, dim=1)
                else:
                    preds = logits
                test_preds.append(preds)
            
            test_preds = torch.cat(test_preds).cpu().numpy()
            test_labels_np = test_labels.cpu().numpy()
        
        # Compute metrics
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(test_labels_np, test_preds)
            metrics["f1_score"] = f1_score(test_labels_np, test_preds, average="macro")
        else:
            metrics["r2_score"] = r2_score(test_labels_np, test_preds)
            metrics["mse"] = np.mean((test_labels_np - test_preds) ** 2)
    
    return metrics 