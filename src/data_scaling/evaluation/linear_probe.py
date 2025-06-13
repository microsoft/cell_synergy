"""
Linear probing module for evaluating embeddings on downstream tasks.
"""
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, r2_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from data_scaling.paths import MODEL_DIR
from scipy.spatial.distance import cdist


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

def auto_detect_checkpoint(cfg):
    """
    Automatically detect the checkpoint path based on config and naming convention.
    Returns the checkpoint path as a Path object. Raises AssertionError if not found.
    """
    ckpt_name = f"Finetune_{cfg.data.dataset}_{cfg.models.method}_{cfg.data.pretrain_split}"
    ckpt_path = MODEL_DIR / "multi" / ckpt_name
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path} (expected from config and naming convention)"
    return ckpt_path


def extract_embeddings_from_fusion_model(cfg, dataset, model_ckpt_path=None, device=None, target_key=None, img_embed_key=None, gex_embed_key=None):
    """
    Extract fused embeddings from a frozen fusion model for all samples in the dataset.
    Embedding keys are configurable via arguments or cfg.data.
    Asserts if required keys are missing in any sample.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_embed_key = img_embed_key or getattr(cfg.data, 'img_embed_key', None)
    gex_embed_key = gex_embed_key or getattr(cfg.data, 'gex_embed_key', None)
    assert img_embed_key is not None, "img_embed_key must be specified in config or as argument."
    assert gex_embed_key is not None, "gex_embed_key must be specified in config or as argument."
    assert target_key is not None, "target_key must be specified."

    # Dynamically import the model class based on cfg (similar to run_alignment.py)
    method = cfg.experiment.method
    if method == 'simclr':
        from data_scaling.models.multimodal.simclr import SimCLRBaseline as FusionModel
    elif method == 'barlow_twins':
        from data_scaling.models.multimodal.barlowtwins import BarlowTwinsBaseline as FusionModel
    elif method == 'vicreg':
        from data_scaling.models.multimodal.vicreg import VicRegBaseline as FusionModel
    elif method == 'comm':
        from data_scaling.models.multimodal.comm import CoMMBaseline as FusionModel
    elif method == 'adversarial':
        from data_scaling.models.multimodal.adversarial import AdversarialBaseline as FusionModel
    elif method == 'concat':
        from data_scaling.models.multimodal.concat import ConcatBaseline as FusionModel
    else:
        raise ValueError(f"Unknown method: {method}")

    # Auto-detect checkpoint if not provided
    if model_ckpt_path is None:
        model_ckpt_path = auto_detect_checkpoint(cfg)
    if isinstance(model_ckpt_path, str):
        model_ckpt_path = Path(model_ckpt_path)
    assert model_ckpt_path.exists(), f"Model checkpoint does not exist: {model_ckpt_path}"

    # Load model
    model = FusionModel(cfg)
    checkpoint = torch.load(model_ckpt_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    # Extract embeddings and targets
    embeddings = []
    targets = []
    sample_names = []
    for row in dataset:
        # Check for required embedding keys
        assert img_embed_key in row, f"Missing image embedding key '{img_embed_key}' in sample {row.get('name', '?')}"
        assert gex_embed_key in row, f"Missing gex embedding key '{gex_embed_key}' in sample {row.get('name', '?')}"
        img_embed = row[img_embed_key]
        gex_embed = row[gex_embed_key]
        assert img_embed is not None, f"Image embedding is None for sample {row.get('name', '?')}"
        assert gex_embed is not None, f"Gex embedding is None for sample {row.get('name', '?')}"
        img_embed = torch.tensor(img_embed, dtype=torch.float32, device=device)
        gex_embed = torch.tensor(gex_embed, dtype=torch.float32, device=device)
        # Forward through fusion model (assume model returns fused embedding)
        with torch.no_grad():
            fused = model.fusion(img_embed.unsqueeze(0), gex_embed.unsqueeze(0))
        embeddings.append(fused.squeeze(0).cpu())
        # Target
        target = row[target_key]
        if isinstance(target, list):
            target = torch.tensor(target)
        targets.append(target)
        sample_names.append(row['name'])
    embeddings = torch.stack(embeddings)
    targets = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in targets])
    return embeddings, targets, sample_names


def run_loocv_linear_probe(cfg, dataset, model_ckpt_path=None, task_type=None, target_key=None, img_embed_key=None, gex_embed_key=None):
    """
    Run LOOCV with a linear probe on fused embeddings.
    Embedding keys and checkpoint path are auto-detected if not provided.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings, targets, sample_names = extract_embeddings_from_fusion_model(
        cfg, dataset, model_ckpt_path, device, target_key, img_embed_key, gex_embed_key
    )
    n_samples = len(sample_names)
    metrics_list = []
    for i in range(n_samples):
        # LOOCV split
        test_idx = i
        train_idx = [j for j in range(n_samples) if j != i]
        train_embeddings = embeddings[train_idx]
        train_targets = targets[train_idx]
        test_embeddings = embeddings[test_idx].unsqueeze(0)
        test_targets = targets[test_idx].unsqueeze(0)
        # Optionally, split off a validation set from train (here, use 10% for val)
        n_train = train_embeddings.shape[0]
        val_size = max(1, int(0.1 * n_train))
        val_embeddings = train_embeddings[:val_size]
        val_targets = train_targets[:val_size]
        train_embeddings_ = train_embeddings[val_size:]
        train_targets_ = train_targets[val_size:]
        # Train linear probe
        metrics = train_linear_probe(
            train_embeddings_, train_targets_,
            val_embeddings, val_targets,
            test_embeddings, test_targets,
            task_type=task_type,
            batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=getattr(cfg.training, 'weight_decay', 0.0),
            max_epochs=cfg.training.max_epochs,
            patience=getattr(cfg.training, 'patience', 10),
            embedding_source=cfg.experiment.method,
            wandb_project=cfg.experiment.wandb_project,
            wandb_entity=cfg.experiment.wandb_entity,
            wandb_group=f"LOOCV_{sample_names[test_idx]}",
            wandb_name=f"LOOCV_{sample_names[test_idx]}"
        )
        metrics['test_sample'] = sample_names[test_idx]
        metrics_list.append(metrics)
    # Aggregate metrics
    agg_metrics = {}
    for key in metrics_list[0].keys():
        if key == 'test_sample':
            continue
        values = [m[key] for m in metrics_list]
        agg_metrics[f"{key}_mean"] = float(np.mean(values))
        agg_metrics[f"{key}_std"] = float(np.std(values))
    print("LOOCV results:", agg_metrics)
    return metrics_list, agg_metrics 

def prepare_neighbor_aggregation_data(embeddings, labels, centers, k, task_type, class_count=None):
    """
    For each patch, aggregate the labels of its k nearest neighbors.
    - For classification: returns a class distribution (soft multi-label, normalized counts).
    - For regression: returns the mean of neighbor cell type composition vectors.
    Args:
        embeddings: np.ndarray [N, D] (patch embeddings)
        labels: np.ndarray [N] (class indices) or [N, D] (regression targets)
        centers: np.ndarray [N, 2] (patch centers)
        k: int (number of neighbors)
        task_type: 'classification' or 'regression'
        class_count: int, number of classes (required for classification)
    Returns:
        X: np.ndarray [N, D] (patch embeddings)
        Y: np.ndarray [N, C] (aggregated neighbor labels)
        neighbor_indices: np.ndarray [N, k] (indices of neighbors for each patch)
    """
    N = embeddings.shape[0]
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    neighbor_indices = np.argsort(dists, axis=1)[:, :k]
    X = embeddings
    if task_type == 'classification':
        assert class_count is not None, "class_count must be provided for classification aggregation."
        Y = np.zeros((N, class_count), dtype=np.float32)
        for i in range(N):
            neighbor_labels = labels[neighbor_indices[i]]
            for c in range(class_count):
                Y[i, c] = np.sum(neighbor_labels == c)
            Y[i] /= k  # Normalize to get distribution
    else:
        # Regression: mean of neighbor vectors
        Y = np.zeros_like(labels)
        for i in range(N):
            neighbor_targets = labels[neighbor_indices[i]]
            Y[i] = neighbor_targets.mean(axis=0)
    return X, Y, neighbor_indices 