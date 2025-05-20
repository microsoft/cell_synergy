"""
Nicheformer training module for data scaling experiments.
"""
import os
import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import pandas as pd

# Import from nicheformer package
from nicheformer.models._nicheformer import Nicheformer as BaseNicheformer
from nicheformer.data.datamodules import MerlinDataModuleDistributed

from data_scaling.paths import MODEL_DIR, PROJECT_DIR


# Create a subclass of Nicheformer that fixes the autoregressive issue
class FixedNicheformer(BaseNicheformer):
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Forward pass of the model with fixed autoregressive handling."""
        token_embedding = self.embeddings(x)

        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device))
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


def train_nicheformer(
    config: DictConfig,
    world_size: int = None,
) -> str:
    """
    Train a Nicheformer model on a specified subset.
    
    Args:
        config: Hydra configuration
        world_size: Number of GPUs for distributed training
    
    Returns:
        Path to the best checkpoint
    """
    # Set random seed for reproducibility
    pl.seed_everything(config.experiment.seed)
    
    # Get GPU count if world_size is not provided
    if world_size is None:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        elif torch.cuda.is_available():
            world_size = torch.cuda.device_count()
        else:
            world_size = 1
    
    print(f"Training with world_size: {world_size}")
    
    # Get nicheformer subset path based on the specified split
    subset_key = config.models.subset
    path_organ = PROJECT_DIR / subset_key
    
    print(f"Data path: {path_organ}")
    
    # Let's make sure the data directory exists with the expected format
    train_dir = os.path.join(path_organ, "train")
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    
    files = os.listdir(train_dir)
    if not any(f.endswith('.parquet') for f in files):
        raise ValueError(f"No parquet files found in {train_dir}")
    
    print(f"Found {len(files)} files in {train_dir}")
    print(f"Example filenames: {files[:3]}")
    
    # Inspect the first parquet file to see column data types
    first_file = os.path.join(train_dir, files[0])
    print(f"Reading first parquet file: {first_file}")
    df = pd.read_parquet(first_file)
    print("Column data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Set up data module
    key_organ = ["X"]  # Only load X column to avoid type conversion issues
    
    # Also check in particular the columns we're trying to use
    for col in key_organ:
        if col in df.columns:
            print(f"Column {col} present with type {df[col].dtype} and sample: {df[col].iloc[0]}")
        else:
            print(f"WARNING: Column {col} not found in dataframe!")
    
    module = MerlinDataModuleDistributed(
        path=str(path_organ),
        columns=key_organ,
        batch_size=config.training.batch_size,
        world_size=world_size,
        splits=False,  # Use training data for validation and testing
    )
    
    # Create nicheformer model
    model = FixedNicheformer(
        dim_model=config.models.dim_model, 
        nheads=config.models.nheads, 
        dim_feedforward=config.models.dim_feedforward, 
        nlayers=config.models.nlayers,
        dropout=config.models.dropout,
        batch_first=config.models.batch_first,
        masking_p=config.models.masking_p,
        n_tokens=config.models.n_tokens,
        context_length=config.models.context_length,
        warmup=config.models.warmup,
        lr=config.training.learning_rate,
        batch_size=config.training.batch_size,
        max_epochs=config.training.max_epochs,
        supervised_task=config.models.supervised_task,
        learnable_pe=config.models.learnable_pe,
        specie=False,    # Disable specie to avoid using this column
        assay=False,     # Disable assay to avoid using this column
        modality=False,  # Set modality to False since our dataset doesn't have this column
        contrastive=config.models.contrastive,
    )
    
    # Set up logging
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        name=f"{subset_key}_dim{config.models.dim_model}_layers{config.models.nlayers}",
        mode=config.wandb.mode,
    )
    
    # Set up checkpoint directory in MODEL_DIR
    checkpoint_dir = MODEL_DIR / subset_key
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{subset_key}_{{epoch:02d}}_{{train_loss:.4f}}",
        monitor="train_loss",
        save_top_k=3,
        mode="min",
        every_n_train_steps=500,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Set up distributed training strategy
    strategy = "ddp_find_unused_parameters_true"
    
    # Set up trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        max_epochs=config.training.max_epochs,
        devices=world_size,  # Use specified number of GPUs
        log_every_n_steps=100,
        check_val_every_n_epoch=50,
        strategy=strategy,
        default_root_dir=checkpoint_dir,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="bf16-mixed",
        gradient_clip_val=1,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
    )
    
    # Train model
    trainer.fit(model=model, datamodule=module)
    
    # Return path to best checkpoint
    return checkpoint_callback.best_model_path


@hydra.main(config_path="../../configs", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main function for training Nicheformer models.
    
    Args:
        config: Hydra configuration
    """
    # If base config model name does not exist or is not nicheformer, choose nicheformer
    if config.models is None or config.models.model_name != "nicheformer":
        config.models = config.models

    # Get world size from SLURM environment variables if available
    world_size = None
    if "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
    elif "SLURM_GPUS" in os.environ:
        world_size = int(os.environ["SLURM_GPUS"])
    
    # Train nicheformer with distributed setup
    checkpoint_path = train_nicheformer(config, world_size=world_size)
    
    print(f"Best checkpoint saved at: {checkpoint_path}")


if __name__ == "__main__":
    main() 