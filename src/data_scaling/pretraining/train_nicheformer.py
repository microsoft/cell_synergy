"""
Nicheformer training module for data scaling experiments.
"""
import os
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from math import ceil
import hydra
import wandb
import pandas as pd
import gc
import random
import string
from torch.utils.data import DataLoader
from torch.distributed import get_rank
# Import from nicheformer package
import nicheformer.data.datamodules 
from nicheformer.models._nicheformer import Nicheformer as BaseNicheformer
from nicheformer.data.datamodules import MerlinDataModuleDistributed

from mpi4py import MPI
MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)

# from data_scaling.paths import MODEL_DIR, PROJECT_DIR, get_project_dir
# import data_scaling.paths
# print(data_scaling.paths.__file__)


def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/hot/Projects/till_richter/")
    resolved = os.path.expandvars(raw)
    return Path(resolved)

# Dynamically override the _get_data_files_distributed function
def robust_get_data_files_distributed(base_path: str, split: str, world_size: int, sub_sample_frac: float = 1):
    files_devices = []

    # Get all valid parquet files (exclude hidden/temp files)
    all_files = [
        file for file in os.listdir(os.path.join(base_path, split))
        if file.endswith('.parquet') and not file.startswith('.')
    ]

    for device in range(world_size):
        try:
            files = [file for file in all_files if ((int(file.split('.')[0].split('-')[1]) % world_size) == device)]
        except (IndexError, ValueError):
            files = [file for i, file in enumerate(sorted(all_files)) if i % world_size == device]

        files = [os.path.join(base_path, split, file) for file in sorted(files)]
        files.sort(reverse=True)
        files_devices.append(files[:ceil(sub_sample_frac * len(files))])

    return files_devices


# Replace the function in the imported module
import nicheformer.data.datamodules
nicheformer.data.datamodules._get_data_files_distributed = robust_get_data_files_distributed


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
    
    def training_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return super().training_step(batch, batch_idx)

import os
import time
from pytorch_lightning.callbacks import Callback

class RankAwareLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        rank = int(os.environ.get("RANK", -1))
        if batch_idx % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Rank {rank} starting batch {batch_idx}")



def train_nicheformer(
    config: DictConfig,
) -> str:
    """
    Train a Nicheformer model on a specified subset.
    
    Args:
        config: Hydra configuration
    
    Returns:
        Path to the best checkpoint
    """

    def get_data_dir():
        return Path(config.models.subset)

    PROJECT_DIR = get_project_dir()
    UNI_EMBEDDINGS_DIR = PROJECT_DIR / "unimodal_embeddings"
    MODEL_DIR = PROJECT_DIR / "trained_models"
    DATA_DIR = get_data_dir()
    # --- Environment Variable Diagnostics ---
    print(f"--- DISTRIBUTED ENV VARS AT START OF train_nicheformer (PID: {os.getpid()}) ---")
    env_vars_to_print = [
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "NODE_RANK", 
        "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"
    ]
    for var in env_vars_to_print:
        print(f"ENV_VAR - {var}: {os.environ.get(var)}")
    print(f"--- END DISTRIBUTED ENV VARS (PID: {os.getpid()}) ---")

    # Set matmul precision for A100 GPUs early
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0) # Check capability of device 0
            if cap[0] >= 8: # Ampere or newer
                print(f"Setting torch.set_float32_matmul_precision('high') for GPU {torch.cuda.get_device_name(0)} (Capability {cap[0]}.{cap[1]}).")
                torch.set_float32_matmul_precision('high')
        except Exception as e:
            print(f"Could not set matmul precision: {e}")

    pl.seed_everything(config.experiment.seed)
    
    global_world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"DataModule will be configured with world_size (total processes for data sharding) = {global_world_size}")
    
    subset_key = config.models.subset
    # Use the get_project_dir function to ensure the latest environment variable value is used
    # current_project_dir = get_project_dir() 
    # path_organ = current_project_dir / subset_key
    # print(f"Data path: {path_organ}")
    path_organ = DATA_DIR 
    if not path_organ.exists():
        raise ValueError(f"Data directory does not exist: {path_organ}. Please check your environment variable or config.")
    print(f"Data path: {path_organ}")

    # Let's make sure the data directory exists with the expected format
    train_dir = os.path.join(path_organ, "train")
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    
    files = os.listdir(train_dir)
    if not any(f.endswith('.parquet') for f in files):
        raise ValueError(f"No parquet files found in {train_dir}")
    
    print(f"Found {len(files)} files in {train_dir}")
    
    # Set up data module
    key_organ = ["X"]  # Only load X column to avoid type conversion issues
    
    print(f"Global world_size (for data module sharding): {global_world_size}")

    # Sleep for 2 mins to ensure data is downloaded
    time.sleep(120)  # Sleep for 2 minutes to ensure data is downloaded
    
    module = MerlinDataModuleDistributed(
        path=str(path_organ),
        columns=key_organ,
        batch_size=config.training.batch_size,
        world_size=global_world_size,
        splits=False,
        dataloader_kwargs_train={"device": "cpu"}, 
        dataloader_kwargs_inference={"device": "cpu"},
    )

    print(f"After init loader", torch.cuda.memory_summary())

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

    # Get base name from config
    wandb_name = config.wandb.get("name", None)

    # Generate a 6-character alphanumeric suffix
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

    # Append suffix if base name exists, otherwise just use suffix
    wandb_name = f"{wandb_name}_{suffix}" if wandb_name else suffix

    # Add random suffix for each model to be unique
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        name=config.wandb.get("name"), # Use name from hydra config if available, else wandb generates one
        mode=config.wandb.mode,
    )
    
    # Extract donor count from subset name and log to wandb
    subset_key = config.models.subset
    donor_count = None
    if "nf_" in subset_key and "donors" in subset_key:
        try:
            donor_count = int(subset_key.split("_")[1].replace("donors", ""))
        except (IndexError, ValueError):
            print(f"Could not extract donor count from subset: {subset_key}")

    # Log the donor count and other experiment metadata
    if donor_count is not None:
        wandb_logger.log_hyperparams({
            "donor_count": donor_count,
            "subset_name": subset_key,
            "model_dim": config.models.dim_model,
            "model_layers": config.models.nlayers,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
        })
        print(f"Logging donor_count: {donor_count} to wandb")

    checkpoint_key = config.models.subset + "lr" + str(config.training.learning_rate).replace('.', '_') + "_wd" + str(config.training.weight_decay).replace('.', '_')
    checkpoint_dir = MODEL_DIR / checkpoint_key

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir), # Ensure dirpath is a string
        filename=f"{checkpoint_key}_{{epoch:02d}}_{{train_loss:.4f}}", # checkpoint_key used here
        monitor="train_loss",
        save_top_k=3,
        mode="min",
        every_n_train_steps=500,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    strategy = "ddp_find_unused_parameters_true" 
    gc.collect()
    torch.cuda.empty_cache()

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        # devices and num_nodes will be auto-detected by PL from env vars like RANK, WORLD_SIZE, LOCAL_RANK
        max_epochs=config.training.max_epochs,
        log_every_n_steps=100,
        check_val_every_n_epoch=50,
        strategy=strategy,
        default_root_dir=str(checkpoint_dir), 
        callbacks=[checkpoint_callback, lr_monitor, RankAwareLogger()],
        precision="bf16-mixed",
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    trainer.fit(model=model, datamodule=module)
    return checkpoint_callback.best_model_path


@hydra.main(config_path="../../../configs", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main function for training Nicheformer models.
    """
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))  # Ensure W&B is logged in before training

    checkpoint_path = train_nicheformer(config)
    print(f"Training complete. Best model saved at: {checkpoint_path}")
    
if __name__ == "__main__":
    main()