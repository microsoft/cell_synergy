# scripts/continue_train.py

import os
import random
import string
import logging
import time
from pathlib import Path

import torch
import wandb
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# print working directory
print(f"Current working directory: {os.getcwd()}")
# Ensure the data_scaling package is in the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from data_scaling.pretraining.train_nicheformer import FixedNicheformer
from data_scaling.data.hf_dataset.simple_hf_datamodule import SimpleHFDataModule

# Get logger for this module
logger = logging.getLogger(__name__)

# Extended Nicheformer for HF dataset compatibility
class HFCompatibleNicheformer(FixedNicheformer):
    """Extended FixedNicheformer with HF dataset compatibility for finetuning."""
    
    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Override to handle our simple individual cell format."""
        # Our simple datamodule provides individual cells as 1D tensors
        # We need to convert them to 2D for the transformer
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
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Override to handle our simple format and skip the original context_length slicing."""
        # Skip the original Nicheformer's batch transfer logic that expects sequences
        # Our data is already in the right format after on_before_batch_transfer
        return batch
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Forward pass expecting properly preprocessed token indices."""
        rank = int(os.environ.get("RANK", -1))
        
        # Data should be 2D tensor with token indices (dtype=torch.long)
        if x.dtype != torch.long:
            raise ValueError(f"Expected token indices with dtype=torch.long, got {x.dtype}")
        
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor [batch_size, seq_len], got shape {x.shape}")
        
        # Debug logging for first few forward passes
        if hasattr(self, '_debug_forward_count') and self._debug_forward_count <= 3:
            print(f"[Rank {rank}] Forward pass with token indices, shape: {x.shape}, dtype: {x.dtype}")
        
        # Use embedding layer to convert token indices to embeddings
        token_embedding = self.embeddings(x)
        
        # Track forward passes for debugging
        if hasattr(self, '_debug_forward_count'):
            self._debug_forward_count += 1
        else:
            self._debug_forward_count = 1

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
    
    def training_step(self, batch, batch_idx):
        """Override training step to add gradient debugging."""
        # Check if parameters are frozen every 100 steps
        if batch_idx % 100 == 0:
            rank = int(os.environ.get("RANK", -1))
            total_params = 0
            trainable_params = 0
            params_with_grad = 0
            
            for name, param in self.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    if param.grad is not None:
                        params_with_grad += param.numel()
            
            # Also check current learning rate from optimizer
            current_lr = self.optimizers().param_groups[0]['lr'] if hasattr(self.optimizers(), 'param_groups') else "unknown"
            
            print(f"[Rank {rank}] Step {batch_idx}: Total params: {total_params}, "
                  f"Trainable: {trainable_params}, With grad: {params_with_grad}, Current LR: {current_lr}")
        
        return super().training_step(batch, batch_idx)
    
    def configure_optimizers(self) -> tuple:
        """Override configure_optimizers to ensure updated learning rate is used."""
        # Use the updated learning rate from hparams
        logger.info(f"HFCompatibleNicheformer.configure_optimizers called with lr: {self.hparams.lr}")
        
        # Verify this method is actually being called
        rank = int(os.environ.get("RANK", -1))
        print(f"[Rank {rank}] HFCompatibleNicheformer.configure_optimizers: lr={self.hparams.lr}")
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        
        # Verify the optimizer actually has the right learning rate
        actual_lr = optimizer.param_groups[0]['lr']
        print(f"[Rank {rank}] Optimizer created with actual lr: {actual_lr}")
        if actual_lr != self.hparams.lr:
            logger.error(f"Learning rate mismatch! Expected: {self.hparams.lr}, Got: {actual_lr}")
        
        # Try to import the scheduler, fallback to appropriate fine-tuning scheduler if not available
        try:
            from nicheformer.models.lr_schedulers import CosineWarmupScheduler
            scheduler = CosineWarmupScheduler(
                optimizer,
                warmup=self.hparams.warmup,
                max_epochs=self.hparams.max_epochs
            )
            print(f"[Rank {rank}] Using CosineWarmupScheduler with warmup={self.hparams.warmup}, max_epochs={self.hparams.max_epochs}")
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
        except ImportError:
            logger.warning("Could not import CosineWarmupScheduler, using cosine annealing scheduler for fine-tuning")
            # Use cosine annealing for fine-tuning - better than step decay
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.hparams.max_epochs * 1000,  # Approximate steps per epoch * epochs
                eta_min=self.hparams.lr * 0.01  # End at 1% of initial learning rate
            )
            print(f"[Rank {rank}] Using CosineAnnealingLR with T_max={self.hparams.max_epochs * 1000}, eta_min={self.hparams.lr * 0.01}")
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/hot/Projects/till_richter/")
    resolved = os.path.expandvars(raw)
    return Path(resolved)

class RankAwareLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        rank = int(os.environ.get("RANK", -1))
        if batch_idx % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Rank {rank} starting batch {batch_idx}")


@rank_zero_only
def get_wandb_logger(config, wandb_name, scale):
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    # Include scale in the group name for better organization
    base_group = config.wandb.group
    group_name = f"{base_group}_{scale}_{os.environ.get('MASTER_ADDR', 'unknown')}"
    if not is_rank_zero:
        return None  # Don't log from non-zero ranks

    return WandbLogger(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=group_name,
        name=wandb_name,
        mode=config.wandb.mode,
        settings=wandb.Settings(init_timeout=600)
    )



def continue_training_nicheformer(config: DictConfig) -> str:
    """Continue training a Nicheformer model from checkpoint."""
    def get_data_dir():
        return Path(config.models.subset)

    PROJECT_DIR = get_project_dir()
    UNI_EMBEDDINGS_DIR = PROJECT_DIR / "unimodal_embeddings"
    MODEL_DIR = PROJECT_DIR / "trained_models"
    DATA_DIR = get_data_dir()
    logger.info(f"--- ENV (PID: {os.getpid()}) ---")
    for var in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"]:
        logger.info(f"{var}: {os.environ.get(var, 'Not set')}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        logger.info("Matmul precision set to medium")

    pl.seed_everything(config.training.seed)

    # --- Resolve sample names ---
    sample_names = config.data.multimodal["finetune"][config.data.finetune_split]
    scale = config.data.finetune_split
    logger.info(f"=== FINETUNE CONFIGURATION ===")
    logger.info(f"Scale: {scale}")
    logger.info(f"Sample names for finetune split '{scale}': {sample_names}")
    logger.info(f"Number of samples: {len(sample_names)}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Max epochs: {config.training.max_epochs}")
    logger.info(f"===============================")

    # Use the main HF dataset name - filtering is done by sample_names
    hf_dataset_name = config.data.hf_datasets.original

    # Load nicheformer checkpoint path
    checkpoint_path = MODEL_DIR / "nicheformer.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure the nicheformer model has been trained first.")
    logger.info(f"Resuming training from: {checkpoint_path}")
    # --- Initialize data ---
    datamodule = SimpleHFDataModule(
        hf_dataset_name=hf_dataset_name,
        sample_names=sample_names,
        batch_size=config.training.batch_size,
        num_workers=4,
        val_ratio=config.training.val_ratio,
        seed=config.training.seed,
        pad_token_id=0  # Padding token ID to identify valid cells
    )

    # Setup for training
    datamodule.setup(stage="fit")

    # --- Load model ---
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint manually to exclude batch_size and lr from hparams but still provide them during loading
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    original_batch_size = None
    original_lr = None
    
    # Also remove optimizer and lr_scheduler states to ensure fresh training
    if 'optimizer_states' in checkpoint:
        del checkpoint['optimizer_states']
        logger.info("Removed optimizer states from checkpoint - will start with fresh optimizer")
    
    if 'lr_schedulers' in checkpoint:
        del checkpoint['lr_schedulers'] 
        logger.info("Removed lr_schedulers from checkpoint - will start with fresh scheduler")
    
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint was saved at epoch {checkpoint['epoch']}")
        # Reset epoch to start fresh
        checkpoint['epoch'] = 0
        logger.info("Reset epoch to 0 for fresh training")
    
    if 'global_step' in checkpoint:
        logger.info(f"Checkpoint was saved at global_step {checkpoint['global_step']}")
        checkpoint['global_step'] = 0
        logger.info("Reset global_step to 0 for fresh training")
    
    if 'hyper_parameters' in checkpoint:
        if 'batch_size' in checkpoint['hyper_parameters']:
            original_batch_size = checkpoint['hyper_parameters']['batch_size']
            del checkpoint['hyper_parameters']['batch_size']
            logger.info(f"Removed batch_size ({original_batch_size}) from checkpoint hparams before loading model")
        
        if 'lr' in checkpoint['hyper_parameters']:
            original_lr = checkpoint['hyper_parameters']['lr']
            del checkpoint['hyper_parameters']['lr']
            logger.info(f"Removed lr ({original_lr}) from checkpoint hparams before loading model - will use config value instead")
    
    # Save the modified checkpoint temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_file:
        torch.save(checkpoint, tmp_file.name)
        temp_checkpoint_path = tmp_file.name
    
    try:
        # Provide batch_size and lr as parameters during loading since they're required by the constructor
        model = HFCompatibleNicheformer.load_from_checkpoint(
            temp_checkpoint_path, 
            strict=False,
            batch_size=config.training.batch_size,  # Use current config batch_size
            lr=config.training.learning_rate  # Use current config learning rate
        )
    finally:
        # Clean up temporary file
        os.unlink(temp_checkpoint_path)

    # Update model hparams to match config to avoid conflicts
    # Set learning rate for continued training - use training.learning_rate from config
    if hasattr(config.training, "learning_rate"):
        old_lr = getattr(model.hparams, 'lr', 'unknown')
        model.hparams.lr = config.training.learning_rate
        logger.info(f"Updated learning rate from checkpoint ({old_lr}) to: {config.training.learning_rate}")
    elif hasattr(config.training, "continued_lr"):
        model.hparams.lr = config.training.continued_lr
        logger.info(f"Set learning rate to: {config.training.continued_lr}")
    
    # Force update max_epochs to match config
    model.hparams.max_epochs = config.training.max_epochs
    
    # Debug: Log remaining model hparams
    logger.info(f"Model hparams after cleanup: {list(model.hparams.keys()) if hasattr(model, 'hparams') else 'No hparams'}")
    logger.info(f"Final learning rate in model.hparams.lr: {getattr(model.hparams, 'lr', 'Not set')}")

    # --- Logging ---
    wandb_name = config.wandb.get("name", "continued_training")
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    scale = config.data.finetune_split
    wandb_name = f"{wandb_name}_continued_{scale}_{suffix}"

    # --- Callbacks ---
    checkpoint_key = (
        f"continued_finetune_{scale}_"
        f"lr{str(config.training.learning_rate).replace('.', '_')}"
    )
    checkpoint_dir = MODEL_DIR / checkpoint_key

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{checkpoint_key}_{{epoch:02d}}_{{val_loss:.4f}}",
        monitor="val_loss",  # Monitor validation loss instead of train loss
        save_top_k=3,
        mode="min",
        every_n_train_steps=500,
        save_last=True,  # Also save the last checkpoint
    )

    # Add early stopping based on validation loss
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=50,  # Stop if no improvement for 50 epochs
        mode="min",
        verbose=True,
    )

    wandb_logger = get_wandb_logger(config, wandb_name, scale)
    if wandb_logger is not None:
        # Log hyperparameters including the updated learning rate and scale
        wandb_logger.log_hyperparams({
            "scale": scale,
            "dataset_name": config.data.dataset_name,
            "checkpoint_path": checkpoint_path,
            "continued_training": True,
            "lr": config.training.learning_rate,  # Explicitly log the updated learning rate
            "max_epochs": config.training.max_epochs,
            "sample_names": sample_names,  # Log which samples are being used
        })
    
    # Add barrier to synchronize all ranks after wandb init
    if dist.is_initialized():
        dist.barrier()
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        max_epochs=config.training.max_epochs,
        log_every_n_steps=config.training.get("log_every_n_steps", 100),
        check_val_every_n_epoch=config.training.get("val_check_interval", 2),  # Check validation every 2 epochs
        strategy="ddp_find_unused_parameters_true",
        default_root_dir=str(checkpoint_dir),
        callbacks=[
            checkpoint_callback,
            early_stopping,
            LearningRateMonitor(logging_interval="step"),
            RankAwareLogger(),
        ],
        precision=config.training.get("precision", "bf16-mixed"),
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
    )

    logger.info("Starting training...")
    logger.info(f"About to call trainer.fit() with model lr: {getattr(model.hparams, 'lr', 'unknown')}")
    
    # Force the model to log its configure_optimizers call
    print(f"Model class: {type(model)}")
    print(f"Model configure_optimizers method: {model.configure_optimizers}")
    
    trainer.fit(model=model, datamodule=datamodule)
    
    logger.info("Training completed successfully!")

    return checkpoint_callback.best_model_path


@hydra.main(config_path="../../../configs", config_name="continued_training", version_base=None)
def main(config: DictConfig) -> None:
    best_ckpt = continue_training_nicheformer(config)
    logger.info(f"Training finished. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
