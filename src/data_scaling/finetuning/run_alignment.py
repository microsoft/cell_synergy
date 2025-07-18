import os
import torch
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from data_scaling.finetuning.align import get_paired_dataloader, AlignmentTrainer, pool_patches

def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/hot/Projects/till_richter/")
    resolved = os.path.expandvars(raw)
    return Path(resolved)


PROJECT_DIR = get_project_dir()
MODEL_DIR = PROJECT_DIR / "trained_models"

# Import available models dynamically
def get_model_class(method):
    if method == 'adversarial':
        from data_scaling.models.multimodal.adversarial import AdversarialBaseline
        return AdversarialBaseline
    elif method == 'barlow_twins':
        from data_scaling.models.multimodal.barlowtwins import BarlowTwinsBaseline
        return BarlowTwinsBaseline
    elif method == 'byol':
        from data_scaling.models.multimodal.byol import BYOLBaseline
        return BYOLBaseline
    elif method == 'comm':
        from data_scaling.models.multimodal.comm import CoMMBaseline
        return CoMMBaseline
    elif method == 'cca':
        from data_scaling.src.data_scaling.models.multimodal.cca import CCABaseline
        return CCABaseline
    elif method == 'dim':
        from data_scaling.models.multimodal.dim import DIMBaseline
        return DIMBaseline
    elif method == 'simclr':
        from data_scaling.models.multimodal.simclr import SimCLRBaseline
        return SimCLRBaseline
    elif method == 'simsiam':
        from data_scaling.models.multimodal.simsiam import SimSiamBaseline
        return SimSiamBaseline
    elif method == 'vicreg':
        from data_scaling.models.multimodal.vicreg import VICRegBaseline
        return VICRegBaseline    
    else:
        raise ValueError(f"Unknown method: {method}")

def validate_config(cfg: DictConfig):
    """Validate configuration before running."""
    required_keys = [
        'data.dataset',
        'models.method',
        'training.batch_size',
        'wandb.project',
        'wandb.entity'
    ]
    
    for key in required_keys:
        if not OmegaConf.select(cfg, key):
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate method is available
    available_methods = ['adversarial', 'barlow_twins', 'comm', 'simclr', 'vicreg', 'byol', 'dcca', 'dim', 'simsiam']
    if cfg.models.method not in available_methods:
        raise ValueError(f"Method {cfg.models.method} not in available methods: {available_methods}")

@hydra.main(version_base=None, config_path="../../../configs", config_name="align.yaml")
def main(cfg: DictConfig):
    # Validate config
    validate_config(cfg)
    
    # Get seed
    seed = cfg.training.seed
    pl.seed_everything(seed)
    
    # Ensure we have a split set
    if not hasattr(cfg.data, 'split'):
        # Default to pretrain mode
        cfg.data.split = 'pretrain'
    
    # Validate split value
    if cfg.data.split not in ['pretrain', 'finetune', 'test']:
        raise ValueError(f"Invalid split value: {cfg.data.split}. Must be one of: pretrain, finetune, test")
    
    # Validate we have the right scale parameter for the split
    if cfg.data.split == 'pretrain' and not hasattr(cfg.data, 'pretrain_split'):
        raise ValueError("Missing pretrain_split in config when using pretrain split")
    elif cfg.data.split == 'finetune' and not hasattr(cfg.data, 'finetune_split'):
        raise ValueError("Missing finetune_split in config when using finetune split")
    
    print(f"Using split: {cfg.data.split}")
    if cfg.data.split != 'test':
        scale = cfg.data.pretrain_split if cfg.data.split == 'pretrain' else cfg.data.finetune_split
        print(f"Using scale: {scale}")
    
    # Load data with proper train/val split
    print("Creating train/validation split...")
    val_fraction = cfg.training.get('val_fraction', 0.2)  # Default 20% validation
    
    # Adaptive batch size for small datasets
    initial_batch_size = cfg.training.batch_size
    
    # Create training dataloader (80% of data)
    train_loader = get_paired_dataloader(
        cfg, 
        batch_size=initial_batch_size, 
        shuffle=True,
        val_split=False,  # Training split
        val_fraction=val_fraction
    )
    
    # Check if dataset is small and adjust batch size if needed
    training_samples = len(train_loader.dataset)
    if training_samples < initial_batch_size:
        print(f"⚠️  Dataset ({training_samples} samples) smaller than batch size ({initial_batch_size})")
        # Use a smaller batch size for small datasets
        adaptive_batch_size = max(16, training_samples // 4)  # At least 16, or 1/4 of dataset
        print(f"Reducing batch size to {adaptive_batch_size} for better training dynamics")
        
        # Recreate dataloaders with smaller batch size
        train_loader = get_paired_dataloader(
            cfg, 
            batch_size=adaptive_batch_size, 
            shuffle=True,
            val_split=False,
            val_fraction=val_fraction
        )
        
        # Create validation dataloader with same smaller batch size
        val_loader = get_paired_dataloader(
            cfg, 
            batch_size=adaptive_batch_size, 
            shuffle=False,
            val_split=True,
            val_fraction=val_fraction
        )
    else:
        # Create validation dataloader (20% of data) with original batch size
        val_loader = get_paired_dataloader(
            cfg, 
            batch_size=initial_batch_size, 
            shuffle=False,  # Don't shuffle validation
            val_split=True,  # Validation split
            val_fraction=val_fraction
        )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    # Get dimensions from first batch
    first_batch = next(iter(train_loader))
    img_embed, gex_embed = first_batch
    print(f"Image embedding dimension: {img_embed.shape[1]}")  # Should be 1024
    print(f"GEX embedding dimension: {gex_embed.shape[1]}")    # Should be 512
    
    # Set embedding dimensions in config
    cfg.models.img_embed_dim = img_embed.shape[1]  # 1024
    cfg.models.gex_embed_dim = gex_embed.shape[1]  # 512

    # Load model
    ModelClass = get_model_class(cfg.models.method)
    model = ModelClass(cfg)

    lightning_module = AlignmentTrainer(model, config=cfg)

    # Create WandbLogger with unique names including hyperparameters
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    seed = cfg.training.seed
    wandb_name = f"Alignment_{cfg.data.dataset}_{cfg.models.method}_{cfg.data.pretrain_split}_lr{lr}_wd{wd}_seed{seed}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=False,  # Set to True if you want to log model artifacts
        save_dir=str(MODEL_DIR)
    )

    # Debug: Check if wandb is properly initialized
    print(f"Wandb logger initialized: {wandb_logger.experiment is not None}")
    print(f"Wandb run name: {wandb_logger.name}")
    print(f"Wandb run id: {wandb_logger.version}")

    # Log alignment method to wandb
    wandb_logger.log_hyperparams({"alignment_method": cfg.models.method})

    # Test logging
    try:
        wandb_logger.experiment.log({"test_metric": 1.0})
        print("✓ Test logging successful")
    except Exception as e:
        print(f"✗ Test logging failed: {e}")

    # Create checkpoint callback - now we have proper validation
    # (lr, wd, seed already defined above for wandb naming)
    
    # Create method-specific directory for organized checkpoints
    method_dir = MODEL_DIR / cfg.models.method
    method_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(method_dir),  # Explicitly set the directory path
        monitor='val_loss',
        mode='min',
        save_top_k=3,  # Keep top 3 checkpoints
        save_last=True,
        filename=f'best_{cfg.models.method}_lr{lr}_wd{wd}-{{epoch:02d}}-{{val_loss:.4f}}',
        every_n_epochs=1  # Save checkpoint every epoch
    )
    
    # Create early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=cfg.training.get('early_stopping_patience', 15),
        verbose=True,
        min_delta=1e-4  # Minimum change to qualify as an improvement
    )
    
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval='epoch'),  # Log LR per epoch
        early_stopping
    ]

    # Create Trainer with improved stability settings
    # Adaptive validation interval based on dataset size
    training_batches = len(train_loader)
    config_val_interval = cfg.training.get('val_check_interval', 1.0)
    
    # If val_check_interval is specified as a number of steps, ensure it's not larger than training batches
    if isinstance(config_val_interval, (int, float)) and config_val_interval > 1:
        if config_val_interval >= training_batches:
            val_check_interval = 1.0  # Fall back to every epoch
            print(f"⚠️  val_check_interval ({config_val_interval}) >= training batches ({training_batches}), using every epoch instead")
        else:
            val_check_interval = config_val_interval
    else:
        val_check_interval = config_val_interval
    
    print(f"Using val_check_interval: {val_check_interval}")
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.get('max_epochs', 100),
        logger=wandb_logger,
        # Don't set default_root_dir to avoid wandb hash directories
        callbacks=callbacks,
        # Logging settings
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=max(1, min(50, training_batches // 10)),  # Adaptive: 1-50 steps, 10 times per epoch max
        # Training stability settings
        precision="32-true",  # Use full precision for stability, especially with adversarial training
        # Gradient settings for stability
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.5),  # Lower clip value for stability
        gradient_clip_algorithm='norm',  # Use gradient norm clipping
        # Performance settings
        deterministic=False,  # Allow non-deterministic operations for speed
        benchmark=True,  # Enable cudnn benchmarking for consistent input sizes
        # Validation settings
        val_check_interval=val_check_interval,  # Use adaptive validation interval
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,  # Quick validation sanity check
    )

    # Fit
    trainer.fit(lightning_module, train_loader, val_loader)

    # Print best checkpoint path
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
    print(f"Best checkpoint score: {checkpoint_callback.best_model_score}")

    # Use the same method directory that was created for the ModelCheckpoint
    # (lr, wd, seed already defined above for wandb naming)
    
    # The method_dir was already created above for the checkpoint callback
    method_dir = MODEL_DIR / cfg.models.method
    
    # Create descriptive checkpoint names with proper extensions
    base_name = f"{cfg.data.dataset}_{cfg.data.pretrain_split}_lr{lr}_wd{wd}_seed{seed}"
    lightning_ckpt_name = f"lightning_{base_name}.ckpt"
    model_state_name = f"model_{base_name}.pt"
    
    # Save the Lightning module checkpoint (contains full trainer state)
    lightning_path = method_dir / lightning_ckpt_name
    trainer.save_checkpoint(str(lightning_path))
    print(f"Lightning checkpoint saved: {lightning_path}")
    
    # Also save just the model state dict for easy loading in downstream tasks
    model_path = method_dir / model_state_name
    torch.save(lightning_module.model.state_dict(), str(model_path))
    print(f"Model state dict saved: {model_path}")
    
    # Save a metadata file with training info
    metadata = {
        "method": cfg.models.method,
        "dataset": cfg.data.dataset,
        "split": cfg.data.pretrain_split,
        "learning_rate": lr,
        "weight_decay": wd,
        "seed": seed,
        "best_val_loss": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        "total_epochs": trainer.current_epoch,
        "img_embed_dim": cfg.models.img_embed_dim,
        "gex_embed_dim": cfg.models.gex_embed_dim,
        "lightning_checkpoint": str(lightning_path),
        "model_state_dict": str(model_path),
        "best_checkpoint": checkpoint_callback.best_model_path  # Add reference to the best checkpoint
    }
    
    metadata_path = method_dir / f"metadata_{base_name}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    print(f"\n📁 All files saved in organized structure:")
    print(f"   Method directory: {method_dir}")
    print(f"   Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"   Lightning checkpoint: {lightning_path}")
    print(f"   Model state dict: {model_path}")
    print(f"   Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
