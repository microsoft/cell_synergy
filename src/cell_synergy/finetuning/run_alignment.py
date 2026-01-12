import os
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import wandb
from tqdm.auto import tqdm
from cell_synergy.finetuning.align import get_paired_dataloader, AlignmentTrainer, pool_patches, get_full_alignment_dataloader
from cell_synergy.paths import PROJECT_DIR, MODEL_DIR
import datetime


# Import available models dynamically
def get_model_class(method):
    """Get model class based on method name."""
    if method == 'adversarial':
        from cell_synergy.models.adversarial import AdversarialBaseline
        return AdversarialBaseline
    elif method == 'barlow_twins':
        from cell_synergy.models.barlowtwins import BarlowTwinsBaseline
        return BarlowTwinsBaseline
    elif method == 'byol':
        from cell_synergy.models.byol import BYOLBaseline
        return BYOLBaseline
    elif method == 'comm':
        from cell_synergy.models.comm import CoMMBaseline
        return CoMMBaseline
    elif method == 'cca':
        from cell_synergy.models.cca import CCABaseline
        return CCABaseline
    elif method == 'dcca':
        from cell_synergy.models.dcca import DCCABaseline
        return DCCABaseline
    elif method == 'dim':
        from cell_synergy.models.dim import DIMBaseline
        return DIMBaseline
    elif method == 'simclr':
        from cell_synergy.models.simclr import SimCLRBaseline
        return SimCLRBaseline
    elif method == 'simsiam':
        from cell_synergy.models.simsiam import SimSiamBaseline
        return SimSiamBaseline
    elif method == 'vicreg':
        from cell_synergy.models.vicreg import VICRegBaseline
        return VICRegBaseline
    else:
        raise ValueError(f"Unknown method: {method}")


def validate_config(cfg: DictConfig):
    """Validate configuration."""
    required_fields = [
        'data.dataset',
        'models.method',
        'training.batch_size',
        'training.learning_rate',
        'training.weight_decay',
        'training.max_epochs',
        'training.seed',
        'wandb.project',
        'wandb.entity'
    ]

    for field in required_fields:
        parts = field.split('.')
        current = cfg
        for part in parts:
            if not hasattr(current, part):
                raise ValueError(f"Missing required config field: {field}")
            current = getattr(current, part)

    # Validate method is available
    available_methods = [
        'adversarial', 'barlow_twins', 'comm', 'simclr', 'vicreg',
        'byol', 'cca', 'dcca', 'dim', 'simsiam'
    ]
    if cfg.models.method not in available_methods:
        raise ValueError(f"Method {cfg.models.method} not in available methods: {available_methods}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="align.yaml")
def main(cfg: DictConfig):
    # Validate config
    validate_config(cfg)

    # Get seed
    seed = cfg.training.seed
    pl.seed_everything(seed)

    # Determine if method is unsupervised (can use full dataset)
    unsupervised_methods = ['comm', 'simclr', 'barlow_twins', 'vicreg', 'byol', 'simsiam']
    is_unsupervised = cfg.models.method in unsupervised_methods

    if is_unsupervised:
        print(f"\n=== Using unsupervised training setup for {cfg.models.method} ===")
        print("Training on full dataset (unsupervised)")

        # Load full dataset for alignment
        full_loader = get_full_alignment_dataloader(
            cfg,
            batch_size=cfg.training.batch_size,
            shuffle=True
        )
        print(f"Full dataset samples: {len(full_loader.dataset)}")
        print(f"Full dataset batches: {len(full_loader)}")

        # Create validation loader for monitoring (small subset of full data)
        val_fraction = 0.1  # Use smaller validation set
        val_loader = get_paired_dataloader(
            cfg,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            val_split=True,
            val_fraction=val_fraction
        )
    else:
        # Standard setup for supervised/semi-supervised methods
        print(f"\n=== Using supervised/semi-supervised training setup for {cfg.models.method} ===")
        print("Creating train/validation split...")
        val_fraction = cfg.training.get('val_fraction', 0.15)

        # Create training dataloader (85% of data by default)
        train_loader = get_paired_dataloader(
            cfg,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            val_split=False,
            val_fraction=val_fraction
        )

        # Create validation dataloader
        val_loader = get_paired_dataloader(
            cfg,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            val_split=True,
            val_fraction=val_fraction
        )

        # For supervised methods, full_loader is the same as train_loader
        full_loader = train_loader

    # Get dimensions from first batch
    # Ensure CUDA is initialized before accessing first batch (fixes DIM CUDA error)
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
    first_batch = next(iter(full_loader))
    img_embed, gex_embed = first_batch
    print(f"Image embedding dimension: {img_embed.shape[1]}")
    print(f"GEX embedding dimension: {gex_embed.shape[1]}")

    # Set embedding dimensions in config
    cfg.models.img_embed_dim = img_embed.shape[1]
    cfg.models.gex_embed_dim = gex_embed.shape[1]

    # Load model
    ModelClass = get_model_class(cfg.models.method)
    model = ModelClass(cfg)

    # Fix CoMM x-attn Sequential call issue without modifying comm.py
    # The CoMM library calls resblocks[0](x1, x2, key_padding_mask) but Sequential only takes 1 arg
    if cfg.models.method == 'comm' and hasattr(model, 'encoder') and hasattr(model.encoder, 'fusion_transformer'):
        ft = model.encoder.fusion_transformer
        if getattr(ft, 'fusion', None) == "x-attn" and isinstance(ft.resblocks, list):
            # Wrap Sequential objects to make them callable with (x, y, key_padding_mask)
            wrapped = []
            for seq in ft.resblocks:
                if isinstance(seq, nn.Sequential):
                    # Create a callable wrapper that applies all blocks in sequence
                    # Ensure blocks are on the same device as inputs
                    def make_wrapper(s):
                        def wrapper(x, y, key_padding_mask=None):
                            out = x
                            for block in s:
                                # Ensure block is on same device as input
                                if hasattr(block, 'to'):
                                    block_device = next(
                                        block.parameters()).device if list(
                                        block.parameters()) else x.device
                                    if block_device != x.device:
                                        block = block.to(x.device)
                                out = block(out, y, key_padding_mask)
                            return out
                        return wrapper
                    wrapped.append(make_wrapper(seq))
                else:
                    wrapped.append(seq)
            ft.resblocks = wrapped

    lightning_module = AlignmentTrainer(model, config=cfg)

    # Create WandbLogger with unique names including hyperparameters
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    seed = cfg.training.seed
    wandb_name = f"Alignment_{cfg.data.dataset}_{cfg.models.method}_lr{lr}_wd{wd}_seed{seed}"
    if is_unsupervised:
        wandb_name += "_full_alignment"

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=False,
        save_dir=str(MODEL_DIR),
        settings=wandb.Settings(_service_wait=120)  # Increase timeout to 120 seconds
    )

    # Check if wandb is properly initialized
    print(f"Wandb logger initialized: {wandb_logger.experiment is not None}")
    print(f"Wandb run name: {wandb_logger.name}")
    print(f"Wandb run id: {wandb_logger.version}")

    # Log alignment method to wandb
    wandb_logger.log_hyperparams({"alignment_method": cfg.models.method})

    # Test logging
    try:
        wandb_logger.experiment.log({"test_metric": 1.0})
        print("Test logging successful")
    except Exception as e:
        print(f"✗ Test logging failed: {e}")

    # Create checkpoint directory
    checkpoint_dir = MODEL_DIR / cfg.data.dataset / "full_ds"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Only CCA is truly nonparametric (no gradient-based training needed)
    nonparametric_methods = ['cca']
    if cfg.models.method not in nonparametric_methods:
        # Configure trainer for gradient-based methods
        # Support multi-GPU training if available
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            # Use DDP for multi-GPU training
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(find_unused_parameters=False)
            devices = num_gpus
            print(f"Using {num_gpus} GPUs with DDP strategy")
        else:
            strategy = "auto"
            devices = 1

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=devices,
            strategy=strategy,
            num_nodes=1,  # Explicitly set for single-node multi-GPU
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=f"{cfg.models.method}{getattr(cfg.models, 'checkpoint_suffix', '')}",
                    save_top_k=1,
                    monitor="val_loss_smooth",  # Use smoothed validation loss for checkpointing
                    mode="min",
                    save_last=True,
                    every_n_epochs=1
                ),
                EarlyStopping(
                    monitor="val_loss_smooth",  # Use smoothed validation loss for early stopping
                    # Patience is in validation checks, not epochs. With val_check_interval=0.5 (2 checks/epoch),
                    # we want ~50 epochs of patience = 50 * 2 = 100 validation checks
                    patience=int(max(cfg.training.early_stopping_patience, 50)
                                 / cfg.training.get('val_check_interval', 0.5)),
                    mode="min",
                    verbose=True
                ),
                LearningRateMonitor(logging_interval="epoch")
            ],
            enable_progress_bar=True,
            log_every_n_steps=10,
            # Use config value (default 0.5 = 2 validations per epoch)
            val_check_interval=cfg.training.get('val_check_interval', 0.5),
            # Limit validation batches for DCCA (10% of val set) due to slow eigendecomposition, or use config override
            limit_val_batches=cfg.training.get('limit_val_batches', 0.1 if cfg.models.method == 'dcca' else 1.0),
            # Disable distributed sampler to prevent deadlocks
            # We handle data distribution manually via drop_last=True in DataLoader
            use_distributed_sampler=False
        )

        # Train model
        # Support resuming from checkpoint if provided
        ckpt_path = getattr(cfg.training, 'ckpt_path', None)
        try:
            if ckpt_path and os.path.exists(ckpt_path):
                print(f"\n=== Resuming training from checkpoint: {ckpt_path} ===")
                trainer.fit(
                    lightning_module,
                    train_dataloaders=full_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=ckpt_path
                )
            else:
                if ckpt_path:
                    print(f"WARNING: Checkpoint path provided but not found: {ckpt_path}")
                    print("Starting training from scratch...")
                trainer.fit(
                    lightning_module,
                    train_dataloaders=full_loader,
                    val_dataloaders=val_loader
                )
        except Exception as e:
            print(f"\nERROR during training: {e}")
            import traceback
            traceback.print_exc()
            raise

        print("\n=== Training Complete ===")
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
            print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
        else:
            print("WARNING: No checkpoint was saved!")
        # Training time calculation removed due to PyTorch Lightning API changes

        # Save metadata
        metadata = {
            "method": cfg.models.method,
            "dataset": cfg.data.dataset,
            "learning_rate": lr,
            "weight_decay": wd,
            "seed": seed,
            "img_embed_dim": cfg.models.img_embed_dim,
            "gex_embed_dim": cfg.models.gex_embed_dim,
            "best_checkpoint": trainer.checkpoint_callback.best_model_path,
            "best_val_loss_smooth": float(trainer.checkpoint_callback.best_model_score),  # Now monitoring smoothed loss
            "total_epochs": trainer.current_epoch,
            "timestamp": datetime.datetime.now().isoformat(),
            "is_unsupervised": is_unsupervised,
            "early_stopping_patience": max(cfg.training.early_stopping_patience, 50)
        }

    else:
        print(f"\n=== Running nonparametric fitting for {cfg.models.method} ===")

        if cfg.models.method == 'cca':
            # For CCA, fit on the entire dataset
            print("Fitting CCA model on full dataset...")

            # Collect all data
            all_img_embeds = []
            all_gex_embeds = []

            for batch in tqdm(full_loader, desc="Collecting data for CCA fitting"):
                img_embed, gex_embed = batch

                # Skip empty batches (all samples had NaN embeddings)
                if img_embed.size(0) == 0 or gex_embed.size(0) == 0:
                    continue

                # Filter out any remaining NaN embeddings
                valid_mask = ~(torch.isnan(img_embed).any(dim=1) | torch.isnan(gex_embed).any(dim=1))
                if valid_mask.any():
                    img_embed_valid = img_embed[valid_mask].cpu().numpy()
                    gex_embed_valid = gex_embed[valid_mask].cpu().numpy()
                    all_img_embeds.append(img_embed_valid)
                    all_gex_embeds.append(gex_embed_valid)

            # Concatenate all batches
            if all_img_embeds:
                all_img_embeds = np.concatenate(all_img_embeds, axis=0)
                all_gex_embeds = np.concatenate(all_gex_embeds, axis=0)
            else:
                raise ValueError("No valid embeddings found for CCA fitting")

            print(f"CCA fitting on {all_img_embeds.shape[0]} samples")
            print(f"Image embeddings shape: {all_img_embeds.shape}")
            print(f"GEX embeddings shape: {all_gex_embeds.shape}")

            # Validate sample count
            min_samples = max(all_img_embeds.shape[1], all_gex_embeds.shape[1]) + 1
            if all_img_embeds.shape[0] < min_samples:
                raise ValueError(
                    f"CCA requires at least {min_samples} samples, but only {all_img_embeds.shape[0]} available")

            # Fit CCA model
            try:
                lightning_module.model.cca_model.fit(
                    all_img_embeds,
                    all_gex_embeds,
                    lightning_module.model.outdim
                )
                lightning_module.model.is_fitted = True
                lightning_module.model._fitted_on_full_dataset = True
                print("CCA model fitted successfully on full dataset")

                # Save CCA model
                checkpoint_path = checkpoint_dir / "cca.pt"
                torch.save({
                    'state_dict': lightning_module.model.state_dict(),
                    'timestamp': datetime.datetime.now().isoformat()
                }, checkpoint_path)
                print(f"CCA model saved to: {checkpoint_path}")

                # Compute validation loss
                val_losses = []
                lightning_module.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        img_embed, gex_embed = batch
                        if img_embed.size(0) > 0 and gex_embed.size(0) > 0:
                            loss = lightning_module.model(img_embed, gex_embed)
                            val_losses.append(loss.item())

                final_val_loss = np.mean(val_losses) if val_losses else float('inf')
                print(f"Final validation loss: {final_val_loss:.4f}")

            except Exception as e:
                print(f"CCA fitting failed: {str(e)}")
                raise

            # Save metadata
            metadata = {
                "method": cfg.models.method,
                "dataset": cfg.data.dataset,
                "img_embed_dim": cfg.models.img_embed_dim,
                "gex_embed_dim": cfg.models.gex_embed_dim,
                "checkpoint": str(checkpoint_path),
                "val_loss": final_val_loss,
                "timestamp": datetime.datetime.now().isoformat(),
                "is_unsupervised": True,
                "samples_used": all_img_embeds.shape[0]
            }

    # Save metadata
    metadata_path = checkpoint_dir / f"{cfg.models.method}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
