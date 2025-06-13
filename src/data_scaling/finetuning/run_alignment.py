import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data_scaling.finetuning.align import get_paired_dataloader, AlignmentTrainer, pool_patches
from data_scaling.paths import PROJECT_DIR, MODEL_DIR

# Import available models dynamically
def get_model_class(method):
    if method == 'simclr':
        from data_scaling.models.multimodal.simclr import SimCLRBaseline
        return SimCLRBaseline
    elif method == 'barlow_twins':
        from data_scaling.models.multimodal.barlowtwins import BarlowTwinsBaseline
        return BarlowTwinsBaseline
    elif method == 'vicreg':
        from data_scaling.models.multimodal.vicreg import VicRegBaseline
        return VicRegBaseline
    elif method == 'comm':
        from data_scaling.models.multimodal.comm import CoMMBaseline
        return CoMMBaseline
    elif method == 'adversarial':
        from data_scaling.models.multimodal.adversarial import AdversarialBaseline
        return AdversarialBaseline
    elif method == 'concat':
        from data_scaling.models.multimodal.concat import ConcatBaseline
        return ConcatBaseline
    else:
        raise ValueError(f"Unknown method: {method}")

@hydra.main(config_path="../../../configs", config_name="base.yaml")
def main(cfg: DictConfig):
    # Get seed
    seed = cfg.training.seed
    pl.seed_everything(seed)

    # Load data (embeddings, splits, etc.)
    train_loader = get_paired_dataloader(cfg, batch_size=cfg.training.batch_size)
    val_loader = None

    first_batch = next(iter(train_loader))
    img_embed, gex_embed = first_batch
    img_embed = pool_patches(img_embed)
    gex_embed = pool_patches(gex_embed)
    print(f"img_embed shape after pooling: {img_embed.shape}, gex_embed shape after pooling: {gex_embed.shape}")
    cfg.models.img_embed_dim = img_embed.shape[1]
    cfg.models.gex_embed_dim = gex_embed.shape[1]

    # Load model
    ModelClass = get_model_class(cfg.models.method)
    model = ModelClass(cfg)

    lightning_module = AlignmentTrainer(model, config=cfg)

    # Create WandbLogger
    wandb_name = f"Finetune_{cfg.data.dataset}_{cfg.models.method}_{cfg.data.pretrain_split}"
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Create Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs if 'max_epochs' in cfg.training else 100,
        logger=wandb_logger,
        default_root_dir=str(MODEL_DIR),
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=True
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
    )

    # Fit
    trainer.fit(lightning_module, train_loader, val_loader)

    # Save checkpoint with config-based name
    ckpt_name = f"Finetune_{cfg.data.dataset}_{cfg.models.method}_{cfg.data.pretrain_split}_seed{seed}"
    trainer.save_checkpoint(str(MODEL_DIR / ckpt_name))
    print(f"Saved checkpoint: {MODEL_DIR / ckpt_name}")

if __name__ == "__main__":
    main() 