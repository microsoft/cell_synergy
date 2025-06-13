import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import pytorch_lightning as pl

class PairedEmbeddingDataset(Dataset):
    def __init__(self, cfg):
        from data_scaling.paths import PROJECT_DIR
        split = cfg.data.pretrain_split
        dataset_path = PROJECT_DIR / cfg.data.hf_dataset_paths[split]
        self.dataset = load_from_disk(str(dataset_path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        img = row['img_uni_pool']
        gex = row['nicheformer_pool']
        return torch.tensor(img, dtype=torch.float32), torch.tensor(gex, dtype=torch.float32)

def get_paired_dataloader(cfg, batch_size=None, shuffle=True, num_workers=0):
    batch_size = batch_size or cfg.training.batch_size
    dataset = PairedEmbeddingDataset(cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def pool_patches(x):
    # If shape is [batch, 1, patches, embed_dim], squeeze and mean over patches
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)  # [batch, patches, embed_dim]
    if x.ndim == 3:
        return x.mean(dim=1)  # mean over patches
    return x

class AlignmentTrainer(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def training_step(self, batch, batch_idx):
        img_embed, gex_embed = batch
        img_embed = pool_patches(img_embed)
        gex_embed = pool_patches(gex_embed)
        if hasattr(self.model, 'compute_loss'):
            loss = self.model.compute_loss(img_embed, gex_embed)
        else:
            loss = self.model(img_embed, gex_embed)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        lr = getattr(self.config.training, 'learning_rate', 1e-3)
        wd = getattr(self.config.training, 'weight_decay', 1e-5)
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)