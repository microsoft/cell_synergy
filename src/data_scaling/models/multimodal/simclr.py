# From https://github.com/sthalles/SimCLR/

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temperature = cfg.models.temperature
        proj_dim = cfg.models.projection_dim

        # Linear projection layers
        self.img_projection = nn.Linear(cfg.models.img_embed_dim, proj_dim)
        self.gex_projection = nn.Linear(cfg.models.gex_embed_dim, proj_dim)

    def forward(self, img_embed, gex_embed):
        img_proj = F.normalize(self.img_projection(img_embed), dim=-1)
        gex_proj = F.normalize(self.gex_projection(gex_embed), dim=-1)
        return img_proj, gex_proj

    def compute_loss(self, img_embed, gex_embed):
        img_proj, gex_proj = self.forward(img_embed, gex_embed)
        batch_size = img_proj.size(0)

        # Similarity matrix (img x gex)
        logits = img_proj @ gex_proj.T / self.temperature
        labels = torch.arange(batch_size, device=img_proj.device)

        # Symmetric contrastive loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i2t + loss_t2i)

    def get_embeddings(self, img_embed, gex_embed):
        img_proj = F.normalize(self.img_projection(img_embed), dim=-1)
        gex_proj = F.normalize(self.gex_projection(gex_embed), dim=-1)
        return img_proj, gex_proj

    def fusion(self, img_embed, gex_embed):
        img_proj, gex_proj = self.get_embeddings(img_embed, gex_embed)
        return torch.cat([img_proj, gex_proj], dim=-1)