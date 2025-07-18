# From https://github.com/facebookresearch/barlowtwins/

import torch
import torch.nn as nn
import torch.nn.functional as F


class BarlowProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.projector(x)


class BarlowTwinsBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lambd = cfg.models.lambda_param
        proj_dim = cfg.models.projection_dim
        hidden_dim = cfg.models.projection_hidden_dim

        # Shared capacity and architecture
        self.img_proj = BarlowProjection(cfg.models.img_embed_dim, hidden_dim, proj_dim)
        self.gex_proj = BarlowProjection(cfg.models.gex_embed_dim, hidden_dim, proj_dim)

        # Final normalization (LayerNorm affine=False is similar to BatchNorm(affine=False))
        self.norm = nn.LayerNorm(proj_dim, elementwise_affine=False)

    def forward(self, img_embed, gex_embed):
        z1 = self.img_proj(img_embed)
        z2 = self.gex_proj(gex_embed)

        z1 = self.norm(z1)
        z2 = self.norm(z2)

        # Cross-correlation matrix
        c = (z1.T @ z2) / z1.size(0)

        # Loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        z1 = F.normalize(self.img_proj(img_embed), dim=-1)
        z2 = F.normalize(self.gex_proj(gex_embed), dim=-1)
        return z1, z2
