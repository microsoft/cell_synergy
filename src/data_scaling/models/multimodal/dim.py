# Based on https://github.com/rdevon/DIM

import torch
import torch.nn as nn
import torch.nn.functional as F


class DIMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, frozen=True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        self.frozen = frozen
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


class DIMBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.temperature = cfg.models.get("temperature", 0.1)
        self.mi_estimator_type = cfg.models.get("mi_estimator", "mlp")
        self.normalize = cfg.models.get("normalize", True)

        proj_dim = cfg.models.projection_dim
        hidden_dim = cfg.models.projection_hidden_dim
        frozen = cfg.models.get("frozen_encoders", True)

        self.img_encoder = DIMEncoder(cfg.models.img_embed_dim, hidden_dim, proj_dim, frozen=frozen)
        self.gex_encoder = DIMEncoder(cfg.models.gex_embed_dim, hidden_dim, proj_dim, frozen=frozen)

        if self.mi_estimator_type == "bilinear":
            self.score_fn = nn.Bilinear(proj_dim, proj_dim, 1)
        elif self.mi_estimator_type == "mlp":
            self.score_fn = nn.Sequential(
                nn.Linear(2 * proj_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, 1)
            )
        elif self.mi_estimator_type == "dot":
            self.score_fn = None  # Use dot product
        else:
            raise ValueError(f"Unknown MI estimator: {self.mi_estimator_type}")

    def forward(self, img_embed, gex_embed):
        return self.compute_loss(img_embed, gex_embed)

    def compute_loss(self, img_embed, gex_embed):
        z1 = self.img_encoder(img_embed)
        z2 = self.gex_encoder(gex_embed)

        if self.normalize:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

        N = z1.size(0)
        if self.score_fn is not None:
            z1_exp = z1.unsqueeze(1).expand(N, N, -1)
            z2_exp = z2.unsqueeze(0).expand(N, N, -1)
            if isinstance(self.score_fn, nn.Bilinear):
                logits = self.score_fn(z1_exp, z2_exp).squeeze(-1) / self.temperature
            else:
                joint = torch.cat([z1_exp, z2_exp], dim=-1)
                logits = self.score_fn(joint).squeeze(-1) / self.temperature
        else:
            logits = z1 @ z2.T / self.temperature

        labels = torch.arange(N, device=z1.device)
        return F.cross_entropy(logits, labels)

    def get_embeddings(self, img_embed, gex_embed):
        z1 = self.img_encoder(img_embed)
        z2 = self.gex_encoder(gex_embed)
        if self.normalize:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
        return z1, z2