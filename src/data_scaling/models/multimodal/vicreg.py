import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearVICRegEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.encoder(x)

class VICRegBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sim_coeff = cfg.models.sim_coeff
        self.std_coeff = cfg.models.std_coeff
        self.cov_coeff = cfg.models.cov_coeff
        self.projection_dim = cfg.models.projection_dim

        self.img_encoder = LinearVICRegEncoder(cfg.models.img_embed_dim, self.projection_dim)
        self.gex_encoder = LinearVICRegEncoder(cfg.models.gex_embed_dim, self.projection_dim)

    def forward(self, img_embed, gex_embed):
        x = self.img_encoder(img_embed)
        y = self.gex_encoder(gex_embed)

        repr_loss = F.mse_loss(x, y)

        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = (F.relu(1 - std_x).mean() + F.relu(1 - std_y).mean()) / 2

        cov_x = self._off_diagonal_cov(x)
        cov_y = self._off_diagonal_cov(y)
        cov_loss = (cov_x + cov_y) / self.projection_dim

        return (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

    def _off_diagonal_cov(self, z):
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
        return self._off_diagonal(cov).pow(2).sum()

    def _off_diagonal(self, x):
        # Extract off-diagonal elements
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def get_embeddings(self, img_embed, gex_embed):
        z1 = F.normalize(self.img_encoder(img_embed), dim=-1)
        z2 = F.normalize(self.gex_encoder(gex_embed), dim=-1)
        return z1, z2