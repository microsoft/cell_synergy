"""VICReg (Variance-Invariance-Covariance Regularization) for multimodal learning.

Based on: Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization for
Self-Supervised Learning" (ICLR 2022) and https://github.com/facebookresearch/vicreg

This module implements VICReg-style self-supervised learning that enforces
variance, invariance, and covariance constraints on representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegBaseline(nn.Module):
    """VICReg baseline for multimodal alignment.

    Enforces variance (prevent collapse), invariance (similarity), and
    covariance (decorrelation) constraints on representations.
    """

    def __init__(self, cfg):
        """Initialize VICReg baseline.

        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()
        # Read from config, fallback to defaults if not present
        img_embed_dim = cfg.models.img_embed_dim
        gex_embed_dim = cfg.models.gex_embed_dim
        projection_dim = cfg.models.projection_dim
        self.sim_coeff = cfg.models.sim_coeff
        self.std_coeff = cfg.models.std_coeff
        self.cov_coeff = cfg.models.cov_coeff
        self.projection_dim = projection_dim

        self.img_proj = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False)
        )

        self.gex_proj = nn.Sequential(
            nn.Linear(gex_embed_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False)
        )

    def forward(self, img_embed, gex_embed):
        """Forward pass: compute VICReg loss.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            VICReg loss (similarity + variance + covariance terms)
        """
        x = self.img_proj(img_embed)
        y = self.gex_proj(gex_embed)

        repr_loss = F.mse_loss(x, y)

        # Variance loss
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Covariance loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (y.size(0) - 1)
        cov_loss = self._off_diagonal(cov_x).pow(2).sum() / self.projection_dim
        cov_loss += self._off_diagonal(cov_y).pow(2).sum() / self.projection_dim

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

    def _off_diagonal(self, x):
        """Extract off-diagonal elements from a square matrix.

        Args:
            x: Square matrix

        Returns:
            Flattened off-diagonal elements
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def get_embeddings(self, img_embed, gex_embed):
        """Get aligned embeddings for downstream tasks.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Concatenated normalized embeddings
        """
        x = F.normalize(self.img_proj(img_embed), dim=-1)
        y = F.normalize(self.gex_proj(gex_embed), dim=-1)
        # Return concatenated embeddings like other multimodal models for consistency
        # This ensures the evaluation pipeline gets the expected 512D embeddings
        return torch.cat([x, y], dim=-1)
