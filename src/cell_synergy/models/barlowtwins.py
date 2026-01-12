"""Barlow Twins for multimodal learning.

Based on: Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
(ICML 2021) and https://github.com/facebookresearch/barlowtwins

This module implements Barlow Twins-style self-supervised learning that
reduces redundancy in representations while maintaining invariance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BarlowTwinsBaseline(nn.Module):
    """Barlow Twins baseline for multimodal alignment.

    Minimizes redundancy between representations while maintaining
    invariance through cross-correlation matrix regularization.
    """

    def __init__(self, cfg):
        """Initialize Barlow Twins baseline.

        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()
        # Read from config
        img_embed_dim = cfg.models.img_embed_dim
        gex_embed_dim = cfg.models.gex_embed_dim
        projection_dim = cfg.models.projection_dim
        self.lambd = cfg.models.lambda_param

        # Project to shared space
        self.img_proj = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim, bias=False)
        )

        self.gex_proj = nn.Sequential(
            nn.Linear(gex_embed_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim, bias=False)
        )

        # Final normalization layer (no affine)
        self.bn = nn.BatchNorm1d(projection_dim, affine=False)

    def forward(self, img_embed, gex_embed):
        """Forward pass: compute Barlow Twins loss.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Barlow Twins loss (on-diagonal + off-diagonal terms)
        """
        z1 = self.img_proj(img_embed)
        z2 = self.gex_proj(gex_embed)

        z1 = self.bn(z1)
        z2 = self.bn(z2)

        # Cross-correlation matrix
        c = z1.T @ z2 / z1.size(0)

        # Loss terms
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        """Get aligned embeddings for downstream tasks.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Tuple of normalized projected embeddings (z1, z2)
        """
        z1 = F.normalize(self.img_proj(img_embed), dim=-1)
        z2 = F.normalize(self.gex_proj(gex_embed), dim=-1)
        return z1, z2
