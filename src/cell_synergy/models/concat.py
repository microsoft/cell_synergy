"""Concatenation baseline for multimodal learning.

This module implements a simple baseline that concatenates image and gene expression
embeddings and projects them to a shared space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatBaseline(nn.Module):
    """Simple concatenation baseline for multimodal embeddings.

    This baseline naively concatenates image and gene expression embeddings
    and projects them to a shared space. Used as a simple baseline comparison.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        Concat baseline model for multimodal learning. Naive concatenation of image and gene expression embeddings.

        Args:
            img_embed_dim: Dimension of image embeddings (default: 1024 from UNIViT)
            gex_embed_dim: Dimension of gene expression embeddings (default: 128 from GCN)
            projection_dim: Dimension of the joint projection space
        """
        super().__init__()

        img_embed_dim = cfg.models.img_embed_dim
        gex_embed_dim = cfg.models.gex_embed_dim
        projection_dim = cfg.models.projection_dim

        # Concatenate embeddings
        self.concat = nn.Linear(img_embed_dim + gex_embed_dim, projection_dim)

        self.projection_dim = projection_dim

    def forward(self, img_embed, gex_embed):
        """Project embeddings and compute MSE loss"""
        # Get normalized embeddings
        features = F.normalize(self.concat(torch.cat([img_embed, gex_embed], dim=-1)), dim=-1)

        # Compute MSE loss between concatenated features and a target (identity)
        target = torch.zeros_like(features)
        loss = F.mse_loss(features, target)
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        """Get projected embeddings in shared space"""
        # Return separate projections for each modality
        img_proj = F.normalize(self.concat(torch.cat([img_embed, torch.zeros_like(gex_embed)], dim=-1)), dim=-1)
        gex_proj = F.normalize(self.concat(torch.cat([torch.zeros_like(img_embed), gex_embed], dim=-1)), dim=-1)
        return img_proj, gex_proj
