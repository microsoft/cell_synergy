"""SimCLR (Simple Contrastive Learning) for multimodal learning.

Based on: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations"
(ICML 2020) and https://github.com/google-research/simclr

This module implements SimCLR-style contrastive learning for aligning
image and gene expression embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRBaseline(nn.Module):
    """SimCLR baseline for multimodal alignment.

    Uses contrastive learning with temperature-scaled similarity to learn
    aligned representations between image and gene expression embeddings.
    """
    def __init__(
        self,
        cfg,
    ):
        """
        SimCLR baseline model for multimodal learning.

        Args:
            img_embed_dim: Dimension of image embeddings (default: 1024 from UNIViT)
            gex_embed_dim: Dimension of gene expression embeddings (default: 128 from GCN)
            projection_dim: Dimension of the joint projection space
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()

        img_embed_dim = cfg.models.img_embed_dim
        gex_embed_dim = cfg.models.gex_embed_dim
        projection_dim = cfg.models.projection_dim
        temperature = cfg.models.temperature

        # Projection heads
        self.img_projection = nn.Sequential(
            nn.Linear(img_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.gex_projection = nn.Sequential(
            nn.Linear(gex_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.temperature = temperature
        self.projection_dim = projection_dim

    def forward(self, img_embed, gex_embed):
        """Project embeddings and compute similarity matrix"""
        # Get normalized embeddings
        img_features = F.normalize(self.img_projection(img_embed), dim=-1)
        gex_features = F.normalize(self.gex_projection(gex_embed), dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(img_features, gex_features.T) / self.temperature

        return logits

    def compute_loss(self, img_embed, gex_embed):
        """
        Compute SimCLR contrastive loss.
        For paired data, use the diagonal elements as positive pairs.
        """
        batch_size = img_embed.shape[0]
        logits = self(img_embed, gex_embed)
        labels = torch.eye(batch_size, device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels.T)

        return (loss_i2t + loss_t2i) / 2

    def get_embeddings(self, img_embed, gex_embed):
        """Get projected embeddings in shared space"""
        img_features = F.normalize(self.img_projection(img_embed), dim=-1)
        gex_features = F.normalize(self.gex_projection(gex_embed), dim=-1)
        return img_features, gex_features
