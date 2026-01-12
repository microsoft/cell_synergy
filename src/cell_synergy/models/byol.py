"""Bootstrap Your Own Latent (BYOL) for multimodal learning.

Based on: Grill et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
(NeurIPS 2020) and https://github.com/deepmind/deepmind-research/tree/master/byol

This module implements BYOL-style self-supervised learning for aligning
image and gene expression embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BYOLEncoder(nn.Module):
    """Encoder network for BYOL.

    Simple linear projection encoder.
    """

    def __init__(self, input_dim, output_dim):
        """Initialize BYOL encoder.

        Args:
            input_dim: Input embedding dimension
            output_dim: Output projection dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        """Forward pass through encoder.

        Args:
            x: Input embeddings

        Returns:
            Encoded embeddings
        """
        return self.encoder(x)


class Predictor(nn.Module):
    """Predictor network for BYOL.

    Predicts target embeddings from online embeddings.
    """

    def __init__(self, dim, pred_dim=None):
        """Initialize predictor.

        Args:
            dim: Input/output dimension
            pred_dim: Hidden dimension (defaults to dim // 2)
        """
        super().__init__()
        pred_dim = pred_dim or dim // 2
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.LayerNorm(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x):
        """Forward pass through predictor.

        Args:
            x: Input embeddings

        Returns:
            Predicted embeddings
        """
        return self.predictor(x)


class BYOLBaseline(nn.Module):
    """BYOL (Bootstrap Your Own Latent) baseline for multimodal alignment.

    Uses online and target encoders with momentum updates to learn
    aligned representations without negative samples.
    """

    def __init__(self, cfg):
        """Initialize BYOL baseline.

        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()
        # Read from config, fallback to defaults if not present
        img_embed_dim = getattr(cfg.models, 'img_embed_dim', 1536)
        gex_embed_dim = getattr(cfg.models, 'gex_embed_dim', 512)
        projection_dim = getattr(cfg.models, 'projection_dim', 256)
        self.momentum = getattr(cfg.models, 'momentum', 0.996)

        # Create encoders to match checkpoint structure (single linear layer)
        self.img_encoder = BYOLEncoder(img_embed_dim, projection_dim)
        self.gex_encoder = BYOLEncoder(gex_embed_dim, projection_dim)

        # Create predictors (from checkpoint)
        self.img_predictor = Predictor(projection_dim)
        self.gex_predictor = Predictor(projection_dim)

        # Create target encoders (from checkpoint)
        self.img_target_encoder = BYOLEncoder(img_embed_dim, projection_dim)
        self.gex_target_encoder = BYOLEncoder(gex_embed_dim, projection_dim)

    def forward(self, img_embed, gex_embed):
        """Forward pass: compute BYOL loss.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            BYOL loss (MSE between predictions and targets)
        """
        # Use online encoders
        z1 = self.img_encoder(img_embed)
        z2 = self.gex_encoder(gex_embed)

        # Apply predictors
        p1 = self.img_predictor(z1)
        p2 = self.gex_predictor(z2)

        # Use target encoders (no gradients)
        with torch.no_grad():
            z1_target = self.img_target_encoder(img_embed)
            z2_target = self.gex_target_encoder(gex_embed)

        # BYOL loss: MSE between predictions and target projections
        loss1 = F.mse_loss(F.normalize(p1, dim=-1), F.normalize(z2_target, dim=-1))
        loss2 = F.mse_loss(F.normalize(p2, dim=-1), F.normalize(z1_target, dim=-1))

        loss = (loss1 + loss2) / 2
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        """Get aligned embeddings for downstream tasks.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Concatenated aligned embeddings
        """
        # For downstream evaluation, use the trained encoders
        # Return concatenated embeddings like other multimodal methods
        z1 = self.img_encoder(img_embed)
        z2 = self.gex_encoder(gex_embed)

        # Concatenate the embeddings to create a joint representation
        fused = torch.cat([z1, z2], dim=-1)
        return fused
