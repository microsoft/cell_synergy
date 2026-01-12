"""SimSiam (Simple Siamese) for multimodal learning.

Based on: Chen & He "Exploring Simple Siamese Representation Learning" (CVPR 2021)
and https://github.com/facebookresearch/simsiam

This module implements SimSiam-style self-supervised learning for aligning
image and gene expression embeddings without negative samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamEncoder(nn.Module):
    """Encoder network for SimSiam.

    Multi-layer encoder with layer normalization and GELU activations.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        """Initialize SimSiam encoder.

        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output projection dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
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
    """Predictor network for SimSiam.

    Predicts one view from another to encourage alignment.
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
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
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


class SimSiamBaseline(nn.Module):
    """SimSiam (Simple Siamese) baseline for multimodal alignment.

    Uses encoder and predictor networks to learn aligned representations
    through stop-gradient and predictor asymmetry.
    """

    def __init__(self, cfg):
        """Initialize SimSiam baseline.

        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()
        proj_dim = cfg.models.projection_dim
        pred_dim = cfg.models.get('pred_dim', proj_dim // 2)

        self.img_encoder = SimSiamEncoder(
            input_dim=cfg.models.img_embed_dim,
            hidden_dims=[cfg.models.projection_hidden_dim],
            output_dim=proj_dim
        )
        self.gex_encoder = SimSiamEncoder(
            input_dim=cfg.models.gex_embed_dim,
            hidden_dims=[cfg.models.projection_hidden_dim],
            output_dim=proj_dim
        )

        self.img_predictor = Predictor(proj_dim, pred_dim)
        self.gex_predictor = Predictor(proj_dim, pred_dim)

    def forward(self, img_embed, gex_embed):
        """Forward pass: get predictions and targets.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Tuple of (p1, p2, z1, z2) where p are predictions and z are targets
        """
        z1 = self.img_encoder(img_embed)
        z2 = self.gex_encoder(gex_embed)

        p1 = self.img_predictor(z1)
        p2 = self.gex_predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

    def compute_loss(self, img_embed, gex_embed):
        """Compute SimSiam loss.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Negative cosine similarity loss
        """
        p1, p2, z1, z2 = self.forward(img_embed, gex_embed)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)

        loss = -(F.cosine_similarity(p1, z2, dim=-1).mean()
                 + F.cosine_similarity(p2, z1, dim=-1).mean()) * 0.5
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        """Get aligned embeddings for downstream tasks.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Concatenated normalized embeddings
        """
        z1 = F.normalize(self.img_encoder(img_embed), dim=-1)
        z2 = F.normalize(self.gex_encoder(gex_embed), dim=-1)
        # Return concatenated embeddings like other multimodal models for consistency
        # This ensures the evaluation pipeline gets the expected 1024D embeddings
        return torch.cat([z1, z2], dim=-1)
