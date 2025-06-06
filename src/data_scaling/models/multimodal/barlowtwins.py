import torch
import torch.nn as nn
import torch.nn.functional as F

class BarlowTwinsBaseline(nn.Module):
    def __init__(
        self,
        img_embed_dim: int = 1024,
        gex_embed_dim: int = 128,
        projection_dim: int = 256,
        lambd: float = 0.0051
    ):
        """
        Barlow Twins-style model for multimodal embeddings.
        """
        super().__init__()
        self.lambd = lambd

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
        z1 = F.normalize(self.img_proj(img_embed), dim=-1)
        z2 = F.normalize(self.gex_proj(gex_embed), dim=-1)
        return z1, z2
