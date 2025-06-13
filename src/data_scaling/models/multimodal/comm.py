import torch
import torch.nn as nn
import torch.nn.functional as F
from comm.models.mmfusion import MMFusion

class SimpleAdapter(nn.Module):
    """Simple adapter to convert embeddings to tokens."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        # If input is [batch, dim], add sequence dimension
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [batch, 1, dim]
        return self.proj(x)

class CoMMBaseline(nn.Module):
    def __init__(self, cfg):
        """
        CoMM-based multimodal model using MMFusion for modality fusion.
        Works with pre-computed embeddings from image and gene expression data.
        """
        super().__init__()
        
        # Read dimensions from config
        img_embed_dim = cfg.models.img_embed_dim
        gex_embed_dim = cfg.models.gex_embed_dim
        embed_dim = cfg.models.embed_dim
        
        # Create dummy encoders that just return the input (since we already have embeddings)
        self.img_encoder = nn.Identity()
        self.gex_encoder = nn.Identity()
        
        # Create input adapters to convert embeddings to tokens
        self.img_adapter = SimpleAdapter(img_embed_dim, embed_dim)
        self.gex_adapter = SimpleAdapter(gex_embed_dim, embed_dim)
        
        # Initialize MMFusion
        self.fusion = MMFusion(
            encoders=[self.img_encoder, self.gex_encoder],
            input_adapters=[self.img_adapter, self.gex_adapter],
            embed_dim=embed_dim,
            fusion=cfg.models.fusion,
            pool=cfg.models.pool,
            n_heads=cfg.models.n_heads,
            n_layers=cfg.models.n_layers
        )

    def forward(self, img_embed, gex_embed):
        # Pass through fusion model
        fused = self.fusion([img_embed, gex_embed])
        
        # Normalize embeddings for contrastive learning
        fused = F.normalize(fused, dim=-1)
        return fused

    def compute_loss(self, img_embed, gex_embed):
        """
        Compute contrastive loss between the fused embeddings.
        """
        batch_size = img_embed.shape[0]
        fused = self.forward(img_embed, gex_embed)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(fused, fused.T)
        
        # Create labels (diagonal is positive pairs)
        labels = torch.arange(batch_size, device=fused.device)
        
        # Compute cross entropy loss in both directions
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        """
        Return fused embeddings in shared space.
        """
        return self.forward(img_embed, gex_embed) 