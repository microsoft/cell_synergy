import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatBaseline(nn.Module):
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
        
    def forward(self, img_embed, gex_embed):
        """Project embeddings and compute similarity matrix"""
        return torch.cat([img_embed, gex_embed], dim=-1)
    
    def get_embeddings(self, img_embed, gex_embed):
        """Get projected embeddings in shared space"""
        return F.normalize(torch.cat([img_embed, gex_embed], dim=-1), dim=-1)
