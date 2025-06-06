import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPBaseline(nn.Module):
    def __init__(
        self,
        img_embed_dim: int = 1024,
        gex_embed_dim: int = 128,
        projection_dim: int = 256,
        temperature: float = 0.07
    ):
        """
        CLIP baseline model for multimodal learning.
        
        Args:
            img_embed_dim: Dimension of image embeddings (default: 1024 from UNIViT)
            gex_embed_dim: Dimension of gene expression embeddings (default: 128 from GCN)
            projection_dim: Dimension of the joint projection space
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        
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
        Compute CLIP contrastive loss.
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
