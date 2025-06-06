import torch
import torch.nn as nn
import torch.nn.functional as F
from comm.models.mmfusion import MMFusion  # Assuming CoMM is installed or in PYTHONPATH

class CoMMBaseline(nn.Module):
    def __init__(
        self,
        img_embed_dim: int = 1024,
        gex_embed_dim: int = 128,
        fusion_dim: int = 256,
        temperature: float = 0.07
    ):
        """
        CoMM-based multimodal model using MMFusion for modality fusion.
        """
        super().__init__()

        # Project to common dimension
        self.img_proj = nn.Linear(img_embed_dim, fusion_dim)
        self.gex_proj = nn.Linear(gex_embed_dim, fusion_dim)

        # CoMM fusion module
        self.fusion = MMFusion(
            input_dims=[fusion_dim, fusion_dim],
            fusion_type='transformer',  # or 'mlp' depending on your needs
            output_dim=fusion_dim
        )

        self.temperature = temperature

    def forward(self, img_embed, gex_embed):
        # Project embeddings
        img_proj = self.img_proj(img_embed)
        gex_proj = self.gex_proj(gex_embed)

        # Fuse modalities
        fused = self.fusion([img_proj, gex_proj])

        # Normalize for contrastive similarity
        fused = F.normalize(fused, dim=-1)
        return fused

    def compute_loss(self, img_embed, gex_embed):
        """
        Contrastive loss using fused embeddings.
        """
        batch_size = img_embed.shape[0]
        fused = self.forward(img_embed, gex_embed)

        # Assume symmetric contrastive loss (like CLIP)
        logits = torch.matmul(fused, fused.T) / self.temperature
        labels = torch.eye(batch_size, device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels.T)

        return (loss_i2t + loss_t2i) / 2

    def get_embeddings(self, img_embed, gex_embed):
        """
        Return fused embeddings in shared space.
        """
        return self.forward(img_embed, gex_embed)
