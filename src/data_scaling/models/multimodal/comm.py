import torch
import torch.nn as nn
import torch.nn.functional as F
from comm.models.mmfusion import MMFusion
from losses.comm_loss import CoMMLoss

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
        super().__init__()

        embed_dim = cfg.models.embed_dim
        img_embed_dim = cfg.models.img_embed_dim
        gex_embed_dim = cfg.models.gex_embed_dim

        self.img_encoder = nn.Identity()
        self.gex_encoder = nn.Identity()

        # Only simple adapter needed as it already receives embeddings
        self.img_adapter = SimpleAdapter(img_embed_dim, embed_dim)
        self.gex_adapter = SimpleAdapter(gex_embed_dim, embed_dim)

        self.encoder = MMFusion(
            encoders=[self.img_encoder, self.gex_encoder],
            input_adapters=[self.img_adapter, self.gex_adapter],
            embed_dim=embed_dim,
            fusion=cfg.models.fusion,
            pool=cfg.models.pool,
            n_heads=cfg.models.n_heads,
            n_layers=cfg.models.n_layers,
        )

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.loss_fn = CoMMLoss(temperature=cfg.models.temperature)

    def compute_loss(self, img_embed, gex_embed):
        """
        Compute CoMM loss between image and gene expression embeddings.
        Treats img_embed and gex_embed as two augmented views of the same object.
        """
        # Each modality is passed separately as one-view input
        x1 = [gex_embed]   # View 1: gex only
        x2 = [img_embed]   # View 2: image only

        masks1 = [[True]]  # One modality: gex
        masks2 = [[True]]  # One modality: image

        z1 = self.encoder(x1, mask_modalities=masks1)
        z2 = self.encoder(x2, mask_modalities=masks2)

        z1_proj = [self.projector(z) for z in z1]  # List with one tensor
        z2_proj = [self.projector(z) for z in z2]

        # Apply CoMM loss — use prototype index 0 (since we have only one modality per view)
        return self.loss_fn({
            "aug1_embed": z1_proj,
            "aug2_embed": z2_proj,
            "prototype": 0
        })

    def forward(self, batch):
        """
        Original CoMM forward method for batch format input.
        """
        x1, x2 = batch["x1"], batch["x2"]  # Each is a list of tensors: [img_embed, gex_embed]

        masks = self.gen_all_possible_masks(n_mod=len(x1))

        z1 = self.encoder(x1, mask_modalities=masks)  # list of embeddings
        z2 = self.encoder(x2, mask_modalities=masks)

        z1_proj = [self.projector(z) for z in z1]
        z2_proj = [self.projector(z) for z in z2]

        return self.loss_fn({
            "aug1_embed": z1_proj,
            "aug2_embed": z2_proj,
            "prototype": len(z1_proj) - 1  # index of the fused modality
        })

    @torch.no_grad()
    def fusion(self, img_embed, gex_embed):
        """Get fused embedding for downstream tasks."""
        x = [img_embed, gex_embed]
        fused = self.encoder(x, mask_modalities=[[True, True]])[0]
        return F.normalize(fused, dim=-1)

    def get_embeddings(self, img_embed, gex_embed):
        """
        Return separate embeddings in shared space for downstream tasks.
        Compatible with other alignment methods.
        """
        x = [img_embed, gex_embed]
        # Get individual modality embeddings (no fusion)
        masks = [[True, False], [False, True]]  # img only, gex only
        individual_embeds = self.encoder(x, mask_modalities=masks)
        
        img_features = F.normalize(individual_embeds[0], dim=-1)
        gex_features = F.normalize(individual_embeds[1], dim=-1)
        
        return img_features, gex_features

    def gen_all_possible_masks(self, n_mod):
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])  # full fusion mask
        return masks
    

class OLDCoMMBaseline(nn.Module):
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
        self.fusion_module = MMFusion(
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
        fused = self.fusion_module([img_embed, gex_embed])
        
        # Normalize embeddings for contrastive learning
        fused = F.normalize(fused, dim=-1)
        return fused

    def compute_loss(self, img_embed, gex_embed):
        """
        Contrastive loss between projected image and gene expression embeddings.
        Only adapters are trained; MMFusion is *not used* here (as in CoMM).
        """
        batch_size = img_embed.shape[0]

        # Adapters = linear projections into shared space
        img_features = self.img_adapter(img_embed)  # [B, 1, D]
        gex_features = self.gex_adapter(gex_embed)  # [B, 1, D]

        # Drop token dimension and normalize
        img_features = F.normalize(img_features.squeeze(1), dim=-1)  # [B, D]
        gex_features = F.normalize(gex_features.squeeze(1), dim=-1)  # [B, D]

        # Contrastive similarity
        sim_matrix = torch.matmul(img_features, gex_features.T)
        labels = torch.arange(batch_size, device=sim_matrix.device)

        # Symmetric InfoNCE loss
        loss_i2g = F.cross_entropy(sim_matrix, labels)
        loss_g2i = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_i2g + loss_g2i) / 2

    def get_embeddings(self, img_embed, gex_embed):
        """
        Return separate embeddings in shared space for downstream tasks.
        This gives the projected embeddings before fusion.
        """
        img_features = self.img_adapter(img_embed)  # [batch, 1, embed_dim]
        gex_features = self.gex_adapter(gex_embed)  # [batch, 1, embed_dim]
        
        # Remove sequence dimension and normalize
        img_features = F.normalize(img_features.squeeze(1), dim=-1)
        gex_features = F.normalize(gex_features.squeeze(1), dim=-1)
        
        return img_features, gex_features

    def fusion(self, img_embed, gex_embed):
        """
        Create fused embedding for downstream tasks using MMFusion.
        This is the original forward pass that combines modalities.
        """
        return self.forward(img_embed, gex_embed) 