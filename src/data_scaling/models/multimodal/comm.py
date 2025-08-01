import torch
import torch.nn as nn
import torch.nn.functional as F
from comm.models.mmfusion import MMFusion, FusionTransformer
from losses.comm_loss import CoMMLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Monkey patch the fusion behavior to define custom augmentations
def patched_forward_augments(self, img_proj, gex_proj):
    """
    Constructs two different augmented embeddings (aug1, aug2) 
    using different combinations of modality-specific and fused embeddings.
    """
    # First fused embedding (acts like an 'augmentation')
    fused_1 = self.fuse_modalities(img_proj, gex_proj)  # you could use self.fusion_model or similar here
    # Second fused embedding with flipped input order (optional)
    fused_2 = self.fuse_modalities(gex_proj, img_proj)

    # Augmentation 1: e.g. [img_proj, fused_1]
    aug1_embed = torch.cat([img_proj, fused_1], dim=-1)
    # Augmentation 2: e.g. [gex_proj, fused_2]
    aug2_embed = torch.cat([gex_proj, fused_2], dim=-1)

    return aug1_embed, aug2_embed

def patched_initialize(self):
    """Corrected initialization method for FusionTransformer layers."""
    proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
    attn_std = self.width ** -0.5
    fc_std = (2 * self.width) ** -0.5

    def init_block(block):
        if hasattr(block, "attn"):
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        if hasattr(block, "mlp"):
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    if isinstance(self.resblocks, nn.Sequential):  # For "concat"
        for block in self.resblocks:
            init_block(block)
    elif isinstance(self.resblocks, list):  # For "x-attn"
        for block_seq in self.resblocks:
            for block in block_seq:
                init_block(block)

def unwrap_sequential_blocks(self):
    """
    Recursively unwrap all nn.Sequential layers from resblocks.
    Supports both list and nn.Sequential container formats.
    Ensures all blocks are callable with (x1, x2, ...) interface.
    """
    def unwrap(block):
        if isinstance(block, nn.Sequential):
            if len(block) == 1:
                return unwrap(block[0])  # keep unwrapping recursively
            else:
                raise ValueError(f"[ERROR] Sequential with >1 block: {block}")
        return block

    if isinstance(self.resblocks, list):
        unwrapped = []
        for i, block in enumerate(self.resblocks):
            try:
                unwrapped_block = unwrap(block)
                print(f"[DEBUG] Unwrapped block {i}: {type(block)} -> {type(unwrapped_block)}")
                unwrapped.append(unwrapped_block)
            except ValueError as e:
                print(e)
                raise
        self.resblocks = unwrapped

    elif isinstance(self.resblocks, nn.Sequential):
        self.resblocks = nn.Sequential(*[unwrap(b) for b in self.resblocks])

    else:
        raise TypeError("resblocks must be list or nn.Sequential")


FusionTransformer.unwrap_blocks = unwrap_sequential_blocks


FusionTransformer.initialize = patched_initialize

# Apply the monkey patch
MMFusion.forward_augments = patched_forward_augments

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

        if cfg.models.fusion == "x-attn" and cfg.models.pool != "mean":
            print("⚠️ Switching pool to 'mean' due to x-attn restriction")
            cfg.models.pool = "mean"

        # Dummy encoders (you provide embeddings)
        self.img_encoder = nn.Identity()
        self.gex_encoder = nn.Identity()

        # Project to common dimension
        self.img_adapter = SimpleAdapter(img_embed_dim, embed_dim)
        self.gex_adapter = SimpleAdapter(gex_embed_dim, embed_dim)

        # CoMM encoder using cross-attention
        self.encoder = MMFusion(
            encoders=[self.img_encoder, self.gex_encoder],
            input_adapters=[self.img_adapter, self.gex_adapter],
            embed_dim=embed_dim,
            fusion=cfg.models.fusion,
            pool=cfg.models.pool,
            n_heads=cfg.models.n_heads,
            n_layers=cfg.models.n_layers,
        )
        
        # ✅ Unwrap Sequential blocks inside resblocks (if needed)
        self.encoder.fusion_transformer.unwrap_blocks()

        print(f"[DEBUG] Fusion resblock types:")
        for i, block in enumerate(self.encoder.fusion_transformer.resblocks):
            print(f"  Block {i}: {type(block)}")


        # ✅ Normalize format: flatten resblocks to a list of blocks
        flat_blocks = []
        resblocks = self.encoder.fusion_transformer.resblocks

        if isinstance(resblocks, nn.Sequential):
            flat_blocks = list(resblocks)
        elif isinstance(resblocks, list):
            for entry in resblocks:
                if isinstance(entry, nn.Sequential):
                    flat_blocks.extend(entry)
                else:
                    flat_blocks.append(entry)
        else:
            raise TypeError("resblocks must be a list or nn.Sequential")

        # ✅ Move all blocks explicitly to device
        for i, block in enumerate(flat_blocks):
            print(f"[DEBUG] Moving block {i} to {device}")
            block.to(device)

        # ✅ Finally move the whole model to device
        self.to(device)


        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # CoMM loss
        self.loss_fn = CoMMLoss(temperature=cfg.models.temperature)

    def compute_loss(self, img_embed, gex_embed):
        # Fused view via encoder
        fused = self.encoder([img_embed, gex_embed], mask_modalities=[[True, True]])[0]
        fused_proj = self.projector(fused)

        # Individual views directly via adapters
        img_proj = self.projector(self.img_adapter(img_embed).squeeze(1))
        gex_proj = self.projector(self.gex_adapter(gex_embed).squeeze(1))

        loss_dict = self.loss_fn({
            "aug1_embed": [img_proj, gex_proj],
            "aug2_embed": [fused_proj, fused_proj],  # One prototype per aug1
            "prototype": 0  # Not really used if both aug2 are the same
        })

        return loss_dict["loss"]

    @torch.no_grad()
    def fusion(self, img_embed, gex_embed):
        """Return fused embedding using x-attn."""
        x = [img_embed, gex_embed]
        fused = self.encoder(x, mask_modalities=[[True, True]])[0]
        return F.normalize(fused, dim=-1)

    @torch.no_grad()
    def get_embeddings(self, img_embed, gex_embed):
        """You cannot use encoder masking with x-attn — return projected inputs."""
        img_z = self.projector(self.img_adapter(img_embed).squeeze(1))
        gex_z = self.projector(self.gex_adapter(gex_embed).squeeze(1))
        return F.normalize(img_z, dim=-1), F.normalize(gex_z, dim=-1)
 