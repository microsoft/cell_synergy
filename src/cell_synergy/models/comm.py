"""CoMM (Contrastive Multimodal) model for multimodal self-supervised learning.
Based on: https://openreview.net/forum?id=Pe3AxLq6Wf
Code adapted from: https://github.com/Duplums/CoMM

This module wraps the CoMM framework for multimodal learning, which combines
image and gene expression embeddings through a fusion transformer.

This implementation requires the CoMM repository to be available.
Install CoMM via 'pip install -e .' from the CoMM repository, or set the
COMM_PATH environment variable to point to the CoMM repository directory.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add CoMM repository to Python path
# Try environment variable first, then default relative path
comm_path = os.getenv("COMM_PATH")
if comm_path is None:
    # Try relative to repository root
    from cell_synergy.paths import ROOT
    comm_path = ROOT.parent / "CoMM"
    if not comm_path.exists():
        comm_path = None

if comm_path is not None:
    comm_path = Path(comm_path)
    if comm_path.exists() and str(comm_path) not in sys.path:
        sys.path.insert(0, str(comm_path))

# Import CoMM modules after adding to path
# These imports work when CoMM is installed via 'pip install -e .' or when
# the CoMM repository is added to sys.path
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
        """Initialize weights for a transformer block.

        Args:
            block: Transformer block to initialize
        """
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
        """Recursively unwrap Sequential layers from a block.

        Args:
            block: Block to unwrap

        Returns:
            Unwrapped block
        """
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
                # Unwrapped block for proper module registration
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
    """Simple adapter to convert embeddings to tokens.

    Converts flat embeddings to token format expected by transformer models.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """Initialize adapter.

        Args:
            in_dim: Input embedding dimension
            out_dim: Output token dimension
        """
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # If input is [batch, dim], add sequence dimension
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [batch, 1, dim]
        return self.proj(x)


class CoMMBaseline(nn.Module):
    """CoMM (Contrastive Multimodal) baseline model.

    This model uses the CoMM framework for multimodal self-supervised learning,
    combining image and gene expression embeddings through a fusion transformer.
    """

    def __init__(self, cfg):
        """Initialize CoMM baseline model.

        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()
        embed_dim = cfg.models.embed_dim
        img_embed_dim = cfg.models.img_embed_dim
        gex_embed_dim = cfg.models.gex_embed_dim

        if cfg.models.fusion == "x-attn" and cfg.models.pool != "mean":
            print("Switching pool to 'mean' due to x-attn restriction")
            cfg.models.pool = "mean"

        # Dummy encoders (you provide embeddings)
        self.img_encoder = nn.Identity()
        self.gex_encoder = nn.Identity()

        # Project to common dimension
        self.img_adapter = SimpleAdapter(img_embed_dim, embed_dim)
        self.gex_adapter = SimpleAdapter(gex_embed_dim, embed_dim)

        # CoMM encoder using cross-attention
        # Support dropout if specified in config
        dropout = getattr(cfg.models, 'dropout', 0.0)
        self.encoder = MMFusion(
            encoders=[self.img_encoder, self.gex_encoder],
            input_adapters=[self.img_adapter, self.gex_adapter],
            embed_dim=embed_dim,
            fusion=cfg.models.fusion,
            pool=cfg.models.pool,
            n_heads=cfg.models.n_heads,
            n_layers=cfg.models.n_layers,
            dropout=dropout,
        )

        # Unwrap Sequential blocks inside resblocks (if needed)
        # Skip unwrap_blocks for x-attn since it expects list of Sequentials
        if cfg.models.fusion != "x-attn":
            print("Unwrapping blocks")
            self.encoder.fusion_transformer.unwrap_blocks()
        else:
            print("Skipping unwrap_blocks for x-attn")


        # Register resblocks as submodules so they're saved in state_dict
        # For x-attn, resblocks is a list of Sequential containers that need to be registered
        if cfg.models.fusion == "x-attn":
            # Sequential.forward() only accepts one input, but we need to pass (x1, x2, key_padding_mask)
            # So we need to wrap each Sequential in a class that handles multiple arguments
            if isinstance(self.encoder.fusion_transformer.resblocks, list):
                resblocks_list = self.encoder.fusion_transformer.resblocks

                class SequentialWrapper(nn.Module):
                    """Wrapper for Sequential containers to handle multiple input arguments.

                    Each ResidualCrossAttentionBlock takes (x, y, mask) and returns transformed x.
                    The Sequential contains multiple such blocks that process x1 using x2 as context.
                    """

                    def __init__(self, sequential):
                        """Initialize wrapper.

                        Args:
                            sequential: Sequential container to wrap
                        """
                        super().__init__()
                        self.sequential = sequential

                    def forward(self, x1, x2, key_padding_mask=None):
                        """Forward pass through wrapped sequential.

                        Args:
                            x1: First input sequence
                            x2: Second input sequence (context)
                            key_padding_mask: Optional padding mask

                        Returns:
                            Transformed x1
                        """
                        # Forward through each layer in the Sequential
                        # Each layer takes (x1, x2, mask) and returns transformed x1
                        for layer in self.sequential:
                            x1 = layer(x1, x2, key_padding_mask)
                        return x1

                # Wrap each Sequential in a SequentialWrapper and store as ModuleList
                wrapped_blocks = nn.ModuleList([SequentialWrapper(seq) for seq in resblocks_list])
                self.encoder.fusion_transformer.resblocks = wrapped_blocks
                # Wrapped Sequential blocks to handle multiple arguments and registered as ModuleList
        else:
            # For other fusion modes, flatten as before
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

            # Move all blocks explicitly to device
            for i, block in enumerate(flat_blocks):
                # Moving block to device
                block.to(device)

        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # CoMM loss
        self.loss_fn = CoMMLoss(temperature=cfg.models.temperature)

        # Finally move the whole model to device (after all submodules are created)
        self.to(device)
        # Explicitly ensure projector is on device
        # (Sequential modules sometimes don't get moved properly via self.to())
        self.projector = self.projector.to(device)

    def compute_loss(self, img_embed, gex_embed):
        """Compute CoMM contrastive loss.

        Args:
            img_embed: Image embeddings
            gex_embed: Gene expression embeddings

        Returns:
            Contrastive loss scalar
        """
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        img_embed = img_embed.to(device)
        gex_embed = gex_embed.to(device)

        # Our adaptation: Instead of two augmentations, we use the same data twice
        # x1 = [img, gex] and x2 = [img, gex] (same data, two modalities)
        x1 = [img_embed, gex_embed]
        x2 = [img_embed, gex_embed]  # Same data, no augmentation

        # For x-attn fusion, single-modality masks cause issues because x-attn requires 2 modalities
        # So we handle single modalities directly via adapters, and only use fusion for the fused case
        # Generate all possible modality masks: [[True, False], [False, True], [True, True]]
        all_masks = [
            [True, False],   # Image only
            [False, True],   # GEX only
            [True, True]     # Both modalities (fused)
        ]

        # Handle encoding: single modalities use adapters directly, fused uses encoder
        z1 = []
        z2 = []

        # Image only: use adapter directly
        img_only_1 = self.img_adapter(img_embed).squeeze(1)  # [batch, embed_dim]
        img_only_2 = self.img_adapter(img_embed).squeeze(1)
        z1.append(img_only_1)
        z2.append(img_only_2)

        # GEX only: use adapter directly
        gex_only_1 = self.gex_adapter(gex_embed).squeeze(1)  # [batch, embed_dim]
        gex_only_2 = self.gex_adapter(gex_embed).squeeze(1)
        z1.append(gex_only_1)
        z2.append(gex_only_2)

        # Fused: use encoder with both modalities
        fused_1 = self.encoder(x1, mask_modalities=[[True, True]])[0]
        fused_2 = self.encoder(x2, mask_modalities=[[True, True]])[0]
        z1.append(fused_1)
        z2.append(fused_2)

        # Project all embeddings through the projector (matching original CoMM)
        z1_projected = [self.projector(z) for z in z1]
        z2_projected = [self.projector(z) for z in z2]

        # CoMM loss expects:
        # - aug1_embed: List of embeddings from first view (z1)
        # - aug2_embed: List of embeddings from second view (z2)
        # - prototype: Index of fused embedding (last = -1)
        # The loss aligns z1[i] with z2[prototype] and z2[i] with z1[prototype]
        loss_dict = self.loss_fn({
            "aug1_embed": z1_projected,
            "aug2_embed": z2_projected,
            "prototype": -1  # Last element is fused (index -1)
        })

        return loss_dict["loss"]

    def forward(self, img_embed, gex_embed):
        """Forward pass through CoMM model.

        Args:
            img_embed: Image embeddings, shape (batch_size, img_embed_dim)
            gex_embed: Gene expression embeddings, shape (batch_size, gex_embed_dim)

        Returns:
            Contrastive loss value
        """
        """Forward method for direct loss computation."""
        return self.compute_loss(img_embed, gex_embed)

    @torch.no_grad()
    def fusion(self, img_embed, gex_embed):
        """Return fused embedding using x-attn."""
        x = [img_embed, gex_embed]
        fused = self.encoder(x, mask_modalities=[[True, True]])[0]
        return F.normalize(fused, dim=-1)

    @torch.no_grad()
    def get_embeddings(self, img_embed, gex_embed):
        """
        This returns separate projections (img_z, gex_z), NOT the fused embedding.

        For downstream evaluation, use fusion() method instead, which returns the actual
        fused embedding from the transformer encoder. Using get_embeddings() and concatenating
        would make CoMM equivalent to simple concatenation, losing the learned fusion benefits.

        This method exists for compatibility but should NOT be used for evaluation.
        """
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        img_embed = img_embed.to(device)
        gex_embed = gex_embed.to(device)

        img_z = self.projector(self.img_adapter(img_embed).squeeze(1))
        gex_z = self.projector(self.gex_adapter(gex_embed).squeeze(1))
        return F.normalize(img_z, dim=-1), F.normalize(gex_z, dim=-1)
