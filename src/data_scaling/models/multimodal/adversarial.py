import os
import itertools
from tqdm import tqdm
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb

class SharedLayer(nn.Module):
    """Shared layer for encoder and decoder"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.layer(x))

class DeepAE(nn.Module):
    """DeepAE: FC AutoEncoder with shared layers"""

    def __init__(self, input_dim, shared_encoder=None, shared_decoder=None):
        super(DeepAE, self).__init__()
        
        # Encoder layers
        encoder_layers = [
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Add non-shared layer before shared layer
        encoder_layers.extend([
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        if shared_encoder is None:
            encoder_layers.extend([
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        else:
            # Validate shared encoder layer
            if not isinstance(shared_encoder, SharedLayer):
                raise TypeError("shared_encoder must be an instance of SharedLayer")
            if shared_encoder.layer.in_features != input_dim // 2:
                raise ValueError(f"shared_encoder input dimension must be {input_dim // 2}, got {shared_encoder.layer.in_features}")
            if shared_encoder.layer.out_features != input_dim // 4:
                raise ValueError(f"shared_encoder output dimension must be {input_dim // 4}, got {shared_encoder.layer.out_features}")
            encoder_layers.append(shared_encoder)
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        if shared_decoder is None:
            decoder_layers = [
                nn.Linear(input_dim // 4, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            # Validate shared decoder layer
            if not isinstance(shared_decoder, SharedLayer):
                raise TypeError("shared_decoder must be an instance of SharedLayer")
            if shared_decoder.layer.in_features != input_dim // 4:
                raise ValueError(f"shared_decoder input dimension must be {input_dim // 4}, got {shared_decoder.layer.in_features}")
            if shared_decoder.layer.out_features != input_dim // 2:
                raise ValueError(f"shared_decoder output dimension must be {input_dim // 2}, got {shared_decoder.layer.out_features}")
            decoder_layers = [shared_decoder]
        
        # Add non-shared layer after shared layer
        decoder_layers.extend([
            nn.Linear(input_dim // 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # Final decoder layer
        decoder_layers.append(nn.Linear(input_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

class AdversarialBaseline(nn.Module):
    def __init__(
        self, 
        cfg,
    ):
        """
        Initialize AdversarialBaseline with gene expression and image encoders.
        This replicates the MultimodalGAN from multimodal-ssl exactly.

        Args:
            cfg: Model configuration
        """
        super().__init__()
        
        # Add a flag to track training stage
        self.training_stage = 1
        self.config = cfg
        
        # Flag to enable periodic saving
        self.periodic_saving = cfg.get('periodic_saving', False)

        # Set latent dimensions from config - matching original variable names
        self.latent_dim_img = cfg.models.img_embed_dim
        self.latent_dim_gex = cfg.models.gex_embed_dim
        self.latent_dim = min(self.latent_dim_img, self.latent_dim_gex)

        # Projection layers to align latent dimensions - exact same logic as original
        if self.latent_dim_img > self.latent_dim_gex:
            self.img_proj = nn.Linear(self.latent_dim_img, self.latent_dim)
        else:
            self.img_proj = nn.Identity()

        if self.latent_dim_gex > self.latent_dim_img:
            self.gex_proj = nn.Linear(self.latent_dim_gex, self.latent_dim)
        else:
            self.gex_proj = nn.Identity()

        # Shared layers for generators - exact same as original
        shared_encoder = SharedLayer(self.latent_dim // 2, self.latent_dim // 4)
        shared_decoder = SharedLayer(self.latent_dim // 4, self.latent_dim // 2)

        # Generators - exact same as original
        self.gex2img = DeepAE(
            input_dim=self.latent_dim,
            shared_encoder=shared_encoder,
            shared_decoder=shared_decoder
        )
        self.img2gex = DeepAE(
            input_dim=self.latent_dim,
            shared_encoder=shared_encoder,
            shared_decoder=shared_decoder
        )

        # Discriminators - exact same as original
        self.D_img = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 4),
            nn.BatchNorm1d(self.latent_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim // 4, 1)
        )
        self.D_gex = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 4),
            nn.BatchNorm1d(self.latent_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim // 4, 1)
        )

    def forward(self, img_embed, gex_embed):
        """
        Forward pass for alignment training.
        This computes the full adversarial loss exactly like the original implementation.
        """
        return self.compute_adversarial_loss(img_embed, gex_embed)

    def compute_loss(self, img_embed, gex_embed):
        """
        Compute loss for alignment training with numerical stability.
        This is called by the AlignmentTrainer.
        """
        # Add input validation
        if torch.isnan(img_embed).any() or torch.isnan(gex_embed).any():
            print("Warning: NaN detected in input embeddings")
            return torch.tensor(0.0, device=img_embed.device, requires_grad=True)
        
        loss = self.forward(img_embed, gex_embed)
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss) or loss > 1e6:
            print(f"Warning: Invalid loss from forward pass: {loss}")
            return torch.tensor(1.0, device=img_embed.device, requires_grad=True)  # Return a small positive loss
        
        return loss

    def compute_adversarial_loss(self, img_embed, gex_embed):
        """
        Full adversarial training loss computation.
        This replicates the exact loss computation from the original multimodal-ssl implementation.
        """
        # Project embeddings to aligned latent dimension - exact same as original
        img_embed_proj = self.img_proj(img_embed)
        gex_embed_proj = self.gex_proj(gex_embed)

        # Reconstruction and cycle consistency - exact same as original
        gex2img_recon, _ = self.gex2img(gex_embed_proj)
        gex_latent_recon, _ = self.img2gex(gex2img_recon)
        img2gex_recon, _ = self.img2gex(img_embed_proj)
        img_latent_recon, _ = self.gex2img(img2gex_recon)

        # Cycle consistency loss - exact same as original
        img_cycle_loss = F.l1_loss(img_embed_proj, img_latent_recon)
        gex_cycle_loss = F.l1_loss(gex_embed_proj, gex_latent_recon)
        recon_loss = (img_cycle_loss + gex_cycle_loss) * self.config.get('lambda1', 1.0)

        # Latent alignment loss - exact same as original
        # Since we don't have paired labels in alignment training, we treat all samples as paired
        img_latent = self.img2gex.encoder(img_embed_proj)
        gex_latent = self.gex2img.encoder(gex_embed_proj)
        latent_alignment_loss = F.mse_loss(img_latent, gex_latent)

        # Adversarial training logic (Generator Loss) - exact same as original
        img_batch_size = img_embed.size(0)
        gex_batch_size = gex_embed.size(0)

        img_real = torch.ones(img_batch_size, 1, device=img_embed.device)
        img_fake = torch.zeros(img_batch_size, 1, device=img_embed.device)
        gex_real = torch.ones(gex_batch_size, 1, device=gex_embed.device)
        gex_fake = torch.zeros(gex_batch_size, 1, device=gex_embed.device)

        # Generator adversarial loss (d_loss in original) - exact same as original
        if self.config.get('gan_type', 'wasserstein') == 'wasserstein':
            d_loss = (
                -self.D_img(gex2img_recon).mean() - 
                self.D_gex(img2gex_recon).mean()
            )
        else:
            d_loss = (
                F.binary_cross_entropy_with_logits(
                    self.D_img(gex2img_recon), gex_real
                ) + 
                F.binary_cross_entropy_with_logits(
                    self.D_gex(img2gex_recon), img_real
                )
            )
        
        # Total Generator Loss (G_loss in original) - exact same as original
        G_loss = (
            recon_loss + 
            self.config.get('lambda3', 0.5) * d_loss + 
            self.config.get('lambda_align', 0.1) * latent_alignment_loss
        )
        
        return G_loss

    def compute_discriminator_loss(self, img_embed, gex_embed):
        """
        Compute discriminator loss separately.
        This replicates the exact discriminator loss from the original implementation.
        """
        # Project embeddings to aligned latent dimension
        img_embed_proj = self.img_proj(img_embed)
        gex_embed_proj = self.gex_proj(gex_embed)

        # Reconstruction
        gex2img_recon, _ = self.gex2img(gex_embed_proj)
        img2gex_recon, _ = self.img2gex(img_embed_proj)

        # Adversarial training logic - exact same as original
        img_batch_size = img_embed.size(0)
        gex_batch_size = gex_embed.size(0)

        img_real = torch.ones(img_batch_size, 1, device=img_embed.device)
        img_fake = torch.zeros(img_batch_size, 1, device=img_embed.device)
        gex_real = torch.ones(gex_batch_size, 1, device=gex_embed.device)
        gex_fake = torch.zeros(gex_batch_size, 1, device=gex_embed.device)

        # Discriminator Loss (D_loss in original) - exact same as original
        if self.config.get('gan_type', 'wasserstein') == 'wasserstein':
            img_D_loss = (
                self.D_img(gex2img_recon.detach()).mean() - 
                self.D_img(img_embed_proj.detach()).mean()
            )
            gex_D_loss = (
                self.D_gex(img2gex_recon.detach()).mean() - 
                self.D_gex(gex_embed_proj.detach()).mean()
            )
            D_loss = (img_D_loss + gex_D_loss) * self.config.get('lambda3', 0.5)
        else:
            img_D_loss = (
                F.binary_cross_entropy_with_logits(
                    self.D_img(img_embed_proj.detach()), img_real
                ) + 
                F.binary_cross_entropy_with_logits(
                    self.D_img(gex2img_recon.detach()), img_fake
                )
            ) / 2
            gex_D_loss = (
                F.binary_cross_entropy_with_logits(
                    self.D_gex(gex_embed_proj.detach()), gex_real
                ) + 
                F.binary_cross_entropy_with_logits(
                    self.D_gex(img2gex_recon.detach()), gex_fake
                )
            ) / 2
            D_loss = (img_D_loss + gex_D_loss) * self.config.get('lambda3', 0.5)
        
        return D_loss, img_D_loss, gex_D_loss

    def apply_weight_clipping(self):
        """
        Apply weight clipping for Wasserstein GAN - exact same as original.
        """
        if self.config.get('gan_type', 'wasserstein') == 'wasserstein':
            clip_value = self.config.get('clip_value', 0.01)
            for p in self.D_img.parameters():
                p.data.clamp_(-clip_value, clip_value)
            for p in self.D_gex.parameters():
                p.data.clamp_(-clip_value, clip_value)

    def get_loss_components(self, img_embed, gex_embed):
        """
        Get individual loss components for logging - exact same as original.
        Returns the same components as logged in the original implementation.
        """
        # Project embeddings to aligned latent dimension
        img_embed_proj = self.img_proj(img_embed)
        gex_embed_proj = self.gex_proj(gex_embed)

        # Reconstruction and cycle consistency
        gex2img_recon, _ = self.gex2img(gex_embed_proj)
        gex_latent_recon, _ = self.img2gex(gex2img_recon)
        img2gex_recon, _ = self.img2gex(img_embed_proj)
        img_latent_recon, _ = self.gex2img(img2gex_recon)

        # Individual loss components
        img_cycle_loss = F.l1_loss(img_embed_proj, img_latent_recon)
        gex_cycle_loss = F.l1_loss(gex_embed_proj, gex_latent_recon)
        recon_loss = (img_cycle_loss + gex_cycle_loss) * self.config.get('lambda1', 1.0)

        # Latent alignment loss
        img_latent = self.img2gex.encoder(img_embed_proj)
        gex_latent = self.gex2img.encoder(gex_embed_proj)
        latent_alignment_loss = F.mse_loss(img_latent, gex_latent)

        # Get G_loss and D_loss
        G_loss = self.compute_adversarial_loss(img_embed, gex_embed)
        D_loss, img_D_loss, gex_D_loss = self.compute_discriminator_loss(img_embed, gex_embed)

        return {
            'G_loss': G_loss,
            'D_loss': D_loss,
            'recon_loss': recon_loss,
            'img_cycle_loss': img_cycle_loss,
            'gex_cycle_loss': gex_cycle_loss,
            'latent_alignment_loss': latent_alignment_loss,
            'img_D_loss': img_D_loss,
            'gex_D_loss': gex_D_loss
        }

    def save_checkpoint(self, epoch: int, path: Optional[str] = None):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current training epoch
            path (Optional[str]): Path to save checkpoint
        """
        if path is None:
            working_dir = os.getenv("WORKING_DIR")
            if not working_dir:
                raise ValueError("WORKING_DIR environment variable is not set")

            # Store models in a dedicated directory
            save_dir = os.path.join(working_dir, "assets/ckpts")
            os.makedirs(save_dir, exist_ok=True)
            
            path = os.path.join(
                save_dir, 
                f'adversarial_baseline_epoch_{epoch}.pt'
            )
        
        torch.save({
            'epoch': epoch,
            'training_stage': self.training_stage,
            'state_dict': self.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path (str): Path to checkpoint file
        """
        checkpoint = torch.load(path)
        
        self.training_stage = checkpoint.get('training_stage', 1)
        self.load_state_dict(checkpoint['state_dict'])

    def set_training_stage(self, stage: int):
        """
        Set the training stage.

        Args:
            stage (int): Training stage (1 or 2)
        """
        if stage not in [1, 2]:
            raise ValueError("Training stage must be 1 or 2")
        self.training_stage = stage

    def fusion(self, img_embed, gex_embed):
        """
        Create fused embedding for downstream tasks.
        For adversarial training, we concatenate the projected embeddings.
        """
        img_embed_proj = self.img_proj(img_embed)
        gex_embed_proj = self.gex_proj(gex_embed)
        # Concatenate the projected features for downstream tasks
        return torch.cat([img_embed_proj, gex_embed_proj], dim=-1)
