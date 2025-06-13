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

class AdversarialBaseline:
    def __init__(
        self, 
        cfg,
    ):
        """
        Initialize MultimodalGAN with gene expression and image encoders.

        Args:
            config (Dict[str, Any]): Model configuration
        """
        # Add a flag to track training stage
        self.training_stage = 1
        self.config = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Flag to enable periodic saving
        self.periodic_saving = cfg.get('periodic_saving', False)

        # Set latent dimensions
        # img_latent_dim defaults to UNIViT embedding dimension (1024)
        # gex_latent_dim defaults to GCN_1 embedding dimension (128)
        self.latent_dim_img = cfg.get("img_latent_dim", 1024)
        self.latent_dim_gex = cfg.get("gex_latent_dim", 128)
        self.latent_dim = min(self.latent_dim_img, self.latent_dim_gex)

        # Projection layers to align latent dimensions
        if self.latent_dim_img > self.latent_dim_gex:
            self.img_proj = nn.Linear(self.latent_dim_img, self.latent_dim).to(self.device)
        else:
            self.img_proj = nn.Identity().to(self.device)

        if self.latent_dim_gex > self.latent_dim_img:
            self.gex_proj = nn.Linear(self.latent_dim_gex, self.latent_dim).to(self.device)
        else:
            self.gex_proj = nn.Identity().to(self.device)

        # Shared layers for generators
        shared_encoder = SharedLayer(self.latent_dim // 2, self.latent_dim // 4).to(self.device)
        shared_decoder = SharedLayer(self.latent_dim // 4, self.latent_dim // 2).to(self.device)

        # Generators
        self.gex2img = DeepAE(
            input_dim=self.latent_dim,
            shared_encoder=shared_encoder,
            shared_decoder=shared_decoder
        ).to(self.device)
        self.img2gex = DeepAE(
            input_dim=self.latent_dim,
            shared_encoder=shared_encoder,
            shared_decoder=shared_decoder
        ).to(self.device)

        # Discriminators
        self.D_img = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 4),
            nn.BatchNorm1d(self.latent_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim // 4, 1)
        ).to(self.device)
        self.D_gex = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 4),
            nn.BatchNorm1d(self.latent_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim // 4, 1)
        ).to(self.device)

        # Optimizers
        self._init_optimizers()

    def _init_optimizers(self):
        """Initialize optimizers for generators and discriminators."""
        self.generator_params = itertools.chain(
            self.gex2img.parameters(), 
            self.img2gex.parameters()
        )

        if self.config.get('gan_type', 'wasserstein') == 'wasserstein':
            self.optimizer_D = optim.RMSprop(
                itertools.chain(self.D_img.parameters(), self.D_gex.parameters()),
                lr=self.config.get('lr_d', 5e-5),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            self.optimizer_G = optim.RMSprop(
                self.generator_params,
                lr=self.config.get('lr_g', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            self.optimizer_D = optim.AdamW(
                itertools.chain(self.D_img.parameters(), self.D_gex.parameters()),
                lr=self.config.get('lr_d', 5e-5),
                betas=(self.config.get('b1', 0.5), self.config.get('b2', 0.999)),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            self.optimizer_G = optim.AdamW(
                self.generator_params,
                lr=self.config.get('lr_g', 1e-4),
                betas=(self.config.get('b1', 0.5), self.config.get('b2', 0.999)),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )

    def train(self, epoch: int):
        """
        Train the MultimodalGAN for a single epoch.

        Args:
            epoch (int): Current training epoch
        """
        if self.training_stage == 1:
            # Stage 1: Pre-train on partially paired data
            self.gex2img.train()
            self.img2gex.train()
            self.D_img.train()
            self.D_gex.train()

            train_loader = self.train_loader_stage1
            with tqdm(train_loader, desc="Training", unit="batch") as train_bar:
                for img_embed, gex_embed, labels in train_bar:
                    img_embed = img_embed.to(self.device)
                    gex_embed = gex_embed.to(self.device)
                    labels = labels.to(self.device)

                    # Project embeddings to aligned latent dimension
                    img_embed_proj = self.img_proj(img_embed)
                    gex_embed_proj = self.gex_proj(gex_embed)

                    # Reconstruction and cycle consistency
                    gex2img_recon, _ = self.gex2img(gex_embed_proj)
                    gex_latent_recon, _ = self.img2gex(gex2img_recon)
                    img2gex_recon, _ = self.img2gex(img_embed_proj)
                    img_latent_recon, _ = self.gex2img(img2gex_recon)

                    # Cycle consistency loss
                    img_cycle_loss = F.l1_loss(img_embed_proj, img_latent_recon)
                    gex_cycle_loss = F.l1_loss(gex_embed_proj, gex_latent_recon)
                    recon_loss = (img_cycle_loss + gex_cycle_loss) * self.config.get('lambda1', 1.0)

                    # Latent alignment loss for paired samples
                    if self.config.get('task_type', 'classification') == 'classification':
                        paired_mask = labels[:, 1] == 1  # For classification, paired flag is a single value
                    else:
                        paired_mask = torch.BoolTensor([all(x == 1 for x in sublist) for sublist in labels[:, 1]])  # For regression, paired flag is a sequence
                    
                    if paired_mask.sum() > 0:
                        img_latent = self.img2gex.encoder(img_embed_proj[paired_mask])
                        gex_latent = self.gex2img.encoder(gex_embed_proj[paired_mask])
                        latent_alignment_loss = F.mse_loss(img_latent, gex_latent)
                    else:
                        latent_alignment_loss = torch.tensor(0.0).to(self.device)

                    # Adversarial training logic
                    img_batch_size = img_embed.size(0)
                    gex_batch_size = gex_embed.size(0)

                    img_real = torch.ones(img_batch_size, 1).to(self.device)
                    img_fake = torch.zeros(img_batch_size, 1).to(self.device)
                    gex_real = torch.ones(gex_batch_size, 1).to(self.device)
                    gex_fake = torch.zeros(gex_batch_size, 1).to(self.device)

                    # Train Generator
                    self.optimizer_G.zero_grad()
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
                    
                    G_loss = (
                        recon_loss + 
                        self.config.get('lambda3', 0.5) * d_loss + 
                        self.config.get('lambda_align', 0.1) * latent_alignment_loss
                    )
                    G_loss.backward()
                    self.optimizer_G.step()

                    # Train Discriminator
                    self.optimizer_D.zero_grad()
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
                                self.D_img(gex2img_recon.detach()), gex_fake
                            )
                        ) / 2
                        gex_D_loss = (
                            F.binary_cross_entropy_with_logits(
                                self.D_gex(gex_embed_proj.detach()), gex_real
                            ) + 
                            F.binary_cross_entropy_with_logits(
                                self.D_gex(img2gex_recon.detach()), img_fake
                            )
                        ) / 2
                        D_loss = (img_D_loss + gex_D_loss) * self.config.get('lambda3', 0.5)
                    
                    D_loss.backward()
                    self.optimizer_D.step()

                    # Weight clipping for Wasserstein GAN
                    if self.config.get('gan_type', 'wasserstein') == 'wasserstein':
                        for p in self.D_img.parameters():
                            p.data.clamp_(-self.config.get('clip_value', 0.01), 
                                        self.config.get('clip_value', 0.01))
                        for p in self.D_gex.parameters():
                            p.data.clamp_(-self.config.get('clip_value', 0.01), 
                                        self.config.get('clip_value', 0.01))

                    # Log metrics
                    wandb.log({
                        'epoch': epoch,
                        'stage': 1,
                        'G_loss': G_loss.item(),
                        'D_loss': D_loss.item(),
                        'recon_loss': recon_loss.item(),
                        'img_cycle_loss': img_cycle_loss.item(),
                        'gex_cycle_loss': gex_cycle_loss.item(),
                        'latent_alignment_loss': latent_alignment_loss.item()
                    })

        elif self.training_stage == 2:
            # Stage 2: Fine-tune generators and train task-specific head
            # This will be implemented in the MultimodalTrainer
            pass

        # Optional: Save checkpoint periodically
        if self.periodic_saving and (epoch + 1) % self.config.get('save_freq', 50) == 0:
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int, path: Optional[str] = None):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current training epoch
            path (Optional[str]): Path to save checkpoint
        """
        import wandb

        if path is None:
            working_dir = os.getenv("WORKING_DIR")
            if not working_dir:
                raise ValueError("WORKING_DIR environment variable is not set")

            # Store models in a dedicated directory
            save_dir = os.path.join(working_dir, "assets/ckpts")
            os.makedirs(save_dir, exist_ok=True)
            
            path = os.path.join(
                save_dir, 
                f'multimodal_gan_{wandb.run.id}_epoch_{epoch}.pt'
            )
        
        torch.save({
            'epoch': epoch,
            'training_stage': self.training_stage,
            'gex2img_state_dict': self.gex2img.state_dict(),
            'img2gex_state_dict': self.img2gex.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path (str): Path to checkpoint file
        """
        checkpoint = torch.load(path)
        
        self.training_stage = checkpoint.get('training_stage', 1)
        self.gex2img.load_state_dict(checkpoint['gex2img_state_dict'])
        self.img2gex.load_state_dict(checkpoint['img2gex_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    def set_training_stage(self, stage: int):
        """
        Set the training stage.

        Args:
            stage (int): Training stage (1 or 2)
        """
        if stage not in [1, 2]:
            raise ValueError("Training stage must be 1 or 2")
        self.training_stage = stage
