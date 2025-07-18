# Based on https://github.com/facebookresearch/simsiam

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
class Predictor(nn.Module):
    def __init__(self, dim, pred_dim=None):
        super().__init__()
        pred_dim = pred_dim or dim // 2
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x):
        return self.predictor(x)

class SimSiamBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        proj_dim = cfg.models.projection_dim
        pred_dim = cfg.models.get('pred_dim', proj_dim // 2)

        self.img_encoder = SimSiamEncoder(
            input_dim=cfg.models.img_embed_dim,
            hidden_dims=[cfg.models.projection_hidden_dim],
            output_dim=proj_dim
        )
        self.gex_encoder = SimSiamEncoder(
            input_dim=cfg.models.gex_embed_dim,
            hidden_dims=[cfg.models.projection_hidden_dim],
            output_dim=proj_dim
        )

        self.img_predictor = Predictor(proj_dim, pred_dim)
        self.gex_predictor = Predictor(proj_dim, pred_dim)

    def forward(self, img_embed, gex_embed):
        z1 = self.img_encoder(img_embed)
        z2 = self.gex_encoder(gex_embed)

        p1 = self.img_predictor(z1)
        p2 = self.gex_predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

    def compute_loss(self, img_embed, gex_embed):
        p1, p2, z1, z2 = self.forward(img_embed, gex_embed)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)

        loss = -(F.cosine_similarity(p1, z2, dim=-1).mean() +
                 F.cosine_similarity(p2, z1, dim=-1).mean()) * 0.5
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        z1 = F.normalize(self.img_encoder(img_embed), dim=-1)
        z2 = F.normalize(self.gex_encoder(gex_embed), dim=-1)
        return z1, z2
