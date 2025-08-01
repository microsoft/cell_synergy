import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BYOLEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class Predictor(nn.Module):
    def __init__(self, dim, pred_dim=None):
        super().__init__()
        pred_dim = pred_dim or dim // 2
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.LayerNorm(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x):
        return self.predictor(x)


class BYOLBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        proj_dim = cfg.models.projection_dim
        pred_dim = cfg.models.get("pred_dim", proj_dim // 2)
        hidden_dim = cfg.models.projection_hidden_dim
        self.ema_decay = cfg.models.get("ema_decay", 0.99)

        # Online encoders
        self.img_encoder = BYOLEncoder(cfg.models.img_embed_dim, proj_dim)
        self.gex_encoder = BYOLEncoder(cfg.models.gex_embed_dim, proj_dim)

        # Predictors
        self.img_predictor = Predictor(proj_dim, pred_dim)
        self.gex_predictor = Predictor(proj_dim, pred_dim)

        # Target encoders (EMA)
        self.img_target_encoder = copy.deepcopy(self.img_encoder)
        self.gex_target_encoder = copy.deepcopy(self.gex_encoder)
        for param in self.img_target_encoder.parameters():
            param.requires_grad = False
        for param in self.gex_target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target_networks(self):
        for online, target in zip(self.img_encoder.parameters(), self.img_target_encoder.parameters()):
            target.data = self.ema_decay * target.data + (1 - self.ema_decay) * online.data
        for online, target in zip(self.gex_encoder.parameters(), self.gex_target_encoder.parameters()):
            target.data = self.ema_decay * target.data + (1 - self.ema_decay) * online.data

    def forward(self, img_embed, gex_embed):
        self._update_target_networks()

        z1 = F.normalize(self.img_encoder(img_embed), dim=-1)
        z2 = F.normalize(self.gex_encoder(gex_embed), dim=-1)

        p1 = self.img_predictor(z1)
        p2 = self.gex_predictor(z2)

        with torch.no_grad():
            t1 = F.normalize(self.img_target_encoder(img_embed), dim=-1)
            t2 = F.normalize(self.gex_target_encoder(gex_embed), dim=-1)

        loss = -(F.cosine_similarity(p1, t2.detach(), dim=-1).mean() +
                 F.cosine_similarity(p2, t1.detach(), dim=-1).mean()) * 0.5
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        z1 = F.normalize(self.img_encoder(img_embed), dim=-1)
        z2 = F.normalize(self.gex_encoder(gex_embed), dim=-1)
        return z1, z2
