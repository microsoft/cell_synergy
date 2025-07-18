# Based on https://github.com/Michaelvll/DeepCCA

import torch
import torch.nn as nn
import numpy as np


class LinearCCA:
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):
        r1, r2 = 1e-4, 1e-4
        m = H1.shape[0]
        o1, o2 = H1.shape[1], H2.shape[1]

        self.m[0] = np.mean(H1, axis=0)
        self.m[1] = np.mean(H2, axis=0)
        H1bar = H1 - self.m[0]
        H2bar = H2 - self.m[1]

        SigmaHat12 = H1bar.T @ H2bar / (m - 1)
        SigmaHat11 = H1bar.T @ H1bar / (m - 1) + r1 * np.eye(o1)
        SigmaHat22 = H2bar.T @ H2bar / (m - 1) + r2 * np.eye(o2)

        D1, V1 = np.linalg.eigh(SigmaHat11)
        D2, V2 = np.linalg.eigh(SigmaHat22)

        Sigma11_rootinv = V1 @ np.diag(D1**-0.5) @ V1.T
        Sigma22_rootinv = V2 @ np.diag(D2**-0.5) @ V2.T

        T = Sigma11_rootinv @ SigmaHat12 @ Sigma22_rootinv
        U, S, Vt = np.linalg.svd(T)

        self.w[0] = Sigma11_rootinv @ U[:, :outdim_size]
        self.w[1] = Sigma22_rootinv @ Vt.T[:, :outdim_size]

    def transform(self, X1, X2):
        Z1 = (X1 - self.m[0]) @ self.w[0]
        Z2 = (X2 - self.m[1]) @ self.w[1]
        return Z1, Z2


class CCABaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.outdim = cfg.models.projection_dim
        self.cca_model = LinearCCA()

    def forward(self, z1, z2):
        # assume z1, z2 are already frozen features
        z1_np, z2_np = z1.detach().cpu().numpy(), z2.detach().cpu().numpy()
        self.cca_model.fit(z1_np, z2_np, self.outdim)
        z1_proj, z2_proj = self.cca_model.transform(z1_np, z2_np)
        z1_proj = torch.from_numpy(z1_proj).to(z1.device)
        z2_proj = torch.from_numpy(z2_proj).to(z2.device)
        loss = -torch.sum(torch.tensor(np.corrcoef(z1_proj.T, z2_proj.T)[self.outdim:, :self.outdim].diagonal()))
        return loss

    def get_embeddings(self, z1, z2):
        z1_np, z2_np = z1.detach().cpu().numpy(), z2.detach().cpu().numpy()
        z1_proj, z2_proj = self.cca_model.transform(z1_np, z2_np)
        return (
            torch.from_numpy(z1_proj).to(z1.device),
            torch.from_numpy(z2_proj).to(z2.device),
        )