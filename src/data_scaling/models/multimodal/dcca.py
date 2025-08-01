# Based on https://github.com/Michaelvll/DeepCCA

import torch
import torch.nn as nn
import numpy as np


class CCALoss:
    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2, eps=1e-12):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-12

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
        # assert m > 1

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.mm(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.mm(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.mm(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size(), posInd2.size())

        SigmaHat11RootInv = torch.mm(torch.mm(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.mm(torch.mm(V2, torch.diag(D2 ** -0.5)), V2.t())
        # assert torch.isnan(SigmaHat11RootInv).sum().item() == 0
        # assert torch.isnan(SigmaHat22RootInv).sum().item() == 0

        Tval = torch.mm(torch.mm(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        # assert torch.isnan(Tval).sum().item() == 0

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.mm(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.mm(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.linalg.eigh(trace_TT)
            U = torch.where(U>eps, U, (torch.ones(U.shape)*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1)
        self.model2 = MlpNet(layer_sizes2, input_size2)

        self.loss = CCALoss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2


class DCCABaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.outdim = cfg.models.projection_dim
        # Default layer sizes for the neural networks
        layer_sizes1 = getattr(cfg.models, 'layer_sizes1', [1024, 1024, self.outdim])
        layer_sizes2 = getattr(cfg.models, 'layer_sizes2', [1024, 1024, self.outdim])
        
        # Use the actual embedding dimensions from config
        input_size1 = cfg.models.img_embed_dim  # Should be 1024 for UNI
        input_size2 = cfg.models.gex_embed_dim  # Should be 512 for Nicheformer
        
        use_all_singular_values = getattr(cfg.models, 'use_all_singular_values', False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dcca_model = DeepCCA(
            layer_sizes1, layer_sizes2, input_size1, input_size2, 
            self.outdim, use_all_singular_values, device
        )

    def forward(self, z1, z2):
        # z1, z2 are input features
        output1, output2 = self.dcca_model(z1, z2)
        loss = self.dcca_model.loss(output1, output2)
        return loss

    def get_embeddings(self, z1, z2):
        with torch.no_grad():
            output1, output2 = self.dcca_model(z1, z2)
        return output1, output2