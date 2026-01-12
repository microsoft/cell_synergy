"""Deep Canonical Correlation Analysis (DCCA) implementation.

Based on: Andrew et al. "Deep Canonical Correlation Analysis" (ICML 2013)
and https://github.com/Michaelvll/DeepCCA

This module implements deep CCA using neural networks to learn non-linear
projections that maximize correlation between two views.
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CCALoss:
    """Loss function for Canonical Correlation Analysis.

    Computes the negative correlation between two views as the loss.
    """

    def __init__(self, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        """Initialize CCA loss.

        Args:
            outdim_size: Dimension of the projected space
            use_all_singular_values: If True, use all singular values; else use top outdim_size
            device: Device to run computations on
        """
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2, eps=1e-12):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        Based on https://github.com/Michaelvll/DeepCCA - simplified version without excessive fallbacks.
        """
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-12

        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)
        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.mm(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.mm(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.mm(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        # Add minimal error handling for ill-conditioned matrices (but not excessive fallbacks)
        try:
            [D1, V1] = torch.linalg.eigh(SigmaHat11)
            [D2, V2] = torch.linalg.eigh(SigmaHat22)
        except torch._C._LinAlgError:
            # If eigendecomposition fails, try with slightly stronger regularization
            SigmaHat11_reg = SigmaHat11 + 1e-2 * torch.eye(o1, device=self.device)
            SigmaHat22_reg = SigmaHat22 + 1e-2 * torch.eye(o2, device=self.device)
            [D1, V1] = torch.linalg.eigh(SigmaHat11_reg)
            [D2, V2] = torch.linalg.eigh(SigmaHat22_reg)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]

        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.mm(torch.mm(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.mm(torch.mm(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.mm(torch.mm(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.mm(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.mm(Tval.t(), Tval)
            trace_TT = torch.add(
                trace_TT,
                (torch.eye(
                    trace_TT.shape[0]) *
                    r1).to(
                    self.device))  # regularization for more stability
            try:
                U, V = torch.linalg.eigh(trace_TT)
            except torch._C._LinAlgError:
                # If eigendecomposition fails, try with slightly stronger regularization
                trace_TT_reg = trace_TT + 1e-2 * torch.eye(trace_TT.shape[0], device=self.device)
                U, V = torch.linalg.eigh(trace_TT_reg)
            U = torch.where(U > eps, U, (torch.ones(U.shape) * eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))

        return -corr


class MlpNet(nn.Module):
    """Multi-layer perceptron network for Deep CCA.

    This network projects input features through multiple layers with
    batch normalization and sigmoid activations.
    """

    def __init__(self, layer_sizes, input_size):
        """Initialize MLP network.

        Args:
            layer_sizes: List of hidden layer sizes
            input_size: Input feature dimension
        """
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
                    nn.Sigmoid(),  # Original uses Sigmoid, not LeakyReLU
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

        # Initialize weights for better stability
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization for better stability."""
        for layer in self.layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    """Deep Canonical Correlation Analysis model.

    Uses two separate MLP networks to project each view, then maximizes
    correlation between the projections.
    """

    def __init__(
            self,
            layer_sizes1,
            layer_sizes2,
            input_size1,
            input_size2,
            outdim_size,
            use_all_singular_values,
            device=torch.device('cpu')):
        """Initialize Deep CCA model.

        Args:
            layer_sizes1: Hidden layer sizes for first view network
            layer_sizes2: Hidden layer sizes for second view network
            input_size1: Input dimension for first view
            input_size2: Input dimension for second view
            outdim_size: Output dimension (projected space size)
            use_all_singular_values: Whether to use all singular values in loss
            device: Device to run on
        """
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1)
        self.model2 = MlpNet(layer_sizes2, input_size2)

        self.loss = CCALoss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """Forward pass: project inputs and return for loss computation.

        Args:
            x1: First view embeddings, shape (batch_size, n_features1)
            x2: Second view embeddings, shape (batch_size, n_features2)

        Returns:
            Tuple of (output1, output2) - projected embeddings
        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2


class DCCABaseline(nn.Module):
    """Deep CCA baseline wrapper for multimodal alignment.

    This class wraps DeepCCA to provide a consistent interface with other
    baseline models in the package.
    """

    def __init__(self, cfg):
        """Initialize Deep CCA baseline.

        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()
        # Read from config, fallback to defaults if not present
        img_embed_dim = getattr(cfg.models, 'img_embed_dim', 1536)
        gex_embed_dim = getattr(cfg.models, 'gex_embed_dim', 512)
        projection_dim = getattr(cfg.models, 'projection_dim', 256)
        # hidden_dim = getattr(cfg.models, 'hidden_dim', 512)  # Unused, kept for potential future use

        # Use the original DeepCCA architecture (MlpNet) instead of custom projections
        # Default layer sizes matching the original implementation
        layer_sizes1 = getattr(cfg.models, 'layer_sizes1', [1024, 1024, projection_dim])
        layer_sizes2 = getattr(cfg.models, 'layer_sizes2', [1024, 1024, projection_dim])

        use_all_singular_values = getattr(cfg.models, 'use_all_singular_values', False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dcca_model = DeepCCA(
            layer_sizes1, layer_sizes2, img_embed_dim, gex_embed_dim,
            projection_dim, use_all_singular_values, device
        )

    def forward(self, img_embed, gex_embed):
        """
        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]
        """
        output1, output2 = self.dcca_model(img_embed, gex_embed)
        loss = self.dcca_model.loss(output1, output2)
        return loss

    def get_embeddings(self, img_embed, gex_embed):
        """Return separate embeddings for proper canonical correlation evaluation."""
        with torch.no_grad():
            output1, output2 = self.dcca_model(img_embed, gex_embed)
        return output1, output2
