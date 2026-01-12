"""Canonical Correlation Analysis (CCA) models for multimodal alignment.

Based on: https://github.com/Michaelvll/DeepCCA

This module implements both linear CCA and a PyTorch wrapper for training.
Linear CCA finds linear projections that maximize correlation between two views.
"""
import torch
import torch.nn as nn
import numpy as np


class LinearCCA:
    """Linear Canonical Correlation Analysis implementation.

    Finds linear projections of two views that maximize their correlation.
    """

    def __init__(self):
        """Initialize Linear CCA model.

        Sets up storage for projection weights and mean vectors.
        """
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):
        """Fit CCA model to find optimal projections.

        Args:
            H1: First view data, shape (n_samples, n_features1)
            H2: Second view data, shape (n_samples, n_features2)
            outdim_size: Dimension of the projected space
        """
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
        """Transform data using fitted CCA projections.

        Args:
            X1: First view data to transform
            X2: Second view data to transform

        Returns:
            Tuple of transformed embeddings (Z1, Z2)
        """
        Z1 = (X1 - self.m[0]) @ self.w[0]
        Z2 = (X2 - self.m[1]) @ self.w[1]
        return Z1, Z2


class CCABaseline(nn.Module):
    """PyTorch wrapper for Linear CCA for multimodal alignment.

    This class wraps LinearCCA to enable integration with PyTorch training loops.
    It automatically fits the CCA model on the first forward pass and reuses
    the fitted model for subsequent passes.
    """

    def __init__(self, cfg):
        """Initialize CCA baseline model.

        Args:
            cfg: Configuration object with models.projection_dim
        """
        super().__init__()
        self.outdim = cfg.models.projection_dim
        self.cca_model = LinearCCA()
        self.is_fitted = False

        # Store fitting data for state dict saving/loading
        self._fitted_on_full_dataset = False

    def forward(self, z1, z2):
        """Forward pass: compute CCA loss.

        Args:
            z1: First modality embeddings
            z2: Second modality embeddings

        Returns:
            Negative sum of canonical correlations (loss to minimize)
        """
        # Properly handle CUDA tensors
        z1_np, z2_np = z1.detach().cpu().numpy(), z2.detach().cpu().numpy()

        # Only fit if not already fitted (for efficiency)
        if not self.is_fitted:
            self.cca_model.fit(z1_np, z2_np, self.outdim)
            self.is_fitted = True

        z1_proj, z2_proj = self.cca_model.transform(z1_np, z2_np)
        z1_proj = torch.from_numpy(z1_proj).to(z1.device)
        z2_proj = torch.from_numpy(z2_proj).to(z2.device)

        # Compute correlation properly on CPU tensors
        z1_cpu = z1_proj.detach().cpu().numpy()
        z2_cpu = z2_proj.detach().cpu().numpy()
        corr_matrix = np.corrcoef(z1_cpu.T, z2_cpu.T)
        canonical_corrs = np.diag(corr_matrix[:self.outdim, self.outdim:])
        loss = -torch.sum(torch.from_numpy(canonical_corrs.copy()))  # Use .copy() to avoid warning
        return loss

    def get_embeddings(self, z1, z2):
        """Get aligned embeddings after CCA projection.

        Args:
            z1: First modality embeddings
            z2: Second modality embeddings

        Returns:
            Tuple of (z1_proj, z2_proj) - aligned embeddings in shared space
        """
        z1_np, z2_np = z1.detach().cpu().numpy(), z2.detach().cpu().numpy()

        # Only fit once during training, not during evaluation
        # During evaluation, the model should already be fitted and loaded from checkpoint
        if not self.is_fitted:
            if hasattr(self, '_fitted_on_full_dataset') and self._fitted_on_full_dataset:
                # Model was loaded from checkpoint - should already be fitted
                print("CCA model loaded from checkpoint but not fitted. This shouldn't happen!")
                print(f"Emergency fitting on {z1_np.shape[0]} samples...")
            else:
                # First time fitting (during training)
                print(f"Fitting CCA model on {z1_np.shape[0]} samples...")

            self.cca_model.fit(z1_np, z2_np, self.outdim)
            self.is_fitted = True
            print("CCA model fitted successfully")

        z1_proj, z2_proj = self.cca_model.transform(z1_np, z2_np)

        # Convert back to tensors
        z1_proj_tensor = torch.from_numpy(z1_proj).to(z1.device)
        z2_proj_tensor = torch.from_numpy(z2_proj).to(z2.device)

        # Return separate embeddings for proper canonical correlation evaluation
        return z1_proj_tensor, z2_proj_tensor

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to include CCA fitting parameters."""
        state_dict = super().state_dict(destination, prefix, keep_vars)

        # ALWAYS save CCA status, even if not fitted
        state_dict[prefix + 'is_fitted'] = torch.tensor(self.is_fitted)
        state_dict[prefix + '_fitted_on_full_dataset'] = torch.tensor(getattr(self, '_fitted_on_full_dataset', False))

        # Save CCA model parameters if fitted
        if self.is_fitted and self.cca_model.w[0] is not None:
            state_dict[prefix + 'cca_w0'] = torch.from_numpy(self.cca_model.w[0])
            state_dict[prefix + 'cca_w1'] = torch.from_numpy(self.cca_model.w[1])
            state_dict[prefix + 'cca_m0'] = torch.from_numpy(self.cca_model.m[0])
            state_dict[prefix + 'cca_m1'] = torch.from_numpy(self.cca_model.m[1])
            print(
                f"Saved CCA parameters to state_dict: w0={self.cca_model.w[0].shape}, w1={self.cca_model.w[1].shape}")
        else:
            print(f"CCA not fitted yet, saving empty state_dict (is_fitted={self.is_fitted})")

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to restore CCA fitting parameters."""
        # Extract CCA parameters from state_dict
        cca_keys = ['cca_w0', 'cca_w1', 'cca_m0', 'cca_m1', 'is_fitted', '_fitted_on_full_dataset']
        cca_state = {}
        remaining_state = {}

        for key, value in state_dict.items():
            if any(cca_key in key for cca_key in cca_keys):
                cca_state[key] = value
            else:
                remaining_state[key] = value

        # Load standard PyTorch parameters
        result = super().load_state_dict(remaining_state, strict=strict)

        # Restore CCA model state if available
        if 'cca_w0' in cca_state:
            # Move tensors to CPU before converting to numpy
            self.cca_model.w[0] = cca_state['cca_w0'].cpu().numpy()
            self.cca_model.w[1] = cca_state['cca_w1'].cpu().numpy()
            self.cca_model.m[0] = cca_state['cca_m0'].cpu().numpy()
            self.cca_model.m[1] = cca_state['cca_m1'].cpu().numpy()
            self.is_fitted = bool(cca_state.get('is_fitted', torch.tensor(False)).cpu().item())
            self._fitted_on_full_dataset = bool(
                cca_state.get(
                    '_fitted_on_full_dataset',
                    torch.tensor(False)).cpu().item())
            print("CCA model parameters restored from checkpoint")
        else:
            print("No CCA parameters found in checkpoint - model will need to be fitted")

        return result
