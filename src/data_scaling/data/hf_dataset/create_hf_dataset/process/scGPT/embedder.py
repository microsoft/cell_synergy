import numpy as np
import torch
import anndata

try:
    import scgpt
except ImportError:
    raise ImportError("Please install scgpt: pip install scgpt")

from ..utils import LUNG_GENE_NAMES, BREAST_GENE_NAMES, THYMUS_GENE_NAMES


class ScGPTEmbedder(torch.nn.Module):
    def __init__(
        self,
        model_dir,
        gene_col="gene_name",
        batch_size=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scgpt = scgpt
        self.model_dir = model_dir
        self.gene_col = gene_col
        self.batch_size = batch_size

    def forward(self, x_real):
        adata = anndata.AnnData(X=x_real.cpu().numpy())
        num_genes = adata.shape[1]
        if num_genes == 343:
            adata.var[self.gene_col] = LUNG_GENE_NAMES
        elif num_genes == 280:
            adata.var[self.gene_col] = BREAST_GENE_NAMES
        elif num_genes == 2000:
            adata.var[self.gene_col] = THYMUS_GENE_NAMES
        else:
            raise NotImplementedError(f"Gene set with {num_genes} genes not supported.")
        embed_adata = self.scgpt.tasks.embed_data(
            adata,
            self.model_dir,
            gene_col=self.gene_col,
            batch_size=self.batch_size,
            device=self.device,
        )
        return torch.tensor(embed_adata.obsm["X_scGPT"], device=self.device)


def compute_scgpt_embeddings(
    gexp,
    cell_mask,
    scgpt_model,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    X = np.array(gexp)
    mask = (
        np.array(cell_mask, dtype=bool)
        if cell_mask is not None
        else (X.sum(axis=1) > 0)
    )
    X_filtered = X[mask]
    if X_filtered.shape[0] == 0:
        print("Warning: No valid cells; using all data.")
        X_filtered = X

    pool_tensor = torch.tensor(X_filtered, dtype=torch.float32).to(scgpt_model.device)
    with torch.no_grad():
        emb_pool = scgpt_model(pool_tensor)
    scgpt_pool = emb_pool.mean(dim=0).cpu().numpy()

    pseudobulk_counts = np.mean(X_filtered, axis=0)
    pseudobulk_tensor = torch.tensor(
        pseudobulk_counts.reshape(1, -1), dtype=torch.float32
    )
    with torch.no_grad():
        emb_pseudobulk = scgpt_model(pseudobulk_tensor.to(device))
    scgpt_pseudobulk = emb_pseudobulk.squeeze(0).cpu().numpy()

    return [scgpt_pool], [scgpt_pseudobulk]


def compute_scgpt_embeddings_batched(
    gexp,
    cell_mask,
    scgpt_model,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Flatten data and mask
    batch_size, num_cells, num_genes = gexp.shape
    x_flat = gexp.view(-1, num_genes)  # Shape: (batch_size * num_cells, num_genes)

    # If no mask is provided, consider all cells as valid
    if cell_mask is None:
        mask_flat = torch.ones(batch_size * num_cells, dtype=torch.bool, device=device)
    else:
        mask_flat = cell_mask.view(-1)  # Shape: (batch_size * num_cells,)

    # Filter real cells using the mask
    x_real = x_flat[mask_flat]

    # Remove empty rows (cells with all-zero values) from x_real
    non_empty_mask = x_real.sum(dim=1) > 0
    x_real = x_real[non_empty_mask]  # Shape: (num_real_cells, num_genes)

    # Ensure x_real has at least one cell
    if x_real.size(0) == 0:
        raise ValueError(
            "No real cells found in the input data. Check your mask or input data."
        )

    # Compute batch indices for real cells
    real_cell_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
    batch_indices = real_cell_indices // num_cells
    batch_indices = batch_indices[non_empty_mask]  # Shape: (num_real_cells,)

    with torch.no_grad():
        emb_pool = scgpt_model(x_real.to(device))  # Shape: (num_real_cells, emb_dim)

    emb_pool = emb_pool.cpu().numpy()
    batch_indices = batch_indices.cpu().numpy()

    # Initialize list of pooled embeddings
    scgpt_pool = []
    for i in range(batch_size):
        indices = np.where(batch_indices == i)[0]
        if len(indices) == 0:
            scgpt_pool.append(np.zeros(emb_pool.shape[1], dtype=np.float32))
        else:
            scgpt_pool.append(np.mean(emb_pool[indices], axis=0).astype(np.float32))

    return scgpt_pool
