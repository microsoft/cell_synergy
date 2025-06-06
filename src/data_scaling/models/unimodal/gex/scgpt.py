import os
import torch
import anndata
import scgpt
from pathlib import Path

from multimodal_ssl.utils import GENE_NAMES

class ScGPTEmbedder(torch.nn.Module):
    def __init__(
        self,
        gene_col="gene_name",
        batch_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        ScGPT gene expression embedding model.
        
        Args:
            gene_col (str): Column name for gene identifiers
            batch_size (int): Batch size for embedding generation
            device (str): Compute device for embedding generation
        """
        super().__init__()

        working_dir = os.getenv("WORKING_DIR")

        if not working_dir:
            raise ValueError("WORKING_DIR environment variable is not set")

        # Define working directory and local checkpoint path
        local_dir = os.path.join(working_dir, "assets/ckpts/scGPT_human/")
        assert os.path.isdir(local_dir), f"The path '{local_dir}' is not a valid directory."
        
        # Check if directory is empty and provide guidance
        if not os.listdir(local_dir):
            raise ValueError(
                f"The directory '{local_dir}' is empty. "
                "Please download the scGPT model from: "
                "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y "
                "and place the model files in this directory."
            )
        
        self.model_dir = local_dir
        self.gene_col = gene_col
        self.batch_size = batch_size
        self.device = device
    
    def forward(self, x_tuple):
        """
        Generate embeddings for input gene expression data.
        
        Args:
            x_tuple (tuple): Tuple containing:
                - x (torch.Tensor): Input gene expression data with shape (batch_size, num_cells, num_genes)
                - mask (torch.Tensor): Boolean mask with shape (batch_size, num_cells), where True indicates 
                  real cells and False indicates padding cells

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Gene expression embeddings with shape (num_real_cells, embedding_dim)
                - torch.Tensor: Batch indices for the real cells
        """
        x, mask = x_tuple

        # Flatten data and mask
        _, num_cells, num_genes = x.shape
        x_flat = x.view(-1, num_genes)  # Shape: (batch_size * num_cells, num_genes)
        mask_flat = mask.view(-1)  # Shape: (batch_size * num_cells,)

        # Filter real cells using the mask
        x_real = x_flat[mask_flat]

        # Remove empty rows (cells with all-zero values) from x_real
        non_empty_mask = x_real.sum(dim=1) > 0
        x_real = x_real[non_empty_mask]  # Shape: (num_real_cells, num_genes)

        # Ensure x_real has at least one cell
        if x_real.size(0) == 0:
            raise ValueError("No real cells found in the input data. Check your mask or input data.")

        # Create AnnData object
        adata = anndata.AnnData(X=x_real.cpu().numpy())  # Use CPU for AnnData processing
        adata.var[self.gene_col] = GENE_NAMES  # Ensure this is a list of length num_genes

        # Generate embeddings
        embed_adata = scgpt.tasks.embed_data(
            adata,
            self.model_dir,
            gene_col=self.gene_col,
            batch_size=self.batch_size,
            device=self.device,
            use_fast_transformer=False
        )
        embedding_tensor = torch.tensor(embed_adata.obsm['X_scGPT'], device=self.device)  # Shape: (num_real_cells, embedding_dim)

        # Compute batch indices for real cells
        real_cell_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        batch_indices = real_cell_indices // num_cells
        batch_indices = batch_indices[non_empty_mask]  # Shape: (num_real_cells,)

        assert embedding_tensor.size(0) == batch_indices.size(0), \
            f"Embedding tensor size {embedding_tensor.size(0)} does not match batch indices size {batch_indices.size(0)}"

        return (embedding_tensor, batch_indices)

