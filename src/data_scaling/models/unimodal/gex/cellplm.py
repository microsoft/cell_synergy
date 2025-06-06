import os
import torch
import anndata
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline

from multimodal_ssl.utils import GENE_NAMES

class CellPLMEmbedder(torch.nn.Module):
    def __init__(
        self, 
        pretrain_version='20231027_85M', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        CellPLM gene expression embedding model.
        
        Args:
            pretrain_version (str): Version of pretrained model
            device (str): Compute device for embedding generation
        """
        super().__init__()

        working_dir = os.getenv("WORKING_DIR")

        if not working_dir:
            raise ValueError("WORKING_DIR environment variable is not set")

        # Define working directory and local checkpoint path
        local_dir = os.path.join(working_dir, "assets/ckpts/CellPLM/")
        assert os.path.isdir(local_dir), f"The path '{local_dir}' is not a valid directory."
        assert os.listdir(local_dir), f"The directory '{local_dir}' is empty."
        
        self.pretrain_version = pretrain_version
        self.pretrain_directory = local_dir
        self.device = device
        
        # Initialize CellPLM pipeline
        self.pipeline = CellEmbeddingPipeline(
            pretrain_prefix=self.pretrain_version,
            pretrain_directory=self.pretrain_directory
        )
    
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
        adata.var.index = GENE_NAMES  # Ensure this is a list of length num_genes

        # Generate embeddings for all real cells
        embedding_tensor = self.pipeline.predict(adata, device=self.device)  # Shape: (num_real_cells, embedding_dim)

        # Compute batch indices for real cells
        real_cell_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        batch_indices = real_cell_indices // num_cells
        batch_indices = batch_indices[non_empty_mask]  # Shape: (num_real_cells,)

        assert embedding_tensor.size(0) == batch_indices.size(0), \
            f"Embedding tensor size {embedding_tensor.size(0)} does not match batch indices size {batch_indices.size(0)}"

        return (embedding_tensor, batch_indices)

