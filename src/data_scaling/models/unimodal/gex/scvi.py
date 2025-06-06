import os
import torch
import scvi
import anndata
import numpy as np
from scipy.sparse import issparse, csr_matrix

from multimodal_ssl.utils import GENE_NAMES

class ScVIEmbedder(torch.nn.Module):
    def __init__(
        self,
        batch_key=None
    ):
        """
        ScVI gene expression embedding model.

        Args:
            batch_key (str, optional): Column name for batch information
        """
        super().__init__()

        self.batch_key = batch_key
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    @staticmethod
    def validate_gene_names(adata):
        """
        Validate that the gene names in the AnnData object match the GENE_NAMES list.
        
        Args:
            adata (anndata.AnnData): Input AnnData object
        
        Raises:
            ValueError: If gene names do not match or are in a different order
        """
        # Convert both to numpy arrays for comparison
        adata_genes = np.array(adata.var.index)
        gene_names = np.array(GENE_NAMES)
        
        # Check if gene names are exactly the same and in the same order
        if not np.array_equal(adata_genes, gene_names):
            raise ValueError(
                "Gene names in the dataset do not match the predefined GENE_NAMES. "
                "Ensure the gene names are in the exact same order and match completely. "
                f"Dataset genes: {adata_genes[:10]}... "
                f"Expected genes: {gene_names[:10]}..."
            )
    
    def train_model(self, x=None):
        """
        Train the scVI model on the input gene expression data.
        
        Args:
            x (anndata.AnnData, optional): Input gene expression data. 
                If None, attempts to load from default dataset location.
        
        Raises:
            ValueError: If no input data is provided and default dataset cannot be loaded.
        
        TODO: Update dataset repository name to reflect current project
        """
        # If no input data provided, attempt to load from default location
        if x is None:
            working_dir = os.getenv("WORKING_DIR")
            if not working_dir:
                raise ValueError("WORKING_DIR environment variable is not set")
            
            dataset_path = os.path.join(working_dir, "data/processed/lung_train.h5ad")
            
            if not os.path.exists(dataset_path):
                raise ValueError(
                    f"Dataset not found at {dataset_path}. "
                    "Please download the dataset from: "
                    "https://huggingface.co/datasets/theislab-multimodal-ssl/human-xenium "
                    "and place it in the data/processed directory."
                )
            
            x = anndata.read_h5ad(dataset_path)
            # IMPORTANT: scVI requires raw count data for accurate modeling
            # Do NOT normalize or transform the data before training
            # Raw count data preserves the original biological variability
            # Normalization and transformations should be done by scVI internally
            x.X = x.layers['counts']

            if issparse(x.X) and not isinstance(x.X, csr_matrix):
                print("Converting X to CSR format...")
                x.X = csr_matrix(x.X)
        
        # Validate gene names before training
        self.validate_gene_names(x)
        
        # Preprocess the gene expression data
        scvi.model.SCVI.setup_anndata(x, batch_key=self.batch_key)
        
        # Initialize and train the scVI model
        self.model = scvi.model.SCVI(x)
        self.model.train()

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
        
        Note:
            Ensure that the var_names (gene names) in the input data 
            are in the SAME ORDER as the data used to train the model.
        """
        # Validate that the model has been trained
        if self.model is None:
            raise ValueError("Model must be trained before generating embeddings. Call train_model() first.")

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

        # Generate embeddings for all real cells
        embedding_tensor = torch.tensor(
            self.model.get_latent_representation(adata), 
            device=self.device
        )  # Shape: (num_real_cells, embedding_dim)

        # Compute batch indices for real cells
        real_cell_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        batch_indices = real_cell_indices // num_cells
        batch_indices = batch_indices[non_empty_mask]  # Shape: (num_real_cells,)

        assert embedding_tensor.size(0) == batch_indices.size(0), \
            f"Embedding tensor size {embedding_tensor.size(0)} does not match batch indices size {batch_indices.size(0)}"

        return (embedding_tensor, batch_indices)
