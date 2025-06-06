import torch
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from typing import Tuple
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

class GraphSequential(nn.Module):
    def __init__(self, *layers):
        super(GraphSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weights=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weights).relu()
        return x

class GCN_1(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        num_hidden_layers: int = 1, 
        type: str = "GCN", 
        num_classes: int = None, 
        predict_nb: bool = False
    ):
        """
        Initialize a Graph Convolutional Neural Network.

        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
            type (str, optional): Type of convolution ("GCN" or "GAT"). Defaults to "GCN".
            num_classes (int, optional): Number of output classes. Defaults to None.
            predict_nb (bool, optional): Flag to predict negative binomial parameters. Defaults to False.
        """
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.type = type
        self.predict_nb = predict_nb
        self.px_r = nn.Parameter(torch.randn(in_channels))

        if type == "GCN":
            layer = GCNConv
        elif type == "GAT":
            layer = GATConv
        else:
            raise ValueError("type must be either GCN or GAT")

        layers = []
        for i in range(num_hidden_layers + 1):
            if i == 0:
                layers.append(layer(in_channels, hidden_channels))
            else:
                layers.append(layer(hidden_channels, hidden_channels))

        self.layers = GraphSequential(*layers)

        # Output layer for negative binomial: 2 channels for mean (mu) and dispersion (theta)
        if predict_nb:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_channels, 2*in_channels),
                nn.Softplus()
            )
        elif num_classes is None:
            self.output_layer = layer(hidden_channels, in_channels)
        else:
            self.output_layer = layer(hidden_channels, num_classes)

        # Add patch-level attention
        self.patch_attention = nn.Sequential(
            nn.Linear(self.hidden_channels, 1),
            nn.Softmax(dim=0)
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor = None) -> Tensor:
        x = self.layers(x, edge_index, edge_weights)
        
        # Predict NB parameters or raw output based on the flag
        if self.predict_nb:
            output = self.output_layer(x)  # Output has shape (batch_size, 2)
            theta = self.px_r.exp()
            mu = output[:, :self.in_channels]
            return mu, theta
        else:
            return self.output_layer(x, edge_index, edge_weights)
        
    def encode(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor = None) -> Tensor:
        """
        Encode the input data into the latent space.
        
        Args:
            x: Input features
            edge_index: Graph edge indices
            edge_weights: Optional edge weights
        
        Returns:
            Tensor: Encoded features
        """
        return self.layers(x, edge_index, edge_weights)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """Load model from checkpoint file"""
        # Load the state dict and get model configuration
        state_dict = torch.load(checkpoint_path, weights_only=True)  # Add weights_only=True for security
        
        # Detect if the saved model used predict_nb by checking for output_layer.0.weight
        predict_nb = 'output_layer.0.weight' in state_dict
        
        # Create model instance with correct configuration
        model = cls(
            in_channels=kwargs.pop('in_channels', 343),
            hidden_channels=kwargs.pop('hidden_channels', 128),
            predict_nb=predict_nb,  # Use detected configuration
            type="GAT",  # The error shows GAT layers in state dict
            num_hidden_layers=3,  # Error shows layers 0-3
            **kwargs
        )
        
        # Load the state dict
        model.load_state_dict(state_dict)
        return model

    def get_patch_embedding(
        self, 
        x: Tensor, 
        edge_index: Tensor, 
        edge_weights: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Get patch-level embedding using attention pooling.

        Args:
            x (Tensor): Input features
            edge_index (Tensor): Graph edge indices
            edge_weights (Tensor, optional): Edge weights. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Patch embedding and attention weights
        """
        # Get cell embeddings
        cell_emb = self.encode(x, edge_index, edge_weights)
        
        # Calculate attention weights
        attention = self.patch_attention(cell_emb)
        
        # Weight and sum cell embeddings
        patch_emb = torch.sum(cell_emb * attention, dim=0)
        
        return patch_emb, attention
    
    def predict_embeddings(
        self,
        adata,
        target_sum,
        radius=20,
        **kwargs
    ):
        """
        Predict cell and patch embeddings using the graph neural network model.

        Args:
            adata (AnnData): AnnData object representing a single patch from the Xenium image 
                             and corresponding gene expression data. 
                             IMPORTANT: Cell coordinates MUST be stored in adata.obsm['spatial'] 
                             as this is where sq.gr.spatial_neighbors looks for coordinates.
            target_sum (float): Target total count for normalization.
            radius (int, optional): Radius for spatial neighborhood graph. Defaults to 20.
            **kwargs: Additional arguments to customize embedding prediction

        Returns:
            AnnData: Updated AnnData object with cell and patch-level embeddings
        """
        # Set model to evaluation mode
        self.eval()
        device = next(self.parameters()).device
        
        # Initialize embedding storage
        embedding_dim = self.hidden_channels
        adata.obsm['X_GEX_enc'] = np.zeros((adata.n_obs, embedding_dim))
        adata.uns['X_GEX_patch'] = {}
        adata.uns['X_GEX_patch_attention'] = {}
        
        with torch.no_grad():
            # Normalize total counts
            sc.pp.normalize_total(adata, target_sum=target_sum)
            
            # Square root transformation
            adata.X = np.sqrt(adata.X.toarray()) + np.sqrt(adata.X.toarray() + 1)
            
            # Create spatial neighborhood graph
            sq.gr.spatial_neighbors(
                adata=adata,
                spatial_key='spatial',
                radius=radius,
                key_added="adjacency_matrix",
                coord_type="generic"
            )
            
            # Convert adjacency matrix to edge index
            edge_index, _ = from_scipy_sparse_matrix(adata.obsp['adjacency_matrix_connectivities'])
            
            # Disregard patches without edges
            if edge_index is None:
                print(f"No neighbouring cells were found, skipping")
                return adata

            # Prepare gene expression data
            gene_expression = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
            
            # Convert to tensor and move to device
            x = torch.tensor(gene_expression, dtype=torch.float32).to(device)
            edge_index = edge_index.to(device)
            
            # Get patch embedding and attention weights
            patch_emb, attention = self.get_patch_embedding(x, edge_index)
            
            # Get cell embeddings
            cell_emb = self.encode(x, edge_index)
            
            # Store embeddings
            cell_idx = pd.Index(adata.obs.index).get_indexer(adata.obs.index)
            adata.obsm['X_GEX_enc'][cell_idx] = cell_emb.cpu().numpy()
            
            adata.uns['X_GEX_patch'] = patch_emb.cpu().numpy()
            adata.uns['X_GEX_patch_attention'] = attention.cpu().numpy()
        
        return adata

def initialize_gcn_model_from_checkpoint(
    checkpoint_path, 
    in_channels=343, 
    hidden_channels=128, 
    num_hidden_layers=3, 
    model_type="GAT", 
    predict_nb=True, 
    **kwargs
):
    """
    Initialize a GCN_1 model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint
        in_channels (int): Number of input channels/features. Defaults to 343.
        hidden_channels (int, optional): Number of hidden channels. Defaults to 128.
        num_hidden_layers (int, optional): Number of hidden layers. Defaults to 3.
        model_type (str, optional): Type of graph convolution. Defaults to "GAT".
        predict_nb (bool, optional): Whether to predict negative binomial parameters. Defaults to True.
        **kwargs: Additional arguments to pass to GCN_1 constructor

    Returns:
        GCN_1: Initialized model loaded from checkpoint
    """
    # Create initial model instance
    model = GCN_1(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_hidden_layers=num_hidden_layers,
        type=model_type,
        predict_nb=predict_nb,
        **kwargs
    )
    
    # Load checkpoint weights
    model = model.load_from_checkpoint(
        checkpoint_path, 
        in_channels=in_channels, 
        hidden_channels=hidden_channels
    )
    
    return model

