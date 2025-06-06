import torch
import torch.nn as nn

class EmbeddingAggregationHead(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        batch_size: int,
        aggregation_strategy: str = 'mean'
    ):
        """
        A head for aggregating cell-level embeddings into sample-level embeddings.
        
        Args:
            embedding_dim (int): Dimensionality of the input cell embeddings
            batch_size (int): Number of samples in the batch
            aggregation_strategy (str): Strategy for aggregating embeddings 
                                        ('mean', 'weighted_mean', 'max')
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.aggregation_strategy = aggregation_strategy
        
        # Optional learnable weights for weighted aggregation
        if aggregation_strategy == 'weighted_mean':
            self.weights = nn.Parameter(torch.ones(embedding_dim))
    
    def forward(self, x_tuple) -> torch.Tensor:
        """
        Aggregate cell-level embeddings to generate sample-level embeddings.

        Args:
            x_tuple (tuple): Tuple containing:
                - cell_embeddings (torch.Tensor): Cell-level embeddings with shape (num_real_cells, embedding_dim)
                - batch_indices (torch.Tensor): Batch indices of each real cell with shape (num_real_cells,)

        Returns:
            torch.Tensor: Aggregated sample-level embeddings with shape (batch_size, embedding_dim)
        """
        cell_embeddings, batch_indices = x_tuple
        device = cell_embeddings.device
        embedding_dim = cell_embeddings.shape[1]
        
        # Initialize aggregation tensor
        batch_embeddings = torch.zeros(self.batch_size, embedding_dim, device=device)  # Shape: (batch_size, embedding_dim)
        batch_counts = torch.zeros(self.batch_size, device=device)  # Shape: (batch_size,)
        
        if self.aggregation_strategy == 'mean':
            # Simple mean aggregation
            batch_embeddings.index_add_(0, batch_indices, cell_embeddings)  # Sum embeddings for each batch
            batch_counts.index_add_(0, batch_indices, torch.ones_like(batch_indices, dtype=torch.float32))  # Count cells per batch
            
            # Avoid division by zero
            batch_counts = batch_counts.clamp_min(1.0)
            batch_embeddings /= batch_counts.unsqueeze(1)  # Average embeddings for each batch
        
        elif self.aggregation_strategy == 'weighted_mean':
            # Weighted mean aggregation with learnable weights
            weighted_embeddings = cell_embeddings * self.weights
            batch_embeddings.index_add_(0, batch_indices, weighted_embeddings)
            batch_counts.index_add_(0, batch_indices, torch.ones_like(batch_indices, dtype=torch.float32))
            
            # Avoid division by zero
            batch_counts = batch_counts.clamp_min(1.0)
            batch_embeddings /= batch_counts.unsqueeze(1)
        
        elif self.aggregation_strategy == 'max':
            # Max pooling aggregation
            batch_embeddings, _ = torch.ops.scatter_max(
                batch_embeddings, 
                batch_indices, 
                cell_embeddings
            )
        
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation_strategy}")
        
        return batch_embeddings  # Shape: (batch_size, embedding_dim)
