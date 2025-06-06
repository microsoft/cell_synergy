import torch
import torch.nn as nn

from multimodal_ssl.models.utils import compute_cell_type_composition

class PatchMLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate, task_type='classification'):
        """
        Multi-layer Perceptron (MLP) for patch embeddings.
        
        Args:
            input_dim (int): Dimensionality of the input feature vector
            hidden_dims (list): List of hidden layer dimensions
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
            task_type (str): Type of task, either 'classification' or 'regression'
        """
        super(PatchMLPHead, self).__init__()
        
        self.task_type = task_type
        
        # Create layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final classification/regression layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the MLP classifier.
        
        Args:
            x (torch.Tensor): Input feature tensor
        
        Returns:
            torch.Tensor: Class logits or normalized regression values
        """
        output = self.classifier(x)
        
        if self.task_type == 'regression':
            # Normalize output to sum to 1 for regression tasks
            output = torch.softmax(output, dim=-1)
        
        return output

class CellMLPHead(nn.Module):
    def __init__(self, cell_embedding_dim, hidden_dims, num_cell_types, dropout_rate, batch_size=None):
        """
        Multi-layer Perceptron (MLP) for cell embeddings.
        
        Args:
            cell_embedding_dim (int): Dimensionality of cell-level embeddings
            hidden_dims (list): List of hidden layer dimensions
            num_cell_types (int): Number of unique cell types
            dropout_rate (float): Dropout rate
            batch_size (int, optional): Number of samples in the batch
        """
        super(CellMLPHead, self).__init__()

        # Store batch size
        self.batch_size = batch_size

        # Create layers dynamically based on hidden_dims
        layers = []
        prev_dim = cell_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_cell_types))

        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x_tuple):
        """
        Forward pass for cell composition regression.
        
        Args:
            x_tuple (tuple): Tuple containing:
                - cell_embeddings (torch.Tensor): Cell-level embeddings with shape (num_real_cells, embedding_dim)
                - batch_indices (torch.Tensor): Batch indices of each real cell with shape (num_real_cells,)
        
        Returns:
            torch.Tensor: Cell composition fractions of shape (batch_size, num_cell_types)
        """
        cell_embeddings, batch_indices = x_tuple
        
        # Use pre-defined batch_size if available, otherwise compute from batch_indices
        if self.batch_size is None:
            batch_size = batch_indices.max().item() + 1
        else:
            batch_size = self.batch_size

        # Predict cell type for each cell embedding
        cell_type_predictions = torch.argmax(self.classifier(cell_embeddings), dim=1)

        # Compute cell type composition
        cell_composition = compute_cell_type_composition(
            cell_types=cell_type_predictions, 
            batch_indices=batch_indices, 
            batch_size=batch_size, 
            num_cell_types=self.classifier[-1].out_features
        )
        
        return cell_composition

