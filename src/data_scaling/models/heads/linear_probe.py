import torch
import torch.nn as nn

from multimodal_ssl.models.utils import compute_cell_type_composition

class PatchLinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, task_type='classification'):
        """
        Linear probe for patch embeddings.
        
        Args:
            input_dim (int): Dimensionality of the input feature vector
            num_classes (int): Number of output classes
            task_type (str): Type of task, either 'classification' or 'regression'
        """
        super(PatchLinearProbe, self).__init__()
        
        self.task_type = task_type
        
        # Simple linear classification layer
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the linear probe classifier.
        
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

class CellLinearProbe(nn.Module):
    def __init__(self, cell_embedding_dim, num_cell_types, batch_size=None):
        """
        Linear probe for cell embeddings.
        
        Args:
            cell_embedding_dim (int): Dimensionality of cell-level embeddings
            num_cell_types (int): Number of unique cell types
            batch_size (int, optional): Number of samples in the batch
        """
        super(CellLinearProbe, self).__init__()

        # Store batch size
        self.batch_size = batch_size

        # Linear classification layer for cell type prediction
        self.classifier = nn.Linear(cell_embedding_dim, num_cell_types)
    
    def forward(self, x_tuple):
        """
        Forward pass for cell embedding analysis.
        
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
            num_cell_types=self.classifier.out_features
        )

        return cell_composition
