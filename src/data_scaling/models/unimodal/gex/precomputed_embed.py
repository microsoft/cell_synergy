import torch
import torch.nn as nn

class PrecomputedGEXEmbedder(nn.Module):
    """
    A simple embedder that uses pre-computed gene expression embeddings.
    
    This model assumes embeddings are already available in the dataset,
    eliminating the need for additional embedding computation.
    """
    def __init__(self):
        super().__init__()
        # No additional initialization needed
        pass

    def forward(self, x_tuple):
        """
        Forward pass using pre-computed embeddings.
        """
        x, _ = x_tuple

        # Directly return the pre-computed embeddings from the input
        # Assuming the first dimension is batch_size, and embeddings are already computed
        embeddings = x

        return embeddings
