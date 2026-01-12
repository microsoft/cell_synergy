import torch
import torch.nn as nn
from torchvision import transforms


def get_default_augmentations():
    """
    Returns a default set of image augmentations.

    Returns:
        torchvision.transforms.Compose: A composition of image augmentations
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
    ])


def get_model_and_transform(model_type='uni'):
    """
    Create a feature extractor and its corresponding transform based on the specified model type.

    Args:
        model_type (str): Type of feature extractor ('uni', 'resnet', or 'ctranspath')

    Returns:
        tuple: (feature_extractor, transform)
    """
    if model_type == 'uni':
        from multimodal_ssl.models.unimodal.img.uni import UNIViT
        # Since the UNI encoder does not have a classification head, we can use the backbone directly
        feature_extractor = UNIViT()
        transform = feature_extractor.transform

    elif model_type == 'resnet':
        from multimodal_ssl.models.unimodal.img.resnet import ResNet50Classifier
        model = ResNet50Classifier()
        # Remove the classification head
        feature_extractor = nn.Sequential(*list(model.backbone.children())[:-1])
        transform = model.transform

        # Add a flatten layer after the feature extractor
        class FeatureExtractorWithFlatten(nn.Module):
            """Wrapper for feature extractors that adds flattening.

            Useful for models that output multi-dimensional features but need
            flattened vectors for downstream tasks.
            """

            def __init__(self, feature_extractor):
                """Initialize feature extractor wrapper.

                Args:
                    feature_extractor: Base feature extractor module
                """
                super(FeatureExtractorWithFlatten, self).__init__()
                self.feature_extractor = feature_extractor
                self.flatten = nn.Flatten(1)  # Flatten all dimensions except batch size

            def forward(self, x_tuple):
                """Forward pass: extract and flatten features.

                Args:
                    x_tuple: Input tensor or tuple

                Returns:
                    Flattened feature vector
                """
                x, _ = x_tuple if isinstance(x_tuple, tuple) else (x_tuple, None)
                x = self.feature_extractor(x)  # Output shape: (batch_size, 2048, 1, 1)
                x = self.flatten(x)  # Flatten to (batch_size, 2048)
                return x

        # Wrap the feature extractor with the flatten operation
        feature_extractor = FeatureExtractorWithFlatten(feature_extractor)

    elif model_type == 'ctranspath':
        from multimodal_ssl.models.unimodal.img.ctranspath import CTransPath
        # Since the CTransPath encoder does not have a classification head, we can use the backbone directly
        feature_extractor = CTransPath()
        transform = feature_extractor.transform

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return feature_extractor, transform


def create_feature_extractor(model_type='uni'):
    """
    Create a feature extractor based on the specified model type.

    Args:
        model_type (str): Type of feature extractor ('uni', 'resnet', or 'ctranspath')

    Returns:
        torch.nn.Module: Feature extraction model
    """
    feature_extractor, _ = get_model_and_transform(model_type)
    return feature_extractor


def compute_cell_type_composition(cell_types, batch_indices, batch_size, num_cell_types):
    """
    Compute the cell type composition vector for each sample.

    Args:
        cell_types (torch.Tensor): Predicted cell types for each real cell with shape (num_real_cells,).
        batch_indices (torch.Tensor): Batch indices of each real cell with shape (num_real_cells,).
        batch_size (int): Number of samples in the batch.
        num_cell_types (int): Number of possible cell types.

    Returns:
        torch.Tensor: Cell type composition vectors with shape (batch_size, num_cell_types).
    """
    device = cell_types.device

    # Initialize cell type counts
    cell_type_counts = torch.zeros(batch_size, num_cell_types, device=device)  # Shape: (batch_size, num_cell_types)

    # Count cell types using scatter_add
    for t in range(num_cell_types):
        type_mask = (cell_types == t)  # Mask for cells of type `t`
        cell_type_counts[:, t].index_add_(0, batch_indices[type_mask], torch.ones_like(
            batch_indices[type_mask], dtype=torch.float32))

    # Normalize counts to get fractions
    total_cells_per_batch = cell_type_counts.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
    total_cells_per_batch = total_cells_per_batch.clamp_min(1.0)  # Avoid division by zero
    cell_type_compositions = cell_type_counts / total_cells_per_batch  # Shape: (batch_size, num_cell_types)

    return cell_type_compositions
