# HuggingFace DataModule

This module provides PyTorch Lightning DataModules for working with HuggingFace datasets in the data_scaling project.

## Features

- **HFDataModule**: Main PyTorch Lightning DataModule for HuggingFace datasets
- **HFDataset**: PyTorch Dataset wrapper for HuggingFace datasets
- **HFDataModuleWithCollate**: Enhanced DataModule with custom collate function for variable-length data
- **Automatic data splitting**: Support for train/val/test splits
- **Sample filtering**: Filter datasets by sample names or donor IDs
- **Multiple data formats**: Support for gene expression, images, embeddings, and spatial data
- **Configurable**: Easy integration with Hydra configs

## Quick Start

```python
from data_scaling.data.hf_dataset import HFDataModule

# Create data module
datamodule = HFDataModule(
    hf_dataset_name="your_dataset_name",
    sample_names=["VUILD106", "VUILD107MF"],
    batch_size=32,
    num_workers=4,
    val_ratio=0.1
)

# Setup and use with PyTorch Lightning
datamodule.setup(stage="fit")
trainer = pl.Trainer()
trainer.fit(model, datamodule=datamodule)
```

## Configuration

The DataModule supports various configuration options:

```python
datamodule = HFDataModule(
    hf_dataset_name="your_dataset_name",  # Required: HF dataset name
    sample_names=["sample1", "sample2"],  # Optional: filter by sample names
    batch_size=32,                        # Batch size for data loaders
    num_workers=4,                        # Number of worker processes
    cache_dir="/path/to/cache",           # Optional: cache directory
    val_ratio=0.1,                        # Validation split ratio
    test_ratio=0.1,                       # Test split ratio
    seed=42,                              # Random seed for splitting
    return_dict=True,                     # Return dict or tuple format
    pin_memory=True,                      # Pin memory for GPU training
    persistent_workers=True               # Use persistent workers
)
```

## Data Format Support

The DataModule automatically handles various data formats commonly found in biological datasets:

### Gene Expression Data
- `gexp`: Raw gene expression data
- `gexp_embed`: Pre-computed gene expression embeddings
- `gene_expression`: Alternative gene expression field
- `mask` or `gene_mask`: Gene masks for attention mechanisms

### Image Data
- `image`: Raw image data (PIL Image or numpy array)
- `img_embed`: Pre-computed image embeddings
- `image_embed`: Alternative image embedding field

### Spatial Data
- `spatial_coords`: Spatial coordinates
- `coordinates`: Alternative coordinate field

### Labels and Annotations
- `annotation`: Cell type annotations (classification)
- `cell_type`: Alternative cell type field
- `cell_type_ratio`: Cell type ratios (regression)

### Metadata
- `name`: Sample name
- `donor_id`: Donor identifier
- `sample_id`: Sample identifier

## Integration with Hydra Configs

For easy integration with Hydra configurations:

```python
from data_scaling.data.hf_dataset import create_hf_datamodule_from_config

# Create from config
datamodule = create_hf_datamodule_from_config(
    config, 
    sample_names=sample_names,
    val_ratio=0.15
)
```

## Environment Variables

Set the following environment variables for HuggingFace authentication:

```bash
export HF_DATASETS_TOKEN="your_hf_token_here"
# or
export HUGGINGFACE_TOKEN="your_hf_token_here"
```

## Advanced Usage

### Custom Collate Function

For datasets with variable-length sequences:

```python
from data_scaling.data.hf_dataset import HFDataModuleWithCollate

datamodule = HFDataModuleWithCollate(
    hf_dataset_name="your_dataset_name",
    # ... other parameters
)
```

### Custom Transforms

Apply custom transforms to data:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datamodule = HFDataModule(
    hf_dataset_name="your_dataset_name",
    transform=transform
)
```

### Tuple Format

For compatibility with older models expecting tuple format:

```python
datamodule = HFDataModule(
    hf_dataset_name="your_dataset_name",
    return_dict=False  # Returns (gexp, mask, labels) tuple
)
```

## Example with Continued Training

This DataModule is designed to work with the continued training script:

```python
# In continued_training.py
from data_scaling.data.hf_dataset.hf_datamodule import HFDataModule

# Initialize data
datamodule = HFDataModule(
    hf_dataset_name=config.data.hf_dataset_name,
    sample_names=sample_names,
    batch_size=config.training.batch_size,
    num_workers=4,
    cache_dir=None,
)

# Use with trainer
trainer.fit(model=model, datamodule=datamodule)
```

## Logging

The DataModule provides detailed logging information:
- Dataset loading progress
- Sample filtering results
- Dataset sizes and shapes
- Error handling and warnings

## Error Handling

The DataModule includes robust error handling for:
- Missing HuggingFace tokens
- Dataset loading failures
- Invalid configurations
- Network issues

## Dependencies

Required packages (already included in requirements.txt):
- `pytorch-lightning>=2.0.0`
- `datasets>=3.0.0`
- `huggingface-hub`
- `torch`
- `numpy`

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all required packages are installed
2. **Authentication errors**: Set HF_DATASETS_TOKEN environment variable
3. **Dataset not found**: Verify dataset name and access permissions
4. **Memory issues**: Reduce batch_size or num_workers
5. **Slow loading**: Set cache_dir to a fast storage location

### Performance Tips

1. Use `persistent_workers=True` for faster data loading
2. Set appropriate `num_workers` based on your CPU cores
3. Use `pin_memory=True` when training on GPU
4. Cache datasets locally with `cache_dir` parameter
5. Use `HFDataModuleWithCollate` only when needed for variable-length data
