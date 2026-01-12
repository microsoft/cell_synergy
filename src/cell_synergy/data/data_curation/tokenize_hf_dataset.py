#!/usr/bin/env python3
"""
Tokenize HuggingFace dataset directly to Merlin format.
Similar to preprocess_lung_data.py but adapted for breast/thymus datasets.
"""
from typing import Dict
from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
import argparse
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gene_metadata(dataset_name: str) -> pd.DataFrame:
    """Load gene metadata - use the same approach as lung."""
    from cell_synergy.paths import ROOT
    gene_metadata_path = ROOT / "data" / "hf_dataset" / "xenium_mean_script.npy"

    if not gene_metadata_path.exists():
        raise FileNotFoundError(f"Gene metadata not found at {gene_metadata_path}")

    logger.info("Loading gene metadata from %s", gene_metadata_path)
    gene_means = np.load(gene_metadata_path)
    logger.info("Loaded gene means with shape: %s", gene_means.shape)

    gene_metadata = pd.DataFrame({
        'gene_id': [f'gene_{i}' for i in range(len(gene_means))],
        'mean_expression': gene_means
    })

    logger.info("Created gene metadata with %s genes", len(gene_metadata))
    return gene_metadata

def create_tokenizer(gene_metadata: pd.DataFrame) -> Dict:
    """Create tokenizer following the Nicheformer approach."""
    logger.info("Creating tokenizer...")

    gene_metadata_sorted = gene_metadata.sort_values('mean_expression', ascending=False)

    vocabulary = {}
    for token_id, (_, row) in enumerate(gene_metadata_sorted.iterrows(), start=1):
        vocabulary[row['gene_id']] = token_id

    special_tokens = {
        'PAD': len(vocabulary),
        'UNK': len(vocabulary) + 1,
        'MASK': len(vocabulary) + 2
    }

    for token_name, token_id in special_tokens.items():
        vocabulary[token_name] = token_id

    logger.info("Created vocabulary with %s tokens", len(vocabulary))
    return vocabulary

def extract_cells_from_hf_dataset(hf_dataset, vocabulary: Dict) -> pd.DataFrame:
    """Extract individual cells from HF dataset and tokenize them."""
    logger.info("Extracting cells from HF dataset...")

    all_cells = []
    cell_ids = []
    sample_ids = []
    gene_ids = []
    token_stats = {'min': float('in'), 'max': 0, 'unique': set()}

    pad_token_id = 0  # Padding token ID

    logger.info("Processing %s patches...", len(hf_dataset))

    for patch_idx in range(len(hf_dataset)):
        if patch_idx % 1000 == 0:
            logger.info("Processed %s/%s patches, extracted %s cells", patch_idx, len(hf_dataset), len(all_cells))

        # Get the gene expression data for this patch
        data = hf_dataset[patch_idx]
        gexp_raw = data['gexp']

        # Convert to numpy array if it's a list
        if isinstance(gexp_raw, list):
            gexp_array = np.array(gexp_raw, dtype=np.float32)
        elif isinstance(gexp_raw, np.ndarray):
            gexp_array = gexp_raw.astype(np.float32)
        else:
            raise ValueError(f"Unexpected gexp type: {type(gexp_raw)}")

        # Convert to tensor
        gexp_tensor = torch.as_tensor(gexp_array)

        # Handle different tensor shapes - get to [num_cells, num_genes]
        if gexp_tensor.dim() == 3 and gexp_tensor.shape[0] == 1:
            gexp_tensor = gexp_tensor.squeeze(0)  # [1, 200, genes] -> [200, genes]
        elif gexp_tensor.dim() == 3:
            gexp_tensor = gexp_tensor[0]  # Take first sample if multiple
        elif gexp_tensor.dim() == 2:
            # Already in [num_cells, num_genes] format
            pass
        else:
            raise ValueError(f"Unexpected gexp tensor shape: {gexp_tensor.shape}")

        # Convert to token indices (argmax = most expressed gene per cell)
        cell_tokens = torch.argmax(gexp_tensor, dim=-1).long()  # [num_cells]

        # Remove padding (cells with token_id = 0)
        valid_mask = (cell_tokens != pad_token_id)
        valid_cells = cell_tokens[valid_mask]

        # Collect token statistics
        if len(valid_cells) > 0:
            token_stats['min'] = min(token_stats['min'], valid_cells.min().item())
            token_stats['max'] = max(token_stats['max'], valid_cells.max().item())
            token_stats['unique'].update(valid_cells.cpu().numpy().tolist())

        # Add each valid cell individually
        for cell_idx, cell_token in enumerate(valid_cells):
            all_cells.append(cell_token.item())
            cell_ids.append(f"patch_{patch_idx}_cell_{cell_idx}")
            sample_ids.append(f"patch_{patch_idx}")
            gene_ids.append(f"gene_{cell_token.item()}")

    logger.info("Extracted %s total cells from %s patches", len(all_cells), len(hf_dataset))
    logger.info("Token statistics: min=%s, max=%s, unique_tokens=%s", token_stats['min'], token_stats['max'], len(token_stats['unique']))

    # Check for data quality issues
    if len(token_stats['unique']) < 100:
        logger.warning("Very few unique tokens detected (%s). This may indicate data preprocessing issues.", len(token_stats['unique']))

    if token_stats['max'] > 20340:
        logger.warning("Token indices exceed model vocabulary size (max=%s, expected<20340)", token_stats['max'])

    # Create DataFrame
    tokenized_data = pd.DataFrame({
        'cell_id': cell_ids,
        'sample_id': sample_ids,
        'token': all_cells,
        'gene_id': gene_ids
    })

    logger.info("Created tokenized data with %s cells", len(tokenized_data))
    logger.info("Token distribution: %s", tokenized_data['token'].value_counts().head())

    return tokenized_data

def create_merlin_datamodule(tokenized_data: pd.DataFrame, output_dir: Path, chunk_size: int = 100000) -> None:
    """Create Merlin datamodule structure following the original Nicheformer approach."""
    logger.info("Creating Merlin datamodule structure...")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Creating Merlin datamodule in: %s", output_dir)

    # Split data into train/val: 90% train, 10% val
    np.random.seed(42)

    # Get unique sample IDs
    sample_ids = tokenized_data['sample_id'].unique()
    np.random.shuffle(sample_ids)

    # Split samples: 90% train, 10% val
    n_samples = len(sample_ids)
    n_train = int(n_samples * 0.9)

    train_samples = sample_ids[:n_train]
    val_samples = sample_ids[n_train:]

    logger.info("Split: %s train, %s val samples", len(train_samples), len(val_samples))

    # Create splits
    splits = {
        'train': tokenized_data[tokenized_data['sample_id'].isin(train_samples)],
        'val': tokenized_data[tokenized_data['sample_id'].isin(val_samples)]
    }

    # Save each split as parquet files
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        # Split into chunks for efficient loading
        n_chunks = (len(split_data) + chunk_size - 1) // chunk_size

        logger.info("Saving %s split in %s chunks...", split_name, n_chunks)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(split_data))

            chunk_data = split_data.iloc[start_idx:end_idx]

            # Save as parquet
            chunk_path = split_dir / f"chunk_{chunk_idx:06d}.parquet"
            chunk_data.to_parquet(chunk_path, index=False)

        logger.info("Saved %s split to %s", split_name, split_dir)

    # Create metadata file
    metadata = {
        'vocabulary_size': len(tokenized_data['token'].unique()),
        'total_cells': len(tokenized_data),
        'train_cells': len(splits['train']),
        'val_cells': len(splits['val']),
        'chunk_size': chunk_size,
        'seed': 42,
        'split_info': {
            'train_val_ratio': '90/10',
            'note': 'Training dataset split into train/val (90/10) for continued training.'
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", metadata_path)
    logger.info("Merlin datamodule created at %s", output_dir)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Tokenize HuggingFace dataset to Merlin format")
    parser.add_argument("--hf-dataset-path", type=str, required=True,
                        help="Path to HuggingFace dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for Merlin datamodule")
    parser.add_argument("--chunk-size", type=int, default=100000,
                        help="Chunk size for parquet files")

    args = parser.parse_args()

    hf_dataset_path = Path(args.hf_dataset_path)
    output_dir = Path(args.output_dir)

    if not hf_dataset_path.exists():
        raise FileNotFoundError(f"HF dataset not found at {hf_dataset_path}")

    logger.info("Loading HF dataset from %s", hf_dataset_path)
    hf_dataset = load_from_disk(str(hf_dataset_path))
    logger.info("Loaded HF dataset with %s samples", len(hf_dataset))

    # Load gene metadata
    dataset_name = hf_dataset_path.parent.name  # Extract dataset name from path
    gene_metadata = load_gene_metadata(dataset_name)

    # Create tokenizer
    vocabulary = create_tokenizer(gene_metadata)

    # Extract and tokenize cells
    tokenized_data = extract_cells_from_hf_dataset(hf_dataset, vocabulary)

    # Create Merlin datamodule
    create_merlin_datamodule(tokenized_data, output_dir, chunk_size=args.chunk_size)

    logger.info("Tokenization completed successfully!")

if __name__ == "__main__":
    main()
