import os
import argparse
from pathlib import Path
import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from nicheformer.data import NicheformerDataset
from nicheformer.models import Nicheformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
# Assuming these exist in your project structure
from data_scaling.paths import PROJECT_DIR 
from data_scaling.config import load_data_splits # Or your equivalent for getting sample lists

pl.seed_everything(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder for HuggingFace dataset details - replace with your actuals
HF_DATASET_NAME = "your-hf-account/your-gex-dataset"
HF_DATASET_COLUMN_SAMPLE_ID = "sample_id" # Column in HF dataset for sample identifier
HF_DATASET_COLUMN_GEX_COUNTS = "gex_counts" # Column for GEX data (e.g., sparse matrix)
HF_DATASET_COLUMN_GEX_GENES = "gex_gene_names" # Column for list of gene names for GEX data

GEX_EMBEDDINGS_DIR = PROJECT_DIR / "unimodal_embeddings" / "gex"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate GEX embeddings using Nicheformer.")
    parser.add_argument("--model_scale", type=str, required=True, help="Model scale identifier (e.g., 'NicheformerBase'). Used for output naming.")
    parser.add_argument("--data_scale", type=str, required=True, choices=["S", "M", "L", "all"], help="Data scale (S, M, L, or all defined in config).")
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "finetune", "test"], help="Dataset split to process.")
    
    parser.add_argument("--reference_var_path", type=str, required=True, help="Path to reference var parquet file (e.g., 'var.parquet').")
    parser.add_argument("--technology_mean_path", type=str, required=True, help="Path to technology mean .npy file (e.g., 'xenium_mean_script.npy').")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Nicheformer model checkpoint (.ckpt).")
    
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for HuggingFace models and datasets.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--max_seq_len", type=int, default=1500, help="Max sequence length for NicheformerDataset.")
    parser.add_argument("--aux_tokens", type=int, default=30, help="Auxiliary tokens for NicheformerDataset.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for NicheformerDataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="PyTorch Lightning precision.")
    parser.add_argument("--embedding_layer", type=int, default=-1, help="Nicheformer layer to extract embeddings from (-1 for last layer).")
    parser.add_argument("--embedding_name_suffix", type=str, default="embeddings", help="Suffix for the embedding key in adata.obsm (e.g., X_niche_{suffix}).")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if output file exists.")
    parser.add_argument("--hf_dataset_token", type=str, default=None, help="HuggingFace token for private datasets.")

    return parser.parse_args()


def create_adata_for_nicheformer(
    gex_counts, gene_names_for_sample, sample_id, config_split_name,
    reference_gene_list, gene_to_token_mapping
):
    """
    Creates an AnnData object for the current sample, aligns its genes to the reference,
    and prepares it for NicheformerDataset.
    Returns two AnnData objects:
    1. adata_orig: AnnData with original sample data, for storing final embeddings.
    2. niche_adata_processed: AnnData processed and aligned, for NicheformerDataset.
    """
    # 1. Create initial AnnData for the current sample
    # Ensure gene_names_for_sample are strings to be used as var_names
    str_gene_names_for_sample = [str(g) for g in gene_names_for_sample]
    adata_var = pd.DataFrame(index=str_gene_names_for_sample)

    num_spots = gex_counts.shape[0] # Assuming gex_counts is [spots, genes]
    obs_names = [f"{sample_id}_{i}" for i in range(num_spots)]
    adata_obs = pd.DataFrame(index=obs_names)
    adata_obs["batch"] = sample_id
    adata_obs["nicheformer_split_original"] = config_split_name 
    adata_obs["modality"] = "spatial_gex" # Example, adjust as needed
    adata_obs["species"] = "human"      # Example, adjust as needed

    adata_orig = ad.AnnData(X=gex_counts, obs=adata_obs, var=adata_var)
    adata_orig.layers["raw_counts"] = adata_orig.X.copy() if hasattr(adata_orig.X, "copy") else adata_orig.X

    # 2. Prepare 'niche_adata' for Nicheformer processing
    niche_adata_prep = adata_orig.copy()

    # 3. Align genes to the reference gene list
    # This ensures niche_adata_prep has exactly the genes from reference_gene_list,
    # in that order, filling with 0 for genes not in the original sample.
    # It also drops genes in the sample that are not in the reference list.
    common_genes = list(set(niche_adata_prep.var_names).intersection(reference_gene_list))
    niche_adata_aligned = niche_adata_prep[:, common_genes].copy()
    niche_adata_aligned = niche_adata_aligned.reindex(columns=reference_gene_list, fill_value=0.0).copy()
    
    # 4. Add token_id
    token_ids = [gene_to_token_mapping.get(gene, -1) for gene in niche_adata_aligned.var_names]
    niche_adata_aligned.var["token_id"] = token_ids

    # 5. Filter out genes with token id -1 (should not happen if reindexed correctly and map is complete)
    niche_adata_processed = niche_adata_aligned[:, niche_adata_aligned.var["token_id"] != -1].copy()

    return adata_orig, niche_adata_processed


def main():
    args = parse_args()

    GEX_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = f"gex_{args.model_scale}_{args.data_scale}_{args.split}.h5ad"
    output_file_path = GEX_EMBEDDINGS_DIR / output_filename

    if output_file_path.exists() and not args.force:
        logger.info(f"Output file {output_file_path} already exists. Skipping. Use --force to overwrite.")
        return

    # Load reference gene list from var.parquet
    try:
        reference_var_df = pd.read_parquet(args.reference_var_path)
        reference_gene_list = reference_var_df.index.tolist() # Assuming gene IDs are in the index
        if not all(isinstance(gene, str) for gene in reference_gene_list):
            logger.warning("Reference gene list contains non-string elements. Casting to string.")
            reference_gene_list = [str(g) for g in reference_gene_list]
    except Exception as e:
        logger.error(f"Failed to load reference genes from {args.reference_var_path}: {e}")
        return
        
    gene_to_token_mapping = {gene: idx for idx, gene in enumerate(reference_gene_list)}
    
    # Load technology mean
    try:
        technology_mean = np.load(args.technology_mean_path)
    except Exception as e:
        logger.error(f"Failed to load technology mean from {args.technology_mean_path}: {e}")
        return

    # Load Nicheformer model
    logger.info(f"Loading Nicheformer model from {args.checkpoint_path}")
    model = Nicheformer.load_from_checkpoint(args.checkpoint_path, strict=False)
    model.eval()

    # Setup PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        logger=False, # Disable PL logging for this script
        enable_checkpointing=False
    )

    # Load HuggingFace dataset
    logger.info(f"Loading HuggingFace dataset: {HF_DATASET_NAME}")
    try:
        # Use token if provided (for private datasets)
        hf_token = args.hf_dataset_token or os.environ.get("HF_DATASETS_TOKEN")
        full_dataset = load_dataset(HF_DATASET_NAME, cache_dir=args.cache_dir, use_auth_token=hf_token)
    except Exception as e:
        logger.error(f"Failed to load dataset {HF_DATASET_NAME}: {e}")
        return

    # Get sample names for the current data_scale and split
    # This assumes load_data_splits returns a dictionary or similar structure
    # And that it's keyed like: all_splits_config[args.split][args.data_scale] -> list of sample names
    try:
        all_splits_config = load_data_splits() # Or however you load your split configuration
        target_sample_ids = all_splits_config[args.split][args.data_scale]
        logger.info(f"Found {len(target_sample_ids)} samples for scale '{args.data_scale}', split '{args.split}'.")
    except KeyError:
        logger.error(f"Could not find sample IDs for scale '{args.data_scale}', split '{args.split}' in configuration.")
        return
    except Exception as e:
        logger.error(f"Error loading data splits: {e}")
        return

    # Filter the dataset
    # Assuming the HF dataset has a 'train', 'test' etc. split, or just one default split (e.g., 'train')
    # Adjust dataset_split_key if your HF dataset uses different split names
    dataset_split_key = 'train' # Default, adjust if HF dataset has 'pretrain', 'test' etc. splits
    if dataset_split_key not in full_dataset:
        logger.error(f"Split '{dataset_split_key}' not found in the loaded HuggingFace dataset. Available: {list(full_dataset.keys())}")
        return
        
    # Filter the dataset by the target_sample_ids
    # This assumes the HF dataset has a column (e.g., HF_DATASET_COLUMN_SAMPLE_ID) with sample identifiers
    try:
        filtered_hf_dataset = full_dataset[dataset_split_key].filter(
            lambda example: example[HF_DATASET_COLUMN_SAMPLE_ID] in target_sample_ids
        )
        logger.info(f"Filtered dataset to {len(filtered_hf_dataset)} samples.")
        if len(filtered_hf_dataset) == 0:
            logger.warning("No samples remaining after filtering. Check sample IDs and dataset content.")
            return
    except Exception as e:
        logger.error(f"Error filtering HuggingFace dataset: {e}")
        return


    all_processed_adata_for_concat = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for sample_data in tqdm(filtered_hf_dataset, desc="Processing samples"):
        sample_id = sample_data[HF_DATASET_COLUMN_SAMPLE_ID]
        gex_counts = sample_data[HF_DATASET_COLUMN_GEX_COUNTS] # Assuming this is a NumPy array or sparse matrix
        # Ensure gex_counts is in a format AnnData can ingest (e.g., np.array, scipy.sparse.csr_matrix)
        if isinstance(gex_counts, list): # Example: if counts are list of lists
            gex_counts = np.array(gex_counts)
            
        gene_names_for_sample = sample_data[HF_DATASET_COLUMN_GEX_GENES] # Assuming list of strings

        adata_orig_sample, niche_adata_for_dataset = create_adata_for_nicheformer(
            gex_counts, gene_names_for_sample, sample_id, args.split,
            reference_gene_list, gene_to_token_mapping
        )
        
        if niche_adata_for_dataset.n_vars == 0:
            logger.warning(f"Sample {sample_id} has no genes left after processing/token mapping. Skipping.")
            continue
        if niche_adata_for_dataset.n_obs == 0:
            logger.warning(f"Sample {sample_id} has no observations. Skipping.")
            continue

        # NicheformerDataset expects split to be 'train', 'val', 'test', or 'predict'
        # Using 'predict' as we are generating embeddings.
        dataset = NicheformerDataset(
            adata=niche_adata_for_dataset,
            technology_mean=technology_mean,
            split="predict", 
            max_seq_len=args.max_seq_len,
            aux_tokens=args.aux_tokens,
            chunk_size=args.chunk_size,
        )

        if len(dataset) == 0:
            logger.warning(f"NicheformerDataset for sample {sample_id} is empty. Skipping.")
            continue

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        logger.debug(f"Extracting embeddings for {sample_id}...")
        sample_embeddings_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Embeddings for {sample_id}", leave=False):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                emb = model.get_embeddings(
                    batch=batch,
                    layer=args.embedding_layer,
                )
                sample_embeddings_list.append(emb.cpu().numpy())
        
        if not sample_embeddings_list:
            logger.warning(f"No embeddings generated for sample {sample_id}. Skipping.")
            continue
            
        concatenated_embeddings = np.concatenate(sample_embeddings_list, axis=0)

        # The embeddings correspond to the cells in niche_adata_for_dataset.obs_names
        # We need to map these back to adata_orig_sample if cell filtering occurred.
        # However, create_adata_for_nicheformer is designed so niche_adata_processed has the same obs as adata_orig.
        # If NicheformerDataset or DataLoader further subsets/reorders observations, this needs careful handling.
        # For now, assume order and number of obs in niche_adata_for_dataset match concatenated_embeddings.
        if concatenated_embeddings.shape[0] == adata_orig_sample.n_obs:
            embedding_key = f"X_niche_{args.embedding_name_suffix}"
            adata_orig_sample.obsm[embedding_key] = concatenated_embeddings
            all_processed_adata_for_concat.append(adata_orig_sample)
        else:
            logger.warning(
                f"Mismatch in embedding count ({concatenated_embeddings.shape[0]}) and original obs count ({adata_orig_sample.n_obs}) for sample {sample_id}. "
                f"This sample's embeddings will be skipped. This might indicate an issue with dataset chunking or processing."
            )


    if not all_processed_adata_for_concat:
        logger.error("No embeddings were successfully generated for any sample. Output file will not be created.")
        return

    logger.info("Concatenating AnnData objects from all processed samples...")
    final_combined_adata = ad.concat(all_processed_adata_for_concat, axis=0, join='outer', merge='unique')
    # 'outer' join for .var to keep all genes, 'unique' merge for .obs, .uns
    # You might need a different merge strategy for .var if there are conflicting columns other than index.

    logger.info(f"Saving combined embeddings to {output_file_path}")
    final_combined_adata.write_h5ad(output_file_path)
    logger.info("Processing complete.")


if __name__ == "__main__":
    main()