#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import numba
import scanpy as sc
from tqdm import tqdm
import pickle
from os.path import join
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from dask.distributed import LocalCluster, Client
# Import tokenization definitions if available, otherwise define them here
HUMAN = 1
MOUSE = 2
RAT = 3
OTHER_SPECIES = 4

DISSOCIATED = 1
SPATIAL = 2

# Constants
CHUNK_SIZE = 10000  # Process chunks of cells at a time
MAX_SEQ_LEN = 4096
AUX_TOKENS = 10
DEBUG = True

def debug_print(message, force=False):
    """Print debug messages when debug mode is enabled."""
    if DEBUG or force:
        print(f"[DEBUG] {message}")

def sf_normalize(X):
    """Normalize by scaling factor."""
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero division error
    counts += counts == 0.
    # normalize to 10000 counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X

@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x: np.array, max_seq_len: int = -1, aux_tokens: int = 5):
    """Core tokenization logic for gene expression data."""
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    for i, cell in enumerate(x):
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
        sorted_indices = sorted_indices + aux_tokens  # reserve tokens for padding etc
        if max_seq_len:
            scores = np.zeros(max_seq_len)
        else:
            scores = np.zeros_like(cell)
        scores[:len(sorted_indices)] = sorted_indices

        scores_final[i, :] = scores

    return scores_final

def tokenize_data(x: np.array, median_counts_per_gene: np.array, max_seq_len: int = None, spatial: bool = False):
    """Tokenize the input gene vector to a vector of 32-bit integers."""
    x = sf_normalize(x)
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))

    scores_final = _sub_tokenize_data(out, max_seq_len or MAX_SEQ_LEN, AUX_TOKENS)

    return scores_final.astype('i4')

def process_h5ad_file(h5ad_path, output_dir, is_spatial=False, median_counts=None, genes_to_drop=None, mapping_dict=None):
    """Process a single h5ad file and save tokenized data to parquet files. Skips already processed chunks."""
    print(f"Processing {h5ad_path}...")

    # Load h5ad file
    try:
        adata = sc.read_h5ad(h5ad_path)
        print(f"Loaded {adata.shape[0]} cells and {adata.shape[1]} genes from {h5ad_path}")
    except Exception as e:
        print(f"Error reading {h5ad_path}: {e}")
        return False

    # Extract filename without extension
    file_basename = os.path.basename(h5ad_path).replace('.h5ad', '')

    # Create output directory
    file_output_dir = join(output_dir, file_basename)
    train_dir = join(file_output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    # Save var file
    var_df = adata.var
    var_df.to_parquet(join(file_output_dir, 'var.parquet'))

    # Prepare observation metadata
    obs_df = adata.obs.copy()

    # Add standard columns if missing
    if 'specie' not in obs_df.columns:
        if 'species' in obs_df.columns:
            obs_df['specie'] = obs_df['species']
        else:
            # Try to determine species from filename or default to MOUSE
            if 'human' in file_basename.lower():
                obs_df['specie'] = HUMAN
            else:
                obs_df['specie'] = MOUSE

    if 'technology' not in obs_df.columns:
        obs_df['technology'] = SPATIAL if is_spatial else DISSOCIATED

    # Ensure donor_id exists
    if 'donor_id' not in obs_df.columns:
        if 'donor' in obs_df.columns:
            obs_df['donor_id'] = obs_df['donor']
        else:
            print(f"Warning: No donor information found in {h5ad_path}. Creating artificial donor IDs.")
            obs_df['donor_id'] = 1  # Default donor ID

    # Process in chunks to avoid memory issues
    total_cells = adata.shape[0]
    num_chunks = (total_cells + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Get or compute median counts if not provided
    if median_counts is None:
        print("Computing median gene counts...")
        if issparse(adata.X):
            gene_counts = adata.X.mean(axis=0).A1
        else:
            gene_counts = adata.X.mean(axis=0)
        median_counts = gene_counts

    # Check which chunks are already processed
    existing_chunks = set()
    if os.path.exists(train_dir):
        for fname in os.listdir(train_dir):
            if fname.startswith(file_basename) and fname.endswith('.parquet'):
                try:
                    idx = int(fname.replace(file_basename+'-','').replace('.parquet',''))
                    existing_chunks.add(idx)
                except Exception:
                    continue

    # Process each chunk
    for chunk_idx in tqdm(range(num_chunks)):
        output_filename = f"{file_basename}-{chunk_idx}.parquet"
        output_path = join(train_dir, output_filename)
        if chunk_idx in existing_chunks and os.path.exists(output_path):
            print(f"[SKIP] Chunk {chunk_idx+1}/{num_chunks} already exists: {output_filename}")
            continue
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * CHUNK_SIZE, total_cells)

        # Extract chunk data
        chunk_adata = adata[start_idx:end_idx]
        chunk_obs = obs_df.iloc[start_idx:end_idx]

        # Extract expression matrix and convert to dense if sparse
        if issparse(chunk_adata.X):
            X = chunk_adata.X.toarray()
        else:
            X = chunk_adata.X

        # Apply gene filtering if needed (for spatial data)
        if is_spatial and genes_to_drop is not None:
            X = np.delete(X, genes_to_drop, axis=1)
            gene_medians = np.delete(median_counts, genes_to_drop)
        else:
            gene_medians = median_counts

        # Tokenize the data
        print(f"Tokenizing chunk {chunk_idx+1}/{num_chunks} ({start_idx}:{end_idx})...")
        tokenized_data = tokenize_data(X, gene_medians, MAX_SEQ_LEN, is_spatial)

        # Apply mapping dictionary if provided (for spatial data)
        if is_spatial and mapping_dict is not None:
            vectorized_map = np.vectorize(lambda value: mapping_dict.get(value, value))
            tokenized_data = vectorized_map(tokenized_data)

        # Convert to list format for parquet storage
        tokenized_list = [arr.tolist() for arr in tokenized_data]

        # Create DataFrame with tokenized data and metadata
        result_df = pd.DataFrame({'X': tokenized_list})

        # Add metadata columns
        for col in chunk_obs.columns:
            result_df[col] = chunk_obs[col].values

        # Save to parquet
        result_df.to_parquet(
            output_path,
            engine='pyarrow',
            row_group_size=1024
        )
        print(f"Saved chunk to {output_filename}")

    print(f"Completed processing {h5ad_path}")
    return True

def process_directory(input_dir, output_dir, is_spatial=False):
    """Process all h5ad files in a directory. Skips files that are already fully processed."""
    print(f"Processing directory: {input_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all h5ad files
    h5ad_files = [f for f in os.listdir(input_dir) if f.endswith('.h5ad')]

    if not h5ad_files:
        print(f"No h5ad files found in {input_dir}")
        return

    print(f"Found {len(h5ad_files)} h5ad files to process")

    # Try to load median counts if available
    median_counts_path = '/lustre/groups/ml01/projects/2023_nicheformer/data/medians/medians_counts_subsample_cxg_mouse.npy'
    spatial_median_counts_path = '/lustre/groups/ml01/projects/2023_nicheformer/medians_counts_brain.npy'

    try:
        if is_spatial:
            median_counts = np.load(spatial_median_counts_path)
            print(f"Loaded spatial median counts from {spatial_median_counts_path}")
        else:
            median_counts = np.load(median_counts_path)
            print(f"Loaded dissociated median counts from {median_counts_path}")
    except Exception as e:
        print(f"Warning: Could not load median counts file: {e}")
        print("Will compute median counts from data")
        median_counts = None

    # Try to load mapping dictionary and genes to drop for spatial data
    mapping_dict = None
    genes_to_drop = None

    if is_spatial:
        # Replace with path to mapping dictionary from original Nicheformer publication
        mapping_dict_path = '/lustre/groups/ml01/projects/2023_nicheformer/data/matching_dictionaries/matching_dictionary_mouse_spatial_dissociated.pkl'
        # Replace with path to genes to drop from original Nicheformer publication
        genes_to_drop_path = '/lustre/groups/ml01/projects/2023_nicheformer/data/matching_dictionaries/brain_spatial_genes_to_drop_mouse_spatial_dissociated.pkl'

        try:
            with open(mapping_dict_path, 'rb') as file:
                mapping_dict = pickle.load(file)
            print(f"Loaded mapping dictionary from {mapping_dict_path}")

            with open(genes_to_drop_path, 'rb') as file:
                genes_to_drop = pickle.load(file)
            print(f"Loaded genes to drop from {genes_to_drop_path}")
        except Exception as e:
            print(f"Warning: Could not load mapping dictionary or genes to drop: {e}")

    # Process each file
    for h5ad_file in h5ad_files:
        h5ad_path = join(input_dir, h5ad_file)
        # Check if already fully processed
        file_basename = os.path.basename(h5ad_path).replace('.h5ad', '')
        file_output_dir = join(output_dir, file_basename)
        train_dir = join(file_output_dir, 'train')
        # If train_dir exists and contains at least one parquet, check if all chunks are present
        already_done = False
        if os.path.exists(train_dir):
            # Try to estimate expected number of chunks from the h5ad file size
            try:
                adata = sc.read_h5ad(h5ad_path, backed='r')
                total_cells = adata.shape[0]
                num_chunks = (total_cells + CHUNK_SIZE - 1) // CHUNK_SIZE
                existing_chunks = [fname for fname in os.listdir(train_dir) if fname.startswith(file_basename) and fname.endswith('.parquet')]
                if len(existing_chunks) >= num_chunks:
                    print(f"[SKIP] {h5ad_file} already fully processed ({len(existing_chunks)}/{num_chunks} chunks)")
                    already_done = True
            except Exception as e:
                print(f"Warning: Could not check existing chunks for {h5ad_file}: {e}")
        if already_done:
            continue
        process_h5ad_file(
            h5ad_path,
            output_dir,
            is_spatial=is_spatial,
            median_counts=median_counts,
            genes_to_drop=genes_to_drop,
            mapping_dict=mapping_dict
        )

    print(f"Completed processing all files in {input_dir}")

def main():
    """Main function to parse arguments and run data processing."""
    parser = argparse.ArgumentParser(description="Tokenize h5ad files for NicheFormer")

    from cell_synergy.paths import ROOT, PROJECT_DIR
    
    parser.add_argument("--dissociated-dir", type=str,
                        default=str(ROOT.parent / "original_data" / "data" / "dissociated_to_tokenize"),
                        help="Directory containing dissociated h5ad files")

    parser.add_argument("--spatial-dir", type=str,
                        default=str(ROOT.parent / "original_data" / "data" / "data_to_tokenize"),
                        help="Directory containing spatial h5ad files")

    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_DIR),
                        help="Output directory for tokenized data")

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Set debug mode
    global DEBUG
    DEBUG = args.debug

    # Start time
    start_time = time.time()

    try:
        # Initialize cluster for parallel processing
        cluster = LocalCluster(n_workers=5)  # 5 workers
        client = Client(cluster)
        print(f"Started Dask cluster with {len(cluster.workers)} workers")
    except Exception as e:
        print(f"Error starting Dask cluster: {e}")
        print("Continuing without Dask cluster...")
        client = None

    # Process dissociated data
    dissociated_output = join(args.output_dir, "dissociated_tokenized")
    process_directory(args.dissociated_dir, dissociated_output, is_spatial=False)

    # Process spatial data
    spatial_output = join(args.output_dir, "nicheformer_tokens")
    process_directory(args.spatial_dir, spatial_output, is_spatial=True)

    # Clean up
    if client:
        client.close()
        cluster.close()

    # End time
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
