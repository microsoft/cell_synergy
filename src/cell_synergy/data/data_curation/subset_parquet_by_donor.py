#!/usr/bin/env python3
import os
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def debug_print(message, force=False):
    """Print a message if debug mode is enabled or if force is True."""
    if getattr(debug_print, 'enabled', False) or force:
        print(f"[DEBUG] {message}")
debug_print.enabled = False
"""Disable debug printing by default."""

def get_all_donors(parquet_dir):
    """Return set of all donor_ids and a donor_id:sex mapping from all Parquet files in a directory."""
    donor_ids = set()
    donor_sex = dict()
    unique_id_counter = 0
    for fname in tqdm(os.listdir(parquet_dir), desc=f"Scanning donors in {os.path.basename(parquet_dir)}"):
        if not fname.endswith('.parquet'):
            continue
        try:
            df = pd.read_parquet(os.path.join(parquet_dir, fname))
            # Check for donor_id column
            if 'donor_id' not in df.columns:
                print(f"Warning: {fname} does not contain 'donor_id'. Assigning unique donor IDs.")
                df['donor_id'] = [f"auto_donor_{unique_id_counter + i}" for i in range(len(df))]
                unique_id_counter += len(df)
            # Check for sex column
            if 'sex' not in df.columns:
                print(f"Warning: {fname} does not contain 'sex'. Assigning 'NA' as sex.")
                df['sex'] = 'NA'
            # Fill missing sex values with 'NA'
            df['sex'] = df['sex'].fillna('NA')
            donor_ids.update(df['donor_id'].unique())
            for did, sex in zip(df['donor_id'], df['sex']):
                donor_sex[did] = sex
        except Exception as e:
            debug_print(f"Error reading {fname}: {e}", force=True)
    return list(donor_ids), donor_sex

def analyze_donors(donor_ids: list[str], donor_sex: dict[str, str]) -> None:
    """Analyze the distribution of donors by sex."""
    total = len(donor_ids)
    n_female = sum(1 for d in donor_ids if str(donor_sex.get(d, '')).lower() == 'female')
    n_male = sum(1 for d in donor_ids if str(donor_sex.get(d, '')).lower() == 'male')
    n_na = sum(1 for d in donor_ids if str(donor_sex.get(d, '')).lower() not in ['male', 'female'])
    print(f"Nicheformer contains {total} donors")
    print(f"{n_female} female donors")
    print(f"{n_male} male donors")
    print(f"{n_na} donors with sex NA or unknown")

def select_donors(donor_ids: list[str], 
                donor_sex: dict[str, str], 
                n_donors: int | None = None, 
                remove_n_female: int | None = None, 
                only_sex: str | None = None, 
                seed: int = 42) -> set[str]:
    """Select donors based on selection criteria."""
    np.random.seed(seed)
    donor_ids = np.array(list(donor_ids))
    # Remove N female donors if requested
    if remove_n_female is not None:
        female_donors = [d for d in donor_ids if str(donor_sex.get(d, '')).lower() == 'female']
        if remove_n_female > len(female_donors):
            raise ValueError(f"Requested to remove {remove_n_female} female donors, but only {len(female_donors)} available.")
        remove = set(np.random.choice(female_donors, remove_n_female, replace=False))
        donor_ids = np.array([d for d in donor_ids if d not in remove])
    # Filter by sex if requested
    if only_sex is not None:
        donor_ids = np.array([d for d in donor_ids if str(donor_sex.get(d, '')).lower() == only_sex.lower()])
    if n_donors is not None:
        if n_donors > len(donor_ids):
            raise ValueError(f"Requested {n_donors} donors, but only {len(donor_ids)} available after filtering.")
        donor_ids = np.random.choice(donor_ids, n_donors, replace=False)
    return set(donor_ids)

def filter_and_write(source_dir: str, output_dir: str, keep_donors: set[str]) -> None:
    """Filter and write Parquet files by donor selection."""
    os.makedirs(output_dir, exist_ok=True)
    n_files = 0
    n_cells_in = 0
    n_cells_out = 0
    for fname in tqdm(os.listdir(source_dir), desc=f"Filtering {os.path.basename(source_dir)}"):
        if not fname.endswith('.parquet'):
            continue
        src = os.path.join(source_dir, fname)
        dst = os.path.join(output_dir, fname)
        try:
            df = pd.read_parquet(src)
            n_cells_in += len(df)
            df_filt = df[df['donor_id'].isin(keep_donors)]
            n_cells_out += len(df_filt)
            if len(df_filt) > 0:
                df_filt.to_parquet(dst)
                n_files += 1
        except Exception as e:
            debug_print(f"Error filtering {fname}: {e}")
    print(f"Wrote {n_files} files, {n_cells_out}/{n_cells_in} cells kept.")

def main():
    """Main function to analyze or subset Parquet files by donor selection."""
    parser = argparse.ArgumentParser(description="Analyze or subset Parquet files by donor selection.")
    parser.add_argument('--project-root', default="/lustre/groups/ml01/projects/2025_nicheformer_subsets", help='Project root directory (contains nf_full)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Analyze mode
#     parser_analyze = subparsers.add_parser('analyze', help='Analyze donor composition in the dataset')  # unused

    # Subset mode
    parser_subset = subparsers.add_parser('subset', help='Subset donors in the dataset')
    parser_subset.add_argument('--n-donors', type=int, default=None, help='Number of donors to keep (after filtering)')
    parser_subset.add_argument('--remove-n-female', type=int, default=None, help='Remove N female donors (randomly)')
    parser_subset.add_argument('--sex', type=str, choices=['male', 'female'], default=None, help='Only keep donors of this sex')
    parser_subset.add_argument('--output-name', type=str, required=True, help='Name for the output subdirectory (e.g., nf_remove20female)')
    parser_subset.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    debug_print.enabled = args.debug

    source_root = os.path.join(args.project_root, 'nf_full')

    if args.command == 'analyze':
        for split in ['train', 'test']:
            src_dir = os.path.join(source_root, split)
            if not os.path.exists(src_dir):
                print(f"Warning: {src_dir} does not exist, skipping.")
                continue
            print(f"Analyzing split: {split}")
            donor_ids, donor_sex = get_all_donors(src_dir)
            analyze_donors(donor_ids, donor_sex)
    elif args.command == 'subset':
        output_root = os.path.join(args.project_root, args.output_name)
        for split in ['train', 'test']:
            src_dir = os.path.join(source_root, split)
            out_dir = os.path.join(output_root, split)
            if not os.path.exists(src_dir):
                print(f"Warning: {src_dir} does not exist, skipping.")
                continue
            print(f"Processing split: {split}")
            donor_ids, donor_sex = get_all_donors(src_dir)
            keep_donors = select_donors(
                donor_ids, donor_sex,
                n_donors=args.n_donors,
                remove_n_female=args.remove_n_female,
                only_sex=args.sex,
                seed=args.seed
            )
            print(f"Keeping {len(keep_donors)} donors out of {len(donor_ids)}")
            analyze_donors(keep_donors, donor_sex)
            filter_and_write(src_dir, out_dir, keep_donors)
        # Copy non-split files (e.g., var.parquet, lookup tables) from source_root to output_root
        for fname in os.listdir(source_root):
            src = os.path.join(source_root, fname)
            dst = os.path.join(output_root, fname)
            if os.path.isfile(src) and not fname in ['train', 'test']:
                shutil.copy2(src, dst)
        print("Done.")

if __name__ == '__main__':
    main()
