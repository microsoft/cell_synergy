#!/usr/bin/env python
import sys
import os
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

def extract_donors(file_path, verbose=True):
    """Extract donor_id and organism info from a parquet file using PyArrow."""
    if verbose:
        print(f"Extracting donor info from: {os.path.basename(file_path)}")
    
    try:
        # Read using PyArrow directly
        table = pq.read_table(file_path, columns=['donor_id', 'organism', 'specie'])
        df = table.to_pandas()
        
        # Print summary
        if verbose:
            print(f"Found {len(df)} rows and {len(df['donor_id'].unique())} unique donors")
            print(f"Donor examples: {df['donor_id'].unique()[:5]}")
            if 'organism' in df.columns:
                print(f"Organism values: {df['organism'].dropna().unique()[:5]}")
            if 'specie' in df.columns:
                print(f"Specie values: {df['specie'].dropna().unique()[:5]}")
        
        return df
    except Exception as e:
        if verbose:
            print(f"Error extracting donor info: {str(e)}")
        return None

def scan_directory(directory_path, sample_limit=None):
    """Scan a directory of parquet files and extract donor and sex-related info."""
    print(f"Scanning directory: {directory_path}")
    
    # Get parquet files
    parquet_files = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    if sample_limit and len(parquet_files) > sample_limit:
        import random
        parquet_files = random.sample(parquet_files, sample_limit)
    
    print(f"Processing {len(parquet_files)} parquet files...")
    
    # Track donor information
    donor_info = {}
    donor_organism = {}
    donor_specie = {}
    files_processed = 0
    files_with_errors = 0
    
    # Process each file
    for fname in tqdm(parquet_files):
        try:
            df = extract_donors(os.path.join(directory_path, fname), verbose=False)
            if df is not None:
                files_processed += 1
                
                # Record donors from this file
                for _, row in df.iterrows():
                    donor_id = row['donor_id']
                    if donor_id not in donor_info:
                        donor_info[donor_id] = {'count': 0}
                    donor_info[donor_id]['count'] += 1
                    
                    # Record organism if available
                    if 'organism' in df.columns and not pd.isna(row['organism']):
                        donor_organism[donor_id] = row['organism']
                    
                    # Record specie if available
                    if 'specie' in df.columns and not pd.isna(row['specie']):
                        donor_specie[donor_id] = row['specie']
            else:
                files_with_errors += 1
        except Exception as e:
            files_with_errors += 1
            print(f"Error processing {fname}: {str(e)}")
    
    # Print summary
    print(f"\nProcessed {files_processed} files successfully ({files_with_errors} with errors)")
    print(f"Found {len(donor_info)} unique donors")
    
    # Print organism distribution
    organism_counts = {}
    for donor, organism in donor_organism.items():
        organism_counts[organism] = organism_counts.get(organism, 0) + 1
    
    print("\nOrganism distribution:")
    for organism, count in sorted(organism_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {organism}: {count} donors")
    
    # Print specie distribution
    specie_counts = {}
    for donor, specie in donor_specie.items():
        specie_counts[specie] = specie_counts.get(specie, 0) + 1
    
    print("\nSpecie distribution:")
    for specie, count in sorted(specie_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {specie}: {count} donors")
    
    # Check if organism contains sex info
    sex_keywords = ['male', 'female', 'man', 'woman', 'boy', 'girl']
    organisms_with_sex = []
    
    for organism in organism_counts.keys():
        organism_str = str(organism).lower()
        for keyword in sex_keywords:
            if keyword in organism_str:
                organisms_with_sex.append(organism)
                break
    
    if organisms_with_sex:
        print("\nPotential sex information found in organism values:")
        for organism in organisms_with_sex:
            print(f"  {organism}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_donors_from_parquet.py <directory_or_file_path> [sample_limit]")
        sys.exit(1)
    
    path = sys.argv[1]
    sample_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if os.path.isdir(path):
        scan_directory(path, sample_limit)
    else:
        extract_donors(path) 