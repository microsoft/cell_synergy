# Data Curation Scripts

This directory contains scripts for data preprocessing, tokenization, and dataset preparation.

## Scripts

### `tokenize_hf_dataset.py`
Tokenize HuggingFace datasets directly to Merlin format for continued training.

Converts HF datasets (breast/thymus/lung) to tokenized parquet format compatible with the Merlin datamodule. Extracts cells from HF dataset patches, creates tokenizer vocabulary based on gene expression means, generates train/val splits (90/10), and saves tokenized data as parquet chunks.

**Usage**:
```bash
python -m cell_synergy.data.data_curation.tokenize_hf_dataset \
    --hf-dataset-path /path/to/hf_dataset \
    --output-dir /path/to/output \
    --chunk-size 100000
```

---

### `convert_hf_to_h5ad.py`
Convert HuggingFace datasets to h5ad (AnnData) format.

Intermediate conversion step when h5ad format is needed for downstream processing. Converts HF dataset to AnnData format, preserves metadata (obs columns), and handles gene expression data extraction.

**Usage**:
```bash
python -m cell_synergy.data.data_curation.convert_hf_to_h5ad \
    --dataset breast \
    --input-dataset scaling_splits/nicheformer_full_200M_train.L \
    --output-path /path/to/output.h5ad
```

---

### `tokenize_h5ad_data.py`
Tokenize h5ad files for NicheFormer training.

Processes h5ad files (spatial or dissociated) and converts them to tokenized parquet format. Handles both spatial and dissociated data, applies gene filtering and mapping for spatial data, normalizes and tokenizes gene expression, processes data in chunks to handle large files, and supports parallel processing with Dask. Requires Nicheformer datasets from original Nicheformer publication.

**Usage**:
```bash
python -m cell_synergy.data.data_curation.tokenize_h5ad_data \
    --dissociated-dir /path/to/dissociated \
    --spatial-dir /path/to/spatial \
    --output-dir /path/to/output
```

---

### `subset_parquet_by_donor.py`
Subset parquet files by donor selection criteria.

Filter datasets by donor characteristics (sex, count, etc.) for controlled experiments. Analyzes donor composition in datasets, filters by sex, donor count, or other criteria, maintains train/test split structure, and preserves metadata files.

**Usage**:
```bash
python -m cell_synergy.data.data_curation.subset_parquet_by_donor \
    --project-root /path/to/project \
    subset \
    --n-donors 10 \
    --output-name nf_subset
```

---

### `extract_donors_from_parquet.py`
Extract donor information from parquet files for analysis.

Utility to understand donor distribution and metadata in datasets. Scans parquet files to extract donor IDs, analyzes organism and specie distribution, and provides summary statistics.

**Usage**:
```bash
python -m cell_synergy.data.data_curation.extract_donors_from_parquet.py \
    /path/to/parquet/directory \
    [sample_limit]
```

---

### `design_data_splits.py`
Design data splits for continued training experiments.

Creates a comprehensive data split strategy that defines a holdout test set (donor-level split to avoid data leakage), defines training subsets for post-training (1pct, 3.16pct, 10pct, 31.6pct, 100pct), ensures test samples never appear in post-training data, and saves the configuration for future use.

**Usage**:
```bash
python -m cell_synergy.data.data_curation.design_data_splits
```

The script generates a JSON file at `project_folder/{dataset}/continued_training_split_design.json` containing the split design.
