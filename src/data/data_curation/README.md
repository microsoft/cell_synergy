# Data Curation for NicheFormer

This directory contains scripts for curating and processing both spatial and dissociated cell data for the NicheFormer project.

## Main Script

The main script is `data_curation.py`, which processes spatial and dissociated data, tokenizes it, and ensures proper mixing to avoid inconsistencies during training. It also supports donor-based subsampling to create smaller datasets.

### Usage

```bash
python data_curation.py --donor-percentages 10,25,50 --no-mix
```

### Arguments

- `--donor-percentages`: Comma-separated list of donor percentages to sample (default: "10")
- `--no-mix`: Flag to disable mixing of files after processing (by default, files are mixed)

## Output Structure

The script creates the following directory structure:

```
/lustre/groups/ml01/projects/2025_nicheformer_subsets/
├── categorical_lookup/
│   └── [categorical columns lookup tables]
├── nf_full/
│   ├── train/
│   │   └── [all data parquet files]
│   └── test/
│       └── [all test data parquet files]
├── nf_10pct_donor/
│   ├── train/
│   │   └── [data with 10% of donors]
│   └── test/
│       └── [test data with 10% of donors]
└── var.parquet
```

## Donor-based Subsampling

The script includes functionality to subsample the dataset based on donor IDs:

1. The full dataset is processed and saved to the `nf_full` directory
2. Unique donor IDs are identified from the full dataset
3. A specified percentage (e.g., 10%) of donors are randomly selected
4. Data from only these donors is extracted to create a subsampled dataset
5. The subsampled dataset is saved to a directory with the percentage in the name (e.g., `nf_10pct_donor`)

## Preprocessing Steps

1. Loads spatial and dissociated data from source directories
2. Tokenizes gene expression data using appropriate normalization
3. Handles differences between spatial and dissociated data formats
4. Combines data and ensures consistent structure
5. Shuffles and mixes data to avoid training inconsistencies
6. Writes output to parquet files with consistent schema
7. Creates subsampled datasets based on donor IDs

## Required Files

The script requires the following files to operate:

- Median counts files for normalization
- Mapping dictionaries between spatial and dissociated data
- Lists of genes to drop from spatial data
- Source data in zarr format

## Dependencies

- Python 3.8+
- dask
- pandas
- numpy
- pyarrow
- numba
- tqdm
- scipy
- scikit-learn

## Additional Information

- The script uses Dask for distributed processing with a local cluster of 5 workers
- Data is processed in chunks of 32768 samples
- Output parquet files use row groups of 1024 samples
- Tokenization uses a maximum sequence length of 4096 