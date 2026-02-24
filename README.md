# Cell Synergy: Multimodal Self-Supervised Learning for Biological Data

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.02.23.707420-blue)](https://doi.org/10.64898/2026.02.23.707420)

This repository contains the code and data processing pipelines for **"Beyond alignment: synergistic integration is required for multimodal cell foundation models"**.

## Overview

This work investigates how multimodal self-supervised learning (SSL) methods align gene expression (GEX) and histopathology image (IMG) representations in spatial transcriptomics data. We use frozen pretrained encoders (UNI2 for images, Nicheformer for gene expression) and train only the alignment interface. We introduce the **Synergistic Information Score (SIS)** to quantify how well alignment methods capture nonlinear interactions between modalities, beyond simple redundancy.

### Key Contributions

1. **Theoretical Framework**: Extends spectral theory to cross-covariance matrices, revealing a "spectral ceiling" that limits linear alignment methods
2. **SIS Metric**: Novel metric to measure synergistic information capture, distinguishing methods that extract nonlinear interactions from those that only capture redundancy
3. **Comprehensive Benchmarking**: Evaluation of 10 alignment methods (spectral: CCA, DCCA; non-spectral: CoMM, SimCLR, BYOL, SimSiam, Barlow Twins, VICReg, DIM, Concat) across three datasets (lung, breast, thymus)
4. **Data Scaling Analysis**: Systematic study of how data scale affects multimodal alignment performance
5. **Spatial Evaluation**: Task-specific evaluation ranging from local redundancy (cell type classification) to long-range spatial organization (neighborhood prediction)

## Paper Abstract

> The vision of a "virtual cell"---a computational model that simulates biological function across modalities and scales---has become a defining goal in computational biology. While powerful unimodal foundation models exist, the lack of large-scale paired data prohibits the joint training of multimodal approaches. This scarcity favors compositional foundation models (CFMs): architectures that fuse frozen unimodal experts via a learned interface. However, it remains unclear when this multimodal fusion adds task-relevant information beyond the strongest unimodal representation and when it merely aggregates redundant signal. Here, we introduce the Synergistic Information Score (SIS), a metric grounded in partial information decomposition (PID), that quantifies the information gain achievable only through cross-modal interactions. Extending theoretical results from self-supervised learning, we show that standard alignment-based fusion objectives on frozen encoders inherently collapse to detecting linear redundancies, limiting their ability to capture nonlinear synergistic states. This distinction is directly relevant for tasks aiming to link tissue morphology and gene expression. Benchmarking ten fusion methods on spatial transcriptomics datasets, we use SIS to demonstrate that tasks dominated by linear redundancies are sufficiently served by unimodal baselines, whereas complex niche definitions benefit from synergy-aware integration objectives that enable cross-modal interactions beyond linear alignment. Finally, we perform a scaling analysis which highlights that fine-tuning a dominant unimodal expert is the most sample-efficient path for standard tasks, suggesting that the benefits of multimodal frameworks only emerge when tasks depend on information distributed across modalities. Together, these results establish that building towards a virtual cell will require a fundamental shift from alignment objectives that emphasize shared structure to synergy-maximizing integration that preserves and exploits complementary cross-modal signal.

## Repository Structure

```
cell_synergy/
├── src/cell_synergy/          # Main Python package
│   ├── models/                # Alignment model implementations
│   │   ├── comm.py            # CoMM (Compositional Multimodal) model
│   │   ├── cca.py             # Canonical Correlation Analysis
│   │   ├── dcca.py            # Deep CCA
│   │   ├── simclr.py          # SimCLR contrastive learning
│   │   ├── byol.py            # Bootstrap Your Own Latent
│   │   ├── simsiam.py         # SimSiam
│   │   ├── barlowtwins.py     # Barlow Twins
│   │   ├── vicreg.py          # VICReg
│   │   ├── dim.py             # Deep InfoMax
│   │   └── concat.py          # Baseline concatenation
│   ├── sis.py                 # Synergistic Information Score computation
│   ├── downstream/            # Downstream task evaluation
│   ├── finetuning/            # Model finetuning utilities
│   └── data/                  # Data processing and dataset management
├── scripts/                   # SLURM batch scripts for experiments
│   ├── training/              # Model training scripts
│   ├── evaluation/           # Evaluation and benchmarking scripts
│   └── data_processing/      # Data preprocessing and embedding generation
├── configs/                   # Hydra configuration files
├── examples/                  # Tutorial-style example scripts
└── project_folder/           # Data storage (symlinked to larger storage)
```

## Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended)
- SLURM workload manager (for running batch jobs)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cell-synergy.git
cd cell-synergy

# Install the package in editable mode
pip install -e .

# Install additional dependencies
pip install -r requirements.txt
```

### Environment Setup

The codebase uses conda for environment management. Create and activate the environment:

```bash
conda env create -f environment.yml  # If available
# Or manually:
conda create -n cell_synergy_env python=3.10
conda activate cell_synergy_env
pip install -e .
```

## Quick Start

### Examples and Tutorials

We provide simple, tutorial-style examples in the `examples/` directory:

- **`example_compute_sis.py`** - Computing the Synergistic Information Score (SIS)
- **`example_finetune_nicheformer.py`** - Finetuning Nicheformer on new data
- **`example_align_comm.py`** - Training a CoMM alignment model
- **`example_evaluate_f1_r2.py`** - Evaluating F1 and R² scores
- **`example_spatial_consistency.py`** - Evaluating spatial consistency
- **`example_spatial_neighbors.py`** - Evaluating spatial neighbor prediction

Run any example:
```bash
python examples/example_compute_sis.py
```

See `examples/README.md` for more details.

### Computing Synergistic Information Score (SIS)

The SIS metric quantifies how much additional information the multimodal representation provides beyond the best unimodal representation:

```python
from cell_synergy.sis import compute_sis

# Load your evaluation results
results = {
    'Unimodal GEX': {'F1 Macro': [0.75], 'R2': [0.68]},
    'Unimodal IMG': {'F1 Macro': [0.72], 'R2': [0.65]},
    'Multimodal CoMM': {'F1 Macro': [0.82], 'R2': [0.75]},
}

# Compute SIS
sis_scores = compute_sis(results, 'Multimodal CoMM')
print(f"SIS (F1 Macro): {sis_scores['F1 Macro']:.4f}")
print(f"SIS (R²): {sis_scores['R2']:.4f}")
```

### Training an Alignment Model

```bash
python -m cell_synergy.finetuning.run_alignment \
    --config-name align \
    models.name=comm \
    data.dataset=lung \
    training.max_epochs=50
```

### Running Downstream Evaluation

```bash
python -m cell_synergy.downstream.run_benchmarks \
    --config-name downstream \
    evaluation.modality=multimodal \
    data.dataset=lung
```

## Reproducing Paper Results

### 1. Data Preparation

The paper uses three spatial transcriptomics datasets:
- **Lung**: Primary dataset with ~71k samples
- **Breast**: Secondary dataset for validation
- **Thymus**: Tertiary dataset for validation

See `scripts/data_processing/` for data preprocessing scripts.

### 2. Training Alignment Models

Train all 10 alignment methods:

```bash
# Train CoMM (main method)
sbatch scripts/training/train_multimodal_alignment/lung/train_comm_cfg1.sbatch

# Train spectral methods (CCA, DCCA)
sbatch scripts/training/train_multimodal_alignment/lung/train_cca.sbatch
sbatch scripts/training/train_multimodal_alignment/lung/train_dcca.sbatch

# Train non-spectral methods (SimCLR, BYOL, etc.)
sbatch scripts/training/train_multimodal_alignment/lung/train_simclr.sbatch
# ... (see scripts/training/ for all methods)
```

### 3. Computing SIS Scores

After training and evaluation, compute SIS scores:

```python
from cell_synergy import compute_sis_all_models, print_sis_summary

# Load all evaluation results
results = load_all_results()  # Your function to aggregate results

# Compute SIS for all models
sis_results = compute_sis_all_models(results)

# Print summary
print_sis_summary(sis_results)
```

### 4. Data Scaling Experiments

Reproduce scaling experiments:

```bash
# Train CoMM on different data scales
sbatch scripts/training/train_comm_data_scaling/train_1pct_cfg1.sbatch
sbatch scripts/training/train_comm_data_scaling/train_3.16pct_cfg1.sbatch
sbatch scripts/training/train_comm_data_scaling/train_10pct_cfg1.sbatch
sbatch scripts/training/train_comm_data_scaling/train_31.6pct_cfg1.sbatch
sbatch scripts/training/train_comm_data_scaling/train_100pct_cfg1.sbatch
```

### 5. Spatial Evaluation

Run spatial neighborhood prediction tasks:

```bash
sbatch scripts/evaluation/spatial_evaluation/evaluate_spatial_lung.sbatch
```

## Key Components

### Alignment Models

All alignment models are implemented in `src/cell_synergy/models/`:

- **Spectral Methods**: `cca.py`, `dcca.py` - Linear alignment via SVD of cross-covariance
- **Non-Spectral Methods**: 
  - `comm.py` - CoMM (Compositional Multimodal) - Main method with highest SIS
  - `simclr.py` - SimCLR - Strong redundancy capture
  - `byol.py`, `simsiam.py`, `barlowtwins.py`, `vicreg.py`, `dim.py` - Other contrastive methods
  - `concat.py` - Baseline concatenation

### SIS Computation

The `sis.py` module implements the Synergistic Information Score:

```
SIS(Y; Z₁, Z₂) = (I(Y; Z₃) - max(I(Y; Z₁), I(Y; Z₂))) / max(I(Y; Z₁), I(Y; Z₂))
```

Where:
- `Z₁, Z₂`: Unimodal representations (IMG and GEX)
- `Z₃`: Multimodal representation
- `I(Y; Z)`: Mutual information approximated by performance metrics (F1 Macro, R²)

### Datasets

The codebase supports three datasets:
- **Lung**: Primary dataset, ~71k samples, 7 test donors
- **Breast**: Validation dataset
- **Thymus**: Validation dataset

Dataset configurations are in `configs/data/`.

## Pretrained Models

The paper uses the following pretrained foundation models:

- **UNI2**: Histopathology image encoder trained on 200M images
  - HuggingFace: [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)
- **Nicheformer**: Gene expression encoder for spatial transcriptomics
  - Trained on large-scale transcriptomics data: [theislab/nicheformer](https://github.com/theislab/nicheformer)

## Model Implementations and Citations

This codebase implements several alignment methods based on original research. We acknowledge the following sources:

- **CoMM**: Based on [CoMM](https://github.com/Duplums/CoMM) (see `models/mmfusion.py`)
- **CCA**: Based on [DeepCCA](https://github.com/Michaelvll/DeepCCA) (see `models/cca.py`)
- **DCCA**: Based on [DeepCCA](https://github.com/Michaelvll/DeepCCA) (see `models/dcca.py`)
- **SimCLR**: Based on [Google Research SimCLR](https://github.com/google-research/simclr) (see `models/simclr.py`)
- **BYOL**: Based on [DeepMind Research](https://github.com/deepmind/deepmind-research/tree/master/byol) (see `models/byol.py`)
- **SimSiam**: Based on [Facebook Research SimSiam](https://github.com/facebookresearch/simsiam) (see `models/simsiam.py`)
- **Barlow Twins**: Based on [Facebook Research Barlow Twins](https://github.com/facebookresearch/barlowtwins) (see `models/barlowtwins.py`)
- **VICReg**: Based on [Facebook Research VICReg](https://github.com/facebookresearch/vicreg) (see `models/vicreg.py`)
- **DIM**: Based on [DIM](https://github.com/rdevon/DIM) (see `models/dim.py`)

## Relevant Citation (BibTeX)
If you use this code or method in your research, please consider citing the following:

```
@article {hce_classification,
	author = {Ritcher, Till and Zimmermann, Eric, and Hall, James and Theis, Fabian J. and Raghavan, Srivatsan and Winter, Peter S. and Amini, Ava P. and Crawford, Lorin},
	title = {Beyond alignment: synergistic integration is required for multimodal cell foundation models},
	elocation-id = {2026.02.23.707420},
	year = {2026},
	doi = {10.64898/2026.02.23.707420},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.64898/2026.02.23.707420v1},
	eprint = {https://www.biorxiv.org/content/10.64898/2026.02.23.707420v1.full.pdf},
	journal = {bioRxiv}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Acknowledgments

We thank the developers of the original alignment methods and the spatial transcriptomics community for making datasets publicly available.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
