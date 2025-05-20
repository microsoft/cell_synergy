# Data Scaling for Multimodal Self-Supervised Learning

Experiments to study the effects of data scaling across gene expression and vision modalities in self-supervised learning.

## Overview

This project investigates how varying data scale affects representation quality and downstream performance in multimodal biological data analysis.

## Components

1. **Pre-Training**: Train Nicheformer on data subsets, use pretrained vision encoders
2. **CLIP Training**: Train contrastive models for each combination
3. **Evaluation**: Linear probing for all models
4. **Fine-Tuning**: Continued pretraining on selected models

## Installation

```bash
git clone https://github.com/yourusername/data-scaling.git
cd data-scaling
pip install -e .
```

## Usage

```bash
# Train Nicheformer on subset
python scripts/train_nicheformer_subset.py --config configs/nicheformer/10pct.yaml

# Train CLIP 
python scripts/train_clip_matrix.py --nf_model checkpoint.ckpt --vision_model UNI

# Evaluate
python scripts/evaluate_downstream.py --model_type clip --checkpoint model.pt

# Fine-tune
python scripts/finetune_models.py --model_type clip --checkpoint model.pt
```

## Vision Encoders

| Model | Images | HuggingFace Link |
|-------|--------|------------------|
| UNI | 100M | MahmoodLab/UNI |
| UNI2 | 200M | MahmoodLab/UNI2-h |
| GigaPath | 1B | prov-gigapath/prov-gigapath |
| H-Optimus-0 | 500K | bioptimus/H-optimus-0 |
| Virchow2 | 3.1M | paige-ai/Virchow2 |
| H-Optimus-1 | ? | bioptimus/H-optimus-1 | 