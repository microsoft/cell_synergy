#!/usr/bin/env python
# This script computes Nicheformer embeddings for S, M, L splits (pretrain only) using the local HF dataset and saves them as .npy files.
# To change paths, edit the variables below.

import os
import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
import sys
import time
from pathlib import Path
import logging
from data_scaling.paths import PROJECT_DIR, ROOT
from data_scaling.config import load_data_splits
from data_scaling.data.hf_dataset.create_hf_dataset.process.nicheformer.embedder import NicheformerEmbedder, compute_nicheformer_embeddings_batched
from data_scaling.data.hf_dataset.create_hf_dataset.process.loader.get_dataloader import ExpressionDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
SPLIT = "pretrain"  # Only use pretrain
BATCH_SIZE = 8
NUM_WORKERS = 4
DATASET_NAME = "lung"  # or "breast", etc. as needed

# Paths using repo structure
MEAN_FILE = ROOT / "data_scaling" / "data" / "hf_dataset" / "xenium_mean_script.npy"
REFERENCE_FILE = ROOT / "data_scaling" / "data" / "hf_dataset" / "nicheformer_reference.h5ad"
CACHE_DIR = PROJECT_DIR / "hf_cache"
OUTPUT_DIR = PROJECT_DIR / "unimodal_embeddings" / "gex"


def get_sample_ids_for_scale(scale, split, config):
    return config.multimodal[split][scale]


def compute_nicheformer_for_scale(data_split, sample_ids, nicheformer_model, batch_size, num_workers, device, out_path):
    logger.info(f"Filtering dataset for {len(sample_ids)} samples...")
    start = time.time()
    filtered = data_split.filter(
        lambda batch: [name in sample_ids for name in batch['name']],
        batched=True,
        batch_size=1000
    )
    logger.info(f"Filtered dataset has {len(filtered)} samples. Filtering took {time.time() - start:.2f} seconds. Initializing dataloader...")
    dataloader = ExpressionDataLoader(filtered).get_loader(
        batch_size=batch_size, num_workers=num_workers
    )
    niche_pool_list = []
    logger.info(f"Starting embedding extraction for {out_path}...")
    for batch in tqdm(dataloader, desc=f"Nicheformer embeddings for {out_path}"):
        gexps = batch["gexp"]
        masks = batch.get("mask")
        pool = compute_nicheformer_embeddings_batched(
            gexps, masks, nicheformer_model, device=device
        )
        niche_pool_list.extend(pool)
    np.save(out_path, np.array(niche_pool_list))
    logger.info(f"Saved Nicheformer pooled embeddings to {out_path}")


def main():
    # Fail fast: check all required files exist before doing anything else
    logger.info("Checking required files...")
    missing = False
    if not MEAN_FILE.exists():
        logger.error(f"Mean file not found: {MEAN_FILE}")
        missing = True
    if not REFERENCE_FILE.exists():
        logger.error(f"Reference file not found: {REFERENCE_FILE}")
        missing = True
    if missing:
        logger.error("Aborting due to missing required files.")
        return

    logger.info("Loading splits config via load_data_splits...")
    cfg = load_data_splits(dataset_name=DATASET_NAME)

    logger.info("Loading HF dataset from Hugging Face...")
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    dataset = load_dataset(
        "theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6",
        split="train", cache_dir=str(CACHE_DIR), use_auth_token=hf_token,
    )
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["gexp", "mask", "name"]])
    data_split = dataset

    logger.info("Initializing Nicheformer embedder...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nicheformer_model = NicheformerEmbedder(
        local_dir=str(REFERENCE_FILE.parent),
        batch_size=BATCH_SIZE,
        device=device,
        technology_mean=str(MEAN_FILE),
        # reference_path=str(REFERENCE_FILE),
    )
    nicheformer_model = nicheformer_model.to(device)
    nicheformer_model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scale in ["S", "M", "L"]:
        logger.info(f"\n--- Processing scale: {scale} ---")
        sample_ids = get_sample_ids_for_scale(scale, SPLIT, cfg)
        out_path = OUTPUT_DIR / f"nicheformer_pool_{scale}.npy"
        compute_nicheformer_for_scale(
            data_split, sample_ids, nicheformer_model, BATCH_SIZE, NUM_WORKERS, device, str(out_path)
        )
    logger.info("All scales processed.")

if __name__ == "__main__":
    main()
