#!/usr/bin/env python3

import os
import argparse
import logging
import numpy as np
import shutil
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/Projects/till_richter/")
    return Path(os.path.expandvars(raw))


def safe_convert_coords(coords):
    if not coords:
        return [[0.0, 0.0]]
    try:
        return [[float(x), float(y)] for x, y in coords]
    except Exception:
        return [[0.0, 0.0]]


def create_annotation_mapping(original_dataset):
    annotation_mapping = {}
    for batch in tqdm(original_dataset.iter(batch_size=1000), desc="Building annotation mapping"):
        for i, name in enumerate(batch["name"]):
            annotation_mapping[name] = {
                "annotation": batch["annotation"][i] or "",
                "cell_type_ratio": batch["cell_type_ratio"][i] or [],
                "cell_coords": safe_convert_coords(batch["cell_coords"][i]),
            }
    return annotation_mapping


def merge_annotations(processed_dataset, annotation_mapping):
    def add_annotations_batch(batch):
        annotations, cell_type_ratios, cell_coords_list = [], [], []
        for name in batch["name"]:
            entry = annotation_mapping.get(name)
            if entry:
                ann = entry["annotation"]
                if isinstance(ann, str) or ann is None:
                    ann = -1  # fallback int class
                annotations.append(ann)

                cell_type_ratios.append(entry["cell_type_ratio"])
                cell_coords_list.append(entry["cell_coords"])
            else:
                annotations.append(-1)  # fallback if name not found
                cell_type_ratios.append([])
                cell_coords_list.append([[0.0, 0.0]])
        batch["annotation"] = annotations
        batch["cell_type_ratio"] = cell_type_ratios
        batch["cell_coords"] = cell_coords_list
        return batch


    merged = processed_dataset.map(add_annotations_batch, batched=True, batch_size=2048)
    logger.info(f"Merged annotations into {len(merged)} samples")
    return merged


def atomic_save(dataset: Dataset, output_path: Path):
    temp_dir = output_path.parent / f"{output_path.name}_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(f"Saving dataset to temporary location: {temp_dir}")
    dataset.save_to_disk(str(temp_dir))

    if output_path.exists():
        backup_path = output_path.parent / f"{output_path.name}_old"
        logger.info(f"Backing up existing dataset to: {backup_path}")
        if backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)
        os.rename(output_path, backup_path)

    os.rename(temp_dir, output_path)
    logger.info(f"✓ Saved final dataset to {output_path}")


def test_dataset_load(path: Path):
    logger.info("Testing load of saved dataset...")
    ds = load_from_disk(str(path))
    logger.info("✓ Reload successful.")
    logger.info(f"Schema: {ds.features}")
    logger.info(f"Sample: {ds[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dataset_name", type=str, default="lung_hf_backup")
    parser.add_argument("--output_dataset_name", type=str, default="lung_hf")
    args = parser.parse_args()

    project_dir = get_project_dir()
    embeddings_path = project_dir / args.embeddings_dataset_name
    output_path = project_dir / args.output_dataset_name

    logger.info(f"Loading embeddings dataset from: {embeddings_path}")
    processed = load_from_disk(str(embeddings_path))

    logger.info("Loading annotation dataset from HF Hub...")
    original = load_dataset("theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6", trust_remote_code=True)
    original = original["train"] if "train" in original else list(original.values())[0]

    annotation_mapping = create_annotation_mapping(original)
    merged = merge_annotations(processed, annotation_mapping)

    atomic_save(merged, output_path)
    test_dataset_load(output_path)
    logger.info("✓ All done.")


if __name__ == "__main__":
    main()
