#!/usr/bin/env python3
"""
Evaluate finetune models with proper train/test splitting.

This script takes a single HF dataset (containing only test data) and splits it
into train/test subsets for linear probe evaluation. This avoids data leakage
when evaluating finetune models that only have test data available.
"""

import argparse
import json

import numpy as np
from datasets import load_from_disk
from omegaconf import DictConfig
from pathlib import Path

from cell_synergy.downstream.eval import (
    extract_embeddings,
    evaluate_simple,
    make_serializable,
    print_results_table,
)


def split_dataset(dataset, train_ratio=0.8, random_state=42):
    """Split a dataset into train and test subsets.

    Args:
        dataset: HuggingFace Dataset to split
        train_ratio: Fraction of data to use for training (default 0.8)
        random_state: Random seed for reproducibility

    Returns:
        train_dataset, test_dataset
    """
    np.random.seed(random_state)
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_ratio)

    train_indices = sorted(indices[:split_idx].tolist())
    test_indices = sorted(indices[split_idx:].tolist())

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    return train_dataset, test_dataset


def load_and_preprocess_data_single(dataset_path: str, cfg: DictConfig, align_method: str):
    """Load and preprocess data from a single dataset path.

    This is a simplified version of load_and_preprocess_data from eval.py
    that works with a single dataset path instead of train/test splits.
    """
    print(f"\nLoading data from: {dataset_path}")

    ds = load_from_disk(dataset_path)
    original_size = len(ds)
    print(f"Loaded {original_size} samples")

    # Apply max_samples limit if specified (for memory optimization)
    max_samples = getattr(cfg.data, 'max_samples', None)
    if max_samples is not None and len(ds) > max_samples:
        print(f"Limiting dataset to {max_samples} samples for memory optimization")
        ds = ds.select(range(min(max_samples, len(ds))))
        original_size = len(ds)
        print(f"Dataset size after limiting: {original_size} samples")

    # Filter out samples with NaN values
    print("  Filtering out samples with NaN values...")
    mask = np.ones(len(ds), dtype=bool)
    img_key = cfg.data.img_embed_key
    gex_key = cfg.data.gex_embed_key

    # Determine which embeddings to check based on method
    check_img = align_method in [
        "unimodal_img", "multimodal_concat"] or (
        align_method and "multimodal_" in align_method)
    check_gex = align_method in [
        "unimodal_gex", "multimodal_concat"] or (
        align_method and "multimodal_" in align_method)

    if check_img:
        arr_img = np.array(ds[img_key])
        mask &= ~np.isnan(arr_img).any(axis=1)
        del arr_img

    if check_gex:
        arr_gex = np.array(ds[gex_key])
        mask &= ~np.isnan(arr_gex).any(axis=1)
        del arr_gex

    valid_indices = np.where(mask)[0]
    ds_clean = ds.select(valid_indices)
    del mask
    del ds

    print(f"  After NaN filtering: {len(ds_clean)}/{original_size} samples kept")

    # Create classification dataset (filter out excluded classes)
    ds_clf = ds_clean
    if cfg.evaluation.tasks.classify:
        excluded_classes = getattr(cfg.data.annotations, "excluded_classes", [])
        if excluded_classes:
            print(f"Filtering out excluded classes: {excluded_classes}")
            try:
                num_proc = min(4, len(ds_clean) // 10000)
                if num_proc < 1:
                    num_proc = 1
                print(f"Using {num_proc} processes for filtering {len(ds_clean)} samples")
                ds_clf = ds_clean.filter(lambda x: x[cfg.training.classification.label_key] not in excluded_classes,
                                         num_proc=num_proc)
                print(f"Classification dataset: {len(ds_clf)} samples (removed {len(ds_clean) - len(ds_clf)})")
            except RuntimeError as e:
                if "subprocesses has abruptly died" in str(e) or "OOM" in str(e):
                    print(f"Multiprocessing failed, falling back to single-threaded filtering: {e}")
                    ds_clf = ds_clean.filter(lambda x: x[cfg.training.classification.label_key] not in excluded_classes)
                    print(f"Classification dataset: {len(ds_clf)} samples (removed {len(ds_clean) - len(ds_clf)})")
                else:
                    raise e
            gc.collect()

    # Create regression dataset (remove NOMAP and renormalize)
    ds_reg = ds_clean
    if cfg.evaluation.tasks.regress and "cell_types" in cfg.data:
        nomap_idx = getattr(cfg.data.cell_types, "nomap_index", None)
        n_classes = getattr(cfg.data.cell_types, "num_classes", None)

        if nomap_idx is not None:
            print(f"Removing NOMAP (index={nomap_idx}) from regression labels")

            def remove_nomap(example):
                ratios = list(example[cfg.training.regression.label_key])
                if 0 <= nomap_idx < len(ratios):
                    ratios = [r for i, r in enumerate(ratios) if i != nomap_idx]
                    total = sum(ratios)
                    if total > 0:
                        ratios = [r / total for r in ratios]
                example[cfg.training.regression.label_key] = ratios
                return example

            try:
                ds_reg = ds_reg.map(remove_nomap, num_proc=16)
            except RuntimeError as e:
                if "subprocesses has abruptly died" in str(e):
                    print(f"Multiprocessing failed, falling back to single-threaded mapping: {e}")
                    ds_reg = ds_reg.map(remove_nomap)
                else:
                    raise e
            gc.collect()

    print(f"Final dataset sizes: classification={len(ds_clf)}, regression={len(ds_reg)}")

    # Assert that we have data for the tasks we're trying to evaluate
    if cfg.evaluation.tasks.classify and len(ds_clf) == 0:
        raise ValueError("No classification samples available after filtering.")
    if cfg.evaluation.tasks.regress and len(ds_reg) == 0:
        raise ValueError("No regression samples available after filtering.")

    return ds_clf, ds_reg


def evaluate_finetune_model(cfg: DictConfig, dataset_path: str, align_method: str, train_ratio: float = 0.8):
    """Evaluate a finetune model with proper train/test splitting.

    Args:
        cfg: Configuration object
        dataset_path: Path to the HF dataset (contains only test data)
        align_method: Alignment method for evaluation
        train_ratio: Fraction of data to use for training (default 0.8)
    """
    print("\n=== Evaluating Finetune Model ===")
    print(f"Dataset: {dataset_path}")
    print(f"Method: {align_method}")
    print(f"Train ratio: {train_ratio}")

    # Load and preprocess data
    ds_clf, ds_reg = load_and_preprocess_data_single(dataset_path, cfg, align_method)

    print(f"Original dataset size: {len(ds_clf)} samples")

    # Split into train/test
    print(f"Splitting into {train_ratio:.1%} train, {1-train_ratio:.1%} test with seed {cfg.evaluation.seed}")
    train_ds_clf, test_ds_clf = split_dataset(ds_clf, train_ratio=train_ratio, random_state=cfg.evaluation.seed)
    train_ds_reg, test_ds_reg = split_dataset(ds_reg, train_ratio=train_ratio, random_state=cfg.evaluation.seed)

    print(f"After split: train={len(train_ds_clf)}, test={len(test_ds_clf)}")

    # Extract embeddings for training data
    print("  Extracting training embeddings...")
    train_data_clf, train_clf_labels = extract_embeddings(
        ds=train_ds_clf, valid_indices=None, cfg=cfg, align_method=align_method, task_type="classification")
    train_data_reg, train_reg_labels = extract_embeddings(
        ds=train_ds_reg, valid_indices=None, cfg=cfg, align_method=align_method, task_type="regression")

    # Extract embeddings for test data
    print("  Extracting test embeddings...")
    test_data_clf, test_clf_labels = extract_embeddings(
        ds=test_ds_clf, valid_indices=None, cfg=cfg, align_method=align_method, task_type="classification")
    test_data_reg, test_reg_labels = extract_embeddings(
        ds=test_ds_reg, valid_indices=None, cfg=cfg, align_method=align_method, task_type="regression")

    # Evaluate
    metrics = evaluate_simple(
        train_data_clf,
        test_data_clf,
        train_data_reg,
        test_data_reg,
        train_clf_labels,
        test_clf_labels,
        train_reg_labels,
        test_reg_labels,
        random_state=cfg.evaluation.seed,
        cfg=cfg
    )

    return {
        'method': align_method,
        'f1_macro': metrics.get('f1_macro'),
        'r2': metrics.get('r2'),
        'train_samples': len(train_ds_clf),
        'test_samples': len(test_ds_clf),
        'train_ratio': train_ratio
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate finetune models with proper train/test splitting')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to HF dataset (contains only test data)')
    parser.add_argument('--gex_model', type=str, required=True,
                        help='GEX model name (e.g., finetune_1pct_cfg1)')
    parser.add_argument('--align_method', type=str, default='unimodal_gex',
                        help='Alignment method (default: unimodal_gex)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of data for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting (default: 42)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')

    args = parser.parse_args()

    # Create a minimal config object
    cfg = OmegaConf.create({
        'data': {
            'img_embed_key': 'img_embed',
            'gex_embed_key': 'nicheformer_pool',
            'annotations': {
                'excluded_classes': [3, 16]
            },
            'cell_types': {
                'nomap_index': 9,
                'num_classes': 10
            }
        },
        'training': {
            'classification': {
                'label_key': 'annotation'
            },
            'regression': {
                'label_key': 'cell_type_ratio'
            }
        },
        'evaluation': {
            'tasks': {
                'classify': True,
                'regress': True
            },
            'seed': args.seed
        }
    })

    print("Starting Finetune Model Evaluation")
    print("Configuration:")
    print(f"  - Dataset: {args.dataset_path}")
    print(f"  - GEX model: {args.gex_model}")
    print(f"  - Method: {args.align_method}")
    print(f"  - Train ratio: {args.train_ratio}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Output: {args.output_dir}")

    # Evaluate
    result = evaluate_finetune_model(cfg, args.dataset_path, args.align_method, args.train_ratio)

    # Print and save results
    print_results_table(result)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_data = {
        "experiment": {
            "gex_model": args.gex_model,
            "align_method": args.align_method,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "dataset_path": args.dataset_path
        },
        "results": [result]
    }

    result_data = make_serializable(result_data)

    json_path = output_dir / f"finetune_eval_{args.gex_model}_{args.align_method}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    print("\n=== Finetune Model Evaluation Complete ===")


if __name__ == "__main__":
    main()
