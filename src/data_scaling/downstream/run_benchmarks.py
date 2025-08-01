import os
import json
import torch
import hydra
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import ConcatDataset, TensorDataset
from datasets import load_from_disk, load_dataset
from torch.serialization import add_safe_globals

# Add TensorDataset to safe globals for deserialization
add_safe_globals([TensorDataset])

from data_scaling.downstream.linear_probe import run_loocv_linear_probe, train_linear_probe
from data_scaling.downstream.spatial_neighbor_distance_bins import run_spatial_neighbor
from data_scaling.downstream.dataset_preparation import prepare_donor_tensor_datasets, load_tensor_dataset


def get_project_dir() -> Path:
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/Projects/till_richter/")
    return Path(os.path.expandvars(raw))


def load_and_filter_datasets(cfg: DictConfig) -> Dict[str, TensorDataset]:
    """Load datasets from enriched directory and filter for specific model combination."""
    project_dir = get_project_dir()
    hf_dir = project_dir / cfg.data.dataset / "hf_datasets"

    modality = cfg.evaluation.modality
    img_model = cfg.evaluation.img_model
    gex_model = cfg.evaluation.gex_model
    split = cfg.evaluation.split
    scale = cfg.evaluation.scale
    label_key = cfg.training.classification.label_key

    print(f"\nLoading and filtering datasets for:")
    print(f"  - Modality: {modality}")
    print(f"  - Image model: {img_model}")
    print(f"  - GEX model: {gex_model}")
    print(f"  - Split: {split}")
    print(f"  - Scale: {scale}")
    print(f"  - Label key: {label_key}")

    # Build expected directory pattern based on modality
    if modality == "multimodal":
        # For multimodal, we keep the existing pattern
        pattern = f"{gex_model}_{img_model}_test.S" if split == "test" else f"{gex_model}_{img_model}_{split}.{scale}"
    elif modality == "unimodal_img":
        # For unimodal image, use the new img_only pattern
        pattern = f"img_only_{img_model}_test.{scale}" if split == "test" else f"img_only_{img_model}_{split}.{scale}"
    elif modality == "unimodal_gex":
        # For unimodal GEX, use the new gex_only pattern with model name
        pattern = f"gex_only_{gex_model}_test.{scale}" if split == "test" else f"gex_only_{gex_model}_{split}.{scale}"
    else:
        raise ValueError(f"Invalid modality: {modality}")

    # Check if dataset exists
    test_dir = hf_dir / pattern
    if not test_dir.exists():
        raise ValueError(
            f"Test directory not found: {test_dir}\n"
            f"Available directories: {[d.name for d in hf_dir.glob('*')]}"
        )

    print(f"\nLoading test dataset from: {test_dir}")
    ds: Dataset = load_from_disk(str(test_dir))

    print("\nDataset features:")
    for feature, info in ds.features.items():
        shape = getattr(info, 'shape', 'N/A')
        print(f"  - {feature}: type={type(info)}, shape={shape}")

    if label_key not in ds.column_names:
        raise KeyError(f"Label key '{label_key}' not found in dataset columns: {ds.column_names}")

    # Sample check before full processing
    sample_label = ds[0][label_key]
    print(f"\nSample label value for key '{label_key}':")
    print(f"  - Value: {sample_label}")
    print(f"  - Type: {type(sample_label)}")
    if isinstance(sample_label, (list, np.ndarray)):
        print(f"  - Shape: {np.array(sample_label).shape}")
        if not np.isscalar(np.array(sample_label)).all():
            print("  - [Warning] Label is a vector, not a scalar. This may cause shape mismatch.")

    # Group by donor name prefix
    donor_datasets: Dict[str, list] = {}
    for sample in ds:
        donor = sample["name"].split("_")[0]
        donor_datasets.setdefault(donor, []).append(sample)

    print("\nFound donors:")
    for donor, samples in donor_datasets.items():
        print(f"  - {donor}: {len(samples)} samples")

    # Convert to tensor datasets
    tensor_datasets: Dict[str, TensorDataset] = {}
    for donor, samples in donor_datasets.items():
        if not samples:
            continue

        if not tensor_datasets:
            print(f"\nFirst sample from donor {donor}:")
            for k, v in samples[0].items():
                if isinstance(v, (list, np.ndarray)):
                    print(f"  - {k}: shape={np.array(v).shape}, type={type(v)}")
                else:
                    print(f"  - {k}: value={v}, type={type(v)}")

        try:
            # Prepare features and labels based on modality
            if modality == "multimodal":
                img_embeds = torch.tensor([s["img_uni_pool"] for s in samples])
                gex_embeds = torch.tensor([s["nicheformer_pool"] for s in samples])
                labels = torch.tensor([s[label_key] if isinstance(s[label_key], (int, float)) else np.argmax(s[label_key]) for s in samples])
                
                # Add regression labels if available and needed
                if cfg.evaluation.tasks.regress:
                    regression_key = cfg.training.regression.label_key
                    if regression_key in samples[0]:
                        # FIX: Use the entire cell type ratio vector, not just the first element
                        reg_labels = torch.tensor([s[regression_key] for s in samples])
                        tensor_datasets[donor] = TensorDataset(img_embeds, gex_embeds, labels, reg_labels)
                    else:
                        # Create dummy regression labels
                        reg_labels = torch.zeros(len(samples), 10)  # 10 cell types
                        tensor_datasets[donor] = TensorDataset(img_embeds, gex_embeds, labels, reg_labels)
                else:
                    tensor_datasets[donor] = TensorDataset(img_embeds, gex_embeds, labels)

            elif modality == "unimodal_img":
                img_embeds = torch.tensor([s["img_uni_pool"] for s in samples])
                labels = torch.tensor([s[label_key] if isinstance(s[label_key], (int, float)) else np.argmax(s[label_key]) for s in samples])
                print(f"\nImage embeddings shape for {donor}: {img_embeds.shape}")
                
                # Add regression labels if available and needed
                if cfg.evaluation.tasks.regress:
                    regression_key = cfg.training.regression.label_key
                    if regression_key in samples[0]:
                        # FIX: Use the entire cell type ratio vector, not just the first element
                        reg_labels = torch.tensor([s[regression_key] for s in samples])
                        tensor_datasets[donor] = TensorDataset(img_embeds, labels, reg_labels)
                    else:
                        # Create dummy regression labels
                        reg_labels = torch.zeros(len(samples), 10)  # 10 cell types
                        tensor_datasets[donor] = TensorDataset(img_embeds, labels, reg_labels)
                else:
                    tensor_datasets[donor] = TensorDataset(img_embeds, labels)

            elif modality == "unimodal_gex":
                gex_embeds = torch.tensor([s["nicheformer_pool"] for s in samples])
                labels = torch.tensor([s[label_key] if isinstance(s[label_key], (int, float)) else np.argmax(s[label_key]) for s in samples])
                
                # Add regression labels if available and needed
                if cfg.evaluation.tasks.regress:
                    regression_key = cfg.training.regression.label_key
                    if regression_key in samples[0]:
                        # FIX: Use the entire cell type ratio vector, not just the first element
                        reg_labels = torch.tensor([s[regression_key] for s in samples])
                        tensor_datasets[donor] = TensorDataset(gex_embeds, labels, reg_labels)
                    else:
                        # Create dummy regression labels
                        reg_labels = torch.zeros(len(samples), 10)  # 10 cell types
                        tensor_datasets[donor] = TensorDataset(gex_embeds, labels, reg_labels)
                else:
                    tensor_datasets[donor] = TensorDataset(gex_embeds, labels)

            else:
                raise ValueError(f"Invalid modality: {modality}")

        except Exception as e:
            print(f"\n[ERROR] Failed to convert donor {donor} to TensorDataset.")
            print(f"  - Example label: {samples[0].get(label_key)}")
            raise e

    # Validate consistency of embedding dimensions
    if tensor_datasets:
        first_tensor = next(iter(tensor_datasets.values()))
        expected_dim = first_tensor.tensors[0].shape[1]
        print(f"\nValidating embedding dimensions (expected dim: {expected_dim})")
        for donor, ds in tensor_datasets.items():
            curr_dim = ds.tensors[0].shape[1]
            print(f"  - {donor}: {curr_dim}")
            if curr_dim != expected_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions:\n"
                    f"Expected: {expected_dim}, Got: {curr_dim} for donor {donor}"
                )

    print(f"\n✅ Successfully loaded data for {len(tensor_datasets)} donors:")
    for donor, ds in tensor_datasets.items():
        print(f"  - {donor}: {len(ds)} samples")

    return tensor_datasets


def setup_results_dir(cfg) -> Path:
    project_dir = get_project_dir()
    # Include model combination in results path
    modality = cfg.evaluation.modality
    if modality == "multimodal":
        model_str = f"{cfg.evaluation.align_method}_{cfg.evaluation.img_model}_{cfg.evaluation.gex_model}"
    elif modality == "unimodal_img":
        model_str = f"img_{cfg.evaluation.img_model}"
    else:  # unimodal_gex
        model_str = f"gex_{cfg.evaluation.gex_model}"
        
    results_dir = project_dir / "results" / cfg.data.dataset / model_str
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def ensure_tensor_datasets(cfg: DictConfig) -> Dict[str, TensorDataset]:
    # Build dataset name based on modality
    modality = cfg.evaluation.modality
    if modality == "multimodal":
        combined_name = f"combined_dataset_img_{cfg.evaluation.img_model}_gex_{cfg.evaluation.gex_model}"
    elif modality == "unimodal_img":
        combined_name = f"img_only_dataset_{cfg.evaluation.img_model}"
    else:  # unimodal_gex
        combined_name = f"gex_only_dataset_{cfg.evaluation.gex_model}"
    
    tensor_datasets = prepare_donor_tensor_datasets(
        dataset_name=combined_name,
        test_donors=list(cfg.data.multimodal.test),
        embed_keys={"img": cfg.data.img_embed_key, "gex": cfg.data.gex_embed_key},
        label_key=cfg.training.classification.label_key,
        num_proc=cfg.data.get("num_proc", 4),
    )
    return tensor_datasets


def get_donor_splits(donor_ids: List[str]) -> List[Tuple[List[str], List[str]]]:
    return [([d for d in donor_ids if d != test], [test]) for test in donor_ids]


def build_concat_dataset(donors: List[str], all_datasets: Dict[str, TensorDataset]) -> ConcatDataset:
    datasets = [all_datasets[d] for d in donors]
    return ConcatDataset(datasets)


def compute_random_baseline(train_labels):
    """Compute random baseline based on class distribution."""
    unique, counts = np.unique(train_labels, return_counts=True)
    probs = counts / len(train_labels)
    random_acc = np.sum(probs ** 2)  # If we always predict most common class
    return {
        'random_accuracy': random_acc,
        'random_f1_macro': 1.0 / len(unique)  # For balanced F1
    }

def run_classification(cfg: DictConfig, test_datasets: Dict[str, TensorDataset]) -> Dict[str, float]:
    print("\n=== Running Classification Evaluation ===")
    task_type = "classification"
    metrics = {}
    
    # Determine evaluation strategy
    eval_strategy = cfg.evaluation.get('strategy', 'test_only')  # Default to test_only for backward compatibility
    print(f"\nUsing evaluation strategy: {eval_strategy}")
    
    # Function to run evaluation for a specific dataset
    def evaluate_dataset(datasets, pca_dim=None, suffix=""):
        if eval_strategy == 'test_only':
            # Use test set for both training and testing
            print(f"Training and testing on test set to evaluate linear separability{suffix}")
            test_data = ConcatDataset([ds for ds in datasets.values()])
            test_samples = [test_data[i] for i in range(len(test_data))]
            # Handle different tensor structures based on modality and tasks
            if len(test_samples[0]) == 4:  # multimodal with regression
                X_img = torch.stack([s[0] for s in test_samples])
                X_gex = torch.stack([s[1] for s in test_samples])
                X = torch.cat([X_img, X_gex], dim=1)
                y = torch.stack([s[2] for s in test_samples])  # classification labels
            elif len(test_samples[0]) == 3:  # unimodal with regression
                X = torch.stack([s[0] for s in test_samples])
                y = torch.stack([s[1] for s in test_samples])  # classification labels
            else:  # standard case
                X = torch.stack([s[0] for s in test_samples])
                y = torch.stack([s[1] for s in test_samples])
            train_data = X
            train_labels = y
            test_data = X
            test_labels = y
            
            # Apply PCA if requested
            if pca_dim is not None:
                from sklearn.decomposition import PCA
                print(f"\nApplying PCA to reduce dimensionality to {pca_dim} components")
                pca = PCA(n_components=pca_dim)
                train_data = torch.tensor(pca.fit_transform(train_data.cpu().numpy())).float()
                test_data = torch.tensor(pca.transform(test_data.cpu().numpy())).float()
                explained_var = sum(pca.explained_variance_ratio_) * 100
                print(f"Explained variance with {pca_dim} components: {explained_var:.2f}%")
            
            # Per-sample analysis if requested
            if cfg.evaluation.per_sample:
                print("\nRunning per-sample analysis...")
                per_sample_metrics = []
                for i in range(len(train_data)):
                    sample_data = train_data[i:i+1]  # Keep batch dimension
                    sample_label = train_labels[i:i+1]
                    
                    # Train and evaluate on single sample
                    sample_metrics = train_linear_probe(
                        cfg=cfg,
                        train_embeddings=sample_data,
                        train_labels=sample_label,
                        test_embeddings=sample_data,
                        test_labels=sample_label,
                        task_type=task_type,
                    )
                    per_sample_metrics.append(sample_metrics)
                
                # Compute statistics over per-sample metrics
                per_sample_summary = {
                    f'per_sample_accuracy_mean{suffix}': np.mean([m['accuracy'] for m in per_sample_metrics]),
                    f'per_sample_accuracy_std{suffix}': np.std([m['accuracy'] for m in per_sample_metrics]),
                    f'per_sample_f1_macro_mean{suffix}': np.mean([m['f1_macro'] for m in per_sample_metrics]),
                    f'per_sample_f1_macro_std{suffix}': np.std([m['f1_macro'] for m in per_sample_metrics]),
                }
                print("\nPer-sample metrics:")
                for k, v in per_sample_summary.items():
                    print(f"  {k}: {v:.4f}")
            
        else:  # 'train_test_split'
            print(f"Training on pretrain/finetune data, testing on test set{suffix}")
            # Load pretrain and finetune data for training
            train_datasets = {}
            for split in ['pretrain', 'finetune']:
                pattern = f"{cfg.evaluation.img_model}_{split}.{cfg.data[f'{split}_split']}"
                path = get_project_dir() / cfg.data.dataset / "tds" / f"{pattern}.pt"
                if path.exists():
                    print(f"Loading {split} data from {path}")
                    train_datasets.update(torch.load(str(path)))
            
            # Prepare training data
            train_data = ConcatDataset([ds for ds in train_datasets.values()])
            train_samples = [train_data[i] for i in range(len(train_data))]
            train_data = torch.stack([s[0] for s in train_samples])
            train_labels = torch.stack([s[1] for s in train_samples])
            
            # Prepare test data
            test_data = ConcatDataset([ds for ds in datasets.values()])
            test_samples = [test_data[i] for i in range(len(test_data))]
            test_data = torch.stack([s[0] for s in test_samples])
            test_labels = torch.stack([s[1] for s in test_samples])
        
        # Add random baseline based on test set distribution
        baseline = compute_random_baseline(test_labels.numpy())
        print("\nRandom baseline metrics:")
        for k, v in baseline.items():
            print(f"  {k}: {v:.4f}")
        
        # Train and evaluate
        metrics = train_linear_probe(
            cfg=cfg,
            train_embeddings=train_data.float(),
            train_labels=train_labels.float(),
            test_embeddings=test_data.float(),
            test_labels=test_labels.float(),
            task_type=task_type,
        )
        
        # Combine metrics
        suffix = f"_pca{pca_dim}" if pca_dim else ""
        summary = {
            f'classification_accuracy{suffix}': metrics['accuracy'],
            f'classification_f1_macro{suffix}': metrics['f1_macro'],
            f'train_accuracy{suffix}': metrics['train_accuracy'],
            f'train_f1_macro{suffix}': metrics['train_f1_macro'],
            f'random_accuracy{suffix}': baseline['random_accuracy'],
            f'random_f1_macro{suffix}': baseline['random_f1_macro']
        }
        
        if pca_dim:
            summary[f'explained_variance_pca{pca_dim}'] = explained_var
            
        if cfg.evaluation.per_sample:
            summary.update(per_sample_summary)
        
        return summary
    
    # Run evaluation on original embeddings
    metrics.update(evaluate_dataset(test_datasets))
    
    # Run evaluation with different PCA dimensions
    for pca_dim in [512, 128, 64, 32]:
        print(f"\n=== Running Classification with PCA ({pca_dim} components) ===")
        pca_metrics = evaluate_dataset(test_datasets, pca_dim=pca_dim)
        metrics.update(pca_metrics)
    
    return metrics

def run_regression(cfg: DictConfig, test_datasets: Dict[str, TensorDataset]) -> Dict[str, float]:
    print("\n=== Running Regression Evaluation ===")
    task_type = "regression"
    
    # Determine evaluation strategy
    eval_strategy = cfg.evaluation.get('strategy', 'test_only')  # Default to test_only for backward compatibility
    print(f"\nUsing evaluation strategy: {eval_strategy}")
    
    # Function to run evaluation for a specific dataset
    def evaluate_dataset(datasets, suffix=""):
        if eval_strategy == 'test_only':
            # Use test set for both training and testing
            print(f"Training and testing on test set to evaluate linear separability{suffix}")
            test_data = ConcatDataset([ds for ds in datasets.values()])
            test_samples = [test_data[i] for i in range(len(test_data))]
            # Handle different tensor structures based on modality and tasks
            if len(test_samples[0]) == 4:  # multimodal with regression
                X_img = torch.stack([s[0] for s in test_samples])
                X_gex = torch.stack([s[1] for s in test_samples])
                X = torch.cat([X_img, X_gex], dim=1)
                y = torch.stack([s[3] for s in test_samples])  # regression labels
            elif len(test_samples[0]) == 3:  # unimodal with regression
                X = torch.stack([s[0] for s in test_samples])
                y = torch.stack([s[2] for s in test_samples])  # regression labels
            else:  # standard case
                X = torch.stack([s[0] for s in test_samples])
                y = torch.stack([s[1] for s in test_samples])
            train_data = X
            train_labels = y
            test_data = X
            test_labels = y
            
        else:  # 'train_test_split'
            print(f"Training on pretrain/finetune data, testing on test set{suffix}")
            # Load pretrain and finetune data for training
            train_datasets = {}
            for split in ['pretrain', 'finetune']:
                pattern = f"{cfg.evaluation.img_model}_{split}.{cfg.data[f'{split}_split']}"
                path = get_project_dir() / cfg.data.dataset / "tds" / f"{pattern}.pt"
                if path.exists():
                    print(f"Loading {split} data from {path}")
                    train_datasets.update(torch.load(str(path)))
            
            # Prepare training data
            train_data = ConcatDataset([ds for ds in train_datasets.values()])
            train_samples = [train_data[i] for i in range(len(train_data))]
            train_data = torch.stack([s[0] for s in train_samples])
            train_labels = torch.stack([s[1] for s in train_samples])
            
            # Prepare test data
            test_data = ConcatDataset([ds for ds in datasets.values()])
            test_samples = [test_data[i] for i in range(len(test_data))]
            test_data = torch.stack([s[0] for s in test_samples])
            test_labels = torch.stack([s[1] for s in test_samples])
        
        # Train and evaluate
        metrics = train_linear_probe(
            cfg=cfg,
            train_embeddings=train_data.float(),
            train_labels=train_labels.float(),
            test_embeddings=test_data.float(),
            test_labels=test_labels.float(),
            task_type=task_type,
        )
        
        # Return metrics with suffix
        return {
            f'regression_r2{suffix}': metrics['r2'],
            f'regression_mse{suffix}': metrics['mse']
        }
    
    # Run evaluation on original embeddings
    original_datasets = {k: v for k, v in test_datasets.items() if "_pca" not in k}
    metrics = evaluate_dataset(original_datasets)
    
    # Run evaluation on PCA variants if enabled
    if hasattr(cfg.evaluation, 'run_pca') and cfg.evaluation.run_pca:
        for n_components in [1024, 512, 128]:
            pca_datasets = {k: v for k, v in test_datasets.items() if f"_pca{n_components}" in k}
            if pca_datasets:  # Only if we have datasets with this number of components
                print(f"\n=== Running Regression with PCA ({n_components} components) ===")
                pca_metrics = evaluate_dataset(pca_datasets, f"_pca{n_components}")
                metrics.update(pca_metrics)
    
    return metrics


def run_spatial(cfg: DictConfig) -> Dict[str, float]:
    print("Running spatial neighbor prediction...")
    try:
        summary = run_spatial_neighbor(cfg)
        out = {}
        for k, v in summary.items():
            if k == "bin_midpoints":
                for dist_key, dist_val in v.items():
                    out[f"spatial_{dist_key}"] = dist_val
            else:
                out[f"spatial_{k}_mean"] = v.get("mean", 0.0)
                out[f"spatial_{k}_std"] = v.get("std", 0.0)
        return out
    except Exception as e:
        print(f"Spatial neighbor evaluation failed: {e}")
        bins = cfg.training.spatial_neighbor.get("num_bins", 5)
        return {
            **{f"spatial_bin_{i}_mean": 0.0 for i in range(bins)},
            **{f"spatial_bin_{i}_std": 0.0 for i in range(bins)},
            **{f"spatial_bin_{i}_distance": 0.0 for i in range(bins)},
        }


def run_unimodal_baselines(cfg: DictConfig) -> Dict[str, float]:
    """Run unimodal baselines for both IMG and GEX encoders."""
    print("\n=== Running Unimodal Baselines ===")
    all_metrics = {}
    
    # Save original modality
    original_modality = cfg.evaluation.modality
    
    # Test IMG encoder
    print("\n--- Testing Image Encoder ---")
    cfg.evaluation.modality = "unimodal_img"
    tensor_datasets = load_and_filter_datasets(cfg)
    donor_ids = list(tensor_datasets.keys())
    donor_splits = get_donor_splits(donor_ids)
    
    if cfg.evaluation.tasks.classify:
        metrics = run_classification(cfg, tensor_datasets)
        all_metrics.update({f"img_{k}": v for k, v in metrics.items()})
    
    if cfg.evaluation.tasks.regress:
        metrics = run_regression(cfg, tensor_datasets)
        all_metrics.update({f"img_{k}": v for k, v in metrics.items()})
    
    # Test GEX encoder
    print("\n--- Testing GEX Encoder ---")
    cfg.evaluation.modality = "unimodal_gex"
    tensor_datasets = load_and_filter_datasets(cfg)
    donor_ids = list(tensor_datasets.keys())
    donor_splits = get_donor_splits(donor_ids)
    
    if cfg.evaluation.tasks.classify:
        metrics = run_classification(cfg, tensor_datasets)
        all_metrics.update({f"gex_{k}": v for k, v in metrics.items()})
    
    if cfg.evaluation.tasks.regress:
        metrics = run_regression(cfg, tensor_datasets)
        all_metrics.update({f"gex_{k}": v for k, v in metrics.items()})
    
    # Restore original modality
    cfg.evaluation.modality = original_modality
    return all_metrics


def make_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return make_serializable(OmegaConf.to_container(obj, resolve=True))
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj


def extract_task_results(metrics: dict, task: str) -> dict:
    """Extract task-specific results from metrics dictionary."""
    if task == "classification":
        return {
            "accuracy": metrics.get("classification_accuracy", metrics.get("accuracy", None)),
            "f1_macro": metrics.get("classification_f1_macro", metrics.get("f1_macro", None)),
            "train_accuracy": metrics.get("train_accuracy", None),
            "train_f1_macro": metrics.get("train_f1_macro", None),
            "random_accuracy": metrics.get("random_accuracy", None),
            "random_f1_macro": metrics.get("random_f1_macro", None),
        }
    elif task == "regression":
        return {
            "r2": metrics.get("regression_r2", metrics.get("r2", None)),
            "mse": metrics.get("regression_mse", metrics.get("mse", None)),
        }
    else:
        return {}

def aggregate_metrics(metrics_list, explained_var=None, cumulative_var=None):
    # metrics_list: list of dicts with keys train_accuracy, train_f1_macro, accuracy, f1_macro, train_r2, train_mse, r2, mse
    keys = [
        "train_accuracy", "train_f1_macro", "accuracy", "f1_macro",
        "train_r2", "train_mse", "r2", "mse"
    ]
    arr = {k: [m[k] for m in metrics_list if k in m and m[k] is not None] for k in keys}
    mean_metrics = {
        "train_accuracy": np.mean(arr["train_accuracy"]) if arr["train_accuracy"] else None,
        "train_f1_macro": np.mean(arr["train_f1_macro"]) if arr["train_f1_macro"] else None,
        "test_accuracy": np.mean(arr["accuracy"]) if arr["accuracy"] else None,
        "test_f1_macro": np.mean(arr["f1_macro"]) if arr["f1_macro"] else None,
        "train_r2": np.mean(arr["train_r2"]) if arr["train_r2"] else None,
        "train_mse": np.mean(arr["train_mse"]) if arr["train_mse"] else None,
        "test_r2": np.mean(arr["r2"]) if arr["r2"] else None,
        "test_mse": np.mean(arr["mse"]) if arr["mse"] else None,
    }
    std_metrics = {
        "train_accuracy": np.std(arr["train_accuracy"]) if arr["train_accuracy"] else None,
        "train_f1_macro": np.std(arr["train_f1_macro"]) if arr["train_f1_macro"] else None,
        "test_accuracy": np.std(arr["accuracy"]) if arr["accuracy"] else None,
        "test_f1_macro": np.std(arr["f1_macro"]) if arr["f1_macro"] else None,
        "train_r2": np.std(arr["train_r2"]) if arr["train_r2"] else None,
        "train_mse": np.std(arr["train_mse"]) if arr["train_mse"] else None,
        "test_r2": np.std(arr["r2"]) if arr["r2"] else None,
        "test_mse": np.std(arr["mse"]) if arr["mse"] else None,
    }
    return {
        "explained_variance": explained_var,
        "cumulative_explained_variance": cumulative_var,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "per_donor": {"all": metrics_list[0] if len(metrics_list) == 1 else {}}
    }

def save_results(results_dir: Path, cfg: DictConfig, metrics: dict):
    """Save evaluation results to JSON."""
    result = {
        "experiment": {
            "aligner": cfg.models.method,
            "embedder": {
                "image": cfg.data.img_embed_key,
                "gex": cfg.data.gex_embed_key,
            },
            "checkpoint_path": cfg.models.checkpoint_path,
            "dataset": cfg.data.dataset,
            "modality": cfg.evaluation.modality,
            "img_model": cfg.evaluation.img_model,
            "gex_model": cfg.evaluation.gex_model,
            "evaluation_strategy": "full_test_set",  # Always using full test set for this benchmark
        },
        "data": {
            "test_donors": cfg.data.multimodal.test,
        },
        "training": {
            "classification": cfg.training.classification,
            "regression": cfg.training.regression,
            "spatial_neighbor": cfg.training.spatial_neighbor,
        },
        "results": {
            "full_dim": {
                "classification": aggregate_metrics(
                    [extract_task_results(metrics, "classification")],
                    metrics.get("explained_variance_pca512"),
                    metrics.get("explained_variance_pca512")
                ),
                "regression": aggregate_metrics(
                    [extract_task_results(metrics, "regression")],
                    metrics.get("explained_variance_pca512"),
                    metrics.get("explained_variance_pca512")
                ),
            }
        }
    }

    # Add PCA results if available
    for dim in [512, 128, 64, 32]:
        pca_metrics = {k: v for k, v in metrics.items() if f"pca{dim}" in k}
        if pca_metrics:
            result["results"][f"pca_{dim}"] = {
                "classification": aggregate_metrics(
                    [extract_task_results(pca_metrics, "classification")],
                    pca_metrics.get(f"explained_variance_pca{dim}"),
                    pca_metrics.get(f"explained_variance_pca{dim}")
                ),
                "regression": aggregate_metrics(
                    [extract_task_results(pca_metrics, "regression")],
                    pca_metrics.get(f"explained_variance_pca{dim}"),
                    pca_metrics.get(f"explained_variance_pca{dim}")
                ),
                "explained_variance": metrics.get(f"explained_variance_pca{dim}", None),
                "cumulative_explained_variance": metrics.get(f"explained_variance_pca{dim}", None),
                "top_components": {
                    f"pc{i+1}": metrics.get(f"pc{i+1}_explained_variance_pca{dim}", None)
                    for i in range(10)  # Store top 10 components
                }
            }

    result = make_serializable(result)
    
    # Print results
    print("\n================== FINAL RESULTS ==================\n")
    print("Evaluation Strategy: Full Test Set")
    
    # Print full dimensionality results
    print("\nFull Dimensionality Results:")
    
    # Print classification results
    if "classification" in result["results"]["full_dim"]:
        print("  Classification:")
        class_metrics = result["results"]["full_dim"]["classification"]
        if class_metrics:
            accuracy = class_metrics.get("mean_metrics", {}).get("test_accuracy")
            f1_macro = class_metrics.get("mean_metrics", {}).get("test_f1_macro")
            train_accuracy = class_metrics.get("mean_metrics", {}).get("train_accuracy")
            train_f1_macro = class_metrics.get("mean_metrics", {}).get("train_f1_macro")
            random_accuracy = class_metrics.get("mean_metrics", {}).get("random_accuracy")
            random_f1_macro = class_metrics.get("mean_metrics", {}).get("random_f1_macro")
            
            if accuracy is not None:
                print(f"    Test Accuracy:  {accuracy:.4f}")
            if f1_macro is not None:
                print(f"    Test F1 Macro:  {f1_macro:.4f}")
            if train_accuracy is not None:
                print(f"    Train Accuracy: {train_accuracy:.4f}")
            if train_f1_macro is not None:
                print(f"    Train F1 Macro: {train_f1_macro:.4f}")
            if random_accuracy is not None:
                print(f"    Random Accuracy: {random_accuracy:.4f}")
            if random_f1_macro is not None:
                print(f"    Random F1 Macro: {random_f1_macro:.4f}")
    
    # Print regression results
    if "regression" in result["results"]["full_dim"]:
        print("  Regression:")
        reg_metrics = result["results"]["full_dim"]["regression"]
        if reg_metrics:
            r2 = reg_metrics.get("mean_metrics", {}).get("test_r2")
            mse = reg_metrics.get("mean_metrics", {}).get("test_mse")
            
            if r2 is not None:
                print(f"    Test R²:        {r2:.4f}")
            if mse is not None:
                print(f"    Test MSE:       {mse:.4f}")

    # Print PCA results
    for dim in [512, 128, 64, 32]:
        pca_key = f"pca_{dim}"
        if pca_key in result["results"]:
            print(f"\nPCA {dim} Components Results:")
            
            # Print explained variance
            explained_var = result["results"][pca_key].get("explained_variance")
            cumulative_var = result["results"][pca_key].get("cumulative_explained_variance")
            if explained_var is not None:
                print(f"  Explained Variance: {explained_var:.2f}%")
                if cumulative_var is not None:
                    print(f"  Cumulative Explained Variance: {cumulative_var:.2f}%")
            
            # Print classification results
            if "classification" in result["results"][pca_key]:
                print("  Classification:")
                class_metrics = result["results"][pca_key]["classification"]
                if class_metrics:
                    accuracy = class_metrics.get("mean_metrics", {}).get("test_accuracy")
                    f1_macro = class_metrics.get("mean_metrics", {}).get("test_f1_macro")
                    train_accuracy = class_metrics.get("mean_metrics", {}).get("train_accuracy")
                    train_f1_macro = class_metrics.get("mean_metrics", {}).get("train_f1_macro")
                    
                    if accuracy is not None:
                        print(f"    Test Accuracy:  {accuracy:.4f}")
                    if f1_macro is not None:
                        print(f"    Test F1 Macro:  {f1_macro:.4f}")
                    if train_accuracy is not None:
                        print(f"    Train Accuracy: {train_accuracy:.4f}")
                    if train_f1_macro is not None:
                        print(f"    Train F1 Macro: {train_f1_macro:.4f}")
            
            # Print regression results
            if "regression" in result["results"][pca_key]:
                print("  Regression:")
                reg_metrics = result["results"][pca_key]["regression"]
                if reg_metrics:
                    r2 = reg_metrics.get("mean_metrics", {}).get("test_r2")
                    mse = reg_metrics.get("mean_metrics", {}).get("test_mse")
                    
                    if r2 is not None:
                        print(f"    Test R²:        {r2:.4f}")
                    if mse is not None:
                        print(f"    Test MSE:       {mse:.4f}")

    print("\n===================================================\n")

    # Save to file
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "full_test_set_results.json"  # Renamed to indicate strategy
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to JSON: {json_path}")


@hydra.main(config_path='../../../configs', config_name='downstream.yaml')
def main(cfg: DictConfig):
    # Add per-sample flag to config if not present
    if not hasattr(cfg.evaluation, 'per_sample'):
        cfg.evaluation.per_sample = False

    print(f"Starting downstream evaluation")
    print(f"Configuration:")
    print(f"  - Modality: {cfg.evaluation.modality}")
    if cfg.evaluation.modality == "multimodal":
        print(f"  - Alignment method: {cfg.evaluation.align_method}")
    print(f"  - Image model: {cfg.evaluation.img_model}")
    print(f"  - GEX model: {cfg.evaluation.gex_model}")
    print(f"  - Evaluation strategy: Full test set")
    print(f"Tasks:")
    print(f"  - Classification: {cfg.evaluation.tasks.classify}")
    print(f"  - Regression: {cfg.evaluation.tasks.regress}")
    print(f"  - Spatial: {cfg.evaluation.tasks.spatial}")
    print(f"  - PCA dimensions: [512, 128, 64, 32]")

    results_dir = setup_results_dir(cfg)
    metrics = {}

    # Only run unimodal baselines if no specific modality is set
    if not hasattr(cfg.evaluation, 'modality') and (cfg.evaluation.tasks.classify or cfg.evaluation.tasks.regress):
        metrics.update(run_unimodal_baselines(cfg))

    # Run the requested modality (multimodal or specific unimodal)
    tensor_datasets = load_and_filter_datasets(cfg)

    if cfg.evaluation.tasks.classify:
        print("\n=== Running Classification Evaluation ===")
        # For each PCA dim and full, collect metrics_list (just one dict for full test set),
        # then aggregate and store in results dict as above.
        # At the end, pass this results dict to save_results, and print as in within-donor/LOOCV.
        for pca_dim in [512, 128, 64, 32]:
            print(f"\n=== Running Classification with PCA ({pca_dim} components) ===")
            pca_metrics = run_classification(cfg, tensor_datasets)
            metrics.update(pca_metrics)
        metrics.update(run_classification(cfg, tensor_datasets)) # Full test set

    if cfg.evaluation.tasks.regress:
        print("\n=== Running Regression Evaluation ===")
        # For each PCA dim and full, collect metrics_list (just one dict for full test set),
        # then aggregate and store in results dict as above.
        # At the end, pass this results dict to save_results, and print as in within-donor/LOOCV.
        for pca_dim in [1024, 512, 128]:
            pca_datasets = {k: v for k, v in tensor_datasets.items() if f"_pca{pca_dim}" in k}
            if pca_datasets:
                print(f"\n=== Running Regression with PCA ({pca_dim} components) ===")
                pca_metrics = run_regression(cfg, pca_datasets)
                metrics.update(pca_metrics)
        metrics.update(run_regression(cfg, tensor_datasets)) # Full test set

    if cfg.evaluation.tasks.spatial:
        print("\n=== Running Distance-Based Spatial Neighbor Evaluation ===")
        metrics.update(run_spatial(cfg))

    # Add evaluation config to results
    metrics.update({
        "modality": cfg.evaluation.modality,
        "img_model": cfg.evaluation.img_model,
        "gex_model": cfg.evaluation.gex_model,
        "align_method": cfg.evaluation.align_method if cfg.evaluation.modality == "multimodal" else None,
        "per_sample": cfg.evaluation.per_sample
    })

    save_results(results_dir, cfg, metrics)
    print("\n=== Evaluation Complete ===")

    return metrics


if __name__ == "__main__":
    main()
