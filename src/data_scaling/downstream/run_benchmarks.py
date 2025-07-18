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
from datasets import load_from_disk

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
    
    print(f"\nLoading and filtering datasets for:")
    print(f"  - Modality: {modality}")
    print(f"  - Image model: {img_model}")
    print(f"  - GEX model: {gex_model}")
    
    # Build expected directory pattern based on modality
    if modality == "multimodal":
        if split == "test":
            # Test split has no scale, kept S for backwards compatibility
            pattern = f"{gex_model}_{img_model}_test.S"
        else:
            pattern = f"{gex_model}_{img_model}_{split}.{scale}"
    elif modality == "unimodal_img":
        if split == "test":
            pattern = f"{img_model}_test.S"
        else:
            pattern = f"{img_model}_{split}.{scale}"
    elif modality == "unimodal_gex":
        if split == "test":
            pattern = f"{gex_model}_test.S"
        else:
            pattern = f"{gex_model}_{split}.{scale}"
    else:
        raise ValueError(f"Invalid modality: {modality}")
    
    # Find matching directory
    test_dir = hf_dir / pattern
    if not test_dir.exists():
        raise ValueError(
            f"Test directory not found: {test_dir}\n"
            f"Available directories: {[d.name for d in hf_dir.glob('*')]}"
        )
    
    print(f"Loading test dataset from {test_dir}")
    ds = load_from_disk(str(test_dir))
    
    # Group by donor
    donor_datasets = {}
    for sample in ds:
        donor = sample["name"].split("_")[0]
        if donor not in donor_datasets:
            donor_datasets[donor] = []
        donor_datasets[donor].append(sample)
    
    # Convert to tensor datasets
    tensor_datasets = {}
    for donor, samples in donor_datasets.items():
        if not samples:
            continue
            
        # Extract embeddings based on modality
        if modality == "multimodal":
            img_embeds = torch.tensor([s["img_uni_pool"] for s in samples])
            gex_embeds = torch.tensor([s["nicheformer_pool"] for s in samples])
            labels = torch.tensor([s[cfg.training.classification.label_key] for s in samples])
            tensor_datasets[donor] = TensorDataset(img_embeds, gex_embeds, labels)
        elif modality == "unimodal_img":
            img_embeds = torch.tensor([s["img_uni_pool"] for s in samples])
            labels = torch.tensor([s[cfg.training.classification.label_key] for s in samples])
            tensor_datasets[donor] = TensorDataset(img_embeds, labels)
        else:  # unimodal_gex
            gex_embeds = torch.tensor([s["nicheformer_pool"] for s in samples])
            labels = torch.tensor([s[cfg.training.classification.label_key] for s in samples])
            tensor_datasets[donor] = TensorDataset(gex_embeds, labels)
    
    # Validate embedding dimensions
    first_donor = next(iter(tensor_datasets.values()))
    first_shape = first_donor.tensors[0].shape[1]  # First embedding dimension
    for donor, ds in tensor_datasets.items():
        if ds.tensors[0].shape[1] != first_shape:
            raise ValueError(
                f"Inconsistent embedding dimensions!\n"
                f"First donor had shape {first_shape}\n"
                f"Donor {donor} has shape {ds.tensors[0].shape[1]}"
            )
    
    print(f"\nSuccessfully loaded data for {len(tensor_datasets)} donors:")
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


def run_classification(cfg: DictConfig, donor_splits, all_datasets):
    print("Starting classification...")
    task_type = "classification"
    all_metrics = []
    donor_results = {}

    for i, (train_donors, test_donor) in enumerate(donor_splits):
        test_name = test_donor[0]
        print(f"[{i+1}/{len(donor_splits)}] ➔ Test donor: {test_name}")
        train_data = build_concat_dataset(train_donors, all_datasets)
        test_data = build_concat_dataset(test_donor, all_datasets)

        train_samples = [train_data[i] for i in range(len(train_data))]
        test_samples = [test_data[i] for i in range(len(test_data))]

        if cfg.evaluation.modality == "multimodal":
            train_img = torch.stack([s[0] for s in train_samples])
            train_gex = torch.stack([s[1] for s in train_samples])
            train_y = torch.stack([s[2] for s in train_samples])
            test_img = torch.stack([s[0] for s in test_samples])
            test_gex = torch.stack([s[1] for s in test_samples])
            test_y = torch.stack([s[2] for s in test_samples])
            X_train = torch.cat([train_img, train_gex], dim=-1)
            X_test = torch.cat([test_img, test_gex], dim=-1)
        else:
            X_train = torch.stack([s[0] for s in train_samples])
            train_y = torch.stack([s[1] for s in train_samples])
            X_test = torch.stack([s[0] for s in test_samples])
            test_y = torch.stack([s[1] for s in test_samples])

        metrics = train_linear_probe(
            cfg=cfg,
            train_embeddings=X_train.float(),
            train_labels=train_y.float(),
            test_embeddings=X_test.float(),
            test_labels=test_y.float(),
            task_type=task_type,
        )

        # Aggregate metrics
        all_metrics.append(metrics)
        for k, v in metrics.items():
            if k in {"y_true", "y_pred", "loss_curve", "grad_norms"}:
                continue
            donor_results[f"classification_donor_{test_name}_{k}"] = float(v)

    accs = [m['accuracy'] for m in all_metrics]
    summary = {
        'classification_accuracy_mean': float(np.mean(accs)),
        'classification_accuracy_std': float(np.std(accs)),
    }
    return {**summary, **donor_results}

def run_regression(cfg: DictConfig, donor_splits: List[Tuple[List[str], List[str]]], all_datasets: Dict[str, TensorDataset]) -> Dict[str, float]:
    print("Starting regression...")
    task_type = "regression"
    all_metrics = []
    donor_results = {}

    for i, (train_donors, test_donor) in enumerate(donor_splits):
        test_name = test_donor[0]
        print(f"[{i+1}/{len(donor_splits)}] ➔ Test donor: {test_name}")
        train_data = build_concat_dataset(train_donors, all_datasets)
        test_data = build_concat_dataset(test_donor, all_datasets)

        train_samples = [train_data[i] for i in range(len(train_data))]
        test_samples = [test_data[i] for i in range(len(test_data))]

        if cfg.evaluation.modality == "multimodal":
            train_img = torch.stack([s[0] for s in train_samples])
            train_gex = torch.stack([s[1] for s in train_samples])
            train_y = torch.stack([s[2] for s in train_samples])
            test_img = torch.stack([s[0] for s in test_samples])
            test_gex = torch.stack([s[1] for s in test_samples])
            test_y = torch.stack([s[2] for s in test_samples])
            X_train = torch.cat([train_img, train_gex], dim=-1)
            X_test = torch.cat([test_img, test_gex], dim=-1)
        else:
            X_train = torch.stack([s[0] for s in train_samples])
            train_y = torch.stack([s[1] for s in train_samples])
            X_test = torch.stack([s[0] for s in test_samples])
            test_y = torch.stack([s[1] for s in test_samples])

        metrics = train_linear_probe(
            cfg=cfg,
            train_embeddings=X_train.float(),
            train_labels=train_y.float(),
            test_embeddings=X_test.float(),
            test_labels=test_y.float(),
            task_type=task_type,
        )
        all_metrics.append(metrics)

        # Include per-donor results
        for k, v in metrics.items():
            if k in {"y_true", "y_pred", "loss_curve", "grad_norms"}:
                continue
            donor_results[f"regression_donor_{test_name}_{k}"] = float(v)

    # Aggregate metrics
    arr = np.array([[m['r2'], m['mse']] for m in all_metrics])
    summary = {
        'regression_r2_mean': float(np.mean(arr[:, 0])),
        'regression_r2_std': float(np.std(arr[:, 0])),
        'regression_mse_mean': float(np.mean(arr[:, 1])),
        'regression_mse_std': float(np.std(arr[:, 1])),
    }

    return {**summary, **donor_results}


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


def save_results(results_dir: Path, cfg: DictConfig, metrics: dict):
    def extract_task_results(metrics: dict, task: str) -> dict:
        task_results = {
            k: v for k, v in metrics.items()
            if k.startswith(f"{task}_mean") or k.startswith(f"{task}_std")
        }

        per_donor = {}
        for k, v in metrics.items():
            if k.startswith(f"{task}_donor_"):
                parts = k.split("_")
                donor = parts[2]
                metric = "_".join(parts[3:])
                per_donor.setdefault(donor, {})[metric] = v

        task_results["per_donor"] = per_donor
        return task_results

    result = {
        "experiment": {
            "aligner": cfg.models.method,
            "embedder": {
                "image": cfg.data.img_embed_key,
                "gex": cfg.data.gex_embed_key,
            },
            "checkpoint_path": cfg.models.checkpoint_path,
            "dataset": cfg.data.dataset,
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
            "classification": extract_task_results(metrics, "classification"),
            "regression": extract_task_results(metrics, "regression"),
        }
    }

    result = make_serializable(result)

    # --- Print results for inspection ---
    print("\n================== FINAL RESULTS ==================\n")
    print(json.dumps(result, indent=2))
    print("\n===================================================\n")

    # --- Write to file ---
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to JSON: {json_path}")


@hydra.main(config_path='../../../configs', config_name='downstream.yaml')
def main(cfg: DictConfig):
    print(f"Starting downstream evaluation")
    print(f"Configuration:")
    print(f"  - Modality: {cfg.evaluation.modality}")
    if cfg.evaluation.modality == "multimodal":
        print(f"  - Alignment method: {cfg.evaluation.align_method}")
    print(f"  - Image model: {cfg.evaluation.img_model}")
    print(f"  - GEX model: {cfg.evaluation.gex_model}")
    print(f"Tasks:")
    print(f"  - Classification: {cfg.evaluation.tasks.classify}")
    print(f"  - Regression: {cfg.evaluation.tasks.regress}")
    print(f"  - Spatial: {cfg.evaluation.tasks.spatial}")

    tensor_datasets = load_and_filter_datasets(cfg)
    results_dir = setup_results_dir(cfg)
    donor_ids = list(tensor_datasets.keys())
    donor_splits = get_donor_splits(donor_ids)

    print(f"\nRunning {len(donor_splits)} leave-one-donor-out splits")
    metrics = {}

    if cfg.evaluation.tasks.classify:
        print("\n=== Running Classification Evaluation ===")
        metrics.update(run_classification(cfg, donor_splits, tensor_datasets))

    if cfg.evaluation.tasks.regress:
        print("\n=== Running Regression Evaluation ===")
        metrics.update(run_regression(cfg, donor_splits, tensor_datasets))

    if cfg.evaluation.tasks.spatial:
        print("\n=== Running Distance-Based Spatial Neighbor Evaluation ===")
        metrics.update(run_spatial(cfg))

    # Add evaluation config to results
    metrics.update({
        "modality": cfg.evaluation.modality,
        "img_model": cfg.evaluation.img_model,
        "gex_model": cfg.evaluation.gex_model,
        "align_method": cfg.evaluation.align_method if cfg.evaluation.modality == "multimodal" else None
    })

    save_results(results_dir, cfg, metrics)
    print("\n=== Evaluation Complete ===")

    return metrics


if __name__ == "__main__":
    main()
