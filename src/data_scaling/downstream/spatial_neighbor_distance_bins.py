import hydra
from omegaconf import DictConfig
import numpy as np
import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score, mean_squared_error
from data_scaling.finetuning.align import ProcessedHFDataset
from data_scaling.downstream.linear_probe import extract_embeddings_from_fusion_model, train_linear_probe


def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/Projects/till_richter/")
    resolved = os.path.expandvars(raw)
    return Path(resolved)


def patch_center(cell_coords):
    coords = np.array([c for c in cell_coords if c[0] != -1 and c[1] != -1])
    return coords.mean(axis=0) if len(coords) > 0 else np.array([np.nan, np.nan])


def get_distance_bins(mu_coords, num_bins=5, metric='euclidean'):
    # Compute pairwise distances using the specified metric and determine bin edges via quantiles
    D = squareform(pdist(mu_coords, metric=metric))
    upper_tri_mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    distance_vals = D[upper_tri_mask]
    bin_edges = np.quantile(distance_vals, np.linspace(0, 1, num_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < num_bins + 1:
        bin_edges = np.linspace(distance_vals.min(), distance_vals.max(), num_bins + 1)
    return D, bin_edges


def assign_bins(D, bin_edges):
    # Assign each pairwise distance to a bin, ignoring self-comparisons
    bin_ids = np.digitize(D, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(bin_edges) - 2)
    np.fill_diagonal(bin_ids, -1)
    return bin_ids


def summarize_neighbors(targets, bin_ids, strategy="mean"):
    # Summarize neighbor targets for each bin by aggregation strategy
    P = bin_ids.shape[0]
    num_bins = np.max(bin_ids) + 1
    if targets.ndim == 1:
        summaries = np.full((P, num_bins), np.nan)
    else:
        summaries = np.full((P, num_bins, targets.shape[1]), np.nan)
    for i in range(P):
        for b in range(num_bins):
            neighbor_idx = np.where(bin_ids[i] == b)[0]
            if len(neighbor_idx) > 0:
                neighbor_targets = targets[neighbor_idx]
                if strategy == "mean":
                    summary = np.nanmean(neighbor_targets, axis=0)
                elif strategy == "median":
                    summary = np.nanmedian(neighbor_targets, axis=0)
                elif strategy == "sum":
                    summary = np.nansum(neighbor_targets, axis=0)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                summaries[i, b] = summary
    return summaries


def run_spatial_neighbor(cfg: DictConfig) -> Dict[str, Dict[str, float]]:
    from data_scaling.downstream.run_benchmarks import load_downstream_dataset

    hf_dataset = load_downstream_dataset(cfg)
    samples = [hf_dataset[i] for i in range(len(hf_dataset))]

    model_ckpt_path = get_project_dir() / cfg.models.checkpoint_path

    task_type = cfg.training.spatial_neighbor.task_type
    label_key = cfg.training[task_type].label_key
    num_bins = cfg.training.spatial_neighbor.num_bins
    distance_metric = cfg.training.spatial_neighbor.distance_metric
    summary_strategy = cfg.training.spatial_neighbor.get("summary_strategy", "mean")

    # Check if cell_coords field exists
    if len(samples) > 0 and "cell_coords" not in samples[0]:
        raise ValueError(
            "Dataset missing 'cell_coords' field required for spatial neighbor evaluation. "
            "Please run merge_annotations.py first to add this field."
        )

    centers = np.array([patch_center(s["cell_coords"]) for s in samples])
    valid_mask = ~np.isnan(centers).any(axis=1)
    samples = [s for s, valid in zip(samples, valid_mask) if valid]
    centers = centers[valid_mask]

    if len(samples) < 10:
        raise ValueError("Too few valid samples after filtering for valid centers.")

    emb, tgt, names = extract_embeddings_from_fusion_model(
        cfg,
        samples,
        str(model_ckpt_path),
        target_key=label_key,
        img_embed_key=cfg.data.img_embed_key,
        gex_embed_key=cfg.data.gex_embed_key,
    )

    emb = emb.numpy()
    tgt = tgt.numpy()
    if emb.shape[0] != centers.shape[0]:
        raise ValueError("Mismatch between embeddings and coordinates.")

    D, bin_edges = get_distance_bins(centers, num_bins, metric=distance_metric)
    bin_ids = assign_bins(D, bin_edges)
    neighbor_summaries = summarize_neighbors(tgt, bin_ids, strategy=summary_strategy)
    bin_midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    donor_names = [s["name"].split("_")[0] for s in samples]
    unique_donors = sorted(set(donor_names))
    bin_r2s = [[] for _ in range(num_bins)]

    for test_donor in unique_donors:
        test_mask = np.array([dn == test_donor for dn in donor_names])
        train_mask = ~test_mask

        if train_mask.sum() < 5 or test_mask.sum() == 0:
            print(f"Skipping donor {test_donor} due to insufficient data.")
            continue

        X_train = torch.tensor(emb[train_mask], dtype=torch.float32)
        Y_train = torch.tensor(neighbor_summaries[train_mask].reshape(train_mask.sum(), -1), dtype=torch.float32)
        X_test = torch.tensor(emb[test_mask], dtype=torch.float32)
        Y_test = neighbor_summaries[test_mask].reshape(test_mask.sum(), -1)

        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

        metrics = train_linear_probe(
            cfg=cfg,
            train_embeddings=X_train,
            train_labels=Y_train,
            test_embeddings=X_test,
            test_labels=Y_test_tensor,
            task_type=task_type,
        )

        y_pred = metrics["y_pred"].reshape(test_mask.sum(), num_bins, -1)
        y_true = Y_test.reshape(test_mask.sum(), num_bins, -1)

        for b in range(num_bins):
            for i in range(y_true.shape[0]):
                if not np.isnan(y_true[i, b]).any():
                    r2 = r2_score(y_true[i, b], y_pred[i, b])
                    bin_r2s[b].append(r2)

    results = {
        f"bin_{b}": {
            "mean": float(np.mean(r2s)) if r2s else 0.0,
            "std": float(np.std(r2s)) if r2s else 0.0
        }
        for b, r2s in enumerate(bin_r2s)
    }

    results["bin_midpoints"] = {f"bin_{b}_distance": float(mp) for b, mp in enumerate(bin_midpoints)}

    return results


@hydra.main(config_path='../../../configs', config_name='downstream.yaml')
def main(cfg: DictConfig):
    """Main function for Hydra compatibility."""
    return run_spatial_neighbor(cfg)


if __name__ == "__main__":
    main()
