import hydra
from omegaconf import DictConfig
from datasets import load_from_disk
import numpy as np
import os
import json
import torch
from data_scaling.evaluation.linear_probe import extract_embeddings_from_fusion_model, prepare_neighbor_aggregation_data, LinearProbe
from data_scaling.paths import PROJECT_DIR
from collections import defaultdict
from scipy.special import rel_entr


def patch_center(cell_coords):
    coords = np.array([c for c in cell_coords if c[0] != -1 and c[1] != -1])
    return coords.mean(axis=0) if len(coords) > 0 else np.array([np.nan, np.nan])


def get_patch_centers(dataset):
    centers = []
    for row in dataset:
        centers.append(patch_center(row['cell_coords']))
    return np.stack(centers)


def kl_divergence(p, q):
    # p, q: [N, C] probability distributions
    # Add small epsilon for numerical stability
    eps = 1e-8
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.mean(np.sum(rel_entr(p, q), axis=1))


def evaluate_neighbor_prediction(y_true, y_pred, task_type):
    if task_type == 'classification':
        # y_true, y_pred: [N, C] probability distributions
        kl = kl_divergence(y_true, y_pred)
        acc = np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
        return {'kl_div': kl, 'argmax_acc': acc}
    else:
        # Regression: y_true, y_pred: [N, D]
        mse = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean(axis=0)) ** 2)
        return {'mse': mse, 'r2': r2}


@hydra.main(config_path='../../../configs', config_name='base.yaml')
def main(cfg: DictConfig):
    """
    For each patch, aggregate the labels of its k nearest neighbors.
    Use sample-level LOOCV: for each sample, train on all other samples, test on held-out sample.
    Train a linear probe to predict the aggregated neighbor label from the patch embedding.
    Save results as JSON in PROJECT_DIR/spatial_neighbor_label_results/.
    """
    k = getattr(cfg, 'k', 8)
    task_type = getattr(cfg, 'task_type', 'classification')
    target_key = getattr(cfg, 'target_key', 'annotation' if task_type == 'classification' else 'cell_type_ratio')
    img_embed_key = getattr(cfg.data, 'img_embed_key', None)
    gex_embed_key = getattr(cfg.data, 'gex_embed_key', None)

    # Load dataset
    dataset_path = getattr(cfg.data, 'hf_dataset_path', None)
    if dataset_path is None:
        from data_scaling.paths import PROJECT_DIR
        dataset_path = PROJECT_DIR / 'hf'
    dataset = load_from_disk(str(dataset_path))

    # Extract embeddings and targets
    embeddings, labels, sample_names = extract_embeddings_from_fusion_model(
        cfg, dataset, None, None, target_key, img_embed_key, gex_embed_key
    )
    embeddings = embeddings.numpy()
    labels = labels.numpy()
    centers = get_patch_centers(dataset)

    # For classification, get number of classes
    if task_type == 'classification':
        # Assume ClassLabel is used in HuggingFace dataset
        class_count = int(np.max(labels)) + 1
    else:
        class_count = None

    # Get sample IDs for sample-level LOOCV
    sample_ids = np.array([row['name'].split('_')[0] for row in dataset])
    unique_samples = np.unique(sample_ids)

    results_dir = PROJECT_DIR / 'spatial_neighbor_label_results'
    os.makedirs(results_dir, exist_ok=True)
    all_results = defaultdict(list)

    for test_sample in unique_samples:
        # Split train/test by sample
        test_mask = (sample_ids == test_sample)
        train_mask = ~test_mask
        X_train, Y_train, _ = prepare_neighbor_aggregation_data(
            embeddings[train_mask], labels[train_mask], centers[train_mask], k, task_type, class_count
        )
        X_test, Y_test, _ = prepare_neighbor_aggregation_data(
            embeddings[test_mask], labels[test_mask], centers[test_mask], k, task_type, class_count
        )
        # Train linear probe
        model = LinearProbe(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            task_type='regression' if task_type == 'regression' else 'regression',  # always regression for multi-target
            learning_rate=cfg.training.learning_rate,
            weight_decay=getattr(cfg.training, 'weight_decay', 0.0),
        )
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
        import pytorch_lightning as pl
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader)
        # Predict on test set
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(model.device)
            Y_pred = model(X_test_tensor).cpu().numpy()
        # Evaluate
        metrics = evaluate_neighbor_prediction(Y_test, Y_pred, task_type)
        all_results['test_sample'].append(test_sample)
        for k_, v in metrics.items():
            all_results[k_].append(v)
        print(f"Test sample {test_sample}: {metrics}")
    # Aggregate and save
    summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in all_results.items() if k != 'test_sample'}
    with open(results_dir / f'neighbor_label_k{k}_results.json', 'w') as f:
        json.dump({'per_sample': dict(all_results), 'summary': summary}, f, indent=2)
    print(f"Results saved to {results_dir}")
    return summary

if __name__ == "__main__":
    main() 