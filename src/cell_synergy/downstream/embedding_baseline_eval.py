import importlib
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from datasets import load_from_disk
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from cell_synergy.downstream.run_benchmarks import get_project_dir, make_serializable
logger = logging.getLogger(__name__)


def generate_random_embeddings(embedding_shape, random_state=42):
    """Generate random embeddings with the same shape as the original embeddings."""
    np.random.seed(random_state)
    random_embeddings = np.random.randn(*embedding_shape)
    return random_embeddings


def load_multimodal_model(cfg: DictConfig, checkpoint_path: Path, device='cpu'):
    """Load trained multimodal model from checkpoint."""
    print(f"Loading multimodal model from: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Model class mapping
    model_paths = {
        "comm": "cell_synergy.models.comm.CoMMBaseline",
    }

    method = "comm"  # We're only dealing with comm models

    if method not in model_paths:
        raise ValueError(f"Unknown method: {method}")

    # Import and instantiate model
    module_path, class_name = model_paths[method].rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    model = ModelClass(cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Successfully loaded {method} model")

    return model


def load_split_data(cfg: DictConfig, split_name, selected_method=None):
    """Load data from a specific split (pretrain or test)."""
    project_dir = get_project_dir()
    hf_dir = project_dir / cfg.data.dataset / "hf_datasets"

    modality = cfg.evaluation.modality
    img_model = cfg.evaluation.img_model
    gex_model = cfg.evaluation.gex_model
    scale = cfg.evaluation.scale

    # Load label keys from config
    class_label_key = cfg.training.classification.label_key
    reg_label_key = cfg.training.regression.label_key

    print(f"\nLoading {split_name} split data for:")
    print(f"  - Modality: {modality}")
    print(f"  - Image model: {img_model}")
    print(f"  - GEX model: {gex_model}")
    print(f"  - Split: {split_name}")
    print(f"  - Scale: {scale}")
    print(f"  - Classification label key: {class_label_key}")
    print(f"  - Regression label key: {reg_label_key}")

    # For random baseline, we need multimodal data to get the embedding dimensions
    # For other methods, use the modality from config
    if selected_method == "random":
        # Use multimodal data to get embedding dimensions for random baseline
        pattern = f"{gex_model}_{img_model}_{split_name}.{scale}"
    else:
        # Build expected directory pattern based on modality
        if modality == "multimodal":
            pattern = f"{gex_model}_{img_model}_{split_name}.{scale}"
        elif modality == "unimodal_img":
            pattern = f"img_only_{img_model}_{split_name}.{scale}"
        elif modality == "unimodal_gex":
            pattern = f"gex_only_{gex_model}_{split_name}.{scale}"
        else:
            raise ValueError(f"Invalid modality: {modality}")

    # Load dataset
    split_dir = hf_dir / pattern
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    print(f"\nLoading dataset from: {split_dir}")
    ds = load_from_disk(str(split_dir))
    print(f"Loaded {len(ds)} samples from {split_name} split")

    return ds


def evaluate_simple(
        X_train,
        X_test,
        y_train,
        y_test,
        train_reg_labels=None,
        test_reg_labels=None,
        random_state=42,
        cfg=None):
    """
    Simplified evaluation function that only computes macro F1 and R2.
    """
    print(f"    Starting evaluation with shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    # Convert tensors to numpy arrays
    X_train = X_train.cpu().numpy().astype(np.float64)
    X_test = X_test.cpu().numpy().astype(np.float64)

    # Check for NaN or infinite values
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        print("    ERROR: Found NaN values in data!")
        print(f"    X_train NaN count: {np.isnan(X_train).sum()}")
        print(f"    X_test NaN count: {np.isnan(X_test).sum()}")
        raise ValueError("Input data contains NaN values!")
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        print("    ERROR: Found infinite values in data!")
        print(f"    X_train inf count: {np.isinf(X_train).sum()}")
        print(f"    X_test inf count: {np.isinf(X_test).sum()}")
        raise ValueError("Input data contains infinite values!")

    print("    Data validation passed - no NaN or infinite values found")

    # Initialize metrics dictionary
    metrics = {}

    # Determine task type from config
    do_classification = cfg.evaluation.tasks.classify
    do_regression = cfg.evaluation.tasks.regress

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle classification task
    if do_classification:
        y_train = y_train.cpu().numpy()
        y_test = y_test.cpu().numpy()

        # If labels are one-hot encoded or multi-dimensional, convert to class indices
        if len(y_train.shape) > 1:
            if np.allclose(y_train.sum(axis=1), 1.0):  # Check if one-hot encoded
                y_train = np.argmax(y_train, axis=1)
            else:
                raise ValueError("Multi-label classification not supported")
        if len(y_test.shape) > 1:
            if np.allclose(y_test.sum(axis=1), 1.0):
                y_test = np.argmax(y_test, axis=1)
            else:
                raise ValueError("Multi-label classification not supported")

        # Train and evaluate classification
        clf = LogisticRegression(max_iter=2000, random_state=random_state)
        clf.fit(X_train_scaled, y_train)

        # Get predictions
        test_preds = clf.predict(X_test_scaled)

        # Calculate classification metrics
        metrics["f1_macro"] = f1_score(y_test, test_preds, average='macro')

    # Handle regression task
    if do_regression:
        # Convert regression labels to numpy
        train_reg_labels = train_reg_labels.cpu().numpy()
        test_reg_labels = test_reg_labels.cpu().numpy()

        # Use Ridge regression
        from sklearn.linear_model import RidgeCV
        model = RidgeCV(alphas=np.logspace(-3, 3, 10), scoring="r2")

        model.fit(X_train_scaled, train_reg_labels)
        test_r2 = model.score(X_test_scaled, test_reg_labels)

        metrics["r2"] = test_r2

    return metrics


def run_embedding_baseline_eval(cfg: DictConfig, pretrain_ds, test_ds, selected_method):
    """Run evaluation for the selected embedding baseline method."""

    print("\n=== Running Embedding Baseline Evaluation ===")
    print(f"Selected method: {selected_method}")
    print(f"Training on: pretrain split ({len(pretrain_ds)} samples)")
    print(f"Testing on: test split ({len(test_ds)} samples)")

    # Filter out samples with NaN values and prepare labels
    print("  Filtering samples with NaN values...")

    # Filter pretrain samples
    valid_pretrain_indices = []
    valid_pretrain_labels = []
    valid_pretrain_reg_labels = []

    for i, sample in enumerate(pretrain_ds):
        # Check for NaN values in embeddings based on method
        has_nan = False
        if selected_method in ["unimodal_gex", "multimodal_concat", "multimodal_comm"]:
            if np.isnan(sample["nicheformer_pool"]).any():
                has_nan = True
        if selected_method in ["unimodal_img", "multimodal_concat", "multimodal_comm"]:
            if np.isnan(sample["img_uni_pool"]).any():
                has_nan = True

        if not has_nan:
            valid_pretrain_indices.append(i)
            valid_pretrain_labels.append(sample[cfg.training.classification.label_key])
            valid_pretrain_reg_labels.append(sample[cfg.training.regression.label_key])

    # Filter test samples
    valid_test_indices = []
    valid_test_labels = []
    valid_test_reg_labels = []

    for i, sample in enumerate(test_ds):
        # Check for NaN values in embeddings based on method
        has_nan = False
        if selected_method in ["unimodal_gex", "multimodal_concat", "multimodal_comm"]:
            if np.isnan(sample["nicheformer_pool"]).any():
                has_nan = True
        if selected_method in ["unimodal_img", "multimodal_concat", "multimodal_comm"]:
            if np.isnan(sample["img_uni_pool"]).any():
                has_nan = True

        if not has_nan:
            valid_test_indices.append(i)
            valid_test_labels.append(sample[cfg.training.classification.label_key])
            valid_test_reg_labels.append(sample[cfg.training.regression.label_key])

    print(
        f"  After filtering: {len(valid_pretrain_indices)}/{len(pretrain_ds)} pretrain samples, "
        f"{len(valid_test_indices)}/{len(test_ds)} test samples")

    # Convert to tensors
    train_labels = torch.tensor(valid_pretrain_labels)
    test_labels = torch.tensor(valid_test_labels)
    train_reg_labels = torch.tensor(valid_pretrain_reg_labels)
    test_reg_labels = torch.tensor(valid_test_reg_labels)

    # Get CoMM checkpoint if needed
    comm_checkpoint = None
    if selected_method == "multimodal_comm":
        project_dir = get_project_dir()
        comm_dir = project_dir / "trained_models" / "multi" / cfg.data.dataset
        # For embedding_baseline_eval.py, use comm_preL.ckpt (pretrain set)
        comm_checkpoint = comm_dir / "comm_preL.ckpt"

        if comm_checkpoint.exists():
            print(f"Using comm checkpoint: {comm_checkpoint.name}")
        else:
            raise FileNotFoundError(f"CoMM preL checkpoint not found: {comm_checkpoint}")

    # Evaluate based on selected method
    if selected_method == "random":
        print("  Evaluating random baseline...")
        sample_img = np.array(pretrain_ds[valid_pretrain_indices[0]]["img_uni_pool"])
        sample_gex = np.array(pretrain_ds[valid_pretrain_indices[0]]["nicheformer_pool"])
        random_shape = (len(valid_pretrain_indices), sample_img.shape[0] + sample_gex.shape[0])

        train_data = torch.from_numpy(generate_random_embeddings(random_shape, random_state=42)).float()
        test_data = torch.from_numpy(
            generate_random_embeddings(
                (len(valid_test_indices),
                 random_shape[1]),
                random_state=43)).float()

    elif selected_method == "unimodal_gex":
        print("  Evaluating unimodal GEX...")
        train_data = torch.stack([torch.tensor(pretrain_ds[i]["nicheformer_pool"]) for i in valid_pretrain_indices])
        test_data = torch.stack([torch.tensor(test_ds[i]["nicheformer_pool"]) for i in valid_test_indices])

    elif selected_method == "unimodal_img":
        print("  Evaluating unimodal IMG...")
        train_data = torch.stack([torch.tensor(pretrain_ds[i]["img_uni_pool"]) for i in valid_pretrain_indices])
        test_data = torch.stack([torch.tensor(test_ds[i]["img_uni_pool"]) for i in valid_test_indices])

    elif selected_method == "multimodal_concat":
        print("  Evaluating multimodal concat...")
        train_img_data = torch.stack([torch.tensor(pretrain_ds[i]["img_uni_pool"]) for i in valid_pretrain_indices])
        test_img_data = torch.stack([torch.tensor(test_ds[i]["img_uni_pool"]) for i in valid_test_indices])
        train_gex_data = torch.stack([torch.tensor(pretrain_ds[i]["nicheformer_pool"]) for i in valid_pretrain_indices])
        test_gex_data = torch.stack([torch.tensor(test_ds[i]["nicheformer_pool"]) for i in valid_test_indices])

        train_data = torch.cat([train_img_data, train_gex_data], dim=-1)
        test_data = torch.cat([test_img_data, test_gex_data], dim=-1)

    elif selected_method == "multimodal_comm":
        print("  Evaluating multimodal CoMM...")
        # Prepare raw embeddings
        train_img_raw = torch.stack([torch.tensor(pretrain_ds[i]["img_uni_pool"]) for i in valid_pretrain_indices])
        test_img_raw = torch.stack([torch.tensor(test_ds[i]["img_uni_pool"]) for i in valid_test_indices])
        train_gex_raw = torch.stack([torch.tensor(pretrain_ds[i]["nicheformer_pool"]) for i in valid_pretrain_indices])
        test_gex_raw = torch.stack([torch.tensor(test_ds[i]["nicheformer_pool"]) for i in valid_test_indices])

        # Set embedding dimensions in config
        cfg.models.img_embed_dim = train_img_raw.shape[1]
        cfg.models.gex_embed_dim = train_gex_raw.shape[1]

        # Load trained multimodal model and get aligned embeddings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        multimodal_model = load_multimodal_model(cfg, Path(comm_checkpoint), device)

        with torch.no_grad():
            # Get aligned embeddings from the trained model
            train_img_aligned, train_gex_aligned = multimodal_model.get_embeddings(
                train_img_raw.to(device), train_gex_raw.to(device)
            )
            test_img_aligned, test_gex_aligned = multimodal_model.get_embeddings(
                test_img_raw.to(device), test_gex_raw.to(device)
            )

            # Move back to CPU and concatenate
            train_data = torch.cat([train_img_aligned.cpu(), train_gex_aligned.cpu()], dim=-1)
            test_data = torch.cat([test_img_aligned.cpu(), test_gex_aligned.cpu()], dim=-1)

    else:
        raise ValueError(f"Unknown method: {selected_method}")

    # Evaluate
    metrics = evaluate_simple(
        train_data, test_data, train_labels, test_labels,
        train_reg_labels, test_reg_labels,
        random_state=42, cfg=cfg
    )

    return {
        'method': selected_method,
        'f1_macro': metrics.get('f1_macro'),
        'r2': metrics.get('r2')
    }


def print_results_table(result):
    """Print results in a nice table format."""
    print("\n" + "=" * 80)
    print("EMBEDDING BASELINE EVALUATION RESULTS")
    print("=" * 80)

    f1_str = f"{result['f1_macro']:.4f}" if result['f1_macro'] is not None else "N/A"
    r2_str = f"{result['r2']:.4f}" if result['r2'] is not None else "N/A"

    # Print table
    headers = ["Method", "F1 Macro", "R²"]
    table_data = [[result['method'], f1_str, r2_str]]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 80)


def save_results(results_dir: Path, cfg: DictConfig, result: dict):
    """Save evaluation results to JSON."""
    result_data = {
        "experiment": {
            "modality": cfg.evaluation.modality,
            "img_model": cfg.evaluation.img_model,
            "gex_model": cfg.evaluation.gex_model,
            "dataset": cfg.data.dataset,
            "evaluation_strategy": "embedding_baseline_eval",
            "train_split": "pretrain",
            "test_split": "test",
            "selected_method": result['method'],
            "tasks": {
                "classification": cfg.evaluation.tasks.classify,
                "regression": cfg.evaluation.tasks.regress
            }
        },
        "results": [result]
    }

    result_data = make_serializable(result_data)

    # Save to file
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"embedding_baseline_{result['method']}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to JSON: {json_path}")


@hydra.main(config_path='../../../configs', config_name='downstream.yaml')
def main(cfg: DictConfig):
    # Get selected method from config
    selected_method = getattr(cfg.evaluation, 'selected_method', None)

    if selected_method is None:
        # Fallback: try to infer from modality
        modality = cfg.evaluation.modality
        if modality == "multimodal":
            selected_method = "multimodal_" + cfg.models.method
        elif modality == "unimodal_img":
            selected_method = "unimodal_img"
        elif modality == "unimodal_gex":
            selected_method = "unimodal_gex"
        elif modality == "random":
            selected_method = "random"
        else:
            raise ValueError(f"Cannot infer method from modality: {modality}")

    # Validate selected method
    valid_methods = ['random', 'unimodal_gex', 'unimodal_img', 'multimodal_concat', 'multimodal_comm']
    if selected_method not in valid_methods:
        raise ValueError(f"Invalid method: {selected_method}. Must be one of: {valid_methods}")

    print("Starting Embedding Baseline Evaluation")
    print("Configuration:")
    print(f"  - Modality: {cfg.evaluation.modality}")
    print(f"  - Image model: {cfg.evaluation.img_model}")
    print(f"  - GEX model: {cfg.evaluation.gex_model}")
    print(f"  - Selected method: {selected_method}")
    print("  - Evaluation strategy: Pretrain-Test (entire splits)")
    print(f"  - Tasks: classify={cfg.evaluation.tasks.classify}, regress={cfg.evaluation.tasks.regress}")

    # Setup results directory
    project_dir = get_project_dir()
    results_dir = project_dir / "results" / cfg.data.dataset / f"embedding_baseline_{cfg.evaluation.img_model}"

    # Load pretrain and test data
    pretrain_ds = load_split_data(cfg, "pretrain", selected_method)
    test_ds = load_split_data(cfg, "test", selected_method)

    # Run evaluation
    result = run_embedding_baseline_eval(cfg, pretrain_ds, test_ds, selected_method)

    # Print and save results
    print_results_table(result)
    save_results(results_dir, cfg, result)
    print("\n=== Embedding Baseline Evaluation Complete ===")


if __name__ == "__main__":
    main()
