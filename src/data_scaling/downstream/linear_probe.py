from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import numpy as np
import importlib

def train_linear_probe(
    cfg,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    task_type: str,
    verbose: bool = True,
) -> dict:
    training_cfg = cfg.training[task_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)

    if task_type == "regression":
        train_labels = train_labels.unsqueeze(1) if train_labels.ndim == 1 else train_labels
        test_labels = test_labels.unsqueeze(1) if test_labels.ndim == 1 else test_labels
        output_dim = train_labels.shape[1]
        criterion = nn.MSELoss()
    elif task_type == "classification":
        output_dim = int(train_labels.max().item()) + 1
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    input_dim = train_embeddings.shape[1]
    batch_size = min(training_cfg.batch_size, 512)

    if verbose:
        print(f"Training linear probe with batch size {batch_size} on device {device}")
        print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
        print(f"Training samples: {len(train_embeddings)}, Test samples: {len(test_embeddings)}")

    # --- Model ---
    model = nn.Linear(input_dim, output_dim).to(device)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training_cfg.learning_rate,
        momentum=0.9,
        weight_decay=training_cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_cfg.max_epochs
    )

    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    grad_norms = []

    if verbose:
        print("Starting training...")

    for epoch in range(training_cfg.max_epochs):
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb.long() if task_type == "classification" else yb)
            loss.backward()

            grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
            total_grad_norm += grad_norm.item()

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)
        losses.append(avg_loss)
        grad_norms.append(avg_grad_norm)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, GradNorm = {avg_grad_norm:.4f}")

    # --- Final evaluation ---
    model.eval()
    with torch.no_grad():
        y_pred = model(test_embeddings).cpu()
        y_true = test_labels.cpu()

    if task_type == "regression":
        return {
            "r2": r2_score(y_true.numpy(), y_pred.numpy(), multioutput='uniform_average'),
            "mse": mean_squared_error(y_true.numpy(), y_pred.numpy()),
            "y_pred": y_pred.numpy(),
            "y_true": y_true.numpy(),
            "loss_curve": losses,
            "grad_norms": grad_norms,
        }
    else:
        acc = accuracy_score(y_true.numpy(), y_pred.argmax(dim=1).numpy())
        f1 = f1_score(y_true.numpy(), y_pred.argmax(dim=1).numpy(), average="macro")
        return {
            "accuracy": acc,
            "f1_macro": f1,
            "y_pred": y_pred.numpy(),
            "y_true": y_true.numpy(),
            "loss_curve": losses,
            "grad_norms": grad_norms,
        }




def extract_embeddings_from_fusion_model(
    cfg,
    dataset: List[Dict],
    model_ckpt_path: str,
    device=None,
    target_key=None,
    img_embed_key=None,
    gex_embed_key=None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Efficient extraction of all embeddings in a single forward pass.
    """
    import importlib

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_embed_key = img_embed_key or cfg.data.img_embed_key
    gex_embed_key = gex_embed_key or cfg.data.gex_embed_key
    target_key = target_key or "annotation"

    method = cfg.models.method
    model_paths = {
        "simclr": "data_scaling.models.multimodal.simclr.SimCLRBaseline",
        "barlow_twins": "data_scaling.models.multimodal.barlowtwins.BarlowTwinsBaseline",
        "vicreg": "data_scaling.models.multimodal.vicreg.VICRegBaseline",
        "comm": "data_scaling.models.multimodal.comm.CoMMBaseline",
        "adversarial": "data_scaling.models.multimodal.adversarial.AdversarialBaseline",
        "byol": "data_scaling.models.multimodal.byol.BYOLBaseline",
        "dcca": "data_scaling.models.multimodal.dcca.DCCABaseline",
        "dim": "data_scaling.models.multimodal.dim.DIMBaseline",
        "simsiam": "data_scaling.models.multimodal.simsiam.SimSiamBaseline",
    }

    if method not in model_paths:
        raise ValueError(f"Unknown method: {method}")
    
    module_path, class_name = model_paths[method].rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    model = ModelClass(cfg).to(device)
    checkpoint = torch.load(model_ckpt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Build tensors
    img_tensor = torch.tensor([s[img_embed_key] for s in dataset], dtype=torch.float32, device=device)
    gex_tensor = torch.tensor([s[gex_embed_key] for s in dataset], dtype=torch.float32, device=device)
    names = [s["name"] for s in dataset]
    targets = torch.tensor(
        [s[target_key] for s in dataset],
        dtype=torch.float32 if not isinstance(dataset[0][target_key], int) else torch.long
    )

    # One forward pass
    with torch.no_grad():
        fused = model.get_embeddings(img_tensor, gex_tensor)
        if isinstance(fused, tuple):
            fused = torch.cat(fused, dim=-1)

    return fused.cpu(), targets, names


def run_loocv_linear_probe(
    cfg,
    samples: List[Dict],
    model_ckpt_path: str,
    task_type: str,
    target_key: str,
    img_embed_key: str,
    gex_embed_key: str,
    test_samples: List[str],
) -> Dict[str, float]:
    """
    Run a linear probe on precomputed embeddings.

    Args:
        cfg: Configuration object.
        samples: All samples (train + test).
        model_ckpt_path: Path to the checkpoint for feature extraction.
        task_type: "classification" or "regression".
        target_key: Key to extract targets (e.g. 'annotation', 'cell_type_ratio').
        img_embed_key, gex_embed_key: Keys for embedding extraction.
        test_samples: List of sample names for the test split.

    Returns:
        Dictionary of evaluation metrics (F1 or R², etc.).
    """
    # Step 1: Extract all embeddings once
    emb, tgt, names = extract_embeddings_from_fusion_model(
        cfg, 
        samples, 
        model_ckpt_path,
        target_key=target_key,
        img_embed_key=img_embed_key,
        gex_embed_key=gex_embed_key,
    )

    # Step 2: Split into train/test tensors
    name_arr = [n.split('/')[-1] for n in names]
    test_mask = [n in test_samples for n in name_arr]
    train_mask = [not m for m in test_mask]

    X_train = emb[train_mask]
    Y_train = tgt[train_mask]
    X_test = emb[test_mask]
    Y_test = tgt[test_mask]

    # Step 3: Run linear probe
    metrics = train_linear_probe(
    cfg=cfg,
    train_embeddings=torch.tensor(X_train, dtype=torch.float32),
    train_labels=torch.tensor(Y_train, dtype=torch.float32),
    test_embeddings=torch.tensor(X_test, dtype=torch.float32),
    test_labels=torch.tensor(Y_test, dtype=torch.float32),
    task_type=task_type,
    )

    return metrics
