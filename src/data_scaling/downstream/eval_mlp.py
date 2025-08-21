import os
import json
import torch
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import logging
from datasets import load_from_disk
from sklearn.preprocessing import StandardScaler
from data_scaling.downstream.run_benchmarks import get_project_dir, make_serializable
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import importlib
from tabulate import tabulate
import glob
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
logger = logging.getLogger(__name__)

class MLPRegressorCV(nn.Module):
    """MLP for regression tasks with Leave-One-Out Cross-Validation like RidgeCV."""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], dropout_rates=[0.0, 0.1, 0.2], lrs=[1e-4, 1e-3, 1e-2], epochs=100, batch_size=32):
        super(MLPRegressorCV, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.lrs = lrs
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Store best hyperparameters
        self.best_dropout_rate = None
        self.best_lr = None
        self.best_model_state = None
        
        # Build layers (will be recreated with best hyperparameters)
        self._build_network(dropout_rates[0])
        
    def _build_network(self, dropout_rate):
        """Build the network with given dropout rate."""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def _train_single_model(self, X_train, y_train, dropout_rate, lr):
        """Train a single model with given hyperparameters."""
        # Rebuild network with new dropout rate
        self._build_network(dropout_rate)
        
        device = next(self.parameters()).device
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup training
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Train without validation (like RidgeCV's LOOCV)
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def fit(self, X, y):
        """Fit the model using k-fold Cross-Validation to find best hyperparameters (like RidgeCV)."""
        print(f"Performing hyperparameter search with {len(self.dropout_rates)} dropout rates and {len(self.lrs)} learning rates...")
        
        best_score = float('inf')
        best_dropout_rate = None
        best_lr = None
        
        # Use k-fold CV instead of LOOCV for efficiency (k=5)
        k_folds = 5
        n_samples = len(X)
        fold_size = n_samples // k_folds
        
        # Grid search over hyperparameters
        for dropout_rate in self.dropout_rates:
            for lr in self.lrs:
                print(f"  Testing dropout={dropout_rate}, lr={lr}")
                
                cv_scores = []
                
                # k-fold cross-validation
                for fold in range(k_folds):
                    # Define fold indices
                    start_idx = fold * fold_size
                    end_idx = start_idx + fold_size if fold < k_folds - 1 else n_samples
                    
                    # Split data
                    X_val = X[start_idx:end_idx]
                    y_val = y[start_idx:end_idx]
                    X_train = np.concatenate([X[:start_idx], X[end_idx:]])
                    y_train = np.concatenate([y[:start_idx], y[end_idx:]])
                    
                    # Train model
                    train_loss = self._train_single_model(X_train, y_train, dropout_rate, lr)
                    
                    # Evaluate on validation fold
                    self.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=next(self.parameters()).device)
                        y_pred = self(X_val_tensor).cpu().numpy()
                        val_loss = np.mean((y_val - y_pred) ** 2)
                        cv_scores.append(val_loss)
                
                # Average CV score
                avg_cv_score = np.mean(cv_scores)
                print(f"    CV MSE: {avg_cv_score:.6f}")
                
                if avg_cv_score < best_score:
                    best_score = avg_cv_score
                    best_dropout_rate = dropout_rate
                    best_lr = lr
                    self.best_model_state = self.state_dict().copy()
        
        print(f"Best hyperparameters: dropout={best_dropout_rate}, lr={best_lr}")
        print(f"Best CV MSE: {best_score:.6f}")
        
        # Train final model with best hyperparameters on full data
        print("Training final model with best hyperparameters...")
        final_loss = self._train_single_model(X, y, best_dropout_rate, best_lr)
        print(f"Final training loss: {final_loss:.6f}")
        
        self.best_dropout_rate = best_dropout_rate
        self.best_lr = best_lr
    
    def predict(self, X):
        """Make predictions."""
        device = next(self.parameters()).device
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            return self(X).cpu().numpy()
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

class MLPClassifier(nn.Module):
    """MLP for classification tasks."""
    
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout_rate=0.1, lr=1e-3):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.lr = lr
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, patience=10):
        """Train the MLP with early stopping."""
        device = next(self.parameters()).device
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=device)
        
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_val = torch.tensor(y_val, dtype=torch.long, device=device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        pbar = tqdm(range(epochs), desc="Training MLP Classifier")
        for epoch in pbar:
            # Training
            self.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if X_val is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                scheduler.step(val_loss)
                
                # Update progress bar
                pbar.set_postfix({
                    'train_loss': f'{train_loss/len(train_loader):.4f}',
                    'val_loss': f'{val_loss/len(val_loader):.4f}',
                    'patience': patience_counter
                })
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
            else:
                # No validation set, just save the last model
                best_model_state = self.state_dict().copy()
                pbar.set_postfix({'train_loss': f'{train_loss/len(train_loader):.4f}'})
        
        # Load best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
    
    def predict(self, X):
        """Make predictions."""
        device = next(self.parameters()).device
        X = torch.tensor(X, dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            logits = self(X)
            return torch.argmax(logits, dim=1).cpu().numpy()

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
        "comm": "data_scaling.models.multimodal.comm.CoMMBaseline",
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
    """Load data from a specific split (train or test)."""
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
    original_size = len(ds)
    
    # 1. Filter out excluded annotation classes
    if 'annotations' in cfg.data and 'excluded_classes' in cfg.data.annotations:
        excluded_classes = cfg.data.annotations.excluded_classes
        print(f"\nFiltering out excluded classes: {excluded_classes}")
        ds = ds.filter(lambda x: x[class_label_key] not in excluded_classes, 
                      num_proc=cfg.data.get('num_proc', 4))
        print(f"After annotation filtering: {len(ds)} samples (removed {original_size - len(ds)} samples)")
    
    # 2. Filter out classes that only appear in test set
    if 'annotations' in cfg.data and cfg.data.annotations.get('exclude_test_only_classes', False):
        # First, get all unique classes in train set
        train_ids = cfg.data.multimodal.train
        train_classes = set(ds.filter(lambda x: x['name'] in train_ids)[class_label_key])
        
        # Filter out classes not in train set
        ds = ds.filter(lambda x: x[class_label_key] in train_classes,
                      num_proc=cfg.data.get('num_proc', 4))
        print(f"After test-only class filtering: {len(ds)} samples")
    
    # 3. Handle regression task - remove NOMAP from cell type ratios
    if cfg.evaluation.tasks.regress and 'cell_types' in cfg.data:
        if 'nomap_index' in cfg.data.cell_types:
            nomap_idx = cfg.data.cell_types.nomap_index
            
            def remove_nomap(example):
                ratios = list(example[reg_label_key])  # Convert to list for modification
                # Remove NOMAP and renormalize remaining ratios
                ratios = ratios[:nomap_idx]  # Exclude NOMAP
                total = sum(ratios)
                if total > 0:  # Avoid division by zero
                    ratios = [r / total for r in ratios]
                example[reg_label_key] = ratios
                return example
            
            ds = ds.map(remove_nomap, num_proc=cfg.data.get('num_proc', 4))
            print(f"Removed NOMAP from cell type ratios and renormalized")
    
    print(f"Final dataset size: {len(ds)} samples")

    return ds

def evaluate_mlp(X_train, X_test, y_train, y_test, train_reg_labels=None, test_reg_labels=None, random_state=42, cfg=None):
    """
    MLP-based evaluation function that computes macro F1 and R2 using MLP heads.
    """
    print(f"    Starting MLP evaluation with shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Convert tensors to numpy arrays
    X_train = X_train.cpu().numpy().astype(np.float64)
    X_test = X_test.cpu().numpy().astype(np.float64)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        print(f"    ERROR: Found NaN values in data!")
        print(f"    X_train NaN count: {np.isnan(X_train).sum()}")
        print(f"    X_test NaN count: {np.isnan(X_test).sum()}")
        raise ValueError("Input data contains NaN values!")
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        print(f"    ERROR: Found infinite values in data!")
        print(f"    X_train inf count: {np.isinf(X_train).sum()}")
        print(f"    X_test inf count: {np.isinf(X_test).sum()}")
        raise ValueError("Input data contains infinite values!")
    
    print(f"    Data validation passed - no NaN or infinite values found")

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

        # Create label mapping to ensure consecutive integers starting from 0
        unique_labels = np.unique(y_train)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        
        # Map labels to consecutive integers
        y_train_mapped = np.array([label_to_idx[label] for label in y_train])
        y_test_mapped = np.array([label_to_idx[label] for label in y_test])
        
        print(f"    Label mapping: {label_to_idx}")
        print(f"    Number of classes: {num_classes}")
        
        # Train and evaluate classification with MLP
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        clf = MLPClassifier(
            input_dim=X_train_scaled.shape[1], 
            num_classes=num_classes,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.1,
            lr=1e-3
        ).to(device)
        
        # Split training data for validation (20% of training data)
        val_size = int(0.2 * len(X_train_scaled))
        X_train_final = X_train_scaled[:-val_size]
        y_train_final = y_train_mapped[:-val_size]
        X_val = X_train_scaled[-val_size:]
        y_val = y_train_mapped[-val_size:]
        
        clf.fit(X_train_final, y_train_final, X_val, y_val, epochs=100, batch_size=32, patience=10)
        
        # Get predictions
        test_preds = clf.predict(X_test_scaled)
        
        # Convert predictions back to original labels for metric calculation
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        test_preds_original = np.array([idx_to_label[pred] for pred in test_preds])
        
        # Calculate classification metrics using original labels
        metrics["f1_macro"] = f1_score(y_test, test_preds_original, average='macro')

    # Handle regression task
    if do_regression:
        # Convert regression labels to numpy
        train_reg_labels = train_reg_labels.cpu().numpy()
        test_reg_labels = test_reg_labels.cpu().numpy() 

        # Ensure labels are 2D for the MLP
        if len(train_reg_labels.shape) == 1:
            train_reg_labels = train_reg_labels.reshape(-1, 1)
            test_reg_labels = test_reg_labels.reshape(-1, 1)
        
        # Use MLPRegressorCV for RidgeCV-like behavior
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_dim = train_reg_labels.shape[1]
        
        model = MLPRegressorCV(
            input_dim=X_train_scaled.shape[1],
            output_dim=output_dim,
            hidden_dims=[512, 256, 128],
            dropout_rates=[0.0, 0.1, 0.2],
            lrs=[1e-4, 1e-3, 1e-2],
            epochs=50,  # Reduced for faster training
            batch_size=32
        ).to(device)
        
        # Fit using CV (like RidgeCV)
        model.fit(X_train_scaled, train_reg_labels)
        
        test_r2 = model.score(X_test_scaled, test_reg_labels)
        metrics["r2"] = test_r2
        
    return metrics

def run_embedding_baseline_eval_mlp(cfg: DictConfig, train_ds, test_ds, selected_method):
    """Run evaluation for the selected embedding baseline method using MLP heads."""
    
    print(f"\n=== Running Embedding Baseline Evaluation with MLP Heads ===")
    print(f"Selected method: {selected_method}")
    print(f"Training on: train split ({len(train_ds)} samples)")
    print(f"Testing on: test split ({len(test_ds)} samples)")
    
    # Filter out samples with NaN values and prepare labels
    print("  Filtering samples with NaN values...")
    
    # Filter train samples
    valid_train_indices = []
    valid_train_labels = []
    valid_train_reg_labels = []
    
    for i, sample in enumerate(train_ds):
        # Check for NaN values in embeddings based on method
        has_nan = False
        if selected_method in ["unimodal_gex", "multimodal_concat", "multimodal_comm"]:
            if np.isnan(sample["nicheformer_pool"]).any():
                has_nan = True
        if selected_method in ["unimodal_img", "multimodal_concat", "multimodal_comm"]:
            if np.isnan(sample["img_uni_pool"]).any():
                has_nan = True
        
        if not has_nan:
            valid_train_indices.append(i)
            valid_train_labels.append(sample[cfg.training.classification.label_key])
            valid_train_reg_labels.append(sample[cfg.training.regression.label_key])
    
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
    
    print(f"  After filtering: {len(valid_train_indices)}/{len(train_ds)} train samples, {len(valid_test_indices)}/{len(test_ds)} test samples")
    
    # Convert to tensors
    train_labels = torch.tensor(valid_train_labels)
    test_labels = torch.tensor(valid_test_labels)
    train_reg_labels = torch.tensor(valid_train_reg_labels)
    test_reg_labels = torch.tensor(valid_test_reg_labels)
    
    # Get CoMM checkpoint if needed
    comm_checkpoint = None
    if selected_method == "multimodal_comm":
        project_dir = get_project_dir()
        # Use the checkpoint path from config instead of hardcoded path
        comm_checkpoint = project_dir / cfg.models.checkpoint_path
        
        if comm_checkpoint.exists():
            print(f"Using comm checkpoint: {comm_checkpoint.name}")
        else:
            raise FileNotFoundError(f"CoMM checkpoint not found: {comm_checkpoint}")
    
    # Evaluate based on selected method
    if selected_method == "random":
        print("  Evaluating random baseline...")
        sample_img = np.array(train_ds[valid_train_indices[0]]["img_uni_pool"])
        sample_gex = np.array(train_ds[valid_train_indices[0]]["nicheformer_pool"])
        random_shape = (len(valid_train_indices), sample_img.shape[0] + sample_gex.shape[0])
        
        train_data = torch.from_numpy(generate_random_embeddings(random_shape, random_state=42)).float()
        test_data = torch.from_numpy(generate_random_embeddings((len(valid_test_indices), random_shape[1]), random_state=43)).float()
        
    elif selected_method == "unimodal_gex":
        print("  Evaluating unimodal GEX...")
        train_data = torch.stack([torch.tensor(train_ds[i]["nicheformer_pool"]) for i in valid_train_indices])
        test_data = torch.stack([torch.tensor(test_ds[i]["nicheformer_pool"]) for i in valid_test_indices])
        
    elif selected_method == "unimodal_img":
        print("  Evaluating unimodal IMG...")
        train_data = torch.stack([torch.tensor(train_ds[i]["img_uni_pool"]) for i in valid_train_indices])
        test_data = torch.stack([torch.tensor(test_ds[i]["img_uni_pool"]) for i in valid_test_indices])
        
    elif selected_method == "multimodal_concat":
        print("  Evaluating multimodal concat...")
        train_img_data = torch.stack([torch.tensor(train_ds[i]["img_uni_pool"]) for i in valid_train_indices])
        test_img_data = torch.stack([torch.tensor(test_ds[i]["img_uni_pool"]) for i in valid_test_indices])
        train_gex_data = torch.stack([torch.tensor(train_ds[i]["nicheformer_pool"]) for i in valid_train_indices])
        test_gex_data = torch.stack([torch.tensor(test_ds[i]["nicheformer_pool"]) for i in valid_test_indices])
        
        train_data = torch.cat([train_img_data, train_gex_data], dim=-1)
        test_data = torch.cat([test_img_data, test_gex_data], dim=-1)
        
    elif selected_method == "multimodal_comm":
        print("  Evaluating multimodal CoMM...")
        # Prepare raw embeddings
        train_img_raw = torch.stack([torch.tensor(train_ds[i]["img_uni_pool"]) for i in valid_train_indices])
        test_img_raw = torch.stack([torch.tensor(test_ds[i]["img_uni_pool"]) for i in valid_test_indices])
        train_gex_raw = torch.stack([torch.tensor(train_ds[i]["nicheformer_pool"]) for i in valid_train_indices])
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
    
    # Evaluate with MLP heads
    metrics = evaluate_mlp(
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
    print("\n" + "="*80)
    print("EMBEDDING BASELINE EVALUATION RESULTS (MLP HEADS)")
    print("="*80)
    
    f1_str = f"{result['f1_macro']:.4f}" if result['f1_macro'] is not None else "N/A"
    r2_str = f"{result['r2']:.4f}" if result['r2'] is not None else "N/A"
    
    # Print table
    headers = ["Method", "F1 Macro", "R²"]
    table_data = [[result['method'], f1_str, r2_str]]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80)

def save_results(results_dir: Path, cfg: DictConfig, result: dict):
    """Save evaluation results to JSON."""
    result_data = {
        "experiment": {
            "modality": cfg.evaluation.modality,
            "img_model": cfg.evaluation.img_model,
            "gex_model": cfg.evaluation.gex_model,
            "dataset": cfg.data.dataset,
            "evaluation_strategy": "embedding_baseline_eval_mlp",
            "train_split": "train",
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
    json_path = results_dir / f"embedding_baseline_mlp_{result['method']}_results.json"
    with open(json_path, 'w') as f:
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
    
    print("Starting Embedding Baseline Evaluation with MLP Heads")
    print(f"Configuration:")
    print(f"  - Modality: {cfg.evaluation.modality}")
    print(f"  - Image model: {cfg.evaluation.img_model}")
    print(f"  - GEX model: {cfg.evaluation.gex_model}")
    print(f"  - Selected method: {selected_method}")
    print(f"  - Evaluation strategy: Pretrain-Test (entire splits) with MLP heads")
    print(f"  - Tasks: classify={cfg.evaluation.tasks.classify}, regress={cfg.evaluation.tasks.regress}")

    # Setup results directory
    project_dir = get_project_dir()
    results_dir = project_dir / "results" / cfg.data.dataset / f"embedding_baseline_mlp_{cfg.evaluation.img_model}"
    
    # Load train and test data
    train_ds = load_split_data(cfg, "train", selected_method)
    test_ds = load_split_data(cfg, "test", selected_method)
    
    # Run evaluation with MLP heads
    result = run_embedding_baseline_eval_mlp(cfg, train_ds, test_ds, selected_method)
    
    # Print and save results
    print_results_table(result)
    save_results(results_dir, cfg, result)
    print("\n=== Embedding Baseline Evaluation with MLP Heads Complete ===")

if __name__ == "__main__":
    main() 