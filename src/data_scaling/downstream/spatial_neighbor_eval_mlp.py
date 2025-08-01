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
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
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
    print(f"Loaded {len(ds)} samples from {split_name} split")

    # Check if cell_coords field exists
    if len(ds) > 0 and "cell_coords" not in ds[0]:
        raise ValueError(
            "Dataset missing 'cell_coords' field required for spatial neighbor evaluation. "
            "Please run merge_annotations.py first to add this field."
        )

    return ds

def patch_center(cell_coords):
    """Calculate the center of a patch from cell coordinates."""
    if len(cell_coords) == 0:
        return np.array([0, 0])
    return np.mean(cell_coords, axis=0)

def create_meaningful_distance_bins(centers, num_bins=5, max_neighbor_factor=10):
    """Create meaningful distance bins based on the distribution of distances."""
    if len(centers) < 2:
        return np.linspace(0, 1000, num_bins + 1)
    
    # Calculate all pairwise distances
    distances = pdist(centers)
    
    if len(distances) == 0:
        return np.linspace(0, 1000, num_bins + 1)
    
    # Use percentiles to create meaningful bins
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(distances, percentiles)
    
    # Ensure the last bin includes some reasonable maximum
    max_distance = np.percentile(distances, 95)  # Use 95th percentile as max
    bin_edges[-1] = max_distance
    
    return bin_edges

def assign_distance_bins(distances, bin_edges):
    """Assign distances to bins."""
    return np.digitize(distances, bin_edges) - 1

def extract_neighbor_prediction_pairs(embeddings, labels, centers, num_bins=5, max_neighbor_factor=10):
    """
    Extract pairs of (source_embedding, target_label) for neighbor prediction.
    
    Args:
        embeddings: Array of embeddings (n_patches, embed_dim)
        labels: Array of labels (n_patches,) or (n_patches, n_classes)
        centers: Array of patch centers (n_patches, 2)
        num_bins: Number of distance bins
        max_neighbor_factor: Maximum number of neighbors per patch as factor of total patches
    
    Returns:
        Dictionary with distance-binned prediction pairs
    """
    n_patches = len(embeddings)
    if n_patches < 2:
        return {}
    
    # Create distance bins
    bin_edges = create_meaningful_distance_bins(centers, num_bins, max_neighbor_factor)
    
    # Calculate all pairwise distances
    distances = squareform(pdist(centers))
    
    # Store pairs by distance bin
    binned_pairs = {}
    for b in range(num_bins):
        binned_pairs[b] = {
            'z_i': [],  # Source embeddings
            'y_j': [],  # Target labels
            'distances': []
        }
    
    # For each patch, find neighbors in each distance bin
    for i in range(n_patches):
        # Get distances from patch i to all other patches
        patch_distances = distances[i]
        
        # Assign distances to bins
        distance_bins = assign_distance_bins(patch_distances, bin_edges)
        
        # For each distance bin, collect neighbor pairs
        for b in range(num_bins):
            # Find patches in this distance bin
            neighbor_indices = np.where(distance_bins == b)[0]
            
            # Limit number of neighbors per patch
            max_neighbors = min(len(neighbor_indices), max_neighbor_factor)
            if max_neighbors > 0:
                # Randomly sample neighbors if too many
                if len(neighbor_indices) > max_neighbors:
                    neighbor_indices = np.random.choice(neighbor_indices, max_neighbors, replace=False)
                
                # Add pairs for this bin
                for j in neighbor_indices:
                    if i != j:  # Don't predict self
                        binned_pairs[b]['z_i'].append(embeddings[i])
                        binned_pairs[b]['y_j'].append(labels[j])
                        binned_pairs[b]['distances'].append(patch_distances[j])
    
    # Convert to numpy arrays
    for b in range(num_bins):
        if len(binned_pairs[b]['z_i']) > 0:
            binned_pairs[b]['z_i'] = np.array(binned_pairs[b]['z_i'])
            binned_pairs[b]['y_j'] = np.array(binned_pairs[b]['y_j'])
            binned_pairs[b]['distances'] = np.array(binned_pairs[b]['distances'])
    
    return binned_pairs, bin_edges

def evaluate_neighbor_prediction_bin_mlp(z_i, y_j, task_type="regression", test_size=0.2, random_state=42):
    """
    Evaluate neighbor prediction for a single distance bin using MLP heads.
    
    Args:
        z_i: Source embeddings (n_pairs, embed_dim)
        y_j: Target labels (n_pairs,) or (n_pairs, n_classes)
        task_type: "regression" or "classification"
        test_size: Fraction of data for testing
        random_state: Random seed
    
    Returns:
        Dictionary with evaluation metrics
    """
    if len(z_i) < 10:
        return None
    
    # Split data
    z_train, z_test, y_train, y_test = train_test_split(
        z_i, y_j, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train)
    z_test_scaled = scaler.transform(z_test)
    
    # Train model with MLP
    if task_type == "regression":
        # Ensure labels are 2D for the MLP
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_dim = y_train.shape[1]
        
        model = MLPRegressorCV(
            input_dim=z_train_scaled.shape[1],
            output_dim=output_dim,
            hidden_dims=[512, 256, 128],
            dropout_rates=[0.0, 0.1, 0.2],
            lrs=[1e-4, 1e-3, 1e-2],
            epochs=50,  # Reduced for faster training
            batch_size=32
        ).to(device)
        
        # Fit using CV (like RidgeCV)
        model.fit(z_train_scaled, y_train)
        
        # Evaluate
        train_r2 = model.score(z_train_scaled, y_train)
        test_r2 = model.score(z_test_scaled, y_test)
        train_mse = mean_squared_error(y_train, model.predict(z_train_scaled))
        test_mse = mean_squared_error(y_test, model.predict(z_test_scaled))
        
        return {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "n_pairs": len(z_i)
        }
    elif task_type == "classification":
        # Handle classification - convert one-hot to class indices if needed
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            # One-hot encoded, convert to class indices
            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test, axis=1)
        
        # Create label mapping to ensure consecutive integers starting from 0
        unique_labels = np.unique(y_train)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        
        # Map labels to consecutive integers
        y_train_mapped = np.array([label_to_idx[label] for label in y_train])
        y_test_mapped = np.array([label_to_idx[label] for label in y_test])
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        clf = MLPClassifier(
            input_dim=z_train_scaled.shape[1], 
            num_classes=num_classes,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.1,
            lr=1e-3
        ).to(device)
        
        # Split training data for validation (20% of training data)
        val_size = int(0.2 * len(z_train_scaled))
        z_train_final = z_train_scaled[:-val_size]
        y_train_final = y_train_mapped[:-val_size]
        z_val = z_train_scaled[-val_size:]
        y_val = y_train_mapped[-val_size:]
        
        clf.fit(z_train_final, y_train_final, z_val, y_val, epochs=100, batch_size=32, patience=10)
        
        # Get predictions
        test_preds = clf.predict(z_test_scaled)
        
        # Convert predictions back to original labels for metric calculation
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        test_preds_original = np.array([idx_to_label[pred] for pred in test_preds])
        
        # Calculate classification metrics using original labels
        train_accuracy = accuracy_score(y_train, test_preds_original[:len(y_train)])
        test_accuracy = accuracy_score(y_test, test_preds_original[len(y_train):])
        train_f1 = f1_score(y_train, test_preds_original[:len(y_train)], average='macro')
        test_f1 = f1_score(y_test, test_preds_original[len(y_train):], average='macro')
        
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_f1_macro": train_f1,
            "test_f1_macro": test_f1,
            "n_pairs": len(z_i)
        }
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def run_spatial_neighbor_eval_mlp(cfg: DictConfig, train_ds, test_ds, selected_method):
    """Run spatial neighbor evaluation using MLP heads."""
    
    print(f"\n=== Running Spatial Neighbor Evaluation with MLP Heads ===")
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
    
    # Get embeddings based on selected method
    if selected_method == "random":
        print("  Evaluating random baseline...")
        sample_img = np.array(train_ds[valid_train_indices[0]]["img_uni_pool"])
        sample_gex = np.array(train_ds[valid_train_indices[0]]["nicheformer_pool"])
        random_shape = (len(valid_train_indices), sample_img.shape[0] + sample_gex.shape[0])
        
        train_embeddings = generate_random_embeddings(random_shape, random_state=42)
        
    elif selected_method == "unimodal_gex":
        print("  Evaluating unimodal GEX...")
        train_embeddings = np.stack([train_ds[i]["nicheformer_pool"] for i in valid_train_indices])
        
    elif selected_method == "unimodal_img":
        print("  Evaluating unimodal IMG...")
        train_embeddings = np.stack([train_ds[i]["img_uni_pool"] for i in valid_train_indices])
        
    elif selected_method == "multimodal_concat":
        print("  Evaluating multimodal concat...")
        train_img_data = np.stack([train_ds[i]["img_uni_pool"] for i in valid_train_indices])
        train_gex_data = np.stack([train_ds[i]["nicheformer_pool"] for i in valid_train_indices])
        train_embeddings = np.concatenate([train_img_data, train_gex_data], axis=-1)
        
    elif selected_method == "multimodal_comm":
        print("  Evaluating multimodal CoMM...")
        # Prepare raw embeddings
        train_img_raw = torch.stack([torch.tensor(train_ds[i]["img_uni_pool"]) for i in valid_train_indices])
        train_gex_raw = torch.stack([torch.tensor(train_ds[i]["nicheformer_pool"]) for i in valid_train_indices])
        
        # Set embedding dimensions in config
        cfg.models.img_embed_dim = train_img_raw.shape[1]
        cfg.models.gex_embed_dim = train_gex_raw.shape[1]
        
        # Load trained multimodal model and get aligned embeddings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        multimodal_model = load_multimodal_model(cfg, comm_checkpoint, device)
        
        with torch.no_grad():
            # Get aligned embeddings from the trained model
            train_img_aligned, train_gex_aligned = multimodal_model.get_embeddings(
                train_img_raw.to(device), train_gex_raw.to(device)
            )
            
            # Move back to CPU and concatenate
            train_embeddings = torch.cat([train_img_aligned.cpu(), train_gex_aligned.cpu()], dim=-1).numpy()
    
    else:
        raise ValueError(f"Unknown method: {selected_method}")
    
    # Get patch centers and labels
    train_centers = [patch_center(train_ds[i]["cell_coords"]) for i in valid_train_indices]
    train_centers = np.array(train_centers)
    
    train_labels = np.array(valid_train_labels)
    train_reg_labels = np.array(valid_train_reg_labels)
    
    # Get configuration
    num_bins = cfg.training.spatial_neighbor.num_bins
    max_neighbor_factor = cfg.training.spatial_neighbor.get("max_neighbor_factor", 10)
    
    # Determine which tasks to run
    run_regression = cfg.evaluation.tasks.regress
    run_classification = cfg.evaluation.tasks.classify
    
    if not run_regression and not run_classification:
        print("Warning: Neither regression nor classification tasks are enabled")
        return {}
    
    # Extract neighbor prediction pairs
    print(f"  Extracting neighbor prediction pairs with {num_bins} distance bins...")
    binned_pairs, bin_edges = extract_neighbor_prediction_pairs(
        train_embeddings, train_labels, train_centers, num_bins, max_neighbor_factor
    )
    
    # Evaluate each distance bin
    metrics = {}
    for b in range(num_bins):
        if b not in binned_pairs or len(binned_pairs[b]['z_i']) == 0:
            print(f"    Bin {b}: No pairs found")
            continue
        
        print(f"    Bin {b}: {len(binned_pairs[b]['z_i'])} pairs")
        
        # Evaluate regression if enabled
        if run_regression:
            reg_metrics = evaluate_neighbor_prediction_bin_mlp(
                binned_pairs[b]['z_i'], train_reg_labels[binned_pairs[b]['y_j']], 
                task_type="regression", test_size=0.2, random_state=42
            )
            if reg_metrics:
                metrics[f'bin_{b}_train_r2'] = reg_metrics['train_r2']
                metrics[f'bin_{b}_test_r2'] = reg_metrics['test_r2']
                metrics[f'bin_{b}_train_mse'] = reg_metrics['train_mse']
                metrics[f'bin_{b}_test_mse'] = reg_metrics['test_mse']
        
        # Evaluate classification if enabled
        if run_classification:
            class_metrics = evaluate_neighbor_prediction_bin_mlp(
                binned_pairs[b]['z_i'], binned_pairs[b]['y_j'], 
                task_type="classification", test_size=0.2, random_state=42
            )
            if class_metrics:
                metrics[f'bin_{b}_train_accuracy'] = class_metrics['train_accuracy']
                metrics[f'bin_{b}_test_accuracy'] = class_metrics['test_accuracy']
                metrics[f'bin_{b}_train_f1_macro'] = class_metrics['train_f1_macro']
                metrics[f'bin_{b}_test_f1_macro'] = class_metrics['test_f1_macro']
        
        metrics[f'bin_{b}_n_pairs'] = len(binned_pairs[b]['z_i'])
        metrics[f'bin_{b}_avg_distance'] = np.mean(binned_pairs[b]['distances'])
    
    # Add distance bin information
    metrics['distance_bin_edges'] = bin_edges.tolist()
    
    return metrics

def print_results_table(metrics, distance_bin_edges=None):
    """Print results in a nice table format."""
    print("\n" + "="*80)
    print("SPATIAL NEIGHBOR EVALUATION RESULTS (MLP HEADS)")
    print("="*80)
    
    if distance_bin_edges is not None:
        print(f"Distance bins: {distance_bin_edges}")
    
    # Print regression results
    reg_bins = [k for k in metrics.keys() if k.startswith('bin_') and k.endswith('_test_r2')]
    if reg_bins:
        print("\nRegression Results (R²):")
        headers = ["Bin", "Train R²", "Test R²", "Train MSE", "Test MSE", "N Pairs", "Avg Distance"]
        table_data = []
        for bin_key in sorted(reg_bins):
            b = bin_key.split('_')[1]
            row = [f"Bin {b}"]
            row.append(f"{metrics.get(f'bin_{b}_train_r2', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_test_r2', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_train_mse', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_test_mse', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_n_pairs', 'N/A')}")
            row.append(f"{metrics.get(f'bin_{b}_avg_distance', 'N/A'):.2f}")
            table_data.append(row)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print classification results
    class_bins = [k for k in metrics.keys() if k.startswith('bin_') and k.endswith('_test_accuracy')]
    if class_bins:
        print("\nClassification Results:")
        headers = ["Bin", "Train Acc", "Test Acc", "Train F1", "Test F1", "N Pairs", "Avg Distance"]
        table_data = []
        for bin_key in sorted(class_bins):
            b = bin_key.split('_')[1]
            row = [f"Bin {b}"]
            row.append(f"{metrics.get(f'bin_{b}_train_accuracy', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_test_accuracy', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_train_f1_macro', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_test_f1_macro', 'N/A'):.4f}")
            row.append(f"{metrics.get(f'bin_{b}_n_pairs', 'N/A')}")
            row.append(f"{metrics.get(f'bin_{b}_avg_distance', 'N/A'):.2f}")
            table_data.append(row)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("="*80)

def save_results(results_dir: Path, cfg: DictConfig, metrics: dict):
    """Save evaluation results to JSON."""
    result_data = {
        "experiment": {
            "modality": cfg.evaluation.modality,
            "img_model": cfg.evaluation.img_model,
            "gex_model": cfg.evaluation.gex_model,
            "dataset": cfg.data.dataset,
            "evaluation_strategy": "spatial_neighbor_eval_mlp",
            "train_split": "train",
            "test_split": "test",
            "selected_method": getattr(cfg.evaluation, 'selected_method', None),
            "tasks": {
                "classification": cfg.evaluation.tasks.classify,
                "regression": cfg.evaluation.tasks.regress
            },
            "spatial_neighbor": {
                "num_bins": cfg.training.spatial_neighbor.num_bins,
                "max_neighbor_factor": cfg.training.spatial_neighbor.get("max_neighbor_factor", 10)
            }
        },
        "results": metrics
    }
    
    result_data = make_serializable(result_data)
    
    # Save to file
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"spatial_neighbor_mlp_{getattr(cfg.evaluation, 'selected_method', 'unknown')}_results.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to JSON: {json_path}")


@hydra.main(config_path='../../../configs', config_name='downstream.yaml')
def main(cfg: DictConfig):
    # Get selected method from config or infer from modality
    selected_method = None
    
    # Try to get from config if it exists
    if hasattr(cfg.evaluation, 'selected_method'):
        selected_method = cfg.evaluation.selected_method
    
    # If not found, infer from modality
    if selected_method is None:
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
    
    print("Starting Spatial Neighbor Evaluation with MLP Heads")
    print(f"Configuration:")
    print(f"  - Modality: {cfg.evaluation.modality}")
    print(f"  - Image model: {cfg.evaluation.img_model}")
    print(f"  - GEX model: {cfg.evaluation.gex_model}")
    print(f"  - Selected method: {selected_method}")
    print(f"  - Evaluation strategy: Spatial Neighbor Prediction with MLP heads")
    print(f"  - Tasks: classify={cfg.evaluation.tasks.classify}, regress={cfg.evaluation.tasks.regress}")
    print(f"  - Distance bins: {cfg.training.spatial_neighbor.num_bins}")

    # Setup results directory
    project_dir = get_project_dir()
    results_dir = project_dir / "results" / cfg.data.dataset / f"spatial_neighbor_mlp_{cfg.evaluation.img_model}"
    
    # Load train and test data
    train_ds = load_split_data(cfg, "train", selected_method)
    test_ds = load_split_data(cfg, "test", selected_method)
    
    # Run evaluation with MLP heads
    metrics = run_spatial_neighbor_eval_mlp(cfg, train_ds, test_ds, selected_method)
    
    # Print and save results
    distance_bin_edges = metrics.get('distance_bin_edges')
    print_results_table(metrics, distance_bin_edges)
    save_results(results_dir, cfg, metrics)
    print("\n=== Spatial Neighbor Evaluation with MLP Heads Complete ===")

if __name__ == "__main__":
    main() 