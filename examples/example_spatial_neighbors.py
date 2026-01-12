#!/usr/bin/env python3
"""
Example: Evaluating Spatial Neighbor Prediction

This example demonstrates how to evaluate spatial neighbor prediction, which
measures whether embeddings can predict the number of neighbors at different
distance scales.

The task: Predict the number of neighbors within distance bins.

Note: This is a simplified example. For production use, see:
  - scripts/evaluation/spatial_evaluation/
  - src/cell_synergy/downstream/spatial_neighbor_distance_bins.py
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import pdist, squareform


def create_synthetic_spatial_data(n_samples=500, n_features=128, n_slides=5):
    """Create synthetic spatial data with coordinates and embeddings."""
    # Generate spatial coordinates (x, y) for each patch
    coords = np.random.rand(n_samples, 2) * 100  # 100x100 spatial grid
    
    # Generate embeddings (in practice, these come from trained models)
    embeddings = np.random.randn(n_samples, n_features)
    
    # Assign patches to slides
    slide_ids = np.random.randint(0, n_slides, n_samples)
    
    return coords, embeddings, slide_ids


def compute_neighbor_counts(coords, distance_bins):
    """
    Compute the number of neighbors within each distance bin for each patch.
    
    Args:
        coords: Array of shape (n_samples, 2) with spatial coordinates
        distance_bins: Array of bin edges (e.g., [0, 10, 20, 30, ...])
    
    Returns:
        neighbor_counts: Array of shape (n_samples, n_bins) with neighbor counts
    """
    # Compute pairwise distances
    distances = squareform(pdist(coords))
    
    n_samples = len(coords)
    n_bins = len(distance_bins) - 1
    neighbor_counts = np.zeros((n_samples, n_bins), dtype=int)
    
    for i in range(n_samples):
        for bin_idx in range(n_bins):
            # Count neighbors within this distance bin
            bin_min = distance_bins[bin_idx]
            bin_max = distance_bins[bin_idx + 1]
            mask = (distances[i] > bin_min) & (distances[i] <= bin_max)
            neighbor_counts[i, bin_idx] = np.sum(mask)
    
    return neighbor_counts


def evaluate_neighbor_prediction(embeddings, coords, slide_ids, distance_bins):
    """
    Evaluate how well embeddings predict neighbor counts at different distances.
    """
    # Compute ground truth neighbor counts
    neighbor_counts = compute_neighbor_counts(coords, distance_bins)
    
    # Split by slide (group K-fold)
    gkf = GroupKFold(n_splits=5)
    
    results = {
        'r2': [],
        'mse': [],
        'per_bin_r2': [[] for _ in range(len(distance_bins) - 1)]
    }
    
    for train_idx, test_idx in gkf.split(embeddings, neighbor_counts, slide_ids):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = neighbor_counts[train_idx], neighbor_counts[test_idx]
        
        # Train regressor for each distance bin
        bin_r2_scores = []
        bin_mse_scores = []
        
        for bin_idx in range(len(distance_bins) - 1):
            # Train Ridge regression for this bin
            reg = Ridge(alpha=1.0, random_state=42)
            reg.fit(X_train, y_train[:, bin_idx])
            
            # Predict
            y_pred = reg.predict(X_test)
            
            # Evaluate
            r2 = r2_score(y_test[:, bin_idx], y_pred)
            mse = mean_squared_error(y_test[:, bin_idx], y_pred)
            
            bin_r2_scores.append(r2)
            bin_mse_scores.append(mse)
            results['per_bin_r2'][bin_idx].append(r2)
        
        # Overall performance (average across bins)
        results['r2'].append(np.mean(bin_r2_scores))
        results['mse'].append(np.mean(bin_mse_scores))
    
    return results, distance_bins


def main():
    print("=" * 60)
    print("Example: Evaluating Spatial Neighbor Prediction")
    print("=" * 60)
    print()
    
    # Create synthetic data
    print("Step 1: Create synthetic spatial data")
    print("-" * 60)
    coords, embeddings, slide_ids = create_synthetic_spatial_data(
        n_samples=500, n_features=128, n_slides=5
    )
    print(f"  Patches: {len(coords)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Slides: {len(np.unique(slide_ids))}")
    print()
    
    # Define distance bins
    print("Step 2: Define distance bins")
    print("-" * 60)
    distance_bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    print(f"  Distance bins: {distance_bins}")
    print(f"  Number of bins: {len(distance_bins) - 1}")
    print()
    
    # Evaluate neighbor prediction
    print("Step 3: Evaluate neighbor prediction")
    print("-" * 60)
    print("  Computing neighbor counts for each distance bin...")
    print("  Training regressors to predict neighbor counts...")
    print()
    
    results, bins = evaluate_neighbor_prediction(embeddings, coords, slide_ids, distance_bins)
    
    print("Results (5-fold cross-validation):")
    print(f"  Overall R²:    {np.mean(results['r2']):.4f} ± {np.std(results['r2']):.4f}")
    print(f"  Overall MSE:    {np.mean(results['mse']):.4f} ± {np.std(results['mse']):.4f}")
    print()
    
    print("Per-bin R² scores:")
    for bin_idx in range(len(bins) - 1):
        bin_r2 = np.mean(results['per_bin_r2'][bin_idx])
        bin_std = np.std(results['per_bin_r2'][bin_idx])
        print(f"  Bin {bin_idx+1} ({bins[bin_idx]:.1f}-{bins[bin_idx+1]:.1f}): "
              f"{bin_r2:.4f} ± {bin_std:.4f}")
    print()
    
    print("Interpretation:")
    print("  - Higher R² indicates better prediction of spatial structure")
    print("  - R² > 0.5 suggests embeddings capture spatial organization")
    print("  - Performance may vary across distance scales")
    print()
    
    print("=" * 60)
    print("For actual evaluation, use:")
    print("  python -m cell_synergy.downstream.spatial_neighbor_distance_bins \\")
    print("      --config-name downstream \\")
    print("      data.dataset=lung")
    print()
    print("Or use SLURM batch script:")
    print("  sbatch scripts/evaluation/spatial_evaluation/run_baseline_gex.sbatch")
    print("=" * 60)


if __name__ == "__main__":
    main()

