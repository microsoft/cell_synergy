#!/usr/bin/env python3
"""
Example: Evaluating Spatial Consistency

This example demonstrates how to evaluate spatial consistency, which measures
whether spatially close patches have similar embeddings.

The task: Predict whether two patches are spatially close (positive) or far (negative).

Note: This is a simplified example. For production use, see:
  - scripts/evaluation/spatial_evaluation/
  - src/cell_synergy/downstream/spatial_consistency_evaluation.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
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


def create_spatial_pairs(coords, embeddings, slide_ids, k=5, dist_thresh=None):
    """
    Create positive and negative pairs for spatial consistency evaluation.
    
    Positive pairs: k-nearest neighbors within the same slide
    Negative pairs: patches far apart (beyond threshold) within the same slide
    """
    positive_pairs = []
    negative_pairs = []
    
    for slide_id in np.unique(slide_ids):
        # Get patches for this slide
        slide_mask = slide_ids == slide_id
        slide_coords = coords[slide_mask]
        slide_embeddings = embeddings[slide_mask]
        slide_indices = np.where(slide_mask)[0]
        
        # Compute pairwise distances
        distances = squareform(pdist(slide_coords))
        
        # Positive pairs: k-nearest neighbors
        for i, idx in enumerate(slide_indices):
            # Get k nearest neighbors (excluding self)
            nearest = np.argsort(distances[i])[1:k+1]
            for j in nearest:
                pos_idx = slide_indices[j]
                positive_pairs.append((idx, pos_idx))
        
        # Negative pairs: far patches
        if dist_thresh is None:
            dist_thresh = np.percentile(distances[distances > 0], 80)
        
        for i, idx_i in enumerate(slide_indices):
            for j, idx_j in enumerate(slide_indices):
                if i != j and distances[i, j] > dist_thresh:
                    negative_pairs.append((idx_i, idx_j))
    
    # Balance positive and negative pairs
    n_pos = len(positive_pairs)
    n_neg = min(len(negative_pairs), n_pos)
    if len(negative_pairs) > n_neg:
        # Randomly sample negative pairs
        neg_indices = np.random.choice(len(negative_pairs), n_neg, replace=False)
        negative_pairs = [negative_pairs[i] for i in neg_indices]
    
    # Limit positive pairs to match negative pairs
    if len(positive_pairs) > n_neg:
        pos_indices = np.random.choice(len(positive_pairs), n_neg, replace=False)
        positive_pairs = [positive_pairs[i] for i in pos_indices]
    
    return positive_pairs, negative_pairs


def create_pair_features(embeddings, pairs):
    """Create features for pairs by concatenating embeddings."""
    features = []
    for i, j in pairs:
        # Concatenate embeddings
        pair_feat = np.concatenate([embeddings[i], embeddings[j]])
        features.append(pair_feat)
    return np.array(features)


def evaluate_spatial_consistency(embeddings, coords, slide_ids):
    """Evaluate spatial consistency using pair classification."""
    # Create pairs
    pos_pairs, neg_pairs = create_spatial_pairs(coords, embeddings, slide_ids)
    
    # Create features and labels
    pos_features = create_pair_features(embeddings, pos_pairs)
    neg_features = create_pair_features(embeddings, neg_pairs)
    
    X = np.vstack([pos_features, neg_features])
    y = np.array([1] * len(pos_features) + [0] * len(neg_features))
    
    # Split by slide (group K-fold)
    groups = []
    for pair in pos_pairs + neg_pairs:
        # Use slide ID of first patch in pair
        groups.append(slide_ids[pair[0]])
    groups = np.array(groups)
    
    # Cross-validation
    gkf = GroupKFold(n_splits=5)
    scores = {'f1': [], 'accuracy': [], 'roc_auc': []}
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Evaluate
        scores['f1'].append(f1_score(y_test, y_pred))
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['roc_auc'].append(roc_auc_score(y_test, y_proba))
    
    return scores


def main():
    print("=" * 60)
    print("Example: Evaluating Spatial Consistency")
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
    
    # Evaluate spatial consistency
    print("Step 2: Evaluate spatial consistency")
    print("-" * 60)
    print("  Creating positive pairs (k-nearest neighbors)...")
    print("  Creating negative pairs (far patches)...")
    print()
    
    scores = evaluate_spatial_consistency(embeddings, coords, slide_ids)
    
    print("Results (5-fold cross-validation):")
    print(f"  F1 Score:     {np.mean(scores['f1']):.4f} ± {np.std(scores['f1']):.4f}")
    print(f"  Accuracy:      {np.mean(scores['accuracy']):.4f} ± {np.std(scores['accuracy']):.4f}")
    print(f"  ROC-AUC:       {np.mean(scores['roc_auc']):.4f} ± {np.std(scores['roc_auc']):.4f}")
    print()
    
    print("Interpretation:")
    print("  - Higher scores indicate better spatial consistency")
    print("  - F1 > 0.7 suggests embeddings capture spatial structure")
    print()
    
    print("=" * 60)
    print("For actual evaluation, use:")
    print("  python -m cell_synergy.downstream.spatial_consistency_evaluation \\")
    print("      --config-name downstream \\")
    print("      data.dataset=lung")
    print()
    print("Or use SLURM batch script:")
    print("  sbatch scripts/evaluation/spatial_evaluation/run_spatial_consistency_gex.sbatch")
    print("=" * 60)


if __name__ == "__main__":
    main()

