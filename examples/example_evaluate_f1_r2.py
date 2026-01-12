#!/usr/bin/env python3
"""
Example: Evaluating F1 and R² Scores

This example demonstrates how to evaluate multimodal models on downstream tasks:
- Classification (F1 Macro score)
- Regression (R² score)

Note: This is a simplified example. For production use, see:
  - scripts/evaluation/
  - src/cell_synergy/downstream/run_benchmarks.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold


def create_synthetic_data(n_samples=1000, n_features=128, n_classes=10):
    """Create synthetic data for demonstration."""
    # Features (embeddings)
    X = np.random.randn(n_samples, n_features)
    
    # Classification labels
    y_clf = np.random.randint(0, n_classes, n_samples)
    
    # Regression labels (continuous)
    y_reg = np.random.randn(n_samples)
    
    return X, y_clf, y_reg


def evaluate_classification(X_train, y_train, X_test, y_test):
    """
    Evaluate classification performance using F1 Macro score.
    
    Uses a linear probe (Logistic Regression) on frozen embeddings.
    """
    # Train linear probe
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Compute F1 Macro
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return f1, y_pred


def evaluate_regression(X_train, y_train, X_test, y_test):
    """
    Evaluate regression performance using R² score.
    
    Uses a linear probe (Ridge Regression) on frozen embeddings.
    """
    # Train linear probe
    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X_train, y_train)
    
    # Predict
    y_pred = reg.predict(X_test)
    
    # Compute R²
    r2 = r2_score(y_test, y_pred)
    
    return r2, y_pred


def cross_validate(X, y_clf, y_reg, n_splits=5):
    """Perform cross-validation for both tasks."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    f1_scores = []
    r2_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_clf_train, y_clf_test = y_clf[train_idx], y_clf[test_idx]
        y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]
        
        # Classification
        f1, _ = evaluate_classification(X_train, y_clf_train, X_test, y_clf_test)
        f1_scores.append(f1)
        
        # Regression
        r2, _ = evaluate_regression(X_train, y_reg_train, X_test, y_reg_test)
        r2_scores.append(r2)
    
    return f1_scores, r2_scores


def main():
    print("=" * 60)
    print("Example: Evaluating F1 and R² Scores")
    print("=" * 60)
    print()
    
    # Create synthetic data
    print("Step 1: Create synthetic data")
    print("-" * 60)
    X, y_clf, y_reg = create_synthetic_data(n_samples=1000, n_features=128, n_classes=10)
    print(f"  Features shape: {X.shape}")
    print(f"  Classification labels: {len(np.unique(y_clf))} classes")
    print(f"  Regression labels: continuous")
    print()
    
    # Split data
    print("Step 2: Split data into train/test")
    print("-" * 60)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_clf_train, y_clf_test = y_clf[:split_idx], y_clf[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()
    
    # Evaluate classification
    print("Step 3: Evaluate classification (F1 Macro)")
    print("-" * 60)
    f1, y_pred_clf = evaluate_classification(X_train, y_clf_train, X_test, y_clf_test)
    print(f"  F1 Macro score: {f1:.4f}")
    print()
    
    # Evaluate regression
    print("Step 4: Evaluate regression (R²)")
    print("-" * 60)
    r2, y_pred_reg = evaluate_regression(X_train, y_reg_train, X_test, y_reg_test)
    print(f"  R² score: {r2:.4f}")
    print()
    
    # Cross-validation
    print("Step 5: Cross-validation (5-fold)")
    print("-" * 60)
    f1_scores, r2_scores = cross_validate(X, y_clf, y_reg, n_splits=5)
    print(f"  F1 Macro (mean ± std): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  R² (mean ± std):       {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print()
    
    print("=" * 60)
    print("For actual evaluation, use:")
    print("  python -m cell_synergy.downstream.run_benchmarks \\")
    print("      --config-name downstream \\")
    print("      evaluation.modality=multimodal \\")
    print("      data.dataset=lung")
    print()
    print("Or use SLURM batch script:")
    print("  sbatch scripts/evaluation/evaluate_hf_datasets/eval_1pct_cfg1.sbatch")
    print("=" * 60)


if __name__ == "__main__":
    main()

