#!/usr/bin/env python3
"""
Example: Computing the Synergistic Information Score (SIS)

This example demonstrates how to compute SIS scores for multimodal models.
SIS quantifies how much additional information the multimodal representation
provides beyond the best unimodal representation.

SIS = (multimodal_performance - max(unimodal_performances)) / max(unimodal_performances)
"""

import sys
from pathlib import Path

# Add parent directory to path to import cell_synergy
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
# Import directly from sis module to avoid __init__.py issues
from cell_synergy.sis import (
    compute_sis,
    compute_sis_all_models,
    print_sis_summary,
    get_sis_ranking,
    analyze_synergy_patterns
)


def create_example_results():
    """
    Create example evaluation results for demonstration.
    
    In practice, these would come from actual model evaluations.
    """
    results = {
        # Unimodal baselines
        "Unimodal GEX": {
            "F1 Macro": [0.72, 0.73, 0.71],  # Multiple cross-validation folds
            "R2": [0.65, 0.66, 0.64]
        },
        "Unimodal IMG": {
            "F1 Macro": [0.68, 0.69, 0.67],
            "R2": [0.62, 0.63, 0.61]
        },
        
        # Multimodal models
        "Multimodal CoMM": {
            "F1 Macro": [0.82, 0.83, 0.81],
            "R2": [0.75, 0.76, 0.74]
        },
        "Multimodal CCA": {
            "F1 Macro": [0.75, 0.76, 0.74],
            "R2": [0.68, 0.69, 0.67]
        },
        "Multimodal SimCLR": {
            "F1 Macro": [0.78, 0.79, 0.77],
            "R2": [0.71, 0.72, 0.70]
        },
        "Multimodal Concat": {
            "F1 Macro": [0.74, 0.75, 0.73],
            "R2": [0.67, 0.68, 0.66]
        }
    }
    return results


def main():
    print("=" * 60)
    print("Example: Computing Synergistic Information Score (SIS)")
    print("=" * 60)
    print()
    
    # Load example results
    results = create_example_results()
    
    print("Example evaluation results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        f1_mean = np.mean(metrics["F1 Macro"])
        r2_mean = np.mean(metrics["R2"])
        print(f"{model_name:20s} | F1 Macro: {f1_mean:.3f} | R²: {r2_mean:.3f}")
    print()
    
    # Example 1: Compute SIS for a single model
    print("Example 1: Computing SIS for CoMM")
    print("-" * 60)
    sis_scores = compute_sis(results, "Multimodal CoMM")
    print(f"SIS (F1 Macro): {sis_scores['F1 Macro']:.4f}")
    print(f"SIS (R²):       {sis_scores['R2']:.4f}")
    print()
    print("Interpretation:")
    print(f"  - CoMM improves F1 by {sis_scores['F1 Macro']*100:.1f}% over the best unimodal model")
    print(f"  - CoMM improves R² by {sis_scores['R2']*100:.1f}% over the best unimodal model")
    print()
    
    # Example 2: Compute SIS for all multimodal models
    print("Example 2: Computing SIS for all multimodal models")
    print("-" * 60)
    all_sis = compute_sis_all_models(results)
    for model_name, scores in all_sis.items():
        print(f"{model_name:20s} | F1 SIS: {scores['F1 Macro']:7.4f} | R² SIS: {scores['R2']:7.4f}")
    print()
    
    # Example 3: Print formatted summary
    print("Example 3: Formatted SIS summary")
    print("-" * 60)
    print_sis_summary(results)
    print()
    
    # Example 4: Get ranking by SIS score
    print("Example 4: Ranking models by SIS (F1 Macro)")
    print("-" * 60)
    ranking = get_sis_ranking(results, metric="F1 Macro")
    for rank, (model_name, sis_score) in enumerate(ranking, 1):
        print(f"{rank}. {model_name:20s} | SIS: {sis_score:7.4f}")
    print()
    
    # Example 5: Analyze synergy patterns
    print("Example 5: Synergy pattern analysis")
    print("-" * 60)
    analysis = analyze_synergy_patterns(results)
    for model_name, patterns in analysis.items():
        print(f"\n{model_name}:")
        print(f"  F1 SIS: {patterns['F1 Macro SIS']:.4f} ({patterns['F1 Macro Synergy Level']})")
        print(f"  R² SIS: {patterns['R2 SIS']:.4f} ({patterns['R2 Synergy Level']})")
        print(f"  Dominant modality (F1): {patterns['Dominant Modality (F1)']}")
        print(f"  Dominant modality (R²): {patterns['Dominant Modality (R2)']}")
    print()
    
    print("=" * 60)
    print("Note: In practice, load results from actual model evaluations")
    print("      using the evaluation scripts in scripts/evaluation/")
    print("=" * 60)


if __name__ == "__main__":
    main()

