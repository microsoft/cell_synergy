"""
Synergistic Information Score (SIS) computation module.

This module implements the SIS metric as defined in the paper:
SIS(Y; z_1, z_2) = (I(Y; z_3) - max(I(Y; z_1), I(Y; z_2))) / max(I(Y; z_1), I(Y; z_2))

Where:
- z_1 is the GEX (gene expression) unimodal representation
- z_2 is the IMG (image) unimodal representation
- z_3 is the multimodal representation
- I(Y; z) is approximated by performance metrics (F1 Macro, R²)
"""

from typing import Dict, List, Union, Tuple
import numpy as np

# Global variable to store results (set by user)
_RESULTS = None


def set_RESULTS(results: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Set the global results dictionary for SIS computation.

    Args:
        results: Dictionary containing performance results for all models
    """
    global _RESULTS
    _RESULTS = results


def SIS(model_name: str, gex_key: str = "Unimodal GEX", img_key: str = "Unimodal IMG") -> None:
    """
    Simple wrapper function to compute and print SIS for a specific model.

    Args:
        model_name: Name of the multimodal model to evaluate
        gex_key: Key for gene expression (GEX) unimodal model results (mod1)
        img_key: Key for image (IMG) unimodal model results (mod2)

    Note:
        Call set_RESULTS() first to set the results dictionary
    """
    if _RESULTS is None:
        print("Error: No results set. Call set_RESULTS() first.")
        return

    sis_scores = compute_sis(_RESULTS, model_name, gex_key, img_key)
    print(f"SIS R2: {sis_scores['R2']:.4f}")
    print(f"SIS Macro F1: {sis_scores['F1 Macro']:.4f}")


def compute_sis(
    results: Dict[str, Dict[str, List[float]]],
    multimodal_model: str,
    gex_key: str = "Unimodal GEX",
    img_key: str = "Unimodal IMG"
) -> Dict[str, float]:
    """
    Compute the Synergistic Information Score (SIS) for a given multimodal model.

    Args:
        results: Dictionary containing performance results for all models
        multimodal_model: Name of the multimodal model to evaluate
        gex_key: Key for gene expression (GEX) unimodal model results (mod1)
        img_key: Key for image (IMG) unimodal model results (mod2)

    Returns:
        Dictionary with SIS scores for each metric (F1 Macro, R²)
    """

    # Extract performance metrics
    # mod1 = GEX (gene expression)
    gex_f1 = np.mean(results[gex_key]["F1 Macro"])
    gex_r2 = np.mean(results[gex_key]["R2"])

    # mod2 = IMG (image)
    img_f1 = np.mean(results[img_key]["F1 Macro"])
    img_r2 = np.mean(results[img_key]["R2"])

    multimodal_f1 = np.mean(results[multimodal_model]["F1 Macro"])
    multimodal_r2 = np.mean(results[multimodal_model]["R2"])

    # Compute SIS for F1 Macro
    max_unimodal_f1 = max(gex_f1, img_f1)
    sis_f1 = (multimodal_f1 - max_unimodal_f1) / max_unimodal_f1

    # Compute SIS for R²
    max_unimodal_r2 = max(gex_r2, img_r2)
    sis_r2 = (multimodal_r2 - max_unimodal_r2) / max_unimodal_r2

    return {
        "F1 Macro": sis_f1,
        "R2": sis_r2
    }


def compute_sis_all_models(
    results: Dict[str, Dict[str, List[float]]],
    gex_key: str = "Unimodal GEX",
    img_key: str = "Unimodal IMG"
) -> Dict[str, Dict[str, float]]:
    """
    Compute SIS for all multimodal models in the results.

    Args:
        results: Dictionary containing performance results for all models
        gex_key: Key for gene expression (GEX) unimodal model results (mod1)
        img_key: Key for image (IMG) unimodal model results (mod2)

    Returns:
        Dictionary mapping model names to their SIS scores
    """

    # Identify multimodal models (exclude unimodal and random)
    multimodal_models = [
        key for key in results.keys()
        if key.startswith("Multimodal") and key not in [gex_key, img_key]
    ]

    sis_RESULTS = {}
    for model in multimodal_models:
        sis_RESULTS[model] = compute_sis(results, model, gex_key, img_key)

    return sis_RESULTS


def print_sis_summary(
    results: Dict[str, Dict[str, List[float]]],
    model_name: str = None,
    gex_key: str = "Unimodal GEX",
    img_key: str = "Unimodal IMG"
) -> None:
    """
    Print a formatted summary of SIS scores.

    Args:
        results: Dictionary containing performance results for all models
        model_name: Specific model to print SIS for (if None, prints all)
        gex_key: Key for gene expression (GEX) unimodal model results (mod1)
        img_key: Key for image (IMG) unimodal model results (mod2)
    """

    if model_name:
        # Print SIS for specific model
        sis_scores = compute_sis(results, model_name, gex_key, img_key)
        print(f"SIS for {model_name}:")
        print(f"  F1 Macro: {sis_scores['F1 Macro']:.4f}")
        print(f"  R²: {sis_scores['R2']:.4f}")
    else:
        # Print SIS for all multimodal models
        sis_RESULTS = compute_sis_all_models(results, gex_key, img_key)
        print("Synergistic Information Score (SIS) Summary:")
        print("=" * 50)

        for model, scores in sis_RESULTS.items():
            print(f"\n{model}:")
            print(f"  F1 Macro: {scores['F1 Macro']:.4f}")
            print(f"  R²: {scores['R2']:.4f}")


def get_sis_ranking(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = "F1 Macro",
    gex_key: str = "Unimodal GEX",
    img_key: str = "Unimodal IMG"
) -> List[Tuple[str, float]]:
    """
    Get ranked list of models by SIS score for a specific metric.

    Args:
        results: Dictionary containing performance results for all models
        metric: Metric to rank by ("F1 Macro" or "R2")
        gex_key: Key for gene expression (GEX) unimodal model results (mod1)
        img_key: Key for image (IMG) unimodal model results (mod2)

    Returns:
        List of (model_name, sis_score) tuples, sorted by SIS descending
    """

    sis_RESULTS = compute_sis_all_models(results, gex_key, img_key)

    # Extract SIS scores for the specified metric
    ranking = [(model, scores[metric]) for model, scores in sis_RESULTS.items()]

    # Sort by SIS score (descending)
    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def analyze_synergy_patterns(
    results: Dict[str, Dict[str, List[float]]],
    gex_key: str = "Unimodal GEX",
    img_key: str = "Unimodal IMG"
) -> Dict[str, Dict[str, Union[str, float]]]:
    """
    Analyze synergy patterns across all models.

    Args:
        results: Dictionary containing performance results for all models
        gex_key: Key for gene expression (GEX) unimodal model results (mod1)
        img_key: Key for image (IMG) unimodal model results (mod2)

    Returns:
        Dictionary with synergy analysis for each model
    """

    sis_RESULTS = compute_sis_all_models(results, gex_key, img_key)

    analysis = {}
    for model, scores in sis_RESULTS.items():
        # Determine synergy level
        f1_synergy = "High" if scores["F1 Macro"] > 0.1 else "Moderate" if scores["F1 Macro"] > 0 else "Low"
        r2_synergy = "High" if scores["R2"] > 0.1 else "Moderate" if scores["R2"] > 0 else "Low"

        # Determine dominant modality
        # mod1 = GEX (gene expression)
        gex_f1 = np.mean(results[gex_key]["F1 Macro"])
        gex_r2 = np.mean(results[gex_key]["R2"])
        # mod2 = IMG (image)
        img_f1 = np.mean(results[img_key]["F1 Macro"])
        img_r2 = np.mean(results[img_key]["R2"])

        dominant_f1 = "GEX" if gex_f1 > img_f1 else "IMG"
        dominant_r2 = "GEX" if gex_r2 > img_r2 else "IMG"

        analysis[model] = {
            "F1 Macro SIS": scores["F1 Macro"],
            "R2 SIS": scores["R2"],
            "F1 Macro Synergy Level": f1_synergy,
            "R2 Synergy Level": r2_synergy,
            "Dominant Modality (F1)": dominant_f1,
            "Dominant Modality (R2)": dominant_r2
        }

    return analysis
