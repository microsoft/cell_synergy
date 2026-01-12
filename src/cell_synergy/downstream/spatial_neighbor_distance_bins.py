import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from sklearn.metrics import f1_score
from cell_synergy.paths import PROJECT_DIR
import importlib
from cell_synergy.downstream.eval import (
    extract_embeddings,
    load_and_preprocess_data,
)
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler


def patch_center(cell_coords):
    coords = np.array([c for c in cell_coords if c[0] != -1 and c[1] != -1])
    return coords.mean(axis=0) if len(coords) > 0 else np.array([np.nan, np.nan])


def get_distance_bins(mu_coords,
                      num_bins=5,
                      metric="euclidean",
                      dataset_name: Optional[str] = None,
                      distance_percentiles: Optional[List[float]] = None,
                      sample_ids: Optional[np.ndarray] = None):
    """
    Create distance bins for spatial neighbor analysis using optimized percentiles.

    Goal: Get target neighbor counts [8, 16, 24, 32, 40, 48, 56, 64] per bin.

    Args:
        mu_coords: Array of patch center coordinates (N, 2) in PIXELS
        num_bins: Number of distance bins (3-7 recommended)
        metric: Distance metric to use
        dataset_name: Dataset name (e.g., 'lung', 'thymus', 'breast') to load optimized percentiles
        distance_percentiles: Optional list of percentiles to use directly (overrides dataset_name)
        sample_ids: Array of sample IDs (N,) - needed for per-sample conversion if dataset has variable resolution

    Returns:
        bin_edges: Array of bin edges for distance ranges in PIXELS (same units as input)
    """
    # Convert coordinates to micrometers before computing distances
    # This matches the test script behavior - percentiles are optimized for micrometer distances
    microns_per_pixel = None
    if dataset_name is not None:
        try:
            config_path = PROJECT_DIR.parent / "configs" / "spatial_scales.yaml"
            if config_path.exists():
                spatial_config = OmegaConf.load(config_path)
                if dataset_name in spatial_config:
                    if "microns_per_pixel" in spatial_config[dataset_name]:
                        microns_per_pixel = spatial_config[dataset_name]["microns_per_pixel"]
                    elif dataset_name == "breast" and sample_ids is not None:
                        # Breast has variable resolution - need per-sample conversion
                        # Try to get from data config
                        try:
                            # PROJECT_DIR is already imported at module level
                            data_config_path = PROJECT_DIR.parent / "configs" / "data" / f"{dataset_name}.yaml"
                            if data_config_path.exists():
                                data_config = OmegaConf.load(data_config_path)
                                if "pixel_sizes_um_per_pixel" in data_config:
                                    # Create dict mapping sample_id -> conversion
                                    microns_per_pixel = {}
                                    pixel_sizes = data_config.pixel_sizes_um_per_pixel
                                    unique_samples = np.unique(sample_ids)
                                    for sample_id in unique_samples:
                                        # Extract base name (remove suffixes)
                                        base_name = sample_id.split('_')[0] if '_' in sample_id else sample_id
                                        if base_name in pixel_sizes:
                                            microns_per_pixel[sample_id] = pixel_sizes[base_name]
                                        else:
                                            # Try direct match
                                            if sample_id in pixel_sizes:
                                                microns_per_pixel[sample_id] = pixel_sizes[sample_id]
                        except Exception as e:
                            print(f"Warning: Could not load per-sample pixel sizes for breast: {e}")
        except Exception as e:
            print(f"Warning: Failed to load microns_per_pixel from config: {e}")

    # Convert coordinates to micrometers if conversion factor is available
    mu_coords_work = mu_coords.copy()
    if microns_per_pixel is not None:
        if isinstance(microns_per_pixel, (int, float)):
            # Uniform conversion for all samples
            print(f"Converting coordinates from pixels to micrometers using {microns_per_pixel} µm/px")
            mu_coords_work = mu_coords_work * microns_per_pixel
        elif isinstance(microns_per_pixel, dict) and sample_ids is not None:
            # Per-sample conversion (for breast)
            print("Converting coordinates from pixels to micrometers using per-sample conversions")
            mu_coords_um = mu_coords_work.copy()
            for sample_id in np.unique(sample_ids):
                slide_mask = sample_ids == sample_id
                if sample_id in microns_per_pixel:
                    conversion = microns_per_pixel[sample_id]
                    mu_coords_um[slide_mask] = mu_coords_work[slide_mask] * conversion
                else:
                    print(f"Warning: No conversion factor for sample {sample_id}, using pixels")
            mu_coords_work = mu_coords_um
        else:
            print("Warning: microns_per_pixel is dict but no sample_ids provided, using pixels")

    # Compute pairwise distances (now in MICROMETERS if conversion was applied)
    D = squareform(pdist(mu_coords_work, metric=metric))
    upper_tri_mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    distance_vals = D[upper_tri_mask]

    # Load optimized percentiles from config if dataset_name is provided
    if distance_percentiles is None and dataset_name is not None:
        config_path = PROJECT_DIR.parent / "configs" / "spatial_scales.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Cannot load optimized percentiles for dataset '{dataset_name}'. "
                "Please ensure spatial_scales.yaml exists in the configs directory."
            )

        spatial_config = OmegaConf.load(config_path)
        if dataset_name not in spatial_config:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in spatial_scales.yaml. "
                f"Available datasets: {list(spatial_config.keys())}. "
                f"Please add optimized percentiles for '{dataset_name}' to the config."
            )

        if "distance_percentiles" not in spatial_config[dataset_name]:
            raise KeyError(
                f"No 'distance_percentiles' found for dataset '{dataset_name}' in spatial_scales.yaml. "
                f"Available keys: {list(spatial_config[dataset_name].keys())}. "
                f"Please add optimized percentiles for '{dataset_name}' to the config."
            )

        distance_percentiles = spatial_config[dataset_name]["distance_percentiles"]
        print(f"Loaded optimized percentiles from config for {dataset_name}: {distance_percentiles}")

    # Use provided percentiles - REQUIRED if dataset_name was provided
    if distance_percentiles is None:
        if dataset_name is not None:
            raise ValueError(
                f"Failed to load percentiles for dataset '{dataset_name}'. "
                "This should not happen - please check the error messages above."
            )
        else:
            raise ValueError(
                "No percentiles provided and no dataset_name specified. "
                "Either provide distance_percentiles directly or specify dataset_name to load from config."
            )

    percentiles = distance_percentiles
    # If we loaded from config, infer num_bins from percentiles (they define the correct number of bins)
    actual_num_bins = len(percentiles) - 1
    if actual_num_bins != num_bins:
        print(f"Info: Using {actual_num_bins} bins from optimized config (config had {len(percentiles)} percentiles), "
              f"overriding num_bins={num_bins} from training config")
        num_bins = actual_num_bins

    # Convert percentiles to actual distance thresholds (in MICROMETERS if conversion was applied, otherwise PIXELS)
    # Use quantile-based indexing to ensure uniqueness (same as test_distance_bins_neighbor_counts.py)
    sorted_distances = np.sort(distance_vals)
    n = len(sorted_distances)

    bin_edges = []
    used_values = set()

    for p in percentiles:
        if p == 0:
            idx = 0
            value = sorted_distances[idx]
            bin_edges.append(value)
            used_values.add(value)
        elif p >= 100:
            idx = n - 1
            value = sorted_distances[idx]
            bin_edges.append(value)
            used_values.add(value)
        else:
            # Convert percentile to index: p/100 * (n-1), rounded to nearest integer
            idx = int(np.round(p / 100.0 * (n - 1)))
            idx = max(0, min(idx, n - 1))  # Clamp to valid range

            value = sorted_distances[idx]

            # If this value was already used, find the next unique value
            if value in used_values and p > 0:
                # Find next index with different value (search forward)
                search_idx = idx + 1
                while search_idx < n and sorted_distances[search_idx] == value:
                    search_idx += 1
                if search_idx < n:
                    idx = search_idx
                    value = sorted_distances[idx]
                else:
                    # If we reached the end, search backward
                    search_idx = idx - 1
                    while search_idx >= 0 and sorted_distances[search_idx] == value:
                        search_idx -= 1
                    if search_idx >= 0:
                        idx = search_idx
                        value = sorted_distances[idx]

            bin_edges.append(value)
            used_values.add(value)

    bin_edges = np.array(bin_edges)

    # Remove duplicates and ensure sorted
    original_len = len(bin_edges)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < original_len:
        print(f"Warning: Removed {original_len - len(bin_edges)} duplicate bin edges")

    # Ensure we have the minimum distance as first edge
    if len(bin_edges) > 0:
        bin_edges[0] = 0.0

    # Ensure sorted
    bin_edges = np.sort(bin_edges)
    bin_edges[0] = 0.0  # Ensure first is 0

    # Validate we have enough bins
    if len(bin_edges) < 2:
        raise ValueError(f"After processing, we have < 2 bin edges: {bin_edges}. Original percentiles: {percentiles}")

    # Convert bin_edges back to pixels if we converted coordinates
    # The bin_edges are currently in micrometers, but we need them in pixels for find_neighbors_in_bins
    # which expects pixel coordinates
    if microns_per_pixel is not None and isinstance(microns_per_pixel, (int, float)):
        # Convert back to pixels: divide by conversion factor
        bin_edges = bin_edges / microns_per_pixel
        print(f"Converted bin edges back to pixels (dividing by {microns_per_pixel} µm/px)")
    elif microns_per_pixel is not None and isinstance(microns_per_pixel, dict):
        # For variable resolution, we can't convert back easily
        # Instead, we need to work in micrometers throughout
        # But find_neighbors_in_bins expects pixels... this is a problem
        # For now, use average conversion factor
        avg_conversion = np.mean(list(microns_per_pixel.values()))
        bin_edges = bin_edges / avg_conversion
        print(f"Converted bin edges back to pixels using average conversion {avg_conversion} µm/px")

    return bin_edges


def find_neighbors_in_bins(mu_coords, bin_edges, sample_ids=None, metric="euclidean", use_knn=False, knn_bins=None):
    """
    Find neighbors for each patch within each distance bin or k-NN bin.

    Only considers neighbors from the same sample (slide).
    This is essential for spatial analysis - we don't want to predict properties
    of patches from different slides.

    Args:
        mu_coords: Array of patch center coordinates (N, 2)
        bin_edges: Array of bin edges for distance ranges (if use_knn=False)
        sample_ids: Array of sample IDs (N,) - patches with same ID are on same slide
        metric: Distance metric to use
        use_knn: If True, use k-nearest-neighbor bins instead of distance bins
        knn_bins: List of k values defining bins, e.g., [10, 35, 85, 185] for 4 bins
                  Bin 0: neighbors 1-10, Bin 1: neighbors 11-35, etc.

    Returns:
        neighbor_lists: List of lists, where neighbor_lists[i][b] contains
                       indices of neighbors of patch i in distance/k-NN bin b
                       (only includes neighbors from the same sample)
    """
    from sklearn.neighbors import NearestNeighbors

    N = len(mu_coords)
    mu_coords = np.asarray(mu_coords, dtype=np.float32)

    if use_knn:
        # K-NN approach: bins are defined by neighbor rank
        num_bins = len(knn_bins) - 1
    else:
        # Distance approach: bins are defined by distance ranges
        num_bins = len(bin_edges) - 1

    # Initialize neighbor lists
    neighbor_lists = [[] for _ in range(N)]

    # Process each slide separately to avoid cross-slide leakage
    for slide_id in np.unique(sample_ids):
        slide_mask = sample_ids == slide_id
        slide_indices = np.where(slide_mask)[0]
        slide_coords = mu_coords[slide_indices]

        if len(slide_coords) < 2:
            # Skip slides with < 2 patches (can't have neighbors)
            continue

        # Use KDTree for efficient neighbor search
        nn = NearestNeighbors(metric=metric, n_jobs=1)
        nn.fit(slide_coords)

        if use_knn:
            # Find max neighbors needed for all bins, but adapt to slide size
            max_k = knn_bins[-1] if knn_bins else 100
            max_k = min(max_k, len(slide_coords) - 1)  # Can't have more neighbors than patches-1

            if max_k > 0:
                distances, indices = nn.kneighbors(slide_coords, n_neighbors=max_k + 1)
            else:
                continue
        else:
            # For distance bins, we need all neighbors within max distance
            max_dist = bin_edges[-1] if bin_edges is not None else np.inf
            distances, indices = nn.radius_neighbors(slide_coords, radius=max_dist)

        # Process each patch in this slide
        for i, global_idx in enumerate(slide_indices):
            if use_knn:
                # K-NN binning: sort by distance and take k-th nearest neighbors
                patch_distances = distances[i]
                patch_indices = indices[i]

                # Exclude self (distance = 0)
                self_mask = patch_distances > 0
                patch_distances = patch_distances[self_mask]
                patch_indices = patch_indices[self_mask]

                # Sort by distance
                sort_order = np.argsort(patch_distances)
                patch_distances = patch_distances[sort_order]
                patch_indices = patch_indices[sort_order]

                # Assign to bins based on rank
                for b in range(num_bins):
                    k_start = knn_bins[b]
                    k_end = knn_bins[b + 1]

                    if k_start < len(patch_indices):
                        # Take neighbors ranked k_start to k_end (0-indexed)
                        bin_indices = patch_indices[k_start:min(k_end, len(patch_indices))]
                        # Convert back to global indices
                        global_bin_indices = slide_indices[bin_indices]
                        neighbor_lists[global_idx].append(global_bin_indices)
                    else:
                        # Not enough neighbors available for this bin
                        neighbor_lists[global_idx].append(np.array([]))
            else:
                # Distance binning: traditional approach
                patch_distances = distances[i]
                patch_indices = indices[i]

                # Exclude self (distance = 0)
                self_mask = patch_distances > 0
                patch_distances = patch_distances[self_mask]
                patch_indices = patch_indices[self_mask]

                # Assign to bins based on distance
                for b in range(num_bins):
                    min_dist = bin_edges[b]
                    max_dist = bin_edges[b + 1]

                    if b == 0:
                        bin_mask = (patch_distances > 0) & (patch_distances <= max_dist)
                    else:
                        bin_mask = (patch_distances > min_dist) & (patch_distances <= max_dist)

                    bin_indices = patch_indices[bin_mask]
                    # Convert back to global indices
                    global_bin_indices = slide_indices[bin_indices]
                    neighbor_lists[global_idx].append(global_bin_indices)

    # Sanity checks to detect bugs
    _validate_neighbor_computation(neighbor_lists, sample_ids, N)

    return neighbor_lists


def _validate_neighbor_computation(neighbor_lists, sample_ids, N):
    """
    Validate that neighbor computation is correct:
    1. No cross-slide leakage
    2. No self-inclusion
    3. Reasonable neighbor counts
    """
    print("Validating neighbor computation...")

    # Check a few random patches
    test_patches = np.random.choice(N, min(10, N), replace=False)

    for patch_idx in test_patches:
        patch_slide = sample_ids[patch_idx]

        for b, neighbors in enumerate(neighbor_lists[patch_idx]):
            if len(neighbors) > 0:
                # Check 1: No cross-slide leakage
                neighbor_slides = sample_ids[neighbors]
                unique_slides = np.unique(neighbor_slides)
                if len(unique_slides) > 1:
                    print(
                        f"ERROR: Cross-slide leakage detected! Patch {patch_idx} (slide {patch_slide}) has neighbors from slides {unique_slides}")
                    return False
                elif unique_slides[0] != patch_slide:
                    print(
                        f"ERROR: Neighbors from wrong slide! Patch {patch_idx} (slide {patch_slide}) has neighbors from slide {unique_slides[0]}")
                    return False

                # Check 2: No self-inclusion
                if patch_idx in neighbors:
                    print(f"ERROR: Self-inclusion detected! Patch {patch_idx} is included in its own neighbors")
                    return False

    # Check 3: Reasonable neighbor counts
    total_neighbors = sum(len(neighbors) for patch_neighbors in neighbor_lists for neighbors in patch_neighbors)
    avg_neighbors = total_neighbors / (N * len(neighbor_lists[0])) if N > 0 and len(neighbor_lists[0]) > 0 else 0

    print("Validation results:")
    print("  - No cross-slide leakage detected")
    print("  - No self-inclusion detected")
    print(f"  - Average neighbors per patch per bin: {avg_neighbors:.1f}")

    # Print detailed info for first few patches
    print("  - Sample validation (first 3 patches):")
    for i in range(min(3, len(neighbor_lists))):
        patch_slide = sample_ids[i]
        for b, neighbors in enumerate(neighbor_lists[i]):
            if len(neighbors) > 0:
                neighbor_slides = sample_ids[neighbors]
                unique_slides = np.unique(neighbor_slides)
                print(f"    Patch {i} (slide {patch_slide}), bin {b}: {len(neighbors)} neighbors from slides {unique_slides}")

    return True


def create_spatial_probe_data(
        embeddings,
        targets,
        neighbor_lists,
        bin_idx,
        strategy="mode",
        task_type="classification"):
    """
    Create training data for a spatial probe at a specific distance bin.

    Args:
        embeddings: Patch embeddings (N, embedding_dim)
        targets: Patch targets (N,) or (N, num_targets)
        neighbor_lists: Output from find_neighbors_in_bins
        bin_idx: Which distance bin to use
        strategy: How to aggregate neighbor targets
            - For classification: only "mode" (majority vote) is supported
            - For regression: "mean", "median", "sum" are supported
        task_type: "classification" or "regression" to determine aggregation strategy

    Returns:
        X: Center patch embeddings (M, embedding_dim) where M is number of patches with neighbors
        y: Aggregated neighbor targets (M,) or (M, num_targets)
        valid_indices: Indices of patches that have neighbors in this bin
    """
    valid_patches = []
    center_embeddings = []
    neighbor_targets = []

    for i, neighbors in enumerate(neighbor_lists):
        if len(neighbors[bin_idx]) > 0:  # This patch has neighbors in this bin
            valid_patches.append(i)
            center_embeddings.append(embeddings[i])

            # Aggregate neighbor targets
            neighbor_target_vals = targets[neighbors[bin_idx]]

            if task_type == "classification":
                # For classification, mode (majority vote) is the only appropriate strategy
                if strategy == "mode":
                    # Find the most common class among neighbors
                    if len(neighbor_target_vals.shape) == 1:
                        # Single class labels
                        aggregated, _ = mode(neighbor_target_vals, keepdims=False)
                        aggregated = aggregated.item() if hasattr(aggregated, 'item') else aggregated
                    else:
                        raise ValueError("Multi-label classification not supported in spatial neighbor evaluation")
                else:
                    raise ValueError(
                        f"For classification tasks, only 'mode' strategy is supported. Got: {strategy}. Use 'mode' for majority vote aggregation of discrete labels.")
            else:
                # For regression, use standard aggregation
                if strategy == "mean":
                    aggregated = np.nanmean(neighbor_target_vals, axis=0)
                elif strategy == "median":
                    aggregated = np.nanmedian(neighbor_target_vals, axis=0)
                elif strategy == "sum":
                    aggregated = np.nansum(neighbor_target_vals, axis=0)
                else:
                    raise ValueError(f"Unknown strategy for regression: {strategy}")

            neighbor_targets.append(aggregated)

    if len(valid_patches) == 0:
        return None, None, None

    return np.array(center_embeddings), np.array(neighbor_targets), np.array(valid_patches)


def evaluate_spatial_neighbor_bin(X_train, X_test, y_train, y_test, task_type, cfg):
    """
    Specialized evaluation function for spatial neighbor distance bins.
    Handles classification and regression tasks separately.
    """
    print(
        f"    Evaluating {task_type} task with shapes: X_train={X_train.shape}, X_test={X_test.shape}"
    )

    # Convert to numpy arrays
    X_train = X_train.cpu().numpy().astype(np.float64)
    X_test = X_test.cpu().numpy().astype(np.float64)

    # Check for NaN or infinite values
    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        raise ValueError(f"{task_type} input data contains NaN values!")
    if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)):
        raise ValueError(f"{task_type} input data contains infinite values!")

    print("    Data validation passed - no NaN or infinite values found")

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if task_type == "classification":
        # Handle classification
        y_train = y_train.cpu().numpy()
        y_test = y_test.cpu().numpy()

        # Ensure labels are integers (not one-hot encoded)
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            raise ValueError("Multi-label classification not supported")
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            raise ValueError("Multi-label classification not supported")

        # Flatten if needed
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # Create label mapping for consecutive integers
        train_labels = np.unique(y_train)
        test_labels = np.unique(y_test)
        common_labels = np.intersect1d(train_labels, test_labels)

        if len(common_labels) < 2:
            raise ValueError(f"Need at least 2 common classes, got {len(common_labels)}")

        label_to_idx = {label: idx for idx, label in enumerate(common_labels)}

        # Filter data to only include common classes
        train_mask = np.isin(y_train, common_labels)
        test_mask = np.isin(y_test, common_labels)

        X_train_filtered = X_train_scaled[train_mask]
        y_train_filtered = y_train[train_mask]
        X_test_filtered = X_test_scaled[test_mask]
        y_test_filtered = y_test[test_mask]

        # Map labels to consecutive integers
        y_train_mapped = np.array([label_to_idx[label] for label in y_train_filtered])
        y_test_mapped = np.array([label_to_idx[label] for label in y_test_filtered])

        # Train and evaluate classification
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_train_filtered, y_train_mapped)

        # Get predictions
        test_preds = clf.predict(X_test_filtered)

        # Convert predictions back to original labels for metric calculation
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        test_preds_original = np.array([idx_to_label[pred] for pred in test_preds])

        # Calculate F1 macro
        f1_score_val = f1_score(y_test_filtered, test_preds_original, average="macro")
        print(f"    Classification F1 Macro: {f1_score_val:.4f}")

        return {"f1_macro": f1_score_val}

    elif task_type == "regression":
        # Handle regression
        y_train = y_train.cpu().numpy()
        y_test = y_test.cpu().numpy()

        # Use Ridge regression
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=np.logspace(-3, 3, 10), scoring="r2")

        model.fit(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)

        print(f"    Regression R²: {test_r2:.4f}")
        return {"r2": test_r2}

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def run_spatial_neighbor(cfg: DictConfig) -> Dict[str, Dict[str, float]]:
    """Run spatial neighbor distance bins evaluation.

    This is a zero-shot spatial coherence analysis:
    1. Load all available data (we can evaluate on any split since embeddings are frozen)
    2. For each patch, predict neighbor properties from center patch embedding
    3. Use cross-validation within the data to get robust estimates
    4. Report performance metrics showing how well embeddings capture spatial context

    This is not a train/test prediction task - it's measuring spatial coherence
    of the learned embeddings in a zero-shot manner.
    """

    # Determine alignment method from modality (always infer, ignore config default)
    modality = cfg.evaluation.modality

    if "multimodal" in modality:
        align_method = "multimodal_" + cfg.models.method
    elif modality == "unimodal_img":
        align_method = "unimodal_img"
    elif modality == "unimodal_gex":
        align_method = "unimodal_gex"
    elif modality == "random":
        align_method = "random"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    print(f"Running spatial neighbor evaluation with method: {align_method}")

    # Use the SAME data split strategy as eval.py for fair comparison
    # This ensures we're not artificially inflating performance by using more data
    print("\n=== Using Same Data Split Strategy as eval.py ===")

    # Check if we should use cross-validation (same logic as eval.py)
    use_cv = getattr(cfg.evaluation, 'cross_validation', False)

    # For unimodal_img, we need spatial coordinates, so temporarily override modality
    # to use the full multimodal dataset (which has cell_coords) but still extract only image embeddings
    original_modality = cfg.evaluation.modality
    if original_modality == "unimodal_img":
        print("NOTE: For spatial evaluation, unimodal_img will use full multimodal dataset")
        print("   (which has spatial coordinates) but extract only image embeddings.")
        # Temporarily set to multimodal to load full dataset, but align_method stays unimodal_img
        cfg.evaluation.modality = "multimodal_concat"

    if use_cv:
        print("Using cross-validation on train split only (same as eval.py)")
        print("Loading train split...")
        ds_clf, ds_reg = load_and_preprocess_data(cfg, "train", align_method)
        print(f"Classification data: {len(ds_clf)} samples")
        print(f"Regression data: {len(ds_reg)} samples")
    else:
        print("Using train-test split evaluation (same as eval.py)")
        print("Loading train split...")
        train_ds_clf, train_ds_reg = load_and_preprocess_data(cfg, "train", align_method)
        print("Loading test split...")
        test_ds_clf, test_ds_reg = load_and_preprocess_data(cfg, "test", align_method)

        # Use train for training, test for evaluation (same as eval.py)
        ds_clf = train_ds_clf
        ds_reg = train_ds_reg
        print(f"Using train data for spatial analysis: clf={len(ds_clf)}, reg={len(ds_reg)}")
        print(f"Test data available for validation: clf={len(test_ds_clf)}, reg={len(test_ds_reg)}")

    # Restore original modality after loading data
    if original_modality == "unimodal_img":
        cfg.evaluation.modality = original_modality

    # Check if cell_coords field exists
    for ds, name in [(ds_clf, "cl"), (ds_reg, "reg")]:
        if len(ds) > 0 and "cell_coords" not in ds[0]:
            raise ValueError(
                f"Dataset {name} missing 'cell_coords' field required for spatial neighbor evaluation. "
                "Please run merge_annotations.py first to add this field."
            )

    # Get configuration parameters
    num_bins = cfg.training.spatial_neighbor.num_bins
    distance_metric = cfg.training.spatial_neighbor.distance_metric
    summary_strategy = cfg.training.spatial_neighbor.get("summary_strategy", "mean")
    use_knn_bins = cfg.training.spatial_neighbor.get("use_knn_bins", False)
    knn_bins = cfg.training.spatial_neighbor.get("knn_bins", [0, 10, 35, 85, 185, 385])

    # Store results for each bin
    bin_r2s = []
    bin_f1s = []

    # Use the same evaluation strategy as eval.py
    if use_cv:
        # Cross-validation on train data only
        n_cv_folds = getattr(cfg.evaluation, "n_cv_folds", 5)
        print(f"Using {n_cv_folds}-fold cross-validation on train data")
    else:
        # Train-test split evaluation (same as eval.py)
        n_cv_folds = 1  # Single train-test split
        print("Using train-test split evaluation (same as eval.py)")

    if use_knn_bins:
        print(f"\nEvaluating {len(knn_bins)-1} k-NN bins using {n_cv_folds}-fold CV")
        print(f"K-NN bin edges: {knn_bins}")
        num_bins = len(knn_bins) - 1
    else:
        print(f"\nEvaluating {num_bins} spatial distance bins using {n_cv_folds}-fold CV")

    # CLASSIFICATION TASK: Zero-shot spatial coherence evaluation
    if cfg.evaluation.tasks.classify:
        print("\n=== CLASSIFICATION TASK ===")
        print(f"Classification dataset: {len(ds_clf)} samples")

        # Extract cell centers from classification dataset
        centers_clf = np.array([patch_center(s["cell_coords"]) for s in ds_clf])
        valid_mask_clf = ~np.isnan(centers_clf).any(axis=1)
        centers_clf = centers_clf[valid_mask_clf]

        if len(centers_clf) < 10:
            raise ValueError("Too few valid samples in classification dataset.")

        # Extract sample names (for same-slide filtering)
        sample_names_clf = np.array([ds_clf[i]["name"] for i in range(len(ds_clf))])
        sample_names_clf = sample_names_clf[valid_mask_clf]

        print(f"Using {len(centers_clf)} samples with valid coordinates")
        print(f"Number of unique slides: {len(np.unique(sample_names_clf))}")

        # Extract embeddings for classification task
        valid_indices_clf = [i for i, valid in enumerate(valid_mask_clf) if valid]

        print("Extracting embeddings for classification task...")
        emb_clf, tgt_clf = extract_embeddings(
            ds=ds_clf, valid_indices=valid_indices_clf,
            cfg=cfg, align_method=align_method, task_type="classification"
        )
        emb_clf = emb_clf.cpu().numpy()
        tgt_clf = tgt_clf.cpu().numpy()

        print(f"Classification shapes: emb={emb_clf.shape}, tgt={tgt_clf.shape}")

        if emb_clf.shape[0] != centers_clf.shape[0]:
            raise ValueError("Mismatch between embeddings and coordinates.")

        # Create bins (distance or k-NN based)
        if use_knn_bins:
            # K-NN approach: bins defined by neighbor rank
            neighbor_lists_clf = find_neighbors_in_bins(
                centers_clf, bin_edges=None, sample_ids=sample_names_clf,
                metric=distance_metric, use_knn=True, knn_bins=knn_bins
            )

            print(f"K-NN bin edges: {knn_bins}")
            print("Bin statistics:")
            for b in range(num_bins):
                patches_with_neighbors = sum(1 for neighbors in neighbor_lists_clf if len(neighbors[b]) > 0)
                total_neighbors = sum(len(neighbors[b]) for neighbors in neighbor_lists_clf)
                avg_neighbors = total_neighbors / patches_with_neighbors if patches_with_neighbors > 0 else 0
                k_start = knn_bins[b]
                k_end = knn_bins[b + 1]
                print(
                    f"  Bin {b} [neighbors {k_start+1}-{k_end}]: {patches_with_neighbors} patches with neighbors, avg {avg_neighbors:.1f} neighbors/patch, {total_neighbors} total")
        else:
            # Distance approach: bins defined by distance ranges
            # Load optimized percentiles from config for this dataset
            dataset_name = cfg.data.dataset
            bin_edges_clf = get_distance_bins(
                centers_clf,
                num_bins,
                metric=distance_metric,
                dataset_name=dataset_name,
                sample_ids=sample_names_clf)
            # Update num_bins based on actual bin_edges (in case config had different number)
            actual_num_bins_clf = len(bin_edges_clf) - 1
            if actual_num_bins_clf != num_bins:
                print(f"Info: Using {actual_num_bins_clf} bins from optimized config (overriding num_bins={num_bins})")
                num_bins = actual_num_bins_clf
            neighbor_lists_clf = find_neighbors_in_bins(
                centers_clf, bin_edges_clf, sample_ids=sample_names_clf, metric=distance_metric
            )

            print(f"Distance bin edges: {bin_edges_clf}")
            print("Bin statistics:")
            for b in range(num_bins):
                patches_with_neighbors = sum(1 for neighbors in neighbor_lists_clf if len(neighbors[b]) > 0)
                total_neighbors = sum(len(neighbors[b]) for neighbors in neighbor_lists_clf)
                avg_neighbors = total_neighbors / patches_with_neighbors if patches_with_neighbors > 0 else 0
                min_dist = bin_edges_clf[b]
                max_dist = bin_edges_clf[b + 1]
                print(
                    f"  Bin {b} [{min_dist:.1f}, {max_dist:.1f}]: {patches_with_neighbors} patches with neighbors, avg {avg_neighbors:.1f} neighbors/patch, {total_neighbors} total")

        # Evaluate each bin
        for b in range(num_bins):
            if use_knn_bins:
                k_start = knn_bins[b]
                k_end = knn_bins[b + 1]
                print(f"\n  --- Classification Bin {b} (k-NN {k_start+1}-{k_end}) ---")
            else:
                bin_midpoint = (bin_edges_clf[b] + bin_edges_clf[b + 1]) / 2
                print(f"\n  --- Classification Bin {b} (distance: {bin_midpoint:.1f}) ---")

            # Create spatial probe data for this bin
            X_bin, y_bin, valid_idx = create_spatial_probe_data(
                emb_clf, tgt_clf, neighbor_lists_clf, b,
                strategy="mode", task_type="classification"
            )

            # Skip if insufficient data
            if X_bin is None or len(X_bin) < 20:
                print(f"    Skipping: insufficient data ({len(X_bin) if X_bin is not None else 0} samples)")
                bin_f1s.append(None)
                continue

            try:
                if use_cv:
                    # Cross-validation approach (same as eval.py)
                    from sklearn.model_selection import train_test_split

                    f1_scores = []

                    for split_idx in range(n_cv_folds):
                        # Split data with different random seeds
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_bin, y_bin, test_size=0.2,
                                random_state=cfg.evaluation.seed + split_idx,
                                stratify=y_bin
                            )
                        except ValueError:
                            # Fallback: no stratification if some classes have too few samples
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_bin, y_bin, test_size=0.2,
                                random_state=cfg.evaluation.seed + split_idx
                            )

                        # Convert to tensors
                        X_train_tensor = torch.from_numpy(X_train).float()
                        X_test_tensor = torch.from_numpy(X_test).float()
                        y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
                        y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

                        # Evaluate
                        metrics = evaluate_spatial_neighbor_bin(
                            X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor,
                            "classification", cfg
                        )

                        f1_scores.append(metrics.get("f1_macro", 0.0))

                    # Calculate mean and std
                    f1_mean = np.mean(f1_scores)
                    f1_std = np.std(f1_scores)

                    print(f"    Samples: {len(X_bin)} ({n_cv_folds}-fold CV)")
                    print(f"    F1 Macro: {f1_mean:.4f} ± {f1_std:.4f}")
                    bin_f1s.append((f1_mean, f1_std))
                else:
                    # Single train-test split approach (same as eval.py)
                    # For spatial neighbor evaluation, we use the same data for both train and test
                    # since we're measuring spatial coherence within the same dataset
                    from sklearn.model_selection import train_test_split

                    # Split the spatial probe data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_bin, y_bin, test_size=0.2,
                        random_state=cfg.evaluation.seed,
                        stratify=y_bin
                    )

                    # Convert to tensors
                    X_train_tensor = torch.from_numpy(X_train).float()
                    X_test_tensor = torch.from_numpy(X_test).float()
                    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
                    y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

                    # Evaluate
                    metrics = evaluate_spatial_neighbor_bin(
                        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor,
                        "classification", cfg
                    )

                    f1_score = metrics.get("f1_macro", 0.0)

                    print(f"    Samples: {len(X_bin)} (train-test split)")
                    print(f"    F1 Macro: {f1_score:.4f}")
                    bin_f1s.append((f1_score, 0.0))  # No std for single split

            except Exception as e:
                print(f"    Error evaluating bin {b}: {e}")
                import traceback
                traceback.print_exc()
                bin_f1s.append(None)

    # REGRESSION TASK: Zero-shot spatial coherence evaluation
    if cfg.evaluation.tasks.regress:
        print("\n=== REGRESSION TASK ===")
        print(f"Regression dataset: {len(ds_reg)} samples")

        # Extract cell centers from regression dataset
        centers_reg = np.array([patch_center(s["cell_coords"]) for s in ds_reg])
        valid_mask_reg = ~np.isnan(centers_reg).any(axis=1)
        centers_reg = centers_reg[valid_mask_reg]

        if len(centers_reg) < 10:
            raise ValueError("Too few valid samples in regression dataset.")

        # Extract sample names (for same-slide filtering)
        sample_names_reg = np.array([ds_reg[i]["name"] for i in range(len(ds_reg))])
        sample_names_reg = sample_names_reg[valid_mask_reg]

        print(f"Using {len(centers_reg)} samples with valid coordinates")
        print(f"Number of unique slides: {len(np.unique(sample_names_reg))}")

        # Extract embeddings for regression task
        valid_indices_reg = [i for i, valid in enumerate(valid_mask_reg) if valid]

        print("Extracting embeddings for regression task...")
        emb_reg, tgt_reg = extract_embeddings(
            ds=ds_reg, valid_indices=valid_indices_reg,
            cfg=cfg, align_method=align_method, task_type="regression"
        )
        emb_reg = emb_reg.cpu().numpy()
        tgt_reg = tgt_reg.cpu().numpy()

        print(f"Regression shapes: emb={emb_reg.shape}, tgt={tgt_reg.shape}")

        if emb_reg.shape[0] != centers_reg.shape[0]:
            raise ValueError("Mismatch between embeddings and coordinates.")

        # Create bins (distance or k-NN based)
        if use_knn_bins:
            # K-NN approach
            neighbor_lists_reg = find_neighbors_in_bins(
                centers_reg, bin_edges=None, sample_ids=sample_names_reg,
                metric=distance_metric, use_knn=True, knn_bins=knn_bins
            )

            print(f"K-NN bin edges: {knn_bins}")
            print("Bin statistics:")
            for b in range(num_bins):
                patches_with_neighbors = sum(1 for neighbors in neighbor_lists_reg if len(neighbors[b]) > 0)
                total_neighbors = sum(len(neighbors[b]) for neighbors in neighbor_lists_reg)
                avg_neighbors = total_neighbors / patches_with_neighbors if patches_with_neighbors > 0 else 0
                k_start = knn_bins[b]
                k_end = knn_bins[b + 1]
                print(
                    f"  Bin {b} [neighbors {k_start+1}-{k_end}]: {patches_with_neighbors} patches with neighbors, avg {avg_neighbors:.1f} neighbors/patch, {total_neighbors} total")
        else:
            # Distance approach
            # Load optimized percentiles from config for this dataset
            dataset_name = cfg.data.dataset
            bin_edges_reg = get_distance_bins(
                centers_reg,
                num_bins,
                metric=distance_metric,
                dataset_name=dataset_name,
                sample_ids=sample_names_reg)
            # Update num_bins based on actual bin_edges (in case config had different number)
            actual_num_bins_reg = len(bin_edges_reg) - 1
            if actual_num_bins_reg != num_bins:
                print(f"Info: Using {actual_num_bins_reg} bins from optimized config (overriding num_bins={num_bins})")
                num_bins = actual_num_bins_reg
            neighbor_lists_reg = find_neighbors_in_bins(
                centers_reg, bin_edges_reg, sample_ids=sample_names_reg, metric=distance_metric
            )

            print(f"Distance bin edges: {bin_edges_reg}")
            print("Bin statistics:")
            for b in range(num_bins):
                patches_with_neighbors = sum(1 for neighbors in neighbor_lists_reg if len(neighbors[b]) > 0)
                total_neighbors = sum(len(neighbors[b]) for neighbors in neighbor_lists_reg)
                avg_neighbors = total_neighbors / patches_with_neighbors if patches_with_neighbors > 0 else 0
                min_dist = bin_edges_reg[b]
                max_dist = bin_edges_reg[b + 1]
                print(
                    f"  Bin {b} [{min_dist:.1f}, {max_dist:.1f}]: {patches_with_neighbors} patches with neighbors, avg {avg_neighbors:.1f} neighbors/patch, {total_neighbors} total")

        # Evaluate each bin
        for b in range(num_bins):
            if use_knn_bins:
                k_start = knn_bins[b]
                k_end = knn_bins[b + 1]
                print(f"\n  --- Regression Bin {b} (k-NN {k_start+1}-{k_end}) ---")
            else:
                bin_midpoint = (bin_edges_reg[b] + bin_edges_reg[b + 1]) / 2
                print(f"\n  --- Regression Bin {b} (distance: {bin_midpoint:.1f}) ---")

            # Create spatial probe data for this bin
            X_bin, y_bin, valid_idx = create_spatial_probe_data(
                emb_reg, tgt_reg, neighbor_lists_reg, b,
                strategy=summary_strategy, task_type="regression"
            )

            # Skip if insufficient data
            if X_bin is None or len(X_bin) < 20:
                print(f"    Skipping: insufficient data ({len(X_bin) if X_bin is not None else 0} samples)")
                bin_r2s.append(None)
                continue

            try:
                if use_cv:
                    # Cross-validation approach (same as eval.py)
                    from sklearn.model_selection import train_test_split

                    r2_scores = []

                    for split_idx in range(n_cv_folds):
                        # Split data with different random seeds
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_bin, y_bin, test_size=0.2,
                            random_state=cfg.evaluation.seed + split_idx
                        )

                        # Convert to tensors
                        X_train_tensor = torch.from_numpy(X_train).float()
                        X_test_tensor = torch.from_numpy(X_test).float()
                        y_train_tensor = torch.from_numpy(y_train).float()
                        y_test_tensor = torch.from_numpy(y_test).float()

                        # Evaluate
                        metrics = evaluate_spatial_neighbor_bin(
                            X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor,
                            "regression", cfg
                        )

                        r2_scores.append(metrics.get("r2", 0.0))

                    # Calculate mean and std
                    r2_mean = np.mean(r2_scores)
                    r2_std = np.std(r2_scores)

                    print(f"    Samples: {len(X_bin)} ({n_cv_folds}-fold CV)")
                    print(f"    R²: {r2_mean:.4f} ± {r2_std:.4f}")
                    bin_r2s.append((r2_mean, r2_std))
                else:
                    # Single train-test split approach (same as eval.py)
                    from sklearn.model_selection import train_test_split

                    # Split the spatial probe data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_bin, y_bin, test_size=0.2,
                        random_state=cfg.evaluation.seed
                    )

                    # Convert to tensors
                    X_train_tensor = torch.from_numpy(X_train).float()
                    X_test_tensor = torch.from_numpy(X_test).float()
                    y_train_tensor = torch.from_numpy(y_train).float()
                    y_test_tensor = torch.from_numpy(y_test).float()

                    # Evaluate
                    metrics = evaluate_spatial_neighbor_bin(
                        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor,
                        "regression", cfg
                    )

                    r2_score = metrics.get("r2", 0.0)

                    print(f"    Samples: {len(X_bin)} (train-test split)")
                    print(f"    R²: {r2_score:.4f}")
                    bin_r2s.append((r2_score, 0.0))  # No std for single split

            except Exception as e:
                print(f"    Error evaluating bin {b}: {e}")
                import traceback
                traceback.print_exc()
                bin_r2s.append(None)

    # Compile results
    results = {}

    # Classification results
    if cfg.evaluation.tasks.classify:
        for b in range(num_bins):
            if b < len(bin_f1s) and bin_f1s[b] is not None:
                # Calculate actual number of patches with neighbors in this bin
                patches_with_neighbors = sum(1 for neighbors in neighbor_lists_clf if len(neighbors[b]) > 0)
                if isinstance(bin_f1s[b], tuple):
                    # Handle (mean, std) tuple from cross-validation
                    mean_val, std_val = bin_f1s[b]
                    results[f"bin_{b}_f1_macro"] = {
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "n_samples": int(patches_with_neighbors),
                    }
                else:
                    # Handle single value (fallback)
                    results[f"bin_{b}_f1_macro"] = {
                        "mean": float(bin_f1s[b]),
                        "std": 0.0,
                        "n_samples": int(patches_with_neighbors),
                    }
            else:
                results[f"bin_{b}_f1_macro"] = {"mean": 0.0, "std": 0.0, "n_samples": 0}

    # Regression results
    if cfg.evaluation.tasks.regress:
        for b in range(num_bins):
            if b < len(bin_r2s) and bin_r2s[b] is not None:
                # Calculate actual number of patches with neighbors in this bin
                patches_with_neighbors = sum(1 for neighbors in neighbor_lists_reg if len(neighbors[b]) > 0)
                if isinstance(bin_r2s[b], tuple):
                    # Handle (mean, std) tuple from cross-validation
                    mean_val, std_val = bin_r2s[b]
                    results[f"bin_{b}_r2"] = {
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "n_samples": int(patches_with_neighbors),
                    }
                else:
                    # Handle single value (fallback)
                    results[f"bin_{b}_r2"] = {
                        "mean": float(bin_r2s[b]),
                        "std": 0.0,
                        "n_samples": int(patches_with_neighbors),
                    }
            else:
                results[f"bin_{b}_r2"] = {"mean": 0.0, "std": 0.0, "n_samples": 0}

    # Add bin distance information (use classification if available, otherwise regression)
    if use_knn_bins:
        # For k-NN bins, we don't have distance midpoints, so use bin indices
        bin_midpoints = list(range(num_bins))
        if cfg.evaluation.tasks.classify:
            total_samples = len(centers_clf)
        else:
            total_samples = len(centers_reg)
    else:
        # For distance bins, calculate midpoints
        if cfg.evaluation.tasks.classify:
            bin_midpoints = [(bin_edges_clf[b] + bin_edges_clf[b + 1]) / 2 for b in range(num_bins)]
            total_samples = len(centers_clf)
        else:
            bin_midpoints = [(bin_edges_reg[b] + bin_edges_reg[b + 1]) / 2 for b in range(num_bins)]
            total_samples = len(centers_reg)

    if use_knn_bins:
        results["bin_midpoints"] = {
            f"bin_{b}_knn_rank": int(b) for b in range(num_bins)
        }
    else:
        results["bin_midpoints"] = {
            f"bin_{b}_distance": float(mp) for b, mp in enumerate(bin_midpoints)
        }

    # Calculate unique donors
    unique_donors = set()
    if cfg.evaluation.tasks.classify:
        for i in valid_indices_clf:
            sample = ds_clf[i]
            sample_name = sample["name"]
            if "_" in sample_name:
                donor_name = sample_name.split("_")[0]
            else:
                donor_name = sample_name
            unique_donors.add(donor_name)
    elif cfg.evaluation.tasks.regress:
        for i in valid_indices_reg:
            sample = ds_reg[i]
            sample_name = sample["name"]
            if "_" in sample_name:
                donor_name = sample_name.split("_")[0]
            else:
                donor_name = sample_name
            unique_donors.add(donor_name)

    results["evaluation_info"] = {
        "method": align_method,
        "eval_split": "train" if use_cv else "train_test_split",
        "num_bins": num_bins,
        "n_cv_folds": n_cv_folds,
        "distance_metric": distance_metric,
        "summary_strategy": summary_strategy,
        "total_samples": total_samples,
        "unique_donors": len(unique_donors),
        "evaluation_strategy": f"zero_shot_spatial_coherence_{'cv' if use_cv else 'train_test'}",
        "use_knn_bins": use_knn_bins,
    }

    return results


def print_spatial_neighbor_results(
    results: Dict, align_method: str, num_bins: int, bin_midpoints: List[float], use_knn_bins: bool = False
):
    """Print spatial neighbor results in a nice table format."""
    print("\n" + "=" * 80)
    if use_knn_bins:
        print("SPATIAL NEIGHBOR K-NN BINS EVALUATION RESULTS")
    else:
        print("SPATIAL NEIGHBOR DISTANCE BINS EVALUATION RESULTS")
    print(f"Method: {align_method}")
    print("=" * 80)

    # Print bin information
    if use_knn_bins:
        print("K-NN Bins:")
        for b in range(num_bins):
            print(f"  Bin {b}: k-NN rank {b}")
    else:
        print("Distance Bins:")
        for b in range(num_bins):
            distance = bin_midpoints[b]
            print(f"  Bin {b}: {distance:.3f} distance units")

    print("\n" + "-" * 80)

    # Print classification results
    if any(f"bin_{b}_f1_macro" in results for b in range(num_bins)):
        print("Classification Results (F1 Macro):")
        headers = ["Bin", "Distance", "F1 Macro (Mean ± Std)", "N Samples"]
        table_data = []

        for b in range(num_bins):
            bin_key = f"bin_{b}_f1_macro"
            if bin_key in results:
                f1_mean = results[bin_key]["mean"]
                f1_std = results[bin_key]["std"]
                n_samples = results[bin_key]["n_samples"]
                distance = bin_midpoints[b]

                table_data.append(
                    [f"Bin {b}", f"{distance:.3f}", f"{f1_mean:.4f} ± {f1_std:.4f}", n_samples]
                )

        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print regression results
    if any(f"bin_{b}_r2" in results for b in range(num_bins)):
        print("\nRegression Results (R²):")
        headers = ["Bin", "Distance", "R² (Mean ± Std)", "N Samples"]
        table_data = []

        for b in range(num_bins):
            bin_key = f"bin_{b}_r2"
            if bin_key in results:
                r2_mean = results[bin_key]["mean"]
                r2_std = results[bin_key]["std"]
                n_samples = results[bin_key]["n_samples"]
                distance = bin_midpoints[b]

                table_data.append(
                    [f"Bin {b}", f"{distance:.3f}", f"{r2_mean:.4f} ± {r2_std:.4f}", n_samples]
                )

        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print evaluation info
    print("\n" + "-" * 80)
    eval_info = results.get("evaluation_info", {})
    print(f"Total samples: {eval_info.get('total_samples', 'N/A')}")
    print(f"Unique donors: {eval_info.get('unique_donors', 'N/A')}")
    print(f"Distance metric: {eval_info.get('distance_metric', 'N/A')}")
    print(f"Summary strategy: {eval_info.get('summary_strategy', 'N/A')}")

    print("=" * 80)


def save_spatial_neighbor_results(
    results_dir: Path, cfg: DictConfig, results: Dict, align_method: str, use_cv: bool = None
):
    """Save spatial neighbor results to JSON."""

    # Determine use_cv if not provided
    if use_cv is None:
        use_cv = getattr(cfg.evaluation, 'cross_validation', False)

    # Get reproducibility info
    checkpoint_info = {}
    if hasattr(cfg.models, 'checkpoint_path') and cfg.models.checkpoint_path:
        checkpoint_info["checkpoint_path"] = str(cfg.models.checkpoint_path)
    if hasattr(cfg.models, 'method') and cfg.models.method:
        checkpoint_info["models_method"] = str(cfg.models.method)
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'dataset_name') and cfg.data.dataset_name:
        checkpoint_info["dataset_name"] = cfg.data.dataset_name
    if hasattr(cfg.data, 'gex_embed_key') and cfg.data.gex_embed_key:
        checkpoint_info["gex_embed_key"] = cfg.data.gex_embed_key
    if hasattr(cfg.data, 'img_embed_key') and cfg.data.img_embed_key:
        checkpoint_info["img_embed_key"] = cfg.data.img_embed_key

    reproducibility_info = {
        "random_seed": cfg.evaluation.seed if hasattr(
            cfg.evaluation,
            'seed') else None,
        "dataset_path": str(
            cfg.data.dataset_name) if hasattr(
                cfg.data,
                'dataset_name') and cfg.data.dataset_name else None,
        "cv_method": "sklearn.KFold" if use_cv else "train_test_split",
        **checkpoint_info}

    result_data = {
        "experiment": {
            "modality": cfg.evaluation.modality,
            "img_model": cfg.evaluation.img_model,
            "gex_model": cfg.evaluation.gex_model,
            "dataset": cfg.data.dataset,
            "evaluation_strategy": "spatial_neighbor_distance_bins",
            "align_method": align_method,
            "tasks": {
                "classification": cfg.evaluation.tasks.classify,
                "regression": cfg.evaluation.tasks.regress,
            },
            "reproducibility": reproducibility_info,
        },
        "results": results,
    }

    # Make results serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj

    result_data = make_serializable(result_data)

    # Save to file
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"spatial_neighbor_{align_method}_results.json"

    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to JSON: {json_path}")


@hydra.main(config_path="../../../configs", config_name="downstream.yaml")
def main(cfg: DictConfig):
    """Main function for Hydra compatibility."""
    print("Starting Spatial Neighbor Distance Bins Evaluation")
    print("Configuration:")
    print(f"  - Modality: {cfg.evaluation.modality}")
    print(f"  - Image model: {cfg.evaluation.img_model}")
    print(f"  - GEX model: {cfg.evaluation.gex_model}")
    print(
        f"  - Tasks: classify={cfg.evaluation.tasks.classify}, regress={cfg.evaluation.tasks.regress}"
    )

    # Run the evaluation
    results = run_spatial_neighbor(cfg)

    # Print results in a nice format
    # Determine method name for results (use modality for unimodal, align_method for multimodal)
    modality = cfg.evaluation.modality
    if "multimodal" in modality:
        align_method = getattr(cfg.evaluation, "align_method", cfg.models.method)
        align_method = "multimodal_" + align_method
    elif modality == "unimodal_img":
        align_method = "unimodal_img"
    elif modality == "unimodal_gex":
        align_method = "unimodal_gex"
    elif modality == "random":
        align_method = "random"
    else:
        align_method = modality

    num_bins = cfg.training.spatial_neighbor.num_bins
    use_knn_bins = cfg.training.spatial_neighbor.get("use_knn_bins", False)
    bin_midpoints = results.get("bin_midpoints", {})

    if use_knn_bins:
        bin_distances = [bin_midpoints.get(f"bin_{b}_knn_rank", b) for b in range(num_bins)]
    else:
        bin_distances = [bin_midpoints.get(f"bin_{b}_distance", 0.0) for b in range(num_bins)]

    print_spatial_neighbor_results(results, align_method, num_bins, bin_distances, use_knn_bins)

    # Save results
    results_dir = (
        PROJECT_DIR / "results" / cfg.data.dataset / f"spatial_neighbor_{cfg.evaluation.img_model}"
    )
    use_cv = getattr(cfg.evaluation, 'cross_validation', False)
    save_spatial_neighbor_results(results_dir, cfg, results, align_method, use_cv)

    print("\n=== Spatial Neighbor Distance Bins Evaluation Complete ===")
    return results


if __name__ == "__main__":
    main()
