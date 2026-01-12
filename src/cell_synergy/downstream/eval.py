import os
import json
import torch
from typing import List, Dict, Tuple
import hydra
from omegaconf import ListConfig
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import logging
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from cell_synergy.paths import PROJECT_DIR
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import importlib
from tabulate import tabulate
import glob
import gc
logger = logging.getLogger(__name__)

# Debug mode (controlled by environment variable)
_DEBUG_MODE = int(os.environ.get("DEBUG_EVAL", 0))

def _debug_print(*args, **kwargs):
    """Print debug messages only if DEBUG_EVAL environment variable is set."""
    if _DEBUG_MODE:
        print(*args, **kwargs)

# Global model cache to avoid reloading models during cross-validation
_global_model_cache = {}

# FusionTransformer.forward() patching is now done per-model-instance
# right after model loading (see extract_embeddings_from_fusion_model)
# This ensures the patch is applied correctly even if CoMM classes are imported elsewhere


def make_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return make_serializable(OmegaConf.to_container(obj, resolve=True))
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj


def generate_random_embeddings(embedding_shape, random_state):
    """Generate random embeddings with the same shape as the original embeddings."""
    np.random.seed(random_state)
    random_embeddings = np.random.randn(*embedding_shape)
    return random_embeddings


def split_dataset(dataset, train_ratio=0.8, random_state=42):
    """Split a dataset into train and test subsets.

    Args:
        dataset: HuggingFace Dataset to split
        train_ratio: Fraction of data to use for training (default 0.8)
        random_state: Random seed for reproducibility

    Returns:
        train_dataset, test_dataset
    """
    np.random.seed(random_state)
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_ratio)

    train_indices = sorted(indices[:split_idx].tolist())
    test_indices = sorted(indices[split_idx:].tolist())

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    return train_dataset, test_dataset


def create_cv_splits(n_samples, n_folds, base_seed, save_path=None):
    """
    Create proper K-fold cross-validation splits with save/load functionality.

    Uses sklearn's KFold to ensure:
    - Each sample appears in exactly one validation set
    - Folds are mutually exclusive
    - Consistent across runs with same seed
    - Splits are saved to disk for reproducibility

    Args:
        n_samples: Number of samples in the dataset
        n_folds: Number of cross-validation folds
        base_seed: Random seed for reproducibility (default: 42)
        save_path: Optional Path to directory where splits should be saved/loaded from

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    from sklearn.model_selection import KFold

    # Use fixed seed (42) for all CV splits to ensure consistency
    if base_seed is None:
        base_seed = 42

    # Try to load saved splits if save_path is provided
    if save_path is not None:
        save_path = Path(save_path)
        splits_file = save_path / f"cv_splits_n{n_samples}_folds{n_folds}_seed{base_seed}.json"

        if splits_file.exists():
            print(f"Loading saved CV splits from {splits_file}")
            try:
                with open(splits_file, encoding='utf-8') as f:
                    saved_splits = json.load(f)
                # Verify the splits match expected format
                if len(saved_splits) == n_folds:
                    # Convert back to tuples
                    splits = [(split["train_indices"], split["val_indices"]) for split in saved_splits]
                    print(f"Loaded {len(splits)} CV splits from disk")
                    return splits
                else:
                    print(f"Saved splits have {len(saved_splits)} folds, expected {n_folds}. Regenerating...")
            except Exception as e:
                print(f"Failed to load saved splits: {e}. Regenerating...")

    # Create KFold object with fixed random state
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)

    # Generate all splits
    splits = []
    indices = np.arange(n_samples)
    for train_indices, val_indices in kf.split(indices):
        splits.append((train_indices.tolist(), val_indices.tolist()))

    # Save splits to disk if save_path is provided
    if save_path is not None:
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            splits_file = save_path / f"cv_splits_n{n_samples}_folds{n_folds}_seed{base_seed}.json"
            # Convert to JSON-serializable format
            splits_dict = [
                {"train_indices": train_idx, "val_indices": val_idx}
                for train_idx, val_idx in splits
            ]
            with open(splits_file, 'w') as f:
                json.dump(splits_dict, f, indent=2)
            print(f"Saved CV splits to {splits_file}")
        except OSError as e:
            if e.errno == 122:  # Disk quota exceeded (ENOSPC)
                print(f"Warning: Could not save CV splits due to disk quota exceeded (errno 122).")
                print(f"   Splits will be regenerated on next run (seed={base_seed} ensures consistency).")
            else:
                raise

    return splits


def _extract_comm_dimensions_from_checkpoint(checkpoint_path: Path) -> Tuple[int, int, int]:
    """Extract img_embed_dim, gex_embed_dim, and embed_dim from CoMM checkpoint.

    Returns:
        (img_embed_dim, gex_embed_dim, embed_dim) where:
        - img_embed_dim: input dimension for image adapter (shape[1] of img_adapter.proj.weight)
        - gex_embed_dim: input dimension for GEX adapter (shape[1] of gex_adapter.proj.weight)
        - embed_dim: output dimension of projections (shape[0] of adapter.proj.weight)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Handle PyTorch Lightning format
        if any(key.startswith("model.") for key in state_dict.keys()):
            img_key = ('model.img_adapter.proj.weight' if 'model.img_adapter.proj.weight' in state_dict
                       else 'img_adapter.proj.weight')
            gex_key = ('model.gex_adapter.proj.weight' if 'model.gex_adapter.proj.weight' in state_dict
                       else 'gex_adapter.proj.weight')
        else:
            img_key = 'img_adapter.proj.weight'
            gex_key = 'gex_adapter.proj.weight'

        img_dim = state_dict[img_key].shape[1] if img_key in state_dict else None
        gex_dim = state_dict[gex_key].shape[1] if gex_key in state_dict else None
        # embed_dim is the output dimension (shape[0]) of the projection
        embed_dim = state_dict[img_key].shape[0] if img_key in state_dict else None
        return img_dim, gex_dim, embed_dim
    except Exception:
        return None, None, None


def extract_embeddings_from_fusion_model(
    cfg,
    hfd,
    valid_indices=None,
    img_embed_key=None,
    gex_embed_key=None,
    target_key=None,
    device=None,
    batch_size: int = 1024,
    task_type="classification",
    align_method=None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract embeddings from a multimodal fusion model using HuggingFace datasets.
    Highly optimized for large datasets via vectorized access and batched forward pass.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_embed_key = img_embed_key or cfg.data.img_embed_key
    gex_embed_key = gex_embed_key or cfg.data.gex_embed_key

    # Choose the correct label key based on task type
    if target_key is None:
        if task_type == "classification":
            target_key = cfg.training.classification.label_key
        elif task_type == "regression":
            target_key = cfg.training.regression.label_key
        else:
            target_key = "annotation"  # fallback

    # Single dict for models + checkpoints
    models_info = {
        "adversarial": {"path": "cell_synergy.models.adversarial.AdversarialBaseline", "ckpt": "adversarial.ckpt"},
        "barlow_twins": {"path": "cell_synergy.models.barlowtwins.BarlowTwinsBaseline", "ckpt": "barlow_twins.ckpt"},
        "byol": {"path": "cell_synergy.models.byol.BYOLBaseline", "ckpt": "byol.ckpt"},
        "cca": {"path": "cell_synergy.models.cca.CCABaseline", "ckpt": "cca.pt"},
        "comm": {"path": "cell_synergy.models.comm.CoMMBaseline", "ckpt": "comm.ckpt"},
        "dcca": {"path": "cell_synergy.models.dcca.DCCABaseline", "ckpt": "dcca.ckpt"},
        "dim": {"path": "cell_synergy.models.dim.DIMBaseline", "ckpt": "dim.ckpt"},
        "simclr": {"path": "cell_synergy.models.simclr.SimCLRBaseline", "ckpt": "simclr.ckpt"},
        "simsiam": {"path": "cell_synergy.models.simsiam.SimSiamBaseline", "ckpt": "simsiam.ckpt"},
        "vicreg": {"path": "cell_synergy.models.vicreg.VICRegBaseline", "ckpt": "vicreg.ckpt"},
    }

    method = cfg.models.method
    if method not in models_info:
        raise ValueError(f"Unknown method: {method}")

    info = models_info[method]
    module_path, class_name = info["path"].rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    # For CoMM, extract dimensions from checkpoint BEFORE creating model to prevent mismatches
    if method == "comm":
        # Try to find checkpoint path and extract dimensions early
        # For regular checkpoints, path is simple
        if not (
            hasattr(
                cfg.evaluation,
                "gex_model") and (
                ("finetune_" in cfg.evaluation.gex_model) or (
                "testset_aligned" in str(
                    cfg.evaluation.gex_model)))):
            # Check if checkpoint_config is specified to use a specific checkpoint file
            if hasattr(cfg.evaluation, 'checkpoint_config') and cfg.evaluation.checkpoint_config is not None:
                checkpoint_config = cfg.evaluation.checkpoint_config
                # Handle "100pct_cfg23" format - check in trained_models/full_ds/
                if "_cfg" in checkpoint_config and "100pct" in checkpoint_config:
                    parts = checkpoint_config.split("_cfg")
                    subset = parts[0]  # e.g., "100pct"
                    config = f"cfg{parts[1]}"  # e.g., "cfg23"

                    # Look for checkpoints in trained_models/{dataset}/full_ds/ with pattern comm_{config}_job*.ckpt
                    base_path = "/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/trained_models"
                    trained_models_dir = Path(f"{base_path}/{cfg.data.dataset}/full_ds")
                    if not trained_models_dir.exists():
                        trained_models_dir = Path(PROJECT_DIR) / "trained_models" / cfg.data.dataset / "full_ds"

                    if trained_models_dir.exists():
                        matching_checkpoints = list(trained_models_dir.glob(f"comm_{config}_job*.ckpt"))
                        if matching_checkpoints:
                            matching_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            model_ckpt_path = matching_checkpoints[0]
                        else:
                            fallback_path = trained_models_dir / f"comm_{config}.ckpt"
                            model_ckpt_path = fallback_path if fallback_path.exists() else None
                    else:
                        model_ckpt_path = None
                else:
                    # Use the specified checkpoint config (e.g., "cfg1_job31975900" -> "comm_cfg1_job31975900.ckpt")
                    checkpoint_name = f"comm_{cfg.evaluation.checkpoint_config}.ckpt"
                    model_ckpt_path = Path(PROJECT_DIR) / "trained_models" / \
                        cfg.data.dataset / "full_ds" / checkpoint_name
            else:
                # Default to comm.ckpt
                model_ckpt_path = Path(PROJECT_DIR) / "trained_models" / cfg.data.dataset / "full_ds" / info["ckpt"]

            # Extract dimensions from checkpoint if it exists
            if model_ckpt_path and model_ckpt_path.exists():
                # Try metadata first
                metadata_path = model_ckpt_path.parent / "comm_metadata.json"
                if metadata_path.exists():
                    import json
                    with open(metadata_path, encoding='utf-8') as mf:
                        metadata = json.load(mf)
                    if "img_embed_dim" in metadata:
                        cfg.models.img_embed_dim = metadata["img_embed_dim"]
                        print(f"   Updated img_embed_dim={cfg.models.img_embed_dim} from metadata")
                    if "gex_embed_dim" in metadata:
                        cfg.models.gex_embed_dim = metadata["gex_embed_dim"]
                        print(f"   Updated gex_embed_dim={cfg.models.gex_embed_dim} from metadata")
                    if "embed_dim" in metadata:
                        cfg.models.embed_dim = metadata["embed_dim"]
                        print(f"   Updated embed_dim={cfg.models.embed_dim} from metadata")
                else:
                    # Extract from checkpoint
                    print(f"   Extracting dimensions from checkpoint: {model_ckpt_path}")
                    img_dim, gex_dim, embed_dim = _extract_comm_dimensions_from_checkpoint(model_ckpt_path)
                    if img_dim:
                        cfg.models.img_embed_dim = img_dim
                        print(f"   Extracted img_embed_dim={img_dim} from checkpoint")
                    if gex_dim:
                        cfg.models.gex_embed_dim = gex_dim
                        print(f"   Extracted gex_embed_dim={gex_dim} from checkpoint")
                    if embed_dim:
                        cfg.models.embed_dim = embed_dim
                        print(f"   Extracted embed_dim={embed_dim} from checkpoint")

                # Force 1536 for img_uni_pool
                if cfg.data.img_embed_key == "img_uni_pool" and cfg.models.img_embed_dim == 1024:
                    cfg.models.img_embed_dim = 1536

    # Check if model is already cached - include dimensions in cache key to prevent wrong-dimension reuse
    cache_key = f"{method}_{cfg.data.dataset}"
    if method == "comm":
        cache_key += f"_img{cfg.models.img_embed_dim}_gex{cfg.models.gex_embed_dim}"

    model = None
    use_cached_model = False

    if cache_key in _global_model_cache:
        print(f"♻️  Using cached {method} model (cache_key={cache_key})")
        cached_model = _global_model_cache[cache_key]

        # Validate cached model dimensions
        if method == "comm" and hasattr(cached_model, 'img_adapter') and hasattr(cached_model.img_adapter, 'proj'):
            if cached_model.img_adapter.proj.weight.shape[1] != cfg.models.img_embed_dim:
                print(f"Cached model has wrong img_embed_dim, reloading...")
                del _global_model_cache[cache_key]
                use_cached_model = False
            elif cfg.data.img_embed_key == "img_uni_pool" and cached_model.img_adapter.proj.weight.shape[1] != 1536:
                print(f"Cached model has wrong img_embed_dim for img_uni_pool, reloading...")
                del _global_model_cache[cache_key]
                use_cached_model = False
            else:
                # Model is valid, use cached version
                model = cached_model
                use_cached_model = True
        else:
            # Non-CoMM model or no adapter to check, use cached version
            model = cached_model
            use_cached_model = True

        # Ensure cached model is in eval mode and on correct device
        if use_cached_model:
            model.eval()
            model = model.to(device)

    if not use_cached_model:
        print(f"🔄 Loading {method} model for the first time (cache_key={cache_key})")

        # Load pre-trained checkpoint for all methods
        if method == "cca":
            # CCA uses .pt format instead of .ckpt
            model_ckpt_path = Path(PROJECT_DIR) / "trained_models" / cfg.data.dataset / "full_ds" / "cca.pt"
            assert model_ckpt_path.exists(), f"CCA checkpoint does not exist: {model_ckpt_path}"

            print(f"Loading CCA checkpoint from: {model_ckpt_path}")
            model = ModelClass(cfg).to(device)

            # Load CCA-specific checkpoint format
            checkpoint = torch.load(model_ckpt_path, map_location=device, weights_only=False)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("CCA checkpoint loaded successfully")
            else:
                # Fallback: load direct state dict
                model.load_state_dict(checkpoint, strict=False)
                print("CCA checkpoint loaded successfully (direct format)")
            model.eval()
        else:
            # Handle scaling experiment CoMM checkpoints
            if method == "comm" and hasattr(
                cfg.evaluation, "gex_model") and (
                ("finetune_" in cfg.evaluation.gex_model) or (
                    "testset_aligned" in str(
                    cfg.evaluation.gex_model))):
                # Extract subset from gex_model (e.g., "finetune_1pct_cfg2" -> "1pct" or
                # "testset_aligned_1pct_cfg2" -> "testset_aligned_1pct_cfg2")
                gex_model = cfg.evaluation.gex_model

                # Handle testset_aligned checkpoints
                if "testset_aligned" in gex_model:
                    # Extract subset (e.g., "100pct") and config (e.g., "cfg1") from gex_model
                    # Example: "testset_aligned_100pct_cfg1"
                    # Checkpoints are saved at:
                    # comm_models_testset_aligned/testset_aligned_{subset}_{config}/{config}/comm.ckpt
                    config = None
                    for cfg_str in ["cfg5", "cfg4", "cfg3", "cfg2", "cfg1"]:
                        if f"_{cfg_str}" in gex_model:
                            config = cfg_str
                            break

                    if config is None:
                        raise ValueError(f"Could not determine config from testset_aligned gex_model: {gex_model}")

                    # For testset_aligned, use the full gex_model name as the subset directory name
                    # Training saves to: results_dir / subset / config where subset = "testset_aligned_100pct_cfg1"
                    # So path is: comm_models_testset_aligned/testset_aligned_100pct_cfg1/cfg1/comm.ckpt
                    subset_dir = gex_model  # e.g., "testset_aligned_100pct_cfg1"

                    # Extract actual subset (e.g., "100pct" from "testset_aligned_100pct_cfg1")
                    actual_subset = subset_dir.replace("testset_aligned_", "").split("_cfg")[0]

                    # Try multiple possible paths (checkpoints might be in different locations)
                    # Check for seed-specific checkpoint first
                    requested_seed = getattr(cfg.evaluation, 'checkpoint_seed', None)
                    ckpt_filename = f"comm_seed{requested_seed}.ckpt" if requested_seed is not None else "comm.ckpt"
                    possible_paths = [
                        # Path 1: testset_aligned directory (expected location)
                        Path(
                            f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/"
                            f"{cfg.data.dataset}/scaling_experiment/comm_models_testset_aligned/"
                            f"{subset_dir}/{config}/{ckpt_filename}"),
                        # Path 2: regular comm_models directory (actual location from training)
                        Path(
                            f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/{cfg.data.dataset}/scaling_experiment/comm_models/{actual_subset}/{config}/{ckpt_filename}"),
                        # Path 3: local testset_aligned directory
                        Path(PROJECT_DIR) / "results" / cfg.data.dataset / "scaling_experiment"
                        / "comm_models_testset_aligned" / subset_dir / config / ckpt_filename,
                        # Path 4: local regular comm_models directory
                        Path(PROJECT_DIR) / "results" / cfg.data.dataset / "scaling_experiment"
                        / "comm_models" / actual_subset / config / ckpt_filename,
                    ]

                    # Try each path until we find a directory that exists
                    ckpt_dir = None
                    for path in possible_paths:
                        if path.parent.exists():
                            ckpt_dir = path.parent
                            break

                    if ckpt_dir is None:
                        paths_str = "\n".join([f"    {i+1}. {p.parent}" for i, p in enumerate(possible_paths)])
                        raise FileNotFoundError(
                            f"Checkpoint directory not found for testset_aligned model {gex_model}.\n"
                            f"  Tried directories:\n{paths_str}\n"
                            f"  Please verify that training completed successfully and checkpoint was saved."
                        )

                    # Check if multiple checkpoints exist (multi-run training)
                    model_ckpt_path = None
                    if ckpt_dir.exists():
                        # Look for checkpoints with seed identifiers
                        seed_checkpoints = list(ckpt_dir.glob("comm_seed*.ckpt"))

                        # Check if a specific checkpoint seed is requested
                        requested_seed = getattr(cfg.evaluation, 'checkpoint_seed', None)

                        if requested_seed is not None:
                            # Use the requested seed checkpoint
                            requested_ckpt = ckpt_dir / f"comm_seed{requested_seed}.ckpt"
                            if requested_ckpt.exists():
                                model_ckpt_path = requested_ckpt
                                print(f"Using requested checkpoint seed: {requested_seed} ({requested_ckpt.name})")
                            else:
                                raise FileNotFoundError(
                                    f"Requested checkpoint seed {requested_seed} not found at {requested_ckpt}\n"
                                    f"Available checkpoints: {[c.name for c in seed_checkpoints]}"
                                )
                        elif len(seed_checkpoints) > 1:
                            # Multiple runs found - select best based on validation loss
                            print(f"Found {len(seed_checkpoints)} checkpoints from multi-run training")
                            best_ckpt = None
                            best_val_loss = float('inf')
                            best_seed = None

                            for seed_ckpt in seed_checkpoints:
                                # Extract seed from filename (comm_seed42.ckpt -> 42)
                                seed_str = seed_ckpt.stem.replace("comm_seed", "")
                                try:
                                    seed = int(seed_str)
                                    metadata_path_seed = ckpt_dir / f"comm_seed{seed}_metadata.json"

                                    if metadata_path_seed.exists():
                                        with open(metadata_path_seed) as mf:
                                            seed_metadata = json.load(mf)
                                        val_loss = seed_metadata.get('best_val_loss', float('inf'))
                                        print(f"   Seed {seed}: val_loss={val_loss:.4f}")

                                        if val_loss < best_val_loss:
                                            best_val_loss = val_loss
                                            best_ckpt = seed_ckpt
                                            best_seed = seed
                                except (ValueError, KeyError):
                                    continue

                            if best_ckpt:
                                model_ckpt_path = best_ckpt
                                print(
                                    f"Selected best checkpoint: {best_ckpt.name} (seed={best_seed}, val_loss={best_val_loss:.4f})")
                            else:
                                print(f"Could not determine best checkpoint, using first found: {model_ckpt_path}")
                        elif len(seed_checkpoints) == 1:
                            # Single seed checkpoint found
                            model_ckpt_path = seed_checkpoints[0]
                            seed_str = model_ckpt_path.stem.replace("comm_seed", "")
                            print(f"Found checkpoint: {model_ckpt_path.name} (seed={seed_str})")
                        else:
                            # No seed checkpoints, try regular naming (backward compatibility)
                            regular_ckpt = ckpt_dir / "comm.ckpt"
                            if regular_ckpt.exists():
                                model_ckpt_path = regular_ckpt
                                print(f"Found checkpoint: {model_ckpt_path.name} (backward compatibility)")
                            else:
                                # Try any of the original possible paths
                                for path in possible_paths:
                                    if path.exists():
                                        model_ckpt_path = path
                                        print(f"Found checkpoint: {model_ckpt_path}")
                                        print(f"   (Using actual subset: {actual_subset})")
                                        break

                    if not model_ckpt_path or not model_ckpt_path.exists():
                        paths_str = "\n".join([f"    {i+1}. {p}" for i, p in enumerate(possible_paths)])
                        raise FileNotFoundError(
                            f"Checkpoint not found for testset_aligned model {gex_model}.\n"
                            f"  Checked directory: {ckpt_dir}\n"
                            f"  Tried paths:\n{paths_str}\n"
                            f"  Please verify that training completed successfully and checkpoint was saved."
                        )

                    ckpt_dir = model_ckpt_path.parent
                    # Try to find metadata - check for seed-specific first, then fallback
                    seed_str = None
                    if "seed" in model_ckpt_path.stem:
                        seed_str = model_ckpt_path.stem.replace("comm_seed", "").replace("comm", "")
                        metadata_path = ckpt_dir / f"comm_seed{seed_str}_metadata.json"
                    else:
                        metadata_path = ckpt_dir / "comm_metadata.json"

                    # Load metadata and update dimensions
                    if metadata_path.exists():
                        import json
                        with open(metadata_path, encoding='utf-8') as mf:
                            metadata = json.load(mf)

                        # Verify metadata matches expected values
                        if "subset" in metadata:
                            if metadata["subset"] != subset_dir:
                                print(
                                    f"Warning: Metadata subset '{metadata['subset']}' does not match expected '{subset_dir}'")
                        if "config" in metadata:
                            if metadata["config"] != config:
                                print(
                                    f"Warning: Metadata config '{metadata['config']}' does not match expected '{config}'")

                        if "img_embed_dim" in metadata:
                            cfg.models.img_embed_dim = metadata["img_embed_dim"]
                        if "gex_embed_dim" in metadata:
                            cfg.models.gex_embed_dim = metadata["gex_embed_dim"]
                        print(
                            f"Using testset-aligned CoMM checkpoint for {subset_dir}/{config} with img_embed_dim={metadata.get('img_embed_dim', 'N/A')}, gex_embed_dim={metadata.get('gex_embed_dim', 'N/A')}")
                    else:
                        print(
                            f"Warning: Metadata not found at {metadata_path}, extracting dimensions from checkpoint...")
                        img_dim, gex_dim, embed_dim = _extract_comm_dimensions_from_checkpoint(model_ckpt_path)
                        if img_dim:
                            cfg.models.img_embed_dim = img_dim
                            print(f"   Extracted img_embed_dim={img_dim} from checkpoint")
                        if gex_dim:
                            cfg.models.gex_embed_dim = gex_dim
                            print(f"   Extracted gex_embed_dim={gex_dim} from checkpoint")
                        if embed_dim:
                            cfg.models.embed_dim = embed_dim
                            print(f"   Extracted embed_dim={embed_dim} from checkpoint")
                else:
                    # Regular finetune checkpoints (existing logic)
                    if any(cfg_str in gex_model for cfg_str in ["cfg1", "cfg2", "cfg3", "cfg4", "cfg5"]):
                        # Extract subset name (handle all configs)
                        subset = gex_model.replace("finetune_", "")
                        for cfg_str in ["_cfg5", "_cfg4", "_cfg3", "_cfg2", "_cfg1"]:
                            subset = subset.replace(cfg_str, "")
                        # Extract config from gex_model (e.g., "finetune_100pct_cfg3" -> "cfg3")
                        config = None
                        for cfg_str in ["cfg5", "cfg4", "cfg3", "cfg2", "cfg1"]:
                            if f"_{cfg_str}" in gex_model:
                                config = cfg_str
                                break

                        # Load checkpoint path from JSON (for cfg1, cfg2) or construct path
                        # directly (for cfg3, cfg4, cfg5)
                        import json
                        checkpoints_json = Path(PROJECT_DIR) / "results" / cfg.data.dataset / \
                            "scaling_experiment" / "comm_models" / "checkpoints.json"
                        if checkpoints_json.exists():
                            with open(checkpoints_json, encoding='utf-8') as f:
                                ckpt_info = json.load(f)
                            if subset in ckpt_info:
                                # Replace dataset name in path from JSON (may have hardcoded 'lung')
                                json_path = ckpt_info[subset]["checkpoint_path"]
                                # Replace any dataset name in the path with the actual dataset
                                import re
                                # Match patterns like /results/lung/ or /results/thymus/ or /results/breast/
                                json_path = re.sub(r'/results/[^/]+/', f'/results/{cfg.data.dataset}/', json_path)
                                model_ckpt_path = Path(json_path)
                                # Load metadata from the same directory as the checkpoint (not from JSON metadata_path which may be wrong)
                                # Extract metadata path from checkpoint path
                                ckpt_dir = model_ckpt_path.parent
                                metadata_path = ckpt_dir / "comm_metadata.json"
                                if not metadata_path.exists():
                                    # Fallback to metadata_path from JSON (also fix dataset name)
                                    json_metadata_path = ckpt_info[subset]["metadata_path"]
                                    json_metadata_path = re.sub(
                                        r'/results/[^/]+/', f'/results/{cfg.data.dataset}/', json_metadata_path)
                                    metadata_path = Path(json_metadata_path)
                            else:
                                # Subset not in JSON, construct path directly
                                if config:
                                    # Check for seed-specific checkpoint first
                                    requested_seed = getattr(cfg.evaluation, 'checkpoint_seed', None)
                                    if requested_seed is not None:
                                        model_ckpt_path = Path(PROJECT_DIR) / "results" / cfg.data.dataset / "scaling_experiment" / \
                                            "comm_models" / subset / config / f"comm_seed{requested_seed}.ckpt"
                                        if not model_ckpt_path.exists():
                                            model_ckpt_path = Path(
                                                f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/{cfg.data.dataset}/scaling_experiment/comm_models/{subset}/{config}/comm_seed{requested_seed}.ckpt")
                                    else:
                                        model_ckpt_path = Path(PROJECT_DIR) / "results" / cfg.data.dataset / \
                                            "scaling_experiment" / "comm_models" / subset / config / "comm.ckpt"
                                        if not model_ckpt_path.exists():
                                            model_ckpt_path = Path(
                                                f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/{cfg.data.dataset}/scaling_experiment/comm_models/{subset}/{config}/comm.ckpt")
                                    ckpt_dir = model_ckpt_path.parent
                                    metadata_path = ckpt_dir / "comm_metadata.json"
                                else:
                                    raise ValueError(f"Could not determine config from gex_model: {gex_model}")
                        else:
                            # JSON doesn't exist, construct path directly
                            if config:
                                # Check for seed-specific checkpoint first
                                requested_seed = getattr(cfg.evaluation, 'checkpoint_seed', None)
                                if requested_seed is not None:
                                    model_ckpt_path = Path(PROJECT_DIR) / "results" / cfg.data.dataset / "scaling_experiment" / \
                                        "comm_models" / subset / config / f"comm_seed{requested_seed}.ckpt"
                                    if not model_ckpt_path.exists():
                                        model_ckpt_path = Path(
                                            f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/{cfg.data.dataset}/scaling_experiment/comm_models/{subset}/{config}/comm_seed{requested_seed}.ckpt")
                                else:
                                    model_ckpt_path = Path(PROJECT_DIR) / "results" / cfg.data.dataset / \
                                        "scaling_experiment" / "comm_models" / subset / config / "comm.ckpt"
                                    if not model_ckpt_path.exists():
                                        model_ckpt_path = Path(
                                            f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/{cfg.data.dataset}/scaling_experiment/comm_models/{subset}/{config}/comm.ckpt")
                                ckpt_dir = model_ckpt_path.parent
                                metadata_path = ckpt_dir / "comm_metadata.json"
                            else:
                                raise ValueError(f"Could not determine config from gex_model: {gex_model}")

                        # Load metadata and update dimensions (for both JSON and direct path cases)
                        if metadata_path.exists():
                            with open(metadata_path, encoding='utf-8') as mf:
                                metadata = json.load(mf)
                            # Update cfg with correct dimensions before model initialization
                            if "img_embed_dim" in metadata:
                                cfg.models.img_embed_dim = metadata["img_embed_dim"]
                            if "gex_embed_dim" in metadata:
                                cfg.models.gex_embed_dim = metadata["gex_embed_dim"]
                            # Extract embed_dim from checkpoint if not in metadata
                            if "embed_dim" not in metadata and model_ckpt_path.exists():
                                _, _, embed_dim = _extract_comm_dimensions_from_checkpoint(model_ckpt_path)
                                if embed_dim:
                                    cfg.models.embed_dim = embed_dim
                                    print(f"   Extracted embed_dim={embed_dim} from checkpoint (not in metadata)")
                            elif "embed_dim" in metadata:
                                cfg.models.embed_dim = metadata["embed_dim"]
                            print(
                                f"Using CoMM checkpoint for {subset} from {model_ckpt_path.parent.name} with img_embed_dim={metadata.get('img_embed_dim', 'N/A')}, gex_embed_dim={metadata.get('gex_embed_dim', 'N/A')}, embed_dim={cfg.models.embed_dim}")
                        else:
                            print(
                                f"Warning: Metadata not found at {metadata_path}, extracting dimensions from checkpoint...")
                            img_dim, gex_dim, embed_dim = _extract_comm_dimensions_from_checkpoint(model_ckpt_path)
                            if img_dim:
                                cfg.models.img_embed_dim = img_dim
                                print(f"   Extracted img_embed_dim={img_dim} from checkpoint")
                            if gex_dim:
                                cfg.models.gex_embed_dim = gex_dim
                                print(f"   Extracted gex_embed_dim={gex_dim} from checkpoint")
                            if embed_dim:
                                cfg.models.embed_dim = embed_dim
                                print(f"   Extracted embed_dim={embed_dim} from checkpoint")
                    else:
                        # No cfg1-5 found in gex_model, use default path
                        model_ckpt_path = Path(PROJECT_DIR) / "trained_models" / \
                            cfg.data.dataset / "full_ds" / info["ckpt"]
            else:
                # Load pre-trained checkpoint
                # Dimension extraction for regular CoMM checkpoints is done earlier (before model creation)
                # to ensure correct model initialization. This path just sets the checkpoint path.
                # Check if checkpoint_config is specified to use a specific checkpoint file
                if hasattr(cfg.evaluation, 'checkpoint_config') and cfg.evaluation.checkpoint_config is not None:
                    # Parse checkpoint_config (e.g., "100pct_cfg23" -> subset="100pct", config="cfg23")
                    checkpoint_config = cfg.evaluation.checkpoint_config
                    if "_cfg" in checkpoint_config:
                        # Extract subset and config from checkpoint_config
                        parts = checkpoint_config.split("_cfg")
                        subset = parts[0]  # e.g., "100pct"
                        config = f"cfg{parts[1]}"  # e.g., "cfg23"

                        # Check for seed-specific checkpoint first
                        requested_seed = getattr(cfg.evaluation, 'checkpoint_seed', None)
                        if requested_seed is not None:
                            model_ckpt_path = Path(
                                f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/{cfg.data.dataset}/scaling_experiment/comm_models/{subset}/{config}/comm_seed{requested_seed}.ckpt")
                            if not model_ckpt_path.exists():
                                model_ckpt_path = Path(PROJECT_DIR) / "results" / cfg.data.dataset / "scaling_experiment" / \
                                    "comm_models" / subset / config / f"comm_seed{requested_seed}.ckpt"
                        else:
                            model_ckpt_path = Path(
                                f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/results/{cfg.data.dataset}/scaling_experiment/comm_models/{subset}/{config}/comm.ckpt")
                            if not model_ckpt_path.exists():
                                model_ckpt_path = Path(PROJECT_DIR) / "results" / cfg.data.dataset / \
                                    "scaling_experiment" / "comm_models" / subset / config / "comm.ckpt"

                        # If still not found and subset is "100pct", check in
                        # trained_models/full_ds/ (where full dataset training saves checkpoints)
                        if not model_ckpt_path.exists() and subset == "100pct":
                            # Look for checkpoints in trained_models/{dataset}/full_ds/ with pattern
                            # comm_{config}_job*.ckpt
                            trained_models_dir = Path(
                                f"/ictstr01/groups/ml01/projects/2025_nicheformer_subsets/trained_models/{cfg.data.dataset}/full_ds")
                            if not trained_models_dir.exists():
                                trained_models_dir = Path(PROJECT_DIR) / "trained_models" / cfg.data.dataset / "full_ds"

                            if trained_models_dir.exists():
                                # Find the most recent checkpoint matching the config
                                matching_checkpoints = list(trained_models_dir.glob(f"comm_{config}_job*.ckpt"))
                                if matching_checkpoints:
                                    # Sort by modification time and use the most recent
                                    matching_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                    model_ckpt_path = matching_checkpoints[0]
                                    print(f"Found checkpoint in trained_models: {model_ckpt_path}")
                                else:
                                    # Try without job ID suffix as fallback
                                    fallback_path = trained_models_dir / f"comm_{config}.ckpt"
                                    if fallback_path.exists():
                                        model_ckpt_path = fallback_path
                                        print(f"Found checkpoint (no job ID): {model_ckpt_path}")

                            # Extract dimensions from checkpoint before model creation
                            if model_ckpt_path.exists():
                                # Try metadata first
                                metadata_path = model_ckpt_path.parent / "comm_metadata.json"
                                if metadata_path.exists():
                                    import json
                                    with open(metadata_path, encoding='utf-8') as mf:
                                        metadata = json.load(mf)
                                    if "img_embed_dim" in metadata:
                                        cfg.models.img_embed_dim = metadata["img_embed_dim"]
                                        print(f"   Updated img_embed_dim={cfg.models.img_embed_dim} from metadata")
                                    if "gex_embed_dim" in metadata:
                                        cfg.models.gex_embed_dim = metadata["gex_embed_dim"]
                                        print(f"   Updated gex_embed_dim={cfg.models.gex_embed_dim} from metadata")
                                    if "embed_dim" in metadata:
                                        cfg.models.embed_dim = metadata["embed_dim"]
                                        print(f"   Updated embed_dim={cfg.models.embed_dim} from metadata")
                                else:
                                    # Extract from checkpoint
                                    print(f"   Extracting dimensions from checkpoint...")
                                    img_dim, gex_dim, embed_dim = _extract_comm_dimensions_from_checkpoint(
                                        model_ckpt_path)
                                    if img_dim:
                                        cfg.models.img_embed_dim = img_dim
                                        print(f"   Extracted img_embed_dim={img_dim} from checkpoint")
                                    if gex_dim:
                                        cfg.models.gex_embed_dim = gex_dim
                                        print(f"   Extracted gex_embed_dim={gex_dim} from checkpoint")
                                    if embed_dim:
                                        cfg.models.embed_dim = embed_dim
                                        print(f"   Extracted embed_dim={embed_dim} from checkpoint")
                    else:
                        # Fallback to old format (e.g., "cfg1_job31975900" -> "comm_cfg1_job31975900.ckpt")
                        checkpoint_name = f"comm_{cfg.evaluation.checkpoint_config}.ckpt"
                        model_ckpt_path = Path(PROJECT_DIR) / "trained_models" / \
                            cfg.data.dataset / "full_ds" / checkpoint_name
                else:
                    # Default to comm.ckpt
                    model_ckpt_path = Path(PROJECT_DIR) / "trained_models" / cfg.data.dataset / "full_ds" / info["ckpt"]

            assert model_ckpt_path.exists(), f"Checkpoint does not exist: {model_ckpt_path}"

            # Load model
            model = ModelClass(cfg).to(device)

        _debug_print(f"[DEBUG] Loading checkpoint from: {model_ckpt_path}")
        checkpoint = torch.load(model_ckpt_path, map_location=device, weights_only=False)

        _debug_print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            _debug_print(f"[DEBUG] State dict keys: {list(state_dict.keys())}")
            _debug_print(f"[DEBUG] State dict shapes:")
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    _debug_print(f"[DEBUG]   {key}: {tensor.shape}")
        else:
            _debug_print(f"[DEBUG] No state_dict in checkpoint, using checkpoint directly")
            state_dict = checkpoint
            _debug_print(f"[DEBUG] Direct checkpoint keys: {list(state_dict.keys())}")
            _debug_print(f"[DEBUG] Direct checkpoint shapes:")
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    _debug_print(f"[DEBUG]   {key}: {tensor.shape}")

        _debug_print(f"[DEBUG] Model state dict before loading:")
        for key, tensor in model.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                _debug_print(f"[DEBUG]   {key}: {tensor.shape}")

        # Handle PyTorch Lightning checkpoint format by removing "model." prefix
        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(key.startswith("model.") for key in state_dict.keys()):
            _debug_print(f"[DEBUG] Detected PyTorch Lightning checkpoint format - removing 'model.' prefix")
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]  # Remove "model." prefix
                    cleaned_state_dict[new_key] = value
                    _debug_print(f"[DEBUG] Mapped {key} -> {new_key}")
                else:
                    cleaned_state_dict[key] = value
            state_dict = cleaned_state_dict

        _debug_print(f"[DEBUG] Final state dict keys after cleaning: {list(state_dict.keys())}")

        # For CoMM, verify adapter weights match before loading
        if method == "comm":
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor) and 'adapter' in key and 'proj.weight' in key:
                    if key in model.state_dict():
                        if model.state_dict()[key].shape != value.shape:
                            raise RuntimeError(
                                f"Adapter dimension mismatch: {key}\n"
                                f"  Checkpoint: {value.shape}, Model: {model.state_dict()[key].shape}\n"
                                f"  Config: img_embed_dim={cfg.models.img_embed_dim}, gex_embed_dim={cfg.models.gex_embed_dim}\n"
                                f"  Root cause: Model initialized with wrong dimensions. Extract dimensions from checkpoint BEFORE model creation.")

        model.load_state_dict(state_dict, strict=False)

        _debug_print(f"[DEBUG] Model state dict after loading:")
        for key, tensor in model.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                _debug_print(f"[DEBUG]   {key}: {tensor.shape}")

        # Additional debugging for transformer weights
        _debug_print(f"[DEBUG] Checking transformer weights specifically:")
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'fusion_transformer'):
            ft = model.encoder.fusion_transformer
            _debug_print(f"[DEBUG] Fusion transformer resblocks type: {type(ft.resblocks)}")
            _debug_print(f"[DEBUG] Fusion transformer resblocks length: {len(ft.resblocks)}")

            # Check if transformer weights are actually loaded
            for i, block in enumerate(ft.resblocks):
                if hasattr(block, 'attn'):
                    attn_weights = block.attn.in_proj_weight
                    _debug_print(
                        f"[DEBUG]   Block {i} attention weights: {attn_weights.shape}, norm: {attn_weights.norm().item():.6f}")
                if hasattr(block, 'mlp'):
                    mlp_weights = block.mlp.c_fc.weight
                    _debug_print(
                        f"[DEBUG]   Block {i} MLP weights: {mlp_weights.shape}, norm: {mlp_weights.norm().item():.6f}")

        # Check if any keys were missing during loading with the cleaned state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            _debug_print(f"[DEBUG] Checkpoint loaded successfully with strict=True")
        except RuntimeError as e:
            _debug_print(f"[DEBUG] Strict loading failed: {e}")
            # Try to parse the error to see what's missing
            if "Missing key(s)" in str(e):
                missing_start = str(e).find("Missing key(s):")
                if missing_start != -1:
                    missing_part = str(e)[missing_start:]
                    _debug_print(f"[DEBUG] Missing keys: {missing_part}")
            if "Unexpected key(s)" in str(e):
                unexpected_start = str(e).find("Unexpected key(s):")
                if unexpected_start != -1:
                    unexpected_part = str(e)[unexpected_start:]
                    _debug_print(f"[DEBUG] Unexpected keys: {unexpected_part}")

            # Fall back to non-strict loading with the cleaned state dict
            _debug_print(f"[DEBUG] Falling back to strict=False loading")
            model.load_state_dict(state_dict, strict=False)

        model.eval()

        # Final validation: adapter dimensions must match
        if method == "comm" and hasattr(model, 'img_adapter') and hasattr(model.img_adapter, 'proj'):
            actual_dim = model.img_adapter.proj.weight.shape[1]
            if actual_dim != cfg.models.img_embed_dim:
                raise RuntimeError(
                    f"Adapter dimension mismatch: model has {actual_dim}D, config expects {cfg.models.img_embed_dim}D")
            if cfg.data.img_embed_key == "img_uni_pool" and actual_dim != 1536:
                raise RuntimeError(f"img_uni_pool requires 1536D adapter, got {actual_dim}D")

        # Fix FusionTransformer.forward() for x-attn mode with Sequential containers
        # This must be done AFTER model loading, right before use
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'fusion_transformer'):
            ft = model.encoder.fusion_transformer
            if hasattr(ft, 'fusion') and ft.fusion == "x-attn":
                # Patch the forward method to handle Sequential containers
                original_forward = ft.forward

                def patched_forward(self, x, key_padding_mask=None):
                    """Patched forward that handles Sequential containers in x-attn mode."""
                    if self.fusion == "x-attn":
                        if self.pool == "cls":
                            raise ValueError("Only `mean` pool is implemented for cross-attention.")
                        if len(x) != 2:
                            raise ValueError("Only 2 modalities are currently accepted for cross-attention")
                        if key_padding_mask is not None:
                            raise NotImplementedError()
                        x1, x2 = x

                        # Handle Sequential containers - manually call each block
                        def forward_sequential(seq_block, query, key, kp_mask=None):
                            """Forward through Sequential container by manually calling each ResidualCrossAttentionBlock."""
                            result = query
                            # resblocks[0] and resblocks[1] are SequentialWrapper objects (or Sequential containers)
                            # Each contains ResidualCrossAttentionBlock layers
                            # If it's a SequentialWrapper, iterate over its .sequential attribute
                            sequential = getattr(seq_block, 'sequential', seq_block)
                            for block in sequential:
                                result = block(result, key, kp_mask)
                            return result

                        # Call with manual Sequential handling
                        out1 = forward_sequential(self.resblocks[0], x1, x2, key_padding_mask)
                        out2 = forward_sequential(self.resblocks[1], x2, x1, key_padding_mask)
                        x = torch.cat([out1, out2], dim=self.token_dim)
                        x = self.norm(x).mean(dim=self.token_dim)
                        return x
                    else:
                        # Use original forward for non-x-attn modes
                        return original_forward(self, x, key_padding_mask)

                # Apply patch to this specific instance
                import types
                ft.forward = types.MethodType(patched_forward, ft)

                # After patching, ensure all Sequential blocks and their components are on device
                # The patch creates a closure that references self.resblocks, so we need to explicitly move them
                for i, seq_block in enumerate(ft.resblocks):
                    seq_block.to(device)
                    # Also ensure all nested modules in the Sequential are on device
                    # If it's a SequentialWrapper, iterate over its .sequential attribute
                    sequential = getattr(seq_block, 'sequential', seq_block)
                    for block in sequential:
                        block.to(device)
                        # Ensure all parameters in each block are on device
                        for param in block.parameters():
                            param.data = param.data.to(device)
                        for buffer in block.buffers():
                            buffer.data = buffer.data.to(device)

                print(
                    f"Patched FusionTransformer.forward() for x-attn Sequential containers and ensured all components on {device}")

        # Cache the model for future use
        _global_model_cache[cache_key] = model
        print(f"Cached {method} model for future use")

    # Build tensors using HF Dataset vectorized access
    def build_tensors(hfd, valid_indices=None):
        if valid_indices is not None:
            # Select only the specified indices
            hfd_subset = hfd.select(valid_indices)
            img_tensor = torch.as_tensor(hfd_subset[img_embed_key], dtype=torch.float32)
            gex_tensor = torch.as_tensor(hfd_subset[gex_embed_key], dtype=torch.float32)
            targets_dtype = torch.float32 if not isinstance(hfd_subset[0][target_key], int) else torch.long
            targets = torch.as_tensor(hfd_subset[target_key], dtype=targets_dtype)
            names = hfd_subset["name"]
        else:
            # Use entire dataset
            img_tensor = torch.as_tensor(hfd[img_embed_key], dtype=torch.float32)
            gex_tensor = torch.as_tensor(hfd[gex_embed_key], dtype=torch.float32)
            targets_dtype = torch.float32 if not isinstance(hfd[0][target_key], int) else torch.long
            targets = torch.as_tensor(hfd[target_key], dtype=targets_dtype)
            names = hfd["name"]
        return img_tensor, gex_tensor, targets, names

    img, gex, targets, names = build_tensors(hfd, valid_indices)

    # Assert that img_uni_pool embeddings are 1536D for 200M model
    if img_embed_key == "img_uni_pool":
        assert img.shape[1] == 1536, (
            f"Expected img_uni_pool embeddings to be 1536D, but got {img.shape[1]}D. "
            f"This indicates the dataset is using outdated embeddings. "
            f"Check that the dataset contains img_uni_pool with correct dimensions."
        )
        print(f"Verified img_uni_pool embedding dimension: {img.shape[1]}D")

    # Batched forward pass
    def get_fused_batched(img_tensor, gex_tensor):
        fused_list = []
        # Ensure model is fully on device before evaluation
        model.eval()
        # Force move all submodules to device (some might not move properly)
        for param in model.parameters():
            param.data = param.data.to(device)
        for buffer in model.buffers():
            buffer.data = buffer.data.to(device)
        for start in range(0, len(img_tensor), batch_size):
            end = start + batch_size
            with torch.no_grad():
                # CoMM's get_embeddings() returns separate projections (img_z, gex_z) tuple,
                # but we need the actual FUSED embedding from the transformer encoder!
                # Using concatenated projections would make CoMM equivalent to simple concatenation.
                #
                # Check if this is CoMM by checking align_method or method name
                is_comm = (align_method == "multimodal_comm"
                           or (hasattr(cfg, 'models') and cfg.models.method == "comm"))
                if hasattr(model, 'fusion') and is_comm:
                    # Use fusion() method which returns the actual fused embedding from the transformer
                    # This is the learned multimodal representation, not just concatenated projections
                    # Ensure inputs are on the same device as model
                    batch_img = img_tensor[start:end].to(device)
                    batch_gex = gex_tensor[start:end].to(device)
                    batch_fused = model.fusion(batch_img, batch_gex)
                    if isinstance(batch_fused, tuple):
                        # Sanity check: fusion() should return a single tensor, not a tuple
                        raise ValueError(
                            "CoMM fusion() returned a tuple - this is unexpected! "
                            "fusion() should return the fused embedding tensor, not separate projections. "
                            "Check CoMM model implementation."
                        )
                else:
                    batch_fused = model.get_embeddings(
                        img_tensor[start:end].to(device),
                        gex_tensor[start:end].to(device)
                    )
                    if isinstance(batch_fused, tuple):
                        # Handle models that return separate embeddings (e.g., other methods)
                        # Concatenate them for downstream evaluation, just like multimodal_concat
                        batch_fused = torch.cat(batch_fused, dim=-1)
                fused_list.append(batch_fused.cpu())
        return torch.cat(fused_list, dim=0)

    fused = get_fused_batched(img, gex)

    _debug_print(f"[DEBUG] Final fused embeddings shape: {fused.shape}")
    _debug_print(f"[DEBUG] Expected shape: batch_size={len(img)} x embedding_dim={fused.shape[1]}")

    return fused, targets, names


def load_and_preprocess_data(cfg: DictConfig, split_name: str, align_method=None):
    """
    Load dataset split and create separate classification and regression datasets.

    Returns:
        ds_clf: Dataset for classification (filtered by excluded classes)
        ds_reg: Dataset for regression (with NOMAP removed from cell type ratios)
    """
    hf_dir = Path("/lustre/groups/ml01/workspace/till.richter/hf_datasets") / cfg.data.dataset

    modality = cfg.evaluation.modality
    img_model = cfg.evaluation.img_model
    gex_model = cfg.evaluation.gex_model
    scale = cfg.evaluation.scale

    class_label_key = cfg.training.classification.label_key
    reg_label_key = cfg.training.regression.label_key

    print(f"\nLoading {split_name} split data for modality={modality}, img={img_model}, gex={gex_model}")

    # Use dataset_name if it looks like a pattern (has underscores/dots and
    # matches expected structure), otherwise construct pattern
    dataset_name = getattr(cfg.data, 'dataset_name', None)
    # Check if dataset_name is a valid pattern (contains expected prefixes
    # like "img_only_", "gex_only_", or has model names)
    is_valid_pattern = (dataset_name and
                        dataset_name != cfg.data.dataset
                        and ('_' in dataset_name or '.' in dataset_name)
                        and (dataset_name.startswith('img_only_') or
                             dataset_name.startswith('gex_only_')
                             or '_' in dataset_name.split('.')[0]))  # Has model names in pattern

    if is_valid_pattern:
        # dataset_name is explicitly set as a pattern (e.g., from script override like "gex_only_full_test.L")
        pattern = dataset_name
    else:
        # Construct pattern from modality/model/scale (default behavior)
        if align_method == "random" or "multimodal" in modality:
            # For multimodal concat, need to find the gex dataset pattern first
            # The actual dataset will be constructed from gex and img embeddings
            if gex_model and gex_model.startswith("finetune_"):
                # Try finetune format: gex_only_finetune_1pct_cfg1_train.S (always _train.S for finetune)
                finetune_pattern = f"gex_only_{gex_model}_train.S"
                finetune_path = hf_dir / finetune_pattern
                if finetune_path.exists():
                    # For concat with finetune models, try combined dataset first
                    # Try both with and without _test suffix (datasets may be named finetune_1pct_cfg1_200M_test.L)
                    combined_pattern = f"{gex_model}_{img_model}.{scale}"
                    combined_pattern_test = f"{gex_model}_{img_model}_test.{scale}"
                    combined_path = hf_dir / combined_pattern
                    combined_path_test = hf_dir / combined_pattern_test
                    if combined_path_test.exists():
                        pattern = combined_pattern_test
                    elif combined_path.exists():
                        pattern = combined_pattern
                    else:
                        # Combined dataset doesn't exist - will load from separate gex and img datasets
                        # Use gex pattern as base, extract_embeddings will handle loading img separately
                        pattern = finetune_pattern
                else:
                    # Try both with and without _test suffix
                    pattern = f"{gex_model}_{img_model}.{scale}"
                    pattern_test = f"{gex_model}_{img_model}_test.{scale}"
                    if (hf_dir / pattern_test).exists():
                        pattern = pattern_test
                    elif not (hf_dir / pattern).exists():
                        # Dataset with this config doesn't exist - try cfg2 or cfg1 fallback
                        # This handles cases where cfg3/cfg5 models were trained on cfg2/cfg1 datasets
                        import re
                        # Extract subset from gex_model (e.g., "finetune_100pct_cfg3" -> "100pct")
                        subset_match = re.search(r'finetune_([^_]+)', gex_model)
                        if subset_match:
                            subset = subset_match.group(1)
                            # Try cfg2 first (most common), then cfg1
                            for fallback_cfg in ["cfg2", "cfg1"]:
                                fallback_gex_model = f"finetune_{subset}_{fallback_cfg}"
                                fallback_pattern = f"{fallback_gex_model}_{img_model}.{scale}"
                                fallback_pattern_test = f"{fallback_gex_model}_{img_model}_test.{scale}"
                                if (hf_dir / fallback_pattern_test).exists():
                                    pattern = fallback_pattern_test
                                    print(f"  Dataset with config from gex_model not found, using fallback: {pattern}")
                                    break
                                elif (hf_dir / fallback_pattern).exists():
                                    pattern = fallback_pattern
                                    print(f"  Dataset with config from gex_model not found, using fallback: {pattern}")
                                    break
            elif gex_model is None or gex_model == "None":
                # For pretrained (0pct), use nicheformer_full pattern
                pattern = f"nicheformer_full_{img_model}.{scale}"
            else:
                pattern = f"{gex_model}_{img_model}.{scale}"
        elif modality == "unimodal_img":
            pattern = f"img_only_{img_model}.{scale}"
        elif modality == "unimodal_gex":
            # For finetune models, datasets are named with _train.S suffix (regardless of split_name used for eval)
            # Check if this is a finetune model and if dataset exists with _train.S format
            if gex_model and gex_model.startswith("finetune_"):
                # Try finetune format: gex_only_finetune_1pct_cfg1_train.S (always _train.S for finetune)
                finetune_pattern = f"gex_only_{gex_model}_train.S"
                finetune_path = hf_dir / finetune_pattern
                if finetune_path.exists():
                    pattern = finetune_pattern
                else:
                    # Fallback to standard format
                    pattern = f"gex_only_{gex_model}.{scale}"
            elif gex_model is None or gex_model == "None":
                # For pretrained (0pct), try nicheformer_full first, then fallback to full
                pretrained_patterns = [
                    f"gex_only_nicheformer_full_{img_model}.{scale}",
                    f"gex_only_full.{scale}",
                    f"gex_only_nicheformer_full_200M.L"  # Common fallback
                ]
                pattern = None
                for pretrained_pattern in pretrained_patterns:
                    pretrained_path = hf_dir / pretrained_pattern
                    if pretrained_path.exists():
                        pattern = pretrained_pattern
                        break
                if pattern is None:
                    # Last resort: use the first pattern
                    pattern = pretrained_patterns[0]
            else:
                pattern = f"gex_only_{gex_model}.{scale}"
        else:
            raise ValueError(f"Invalid modality: {modality}")

    # Special handling for multimodal_concat with finetune models when combined dataset doesn't exist
    # We need to load from separate gex and img datasets and merge them
    load_separate_for_concat = False
    if (align_method == "multimodal_concat"
        and gex_model and gex_model.startswith("finetune_")
            and pattern.startswith("gex_only_")):
        # Check if combined dataset exists
        combined_pattern = f"{gex_model}_{img_model}.{scale}"
        combined_path = hf_dir / combined_pattern
        if not combined_path.exists():
            # Need to load from separate datasets
            load_separate_for_concat = True

    split_dir = hf_dir / pattern
    # Fallback to old path structure if not found in new location
    if not split_dir.exists():
        # Try with _test suffix if pattern doesn't already have it
        pattern_test = pattern.replace(f".{scale}", f"_test.{scale}") if f"_test.{scale}" not in pattern else None
        if pattern_test and (hf_dir / pattern_test).exists():
            split_dir = hf_dir / pattern_test
            pattern = pattern_test
            print(f"  Found dataset with _test suffix: {split_dir}")
        else:
            # Try hf_datasets_full location (where full datasets with img_uni_pool are stored)
            old_hf_dir = PROJECT_DIR / cfg.data.dataset / "hf_datasets_full"
            old_split_dir = old_hf_dir / pattern
            old_split_dir_test = old_hf_dir / pattern_test if pattern_test else None
            # Check if it's a symlink and resolve it
            if old_split_dir.exists() or old_split_dir.is_symlink():
                if old_split_dir.is_symlink():
                    # Resolve symlink to actual path
                    resolved = old_split_dir.resolve()
                    if resolved.exists():
                        print(f"  Using dataset from hf_datasets_full path (resolved symlink): {resolved}")
                        split_dir = resolved
                    else:
                        print(f"  Using dataset from hf_datasets_full path (symlink): {old_split_dir}")
                        split_dir = old_split_dir
                else:
                    print(f"  Using dataset from hf_datasets_full path: {old_split_dir}")
                    split_dir = old_split_dir
            elif old_split_dir_test and (old_split_dir_test.exists() or old_split_dir_test.is_symlink()):
                if old_split_dir_test.is_symlink():
                    resolved = old_split_dir_test.resolve()
                    if resolved.exists():
                        print(
                            f"  Using dataset from hf_datasets_full path with _test suffix (resolved symlink): {resolved}")
                        split_dir = resolved
                        pattern = pattern_test
                    else:
                        print(
                            f"  Using dataset from hf_datasets_full path with _test suffix (symlink): {old_split_dir_test}")
                        split_dir = old_split_dir_test
                        pattern = pattern_test
                else:
                    print(f"  Using dataset from hf_datasets_full path with _test suffix: {old_split_dir_test}")
                    split_dir = old_split_dir_test
                    pattern = pattern_test
            else:
                # Try old hf_datasets location
                old_hf_dir = PROJECT_DIR / cfg.data.dataset / "hf_datasets"
                old_split_dir = old_hf_dir / pattern
                old_split_dir_test = old_hf_dir / pattern_test if pattern_test else None
                if old_split_dir_test and old_split_dir_test.exists():
                    print(f"  Using dataset from old path with _test suffix: {old_split_dir_test}")
                    split_dir = old_split_dir_test
                    pattern = pattern_test
                elif old_split_dir.exists():
                    print(f"  Using dataset from old path: {old_split_dir}")
                    split_dir = old_split_dir
                else:
                    # Last resort: try cfg2/cfg1 fallback for finetune models
                    if gex_model and gex_model.startswith("finetune_") and "multimodal" in modality:
                        import re
                        subset_match = re.search(r'finetune_([^_]+)', gex_model)
                        if subset_match:
                            subset = subset_match.group(1)
                            found_fallback = False
                            for fallback_cfg in ["cfg2", "cfg1"]:
                                fallback_gex_model = f"finetune_{subset}_{fallback_cfg}"
                                fallback_pattern = f"{fallback_gex_model}_{img_model}_test.{scale}"
                                fallback_path = hf_dir / fallback_pattern
                                if fallback_path.exists():
                                    split_dir = fallback_path
                                    pattern = fallback_pattern
                                    print(
                                        f"  Using fallback dataset (cfg3/cfg5 models trained on {fallback_cfg} datasets): {split_dir}")
                                    found_fallback = True
                                    break
                            if found_fallback:
                                pass  # Continue with fallback dataset
                            else:
                                raise ValueError(
                                    f"Split directory not found in new ({split_dir}), hf_datasets_full ({PROJECT_DIR / cfg.data.dataset / 'hf_datasets_full' / pattern}), or old ({old_split_dir}) location, and no fallback cfg2/cfg1 dataset found")
                        else:
                            raise ValueError(
                                f"Split directory not found in new ({split_dir}), hf_datasets_full ({PROJECT_DIR / cfg.data.dataset / 'hf_datasets_full' / pattern}), or old ({old_split_dir}) location")
                    else:
                        raise ValueError(
                            f"Split directory not found in new ({split_dir}), hf_datasets_full ({PROJECT_DIR / cfg.data.dataset / 'hf_datasets_full' / pattern}), or old ({old_split_dir}) location")

    ds = load_from_disk(str(split_dir))
    original_size = len(ds)
    print(f"Loaded {original_size} samples")

    # If we need to load separate datasets for concat, merge them now
    if load_separate_for_concat:
        print(f"  Loading separate img dataset for concat...")
        img_pattern = f"img_only_{img_model}.{scale}"
        img_dir = hf_dir / img_pattern
        if not img_dir.exists():
            # Try fallback locations
            old_hf_dir = PROJECT_DIR / cfg.data.dataset / "hf_datasets_full"
            old_img_dir = old_hf_dir / img_pattern
            if old_img_dir.exists():
                img_dir = old_img_dir
            else:
                old_hf_dir = PROJECT_DIR / cfg.data.dataset / "hf_datasets"
                old_img_dir = old_hf_dir / img_pattern
                if old_img_dir.exists():
                    img_dir = old_img_dir
                else:
                    raise ValueError(f"Image dataset not found for concat: {img_pattern}")

        img_ds = load_from_disk(str(img_dir))
        print(f"  Loaded {len(img_ds)} samples from img dataset")

        # Merge datasets by sample name/id
        # Assuming both datasets have a 'sample_name' or similar identifier column
        # Create a mapping from sample name to index for both datasets
        gex_key = cfg.data.gex_embed_key
        img_key = cfg.data.img_embed_key

        # Find common identifier column (try common names)
        id_cols = ['sample_name', 'sample_id', 'id', 'name']
        gex_id_col = None
        img_id_col = None
        for col in id_cols:
            if col in ds.column_names:
                gex_id_col = col
            if col in img_ds.column_names:
                img_id_col = col
            if gex_id_col and img_id_col:
                break

        if not gex_id_col or not img_id_col:
            # Fallback: use index if no ID column found
            print(f"  Warning: No common ID column found, using index-based merging")
            # Create temporary ID columns
            ds = ds.add_column("_temp_id", [f"gex_{i}" for i in range(len(ds))])
            img_ds = img_ds.add_column("_temp_id", [f"img_{i}" for i in range(len(img_ds))])
            gex_id_col = "_temp_id"
            img_id_col = "_temp_id"

        # Create mapping from ID to img embeddings
        img_embeddings_map = {}
        for i, sample_id in enumerate(img_ds[img_id_col]):
            img_embeddings_map[sample_id] = img_ds[i][img_key]

        # Add img embeddings to gex dataset
        def add_img_embeddings(example):
            sample_id = example[gex_id_col]
            if sample_id in img_embeddings_map:
                example[img_key] = img_embeddings_map[sample_id]
            else:
                # If img embedding not found, use NaN (will be filtered later)
                example[img_key] = np.full(1536, np.nan, dtype=np.float32)
            return example

        ds = ds.map(add_img_embeddings, num_proc=1)
        print(f"  Merged datasets: {len(ds)} samples with both gex and img embeddings")

        # Remove temporary ID column if we added it
        if gex_id_col == "_temp_id":
            ds = ds.remove_columns(["_temp_id"])

    # VERIFICATION: Check dataset metadata if available
    if hasattr(ds, 'info') and hasattr(ds.info, 'metadata') and ds.info.metadata:
        metadata = ds.info.metadata
        print(f"Dataset metadata:")
        if "gex_model_scale" in metadata:
            expected_gex = gex_model if gex_model else "pretrained"
            actual_gex = metadata["gex_model_scale"]
            # For testset_aligned, the dataset might have finetune_ prefix but eval uses testset_aligned
            if actual_gex == expected_gex or (expected_gex and expected_gex.startswith(
                    "testset_aligned") and actual_gex.startswith("finetune")):
                print(f"  GEX model matches: {actual_gex}")
            else:
                print(f"  WARNING: GEX model mismatch! Expected: {expected_gex}, Found: {actual_gex}")
        if "img_model_scale" in metadata:
            expected_img = img_model if img_model else "200M"
            actual_img = metadata["img_model_scale"]
            if actual_img == expected_img:
                print(f"  IMG model matches: {actual_img}")
            else:
                print(f"  WARNING: IMG model mismatch! Expected: {expected_img}, Found: {actual_img}")
        if "n_samples" in metadata:
            if metadata["n_samples"] == original_size:
                print(f"  Sample count matches: {original_size}")
            else:
                print(
                    f"  WARNING: Sample count mismatch! Expected: {metadata['n_samples']}, Found: {original_size}")

    # Apply max_samples limit if specified (for memory optimization)
    max_samples = getattr(cfg.data, 'max_samples', None)
    if max_samples is not None and len(ds) > max_samples:
        print(f"Limiting dataset to {max_samples} samples for memory optimization")
        ds = ds.select(range(min(max_samples, len(ds))))
        original_size = len(ds)
        print(f"Dataset size after limiting: {original_size} samples")

    # FIRST: Efficiently filter out samples with NaN values from the original dataset
    print("  Filtering out samples with NaN values...")
    import numpy as np

    # Get embedding keys from config
    mask = np.ones(len(ds), dtype=bool)
    img_key = cfg.data.img_embed_key
    gex_key = cfg.data.gex_embed_key

    # Determine which embeddings to check based on method
    check_img = align_method in [
        "unimodal_img", "multimodal_concat"] or (
        align_method and "multimodal_" in align_method)
    check_gex = align_method in [
        "unimodal_gex", "multimodal_concat"] or (
        align_method and "multimodal_" in align_method)

    if check_img:
        arr_img = np.array(ds[img_key])
        mask &= ~np.isnan(arr_img).any(axis=1)
        del arr_img  # Clean up large array

    if check_gex:
        arr_gex = np.array(ds[gex_key])
        mask &= ~np.isnan(arr_gex).any(axis=1)
        del arr_gex  # Clean up large array

    valid_indices = np.where(mask)[0]
    ds_clean = ds.select(valid_indices)
    del mask  # Clean up mask
    del ds

    print(f"  After NaN filtering: {len(ds_clean)}/{original_size} samples kept")

    # Create classification dataset (filter out excluded classes) from cleaned dataset
    ds_clf = ds_clean
    if cfg.evaluation.tasks.classify:
        excluded_classes = getattr(cfg.data.annotations, "excluded_classes", [])
        if excluded_classes:
            print(f"Filtering out excluded classes: {excluded_classes}")
            try:
                # Try with reduced multiprocessing first (use fewer processes to avoid OOM)
                num_proc = min(4, len(ds_clean) // 10000)  # Scale processes with dataset size
                if num_proc < 1:
                    num_proc = 1
                print(f"Using {num_proc} processes for filtering {len(ds_clean)} samples")
                ds_clf = ds_clean.filter(lambda x: x[class_label_key] not in excluded_classes,
                                         num_proc=num_proc)
                print(f"Classification dataset: {len(ds_clf)} samples (removed {len(ds_clean) - len(ds_clf)})")
            except RuntimeError as e:
                if "subprocesses has abruptly died" in str(e) or "OOM" in str(e):
                    print(f"Multiprocessing failed, falling back to single-threaded filtering: {e}")
                    ds_clf = ds_clean.filter(lambda x: x[class_label_key] not in excluded_classes)
                    print(f"Classification dataset: {len(ds_clf)} samples (removed {len(ds_clean) - len(ds_clf)})")
                else:
                    raise e
            gc.collect()  # Clean up after heavy operation

        # Skip train class filtering for finetune models (they only have test data, so no train-only classes exist)
        is_finetune = gex_model is not None and gex_model.startswith("finetune_")
        # Also skip for pretrained models (gex_model=None) when using test datasets
        is_pretrained = gex_model is None or gex_model == "None" or str(gex_model).lower() in ["unknown", "pretrained"]

        if split_name == "train" and cfg.data.annotations.get(
            "exclude_test_only_classes",
                False) and modality == "multimodal" and not is_finetune and not is_pretrained:
            # Only apply train class filtering for multimodal mode when NOT using finetune models
            train_ids = cfg.data.multimodal.train[cfg.data.train_split]
            # Handle both donor-level and cell-level names
            train_classes = set()
            for s in ds_clean:
                sample_name = s["name"]
                # Extract donor part if it's a cell-level name (contains underscore)
                if "_" in sample_name:
                    donor_name = sample_name.split("_")[0]
                else:
                    donor_name = sample_name

                if donor_name in train_ids:
                    train_classes.add(s[class_label_key])

            try:
                ds_clf = ds_clf.filter(lambda x: x[class_label_key] in train_classes,
                                       num_proc=16)
                print(f"Kept {len(train_classes)} train classes")
            except RuntimeError as e:
                if "subprocesses has abruptly died" in str(e):
                    print(f"Multiprocessing failed, falling back to single-threaded filtering: {e}")
                    ds_clf = ds_clf.filter(lambda x: x[class_label_key] in train_classes)
                    print(f"Kept {len(train_classes)} train classes")
                else:
                    raise e
            gc.collect()  # Clean up after heavy operation
        else:
            # For unimodal mode, finetune models, or when not excluding test-only classes, use all available classes
            if is_finetune:
                print(f"Using all available classes for finetune model (train class filtering skipped)")
            else:
                print(f"Using all available classes for {modality} mode")

    # Create regression dataset (remove NOMAP and renormalize) from cleaned dataset
    # Regression should use all samples (after NaN filtering), not filtered by classification criteria
    ds_reg = ds_clean
    if cfg.evaluation.tasks.regress and "cell_types" in cfg.data:
        nomap_idx = getattr(cfg.data.cell_types, "nomap_index", None)
        n_classes = getattr(cfg.data.cell_types, "num_classes", None)

        if nomap_idx is not None:
            # Check if reg_label_key exists in dataset before processing
            if len(ds_reg) > 0 and reg_label_key not in ds_reg[0]:
                _debug_print(f"[DEBUG] Warning: reg_label_key '{reg_label_key}' not found in dataset. Skipping NOMAP removal.")
                _debug_print(f"[DEBUG] Available keys: {list(ds_reg[0].keys())}")
            else:
                print(f"Removing NOMAP (index={nomap_idx}) from regression labels")

                # Check some original labels before processing
                if len(ds_reg) > 0:
                    sample_original = ds_reg[0][reg_label_key]
                    _debug_print(
                        f"[DEBUG] Original regression label example: {sample_original[:10] if len(sample_original) > 10 else sample_original}")
                    _debug_print(f"[DEBUG] Original regression label length: {len(sample_original)}")
                    _debug_print(f"[DEBUG] Original regression label sum: {sum(sample_original):.6f}")

                def remove_nomap(example):
                    ratios = list(example[reg_label_key])
                    if 0 <= nomap_idx < len(ratios):
                        # Always remove NOMAP index to maintain consistent dimensions
                        ratios = [r for i, r in enumerate(ratios) if i != nomap_idx]
                        total = sum(ratios)
                        if total > 0:
                            # Renormalize if there are non-zero values
                            ratios = [r / total for r in ratios]
                        # If total is 0, keep the zero vector but with consistent length (9 elements)
                        # This maintains dimensional consistency for tensor creation
                    example[reg_label_key] = ratios
                    return example

                try:
                    ds_reg = ds_reg.map(remove_nomap, num_proc=16)
                except RuntimeError as e:
                    if "subprocesses has abruptly died" in str(e):
                        print(f"Multiprocessing failed, falling back to single-threaded mapping: {e}")
                        ds_reg = ds_reg.map(remove_nomap)
                    else:
                        raise e
                gc.collect()  # Clean up after heavy operation

                # Check some processed labels after processing
                if len(ds_reg) > 0:
                    sample_processed = ds_reg[0][reg_label_key]
                    _debug_print(
                        f"[DEBUG] Processed regression label example: {sample_processed[:10] if len(sample_processed) > 10 else sample_processed}")
                    _debug_print(f"[DEBUG] Processed regression label length: {len(sample_processed)}")
                    _debug_print(f"[DEBUG] Processed regression label sum: {sum(sample_processed):.6f}")

                if len(ds_reg) > 0 and n_classes and reg_label_key in ds_reg[0]:
                    expected_len = n_classes - 1
                    actual_len = len(ds_reg[0][reg_label_key])
                    if actual_len != expected_len:
                        print(f"WARNING: regression vector length {actual_len} != expected {expected_len}")

    print(f"Final dataset sizes: classification={len(ds_clf)}, regression={len(ds_reg)}")

    # Assert that we have data for the tasks we're trying to evaluate
    if cfg.evaluation.tasks.classify and len(ds_clf) == 0:
        print(f"WARNING: No classification samples available after filtering!")
        print(f"   Original dataset size: {original_size}")
        print(f"   After NaN filtering: {len(ds_clean)}")
        print(f"   After class filtering: {len(ds_clf)}")
        print(f"   Modality: {modality}, GEX model: {gex_model}, Split: {split_name}")
        raise ValueError(
            f"No classification samples available after filtering. This indicates a data or configuration issue.")
    if cfg.evaluation.tasks.regress and len(ds_reg) == 0:
        print(f"WARNING: No regression samples available after filtering!")
        print(f"   Original dataset size: {original_size}")
        print(f"   After NaN filtering: {len(ds_clean)}")
        print(f"   Modality: {modality}, GEX model: {gex_model}, Split: {split_name}")
        raise ValueError(f"No regression samples available after filtering. This indicates a data or configuration issue.")

    return ds_clf, ds_reg


def extract_embeddings(ds, valid_indices, cfg, align_method, task_type="classification"):
    """
    Extract embeddings based on the alignment method.
    Returns tensors for both image and gene expression data.

    Args:
        ds: Dataset
        valid_indices: Indices to extract
        cfg: Configuration
        align_method: Alignment method
        task_type: "classification" or "regression" to determine which labels to extract
    """
    img_key = cfg.data.img_embed_key
    gex_key = cfg.data.gex_embed_key

    # Choose the correct label key based on task type
    if task_type == "regression":
        label_key = cfg.training.regression.label_key
    else:
        label_key = cfg.training.classification.label_key
    # Handle case where valid_indices is None (use all samples in dataset)
    if valid_indices is None:
        valid_indices = list(range(len(ds)))

    if align_method == "random":
        # Generate random embeddings
        sample_img = np.array(ds[valid_indices[0]][img_key])
        sample_gex = np.array(ds[valid_indices[0]][gex_key])
        random_shape = (len(valid_indices), sample_img.shape[0] + sample_gex.shape[0])
        targets = torch.tensor(ds[label_key])[valid_indices]
        return torch.from_numpy(generate_random_embeddings(
            random_shape, random_state=cfg.training.classification.seed)).float(), targets

    elif align_method == "unimodal_gex":
        # Only gene expression embeddings - use vectorized extraction
        print(f"    Extracting GEX embeddings...")
        embeddings = np.array(ds[gex_key])[valid_indices]
        targets = torch.tensor(ds[label_key])[valid_indices]
        return torch.from_numpy(embeddings).float(), targets

    elif align_method == "unimodal_img":
        # Only image embeddings - use vectorized extraction
        print(f"    Extracting image embeddings...")
        embeddings = np.array(ds[img_key])[valid_indices]
        # Assert that img_uni_pool embeddings are 1536D for 200M model
        if img_key == "img_uni_pool":
            assert embeddings.shape[1] == 1536, (
                f"Expected img_uni_pool embeddings to be 1536D, but got {embeddings.shape[1]}D. "
                f"This indicates the dataset is using outdated embeddings."
            )
            print(f"    Verified img_uni_pool embedding dimension: {embeddings.shape[1]}D")
        targets = torch.tensor(ds[label_key])[valid_indices]
        return torch.from_numpy(embeddings).float(), targets

    elif align_method == "multimodal_concat":
        # Concatenate image and gene expression embeddings - use vectorized extraction
        print(f"    Extracting image embeddings...")
        img_emb = np.array(ds[img_key])[valid_indices]
        # Assert that img_uni_pool embeddings are 1536D for 200M model
        if img_key == "img_uni_pool":
            assert img_emb.shape[1] == 1536, (
                f"Expected img_uni_pool embeddings to be 1536D, but got {img_emb.shape[1]}D. "
                f"This indicates the dataset is using outdated embeddings."
            )
            print(f"    Verified img_uni_pool embedding dimension: {img_emb.shape[1]}D")
        print(f"    Extracting GEX embeddings...")
        gex_emb = np.array(ds[gex_key])[valid_indices]

        # Convert to tensors and concatenate
        img_tensor = torch.from_numpy(img_emb).float()
        gex_tensor = torch.from_numpy(gex_emb).float()
        targets = torch.tensor(ds[label_key])[valid_indices]
        return torch.cat([img_tensor, gex_tensor], dim=-1), targets

    elif align_method in ["multimodal_barlow_twins", "multimodal_byol", "multimodal_cca", "multimodal_comm", "multimodal_dcca", "multimodal_dim", "multimodal_simclr", "multimodal_simsiam", "multimodal_vicreg"]:
        print(f"    Extracting embeddings using {align_method}...")

        fused, targets, _names = extract_embeddings_from_fusion_model(
            cfg, ds, valid_indices, task_type=task_type, align_method=align_method)
        return fused, targets

    else:
        raise ValueError(f"Unknown method: {align_method}")


def evaluate_simple(X_train_clf,
                    X_test_clf,
                    X_train_reg,
                    X_test_reg,
                    y_train_clf,
                    y_test_clf,
                    y_train_reg,
                    y_test_reg,
                    random_state=42,
                    cfg=None):
    """
    Simplified evaluation function that only computes macro F1 and R2.
    """
    print(
        f"    Starting evaluation with shapes: X_train_clf={X_train_clf.shape}, X_test_clf={X_test_clf.shape}, X_train_reg={X_train_reg.shape}, X_test_reg={X_test_reg.shape}")

    # Check if tensors are dummy tensors (size 1) and skip those tasks
    do_classification = cfg.evaluation.tasks.classify and X_train_clf.shape[0] > 1 and X_test_clf.shape[0] > 1
    do_regression = cfg.evaluation.tasks.regress and X_train_reg.shape[0] > 1 and X_test_reg.shape[0] > 1

    if not do_classification and not do_regression:
        raise ValueError("No valid tasks to evaluate. Both classification and regression tensors are dummy tensors.")

    # Convert tensors to numpy arrays only for tasks being evaluated
    if do_classification:
        X_train_clf = X_train_clf.cpu().numpy().astype(np.float64)
        X_test_clf = X_test_clf.cpu().numpy().astype(np.float64)
    else:
        X_train_clf = X_test_clf = None

    if do_regression:
        X_train_reg = X_train_reg.cpu().numpy().astype(np.float64)
        X_test_reg = X_test_reg.cpu().numpy().astype(np.float64)
    else:
        X_train_reg = X_test_reg = None

    _debug_print(f"[DEBUG] Linear probe input shapes after conversion:")
    _debug_print(f"[DEBUG]   X_train_clf: {X_train_clf.shape if X_train_clf is not None else 'None'} (numpy)")
    _debug_print(f"[DEBUG]   X_test_clf: {X_test_clf.shape if X_test_clf is not None else 'None'} (numpy)")
    _debug_print(f"[DEBUG]   X_train_reg: {X_train_reg.shape if X_train_reg is not None else 'None'} (numpy)")
    _debug_print(f"[DEBUG]   X_test_reg: {X_test_reg.shape if X_test_reg is not None else 'None'} (numpy)")

    # Check for NaN or infinite values
    if X_train_clf is not None and (np.any(np.isnan(X_train_clf)) or np.any(np.isnan(X_test_clf))):
        print(f"    ERROR: Found NaN values in data!")
        print(f"    X_train_clf NaN count: {np.isnan(X_train_clf).sum()}")
        print(f"    X_test_clf NaN count: {np.isnan(X_test_clf).sum()}")
        raise ValueError("Input data contains NaN values!")
    if X_train_clf is not None and (np.any(np.isinf(X_train_clf)) or np.any(np.isinf(X_test_clf))):
        print(f"    ERROR: Found infinite values in data!")
        print(f"    X_train_clf inf count: {np.isinf(X_train_clf).sum()}")
        print(f"    X_test_clf inf count: {np.isinf(X_test_clf).sum()}")
        raise ValueError("Input data contains infinite values!")

    print(f"    Data validation passed - no NaN or infinite values found")

    # Initialize metrics dictionary
    metrics = {}

    # Determine task type from config
    do_classification = cfg.evaluation.tasks.classify
    do_regression = cfg.evaluation.tasks.regress

    # Scale the data
    scaler = StandardScaler()

    # Handle classification task
    if do_classification:
        y_train_clf = y_train_clf.cpu().numpy()
        y_test_clf = y_test_clf.cpu().numpy()

        # If labels are one-hot encoded or multi-dimensional, convert to class indices
        if len(y_train_clf.shape) > 1:
            if np.allclose(y_train_clf.sum(axis=1), 1.0):  # Check if one-hot encoded
                y_train_clf = np.argmax(y_train_clf, axis=1)
            else:
                raise ValueError("Multi-label classification not supported")
        if len(y_test_clf.shape) > 1:
            if np.allclose(y_test_clf.sum(axis=1), 1.0):
                y_test_clf = np.argmax(y_test_clf, axis=1)
            else:
                raise ValueError("Multi-label classification not supported")

        # Create label mapping to ensure consecutive integers starting from 0
        # Only use labels that appear in BOTH train and test sets
        train_labels = np.unique(y_train_clf)
        test_labels = np.unique(y_test_clf)
        common_labels = np.intersect1d(train_labels, test_labels)

        _debug_print(f"Class mapping debug:")
        _debug_print(f"[DEBUG]   Original train labels: {train_labels}")
        _debug_print(f"[DEBUG]   Original test labels: {test_labels}")
        _debug_print(f"[DEBUG]   Common labels: {common_labels}")
        if y_train_clf is not None and len(y_train_clf) > 0:
            _debug_print(f"[DEBUG]   Train label counts: {np.bincount(y_train_clf.astype(int))}")
        else:
            _debug_print(f"[DEBUG]   Train label counts: None (no classification samples)")
        if y_test_clf is not None and len(y_test_clf) > 0:
            _debug_print(f"[DEBUG]   Test label counts: {np.bincount(y_test_clf.astype(int))}")
        else:
            _debug_print(f"[DEBUG]   Test label counts: None (no classification samples)")

        if len(common_labels) < len(train_labels):
            print(f"    WARNING: {len(train_labels) - len(common_labels)} train-only classes will be excluded")
        if len(common_labels) < len(test_labels):
            print(f"    WARNING: {len(test_labels) - len(common_labels)} test-only classes will be excluded")

        # Create mapping only for common labels
        label_to_idx = {label: idx for idx, label in enumerate(common_labels)}
        num_classes = len(common_labels)

        print(f"    Common labels: {common_labels}")
        print(f"    Label mapping: {label_to_idx}")
        print(f"    Number of classes: {num_classes}")

        # Filter data to only include common classes
        if X_train_clf is not None and y_train_clf is not None:
            train_mask = np.isin(y_train_clf, common_labels)
            test_mask = np.isin(y_test_clf, common_labels)

            X_train_filtered = X_train_clf[train_mask]
            y_train_filtered = y_train_clf[train_mask]
            X_test_filtered = X_test_clf[test_mask]
            y_test_filtered = y_test_clf[test_mask]
        else:
            # No classification data available
            X_train_filtered = X_test_filtered = None
            y_train_filtered = y_test_filtered = None

        if X_train_filtered is not None:
            scaler = StandardScaler()
            X_train_filtered = scaler.fit_transform(X_train_filtered)
            X_test_filtered = scaler.transform(X_test_filtered)

        if X_train_filtered is not None:
            print(f"    After filtering: train={len(X_train_filtered)}, test={len(X_test_filtered)}")
            _debug_print(f"[DEBUG] After filtering shapes:")
            _debug_print(f"[DEBUG]   X_train_filtered: {X_train_filtered.shape}")
            _debug_print(f"[DEBUG]   X_test_filtered: {X_test_filtered.shape}")
        else:
            print(f"    No classification data available after filtering")
        if y_train_filtered is not None:
            _debug_print(f"[DEBUG]   y_train_filtered: {y_train_filtered.shape}")
            _debug_print(f"[DEBUG]   y_test_filtered: {y_test_filtered.shape}")

        # Check if we have enough samples
        if X_train_filtered is not None and (len(X_train_filtered) < 10 or len(X_test_filtered) < 5):
            print(f"    ERROR: Too few samples after filtering. Need at least 10 train and 5 test samples.")
            raise ValueError("Insufficient samples for evaluation")

        # Skip classification if no data available
        if X_train_filtered is None:
            print(f"    Skipping classification evaluation - no samples available")
            return {"f1_macro": None, "r2": None, "classification_skipped": True}

        # Verify shapes are consistent
        print(f"    Shape check: X_train_filtered={X_train_filtered.shape}, y_train_filtered={y_train_filtered.shape}")
        print(f"    Shape check: X_test_filtered={X_test_filtered.shape}, y_test_filtered={y_test_filtered.shape}")

        # Map labels to consecutive integers
        y_train_mapped = np.array([label_to_idx[label] for label in y_train_filtered])
        y_test_mapped = np.array([label_to_idx[label] for label in y_test_filtered])

        _debug_print(f"[DEBUG] Mapped labels:")
        _debug_print(f"[DEBUG]   y_train_mapped: {y_train_mapped.shape}, unique: {np.unique(y_train_mapped)}")
        _debug_print(f"[DEBUG]   y_test_mapped: {y_test_mapped.shape}, unique: {np.unique(y_test_mapped)}")

        # Train and evaluate classification with mapped labels
        clf = LogisticRegression(max_iter=2000, random_state=random_state)
        clf.fit(X_train_filtered, y_train_mapped)

        # Get predictions
        test_preds = clf.predict(X_test_filtered)

        # Convert predictions back to original labels for metric calculation
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        test_preds_original = np.array([idx_to_label[pred] for pred in test_preds])

        # Calculate classification metrics using ORIGINAL labels
        metrics["f1_macro"] = f1_score(y_test_filtered, test_preds_original, average='macro')
    else:
        # Explicitly set f1_macro to None when classification is disabled
        metrics["f1_macro"] = None

    # Handle regression task
    if do_regression:
        # Convert regression labels to numpy
        y_train_reg = y_train_reg.cpu().numpy()
        y_test_reg = y_test_reg.cpu().numpy()

        # Use all regression data (no classification filtering applied)
        # Regression should be evaluated independently from classification
        X_train_reg_filtered = X_train_reg
        X_test_reg_filtered = X_test_reg
        y_train_reg_filtered = y_train_reg
        y_test_reg_filtered = y_test_reg
        print(f"    Using all regression data: train={len(X_train_reg_filtered)}, test={len(X_test_reg_filtered)}")

        # Convert to numpy if needed
        X_train_reg_filtered = X_train_reg_filtered.cpu().numpy().astype(
            np.float64) if hasattr(X_train_reg_filtered, 'cpu') else X_train_reg_filtered
        X_test_reg_filtered = X_test_reg_filtered.cpu().numpy().astype(
            np.float64) if hasattr(X_test_reg_filtered, 'cpu') else X_test_reg_filtered

        scaler = StandardScaler()
        X_train_reg_filtered = scaler.fit_transform(X_train_reg_filtered)
        X_test_reg_filtered = scaler.transform(X_test_reg_filtered)

        # Debug regression labels
        _debug_print(f"[DEBUG] Regression evaluation:")
        _debug_print(f"[DEBUG]   X_train_reg_filtered: {X_train_reg_filtered.shape}")
        _debug_print(f"[DEBUG]   X_test_reg_filtered: {X_test_reg_filtered.shape}")
        _debug_print(f"[DEBUG]   y_train_reg_filtered: {y_train_reg_filtered.shape}")
        _debug_print(f"[DEBUG]   y_test_reg_filtered: {y_test_reg_filtered.shape}")
        if len(y_train_reg_filtered) > 0:
            first_label = y_train_reg_filtered[0]
            if hasattr(first_label, '__len__') and not isinstance(first_label, (int, float, np.integer, np.floating)):
                # Vector regression labels (e.g., cell type ratios)
                _debug_print(
                    f"[DEBUG]   First training regression label: {first_label[:5] if len(first_label) > 5 else first_label}")
                _debug_print(f"[DEBUG]   Training regression label sum: {np.sum(first_label):.6f}")
            else:
                # Scalar regression labels
                _debug_print(f"[DEBUG]   First training regression label (scalar): {first_label}")
                _debug_print(f"[DEBUG]   Training regression label type: {type(first_label)}")

        # Use Ridge regression
        from sklearn.linear_model import RidgeCV
        model = RidgeCV(alphas=np.logspace(-3, 3, 10), scoring="r2")

        model.fit(X_train_reg_filtered, y_train_reg_filtered)
        test_r2 = model.score(X_test_reg_filtered, y_test_reg_filtered)

        _debug_print(f"[DEBUG] Regression R² score: {test_r2:.6f}")
        metrics["r2"] = test_r2
    else:
        # Explicitly set r2 to None when regression is disabled
        metrics["r2"] = None

    return metrics


def run_embedding_baseline_eval(cfg: DictConfig, train_ds_clf, train_ds_reg, test_ds_clf, test_ds_reg, align_method):
    """Run evaluation for the selected embedding baseline method."""

    print(f"\n=== Running Embedding Baseline Evaluation ===")
    print(f"Selected method: {align_method}")
    print(f"Classification training on: train split ({len(train_ds_clf)} samples)")
    print(f"Regression training on: train split ({len(train_ds_reg)} samples)")
    print(f"Classification testing on: test split ({len(test_ds_clf)} samples)")
    print(f"Regression testing on: test split ({len(test_ds_reg)} samples)")

    # Use separate indices for classification and regression datasets
    # Classification and regression datasets may have different sizes after filtering
    train_valid_indices_clf = list(range(len(train_ds_clf)))
    test_valid_indices_clf = list(range(len(test_ds_clf)))
    train_valid_indices_reg = list(range(len(train_ds_reg)))
    test_valid_indices_reg = list(range(len(test_ds_reg)))

    print(f"  Using samples: train_clf={len(train_valid_indices_clf)}, test_clf={len(test_valid_indices_clf)}")
    print(f"  Using samples: train_reg={len(train_valid_indices_reg)}, test_reg={len(test_valid_indices_reg)}")

    # Extract embeddings for training data
    print(f"  Extracting training embeddings...")
    train_data_clf, train_clf_labels = extract_embeddings(
        ds=train_ds_clf, valid_indices=train_valid_indices_clf, cfg=cfg, align_method=align_method, task_type="classification")
    if cfg.evaluation.tasks.regress:
        train_data_reg, train_reg_labels = extract_embeddings(
            ds=train_ds_reg, valid_indices=train_valid_indices_reg, cfg=cfg, align_method=align_method, task_type="regression")
    else:
        train_data_reg, train_reg_labels = torch.zeros(1, 1), torch.zeros(1, 1)

    # Extract embeddings for test data
    print(f"  Extracting test embeddings...")
    test_data_clf, test_clf_labels = extract_embeddings(
        ds=test_ds_clf, valid_indices=test_valid_indices_clf, cfg=cfg, align_method=align_method, task_type="classification")
    if cfg.evaluation.tasks.regress:
        test_data_reg, test_reg_labels = extract_embeddings(
            ds=test_ds_reg, valid_indices=test_valid_indices_reg, cfg=cfg, align_method=align_method, task_type="regression")
    else:
        test_data_reg, test_reg_labels = torch.zeros(1, 1), torch.zeros(1, 1)

    # Evaluate
    metrics = evaluate_simple(
        train_data_clf,
        test_data_clf,
        train_data_reg,
        test_data_reg,
        train_clf_labels,
        test_clf_labels,
        train_reg_labels,
        test_reg_labels,
        random_state=cfg.evaluation.seed,
        cfg=cfg
    )

    return {
        'method': align_method,
        'f1_macro': metrics.get('f1_macro'),
        'r2': metrics.get('r2')
    }


def run_cv_evaluation(cfg: DictConfig, train_ds_clf, train_ds_reg, align_method):
    """Run cross-validation evaluation for the selected embedding baseline method."""

    print(f"\n=== Running Cross-Validation Evaluation ===")
    print(f"Selected method: {align_method}")
    print(f"Training on: train split ({len(train_ds_clf)} samples)")
    print(f"Cross-validation folds: {cfg.evaluation.n_cv_folds}")

    # Use fixed seed (42) for all CV splits to ensure consistency across all evaluations
    cv_seed = 42
    if hasattr(cfg.evaluation, 'seed') and cfg.evaluation.seed is not None:
        cv_seed = cfg.evaluation.seed
    print(f"Using CV seed: {cv_seed} (ensures consistent splits across all evaluations)")

    # Create directory for saving CV splits
    splits_save_dir = PROJECT_DIR / "results" / cfg.data.dataset / "cv_splits"

    # Create separate CV splits for classification and regression datasets
    # They may have different sizes after filtering
    # Save/load splits for reproducibility
    cv_splits_clf = create_cv_splits(len(train_ds_clf), cfg.evaluation.n_cv_folds, cv_seed, save_path=splits_save_dir)
    cv_splits_reg = create_cv_splits(len(train_ds_reg), cfg.evaluation.n_cv_folds, cv_seed, save_path=splits_save_dir)

    # Store results for each fold
    fold_results = []

    for fold_idx in range(cfg.evaluation.n_cv_folds):
        print(f"\n  --- Fold {fold_idx + 1}/{cfg.evaluation.n_cv_folds} ---")

        # Get CV splits for both datasets
        fold_splits_clf = cv_splits_clf[fold_idx]
        fold_splits_reg = cv_splits_reg[fold_idx]

        # For each fold, we'll use one split as train and the other as validation
        train_indices_clf = fold_splits_clf[0]  # First split, train indices
        val_indices_clf = fold_splits_clf[1]    # First split, validation indices
        train_indices_reg = fold_splits_reg[0]  # First split, train indices
        val_indices_reg = fold_splits_reg[1]    # First split, validation indices

        print(f"    Train samples: clf={len(train_indices_clf)}, reg={len(train_indices_reg)}")
        print(f"    Validation samples: clf={len(val_indices_clf)}, reg={len(val_indices_reg)}")

        # Extract embeddings for this fold (fast, batch style)
        print(f"    Extracting training embeddings...")
        train_data_clf, train_clf_labels = extract_embeddings(ds=train_ds_clf.select(train_indices_clf),
                                                              valid_indices=None,
                                                              cfg=cfg,
                                                              align_method=align_method,
                                                              task_type="classification")
        if cfg.evaluation.tasks.regress:
            train_data_reg, train_reg_labels = extract_embeddings(ds=train_ds_reg.select(train_indices_reg),
                                                                  valid_indices=None,
                                                                  cfg=cfg,
                                                                  align_method=align_method,
                                                                  task_type="regression")
        else:
            train_data_reg, train_reg_labels = torch.zeros(1, 1), torch.zeros(1, 1)

        print(f"    Extracting validation embeddings...")
        val_data_clf, val_clf_labels = extract_embeddings(ds=train_ds_clf.select(val_indices_clf),
                                                          valid_indices=None,
                                                          cfg=cfg,
                                                          align_method=align_method,
                                                          task_type="classification")
        if cfg.evaluation.tasks.regress:
            val_data_reg, val_reg_labels = extract_embeddings(ds=train_ds_reg.select(val_indices_reg),
                                                              valid_indices=None,
                                                              cfg=cfg,
                                                              align_method=align_method,
                                                              task_type="regression")
        else:
            val_data_reg, val_reg_labels = torch.zeros(1, 1), torch.zeros(1, 1)

        # Evaluate this fold
        fold_metrics = evaluate_simple(
            train_data_clf,
            val_data_clf,
            train_data_reg,
            val_data_reg,
            train_clf_labels,
            val_clf_labels,
            train_reg_labels,
            val_reg_labels,
            random_state=cfg.evaluation.seed + fold_idx,
            cfg=cfg
        )

        fold_results.append(fold_metrics)
        f1_val = fold_metrics.get('f1_macro')
        r2_val = fold_metrics.get('r2')
        f1_str = f"{f1_val:.4f}" if f1_val is not None and isinstance(f1_val, (int, float)) else "N/A"
        r2_str = f"{r2_val:.4f}" if r2_val is not None and isinstance(r2_val, (int, float)) else "N/A"
        print(f"    Fold {fold_idx + 1} results: F1={f1_str}, R²={r2_str}")

    # Calculate mean and std across folds
    cv_summary = {}

    if cfg.evaluation.tasks.classify:
        f1_scores = [r.get('f1_macro') for r in fold_results if r.get('f1_macro') is not None]
        if f1_scores:
            cv_summary['f1_macro_mean'] = np.mean(f1_scores)
            cv_summary['f1_macro_std'] = np.std(f1_scores)
            cv_summary['f1_macro_folds'] = f1_scores

    if cfg.evaluation.tasks.regress:
        r2_scores = [r.get('r2') for r in fold_results if r.get('r2') is not None]
        if r2_scores:
            cv_summary['r2_mean'] = np.mean(r2_scores)
            cv_summary['r2_std'] = np.std(r2_scores)
            cv_summary['r2_folds'] = r2_scores

    return {
        'method': align_method,
        'cv_folds': cfg.evaluation.n_cv_folds,
        'fold_results': fold_results,
        **cv_summary
    }


def print_results_table(result):
    """Print results in a nice table format."""
    print("\n" + "=" * 80)

    # Check if this is cross-validation results
    if 'cv_folds' in result:
        print("CROSS-VALIDATION EVALUATION RESULTS")
        print(f"Method: {result['method']} ({result['cv_folds']} folds)")
    else:
        print("EMBEDDING BASELINE EVALUATION RESULTS")
        print(f"Method: {result['method']}")

    print("=" * 80)

    if 'cv_folds' in result:
        # Cross-validation results
        if 'f1_macro_mean' in result:
            f1_mean = f"{result['f1_macro_mean']:.4f}"
            f1_std = f"{result['f1_macro_std']:.4f}"
            f1_str = f"{f1_mean} ± {f1_std}"
        else:
            f1_str = "N/A"

        if 'r2_mean' in result:
            r2_mean = f"{result['r2_mean']:.4f}"
            r2_std = f"{result['r2_std']:.4f}"
            r2_str = f"{r2_mean} ± {r2_std}"
        else:
            r2_str = "N/A"

        # Print summary table
        headers = ["Method", "F1 Macro (Mean ± Std)", "R² (Mean ± Std)"]
        table_data = [[result['method'], f1_str, r2_str]]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Print individual fold results
        print(f"\nIndividual Fold Results:")
        fold_headers = ["Fold", "F1 Macro", "R²"]
        fold_data = []
        for i, fold_result in enumerate(result['fold_results']):
            f1_fold = f"{fold_result.get('f1_macro', 'N/A'):.4f}" if fold_result.get('f1_macro') is not None else "N/A"
            r2_fold = f"{fold_result.get('r2', 'N/A'):.4f}" if fold_result.get('r2') is not None else "N/A"
            fold_data.append([f"Fold {i+1}", f1_fold, r2_fold])
        print(tabulate(fold_data, headers=fold_headers, tablefmt="simple"))

    else:
        # Single evaluation results
        f1_str = f"{result['f1_macro']:.4f}" if result['f1_macro'] is not None else "N/A"
        r2_str = f"{result['r2']:.4f}" if result['r2'] is not None else "N/A"

        headers = ["Method", "F1 Macro", "R²"]
        table_data = [[result['method'], f1_str, r2_str]]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("=" * 80)


def save_results(results_dir: Path, cfg: DictConfig, result: dict):
    """Save evaluation results to JSON."""

    # Determine evaluation strategy
    if 'cv_folds' in result:
        evaluation_strategy = "cross_validation_eval"
        train_split = "train"  # CV only uses train split
        test_split = "cv_folds"
    else:
        evaluation_strategy = "embedding_baseline_eval"
        train_split = "train"
        test_split = "test"

    # Get checkpoint/model paths for reproducibility
    checkpoint_info = {}
    if hasattr(cfg.models, 'checkpoint_path') and cfg.models.checkpoint_path:
        checkpoint_info["checkpoint_path"] = str(cfg.models.checkpoint_path)
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'dataset_name') and cfg.data.dataset_name:
        checkpoint_info["dataset_name"] = cfg.data.dataset_name

    result_data = {
        "experiment": {
            "modality": cfg.evaluation.modality,
            "img_model": cfg.evaluation.img_model,
            "gex_model": cfg.evaluation.gex_model,
            "dataset": cfg.data.dataset,
            "evaluation_strategy": evaluation_strategy,
            "train_split": train_split,
            "test_split": test_split,
            "align_method": result['method'],
            "tasks": {
                "classification": cfg.evaluation.tasks.classify,
                "regression": cfg.evaluation.tasks.regress},
            "reproducibility": {
                "random_seed": cfg.evaluation.seed if hasattr(
                    cfg.evaluation,
                    'seed') else None,
                "dataset_path": str(
                    cfg.data.dataset) if hasattr(
                    cfg.data,
                    'dataset') else (
                    str(
                        cfg.data.dataset_name) if hasattr(
                        cfg.data,
                        'dataset_name') and cfg.data.dataset_name else None),
                **checkpoint_info}},
        "results": [result]}

    # Add cross-validation specific info
    if 'cv_folds' in result:
        result_data["experiment"]["cv_folds"] = result['cv_folds']
        result_data["experiment"]["cv_strategy"] = "k_fold"
        result_data["experiment"]["reproducibility"]["cv_method"] = "sklearn.KFold"

    result_data = make_serializable(result_data)

    # Save to file
    results_dir.mkdir(parents=True, exist_ok=True)

    # Include gex_model in filename to avoid overwriting results from different subsets
    gex_model_safe = cfg.evaluation.gex_model.replace(
        '/',
        '_').replace(
        '\\',
        '_') if cfg.evaluation.gex_model else "unknown"

    # Include checkpoint seed in filename if specified
    checkpoint_seed_suffix = ""
    if hasattr(cfg.evaluation, 'checkpoint_seed') and cfg.evaluation.checkpoint_seed is not None:
        checkpoint_seed_suffix = f"_seed{cfg.evaluation.checkpoint_seed}"

    # Include checkpoint version/config in filename if specified (e.g., cfg4, cfg5, cfg6, comm-v3, etc.)
    checkpoint_config_suffix = ""
    if hasattr(cfg.evaluation, 'checkpoint_config') and cfg.evaluation.checkpoint_config is not None:
        checkpoint_config_suffix = f"_{cfg.evaluation.checkpoint_config}"
    elif hasattr(cfg.evaluation, 'checkpoint_version') and cfg.evaluation.checkpoint_version is not None:
        checkpoint_config_suffix = f"_{cfg.evaluation.checkpoint_version}"

    if 'cv_folds' in result:
        json_path = results_dir / \
            f"cv_eval_{result['method']}_{gex_model_safe}{checkpoint_seed_suffix}{checkpoint_config_suffix}_results.json"
    else:
        json_path = results_dir / \
            f"embedding_baseline_{result['method']}_{gex_model_safe}{checkpoint_seed_suffix}{checkpoint_config_suffix}_results.json"

    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to JSON: {json_path}")


def is_finetune_model(gex_model: str) -> bool:
    """Check if the GEX model is a finetune model that has seen test data."""
    return gex_model is not None and (gex_model.startswith("finetune_") or "testset_aligned" in gex_model)


@hydra.main(config_path='../../../configs', config_name='downstream.yaml')
def main(cfg: DictConfig):
    # Disable regression for thymus and breast datasets (they don't have cell_type_ratio column)
    dataset_name = getattr(cfg.data, 'dataset', None)
    if dataset_name in ['thymus', 'breast']:
        if cfg.evaluation.tasks.regress:
            print(f"Disabling regression task for {dataset_name} dataset (no cell_type_ratio column available)")
            cfg.evaluation.tasks.regress = False

    # Get selected method from config
    align_method = getattr(cfg.evaluation, 'align_method', None)

    if align_method is None:
        # Fallback: try to infer from modality
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
            raise ValueError(f"Cannot infer method from modality: {modality}")

    # Validate selected method
    valid_methods = [
        'random',
        'unimodal_gex',
        'unimodal_img',
        'multimodal_concat',
        'multimodal_comm',
        'multimodal_barlow_twins',
        'multimodal_byol',
        'multimodal_cca',
        'multimodal_dcca',
        'multimodal_dim',
        'multimodal_simclr',
        'multimodal_simsiam',
        'multimodal_vicreg']
    if align_method not in valid_methods:
        raise ValueError(f"Invalid method: {align_method}. Must be one of: {valid_methods}")

    # Check if this is a finetune model that has seen test data
    is_finetune = is_finetune_model(cfg.evaluation.gex_model)
    # For finetune models, we'll do CV on test data (see below), so don't disable CV here
    # Instead, we handle finetune models specially in the evaluation section
    if is_finetune and cfg.evaluation.cross_validation:
        print("ℹ️  Detected finetune model - will use cross-validation on test data.")
        print("   This provides robust evaluation while using the available test dataset.")

        # Use fixed seed (42) for all evaluations to ensure consistency
        # This ensures CV splits are identical across all evaluations
        cfg.evaluation.seed = 42
        print(f"   Using fixed seed 42 for consistent CV splits across all evaluations.")

    print("Starting Embedding Baseline Evaluation")
    print(f"Configuration:")
    print(f"  - Modality: {cfg.evaluation.modality}")
    print(f"  - Image model: {cfg.evaluation.img_model}")
    print(f"  - GEX model: {cfg.evaluation.gex_model}")
    print(f"  - Selected method: {align_method}")
    print(f"  - Is finetune model: {is_finetune}")
    print(f"  - Cross-validation: {cfg.evaluation.cross_validation}")
    if cfg.evaluation.cross_validation:
        print(f"  - CV folds: {cfg.evaluation.n_cv_folds}")
        print(f"  - Evaluation strategy: Cross-Validation on train split")
    else:
        print(f"  - Evaluation strategy: Train-Test split evaluation")
        if is_finetune:
            print(f"  - Note: Finetune model has seen test data during embedding generation")
            print(f"  - This evaluation uses the same train-test split as continued training")
    print(f"  - Tasks: classify={cfg.evaluation.tasks.classify}, regress={cfg.evaluation.tasks.regress}")
    # Ensure fixed seed (42) for consistency
    if not hasattr(cfg.evaluation, 'seed') or cfg.evaluation.seed is None:
        cfg.evaluation.seed = 42
    print(f"  - Evaluation seed: {cfg.evaluation.seed} (fixed for consistency)")

    # Setup results directory
    results_dir = PROJECT_DIR / "results" / cfg.data.dataset / f"embedding_baseline_{cfg.evaluation.img_model}"

    if cfg.evaluation.cross_validation:
        # For finetune/testset_aligned models, use test split for CV (they only have test data)
        if is_finetune_model(cfg.evaluation.gex_model):
            print("Detected finetune/testset_aligned model - using test split for cross-validation.")
            print("   This avoids data leakage since these models were trained on test data.")
            cv_ds_clf, cv_ds_reg = load_and_preprocess_data(cfg, "test", align_method)
        else:
            # Load only train data for cross-validation (standard case)
            cv_ds_clf, cv_ds_reg = load_and_preprocess_data(cfg, "train", align_method)

        # Run cross-validation evaluation
        result = run_cv_evaluation(cfg, cv_ds_clf, cv_ds_reg, align_method)
    else:
        # Load train and test data for single evaluation
        train_ds_clf, train_ds_reg = load_and_preprocess_data(cfg, "train", align_method)
        test_ds_clf, test_ds_reg = load_and_preprocess_data(cfg, "test", align_method)

        # If we're evaluating finetune models, they only have test data available
        # For finetune models, use 5-fold CV on the test data instead of single train-test split
        if is_finetune_model(cfg.evaluation.gex_model):
            print("Detected finetune model - only test data available.")
            print("   Using 5-fold cross-validation on test data for robust evaluation.")
            print(f"   Test dataset size: {len(test_ds_clf)} samples")

            # Ensure we use 5-fold CV (override any setting)
            original_cv_folds = getattr(cfg.evaluation, 'n_cv_folds', 5)
            cfg.evaluation.n_cv_folds = 5
            if original_cv_folds != 5:
                print(f"   Setting CV folds to 5 (was {original_cv_folds})")

            # Check if train and test are identical (they should be for finetune models)
            if len(train_ds_clf) == len(test_ds_clf) and len(train_ds_clf) > 0:
                print(f"   Train and test datasets are identical - using test data for CV")
                # Use test data as the single dataset for CV
                cv_ds_clf = test_ds_clf
                cv_ds_reg = test_ds_reg
            else:
                # Fallback: use test data
                cv_ds_clf = test_ds_clf
                cv_ds_reg = test_ds_reg

            # Run cross-validation evaluation on test data
            result = run_cv_evaluation(cfg, cv_ds_clf, cv_ds_reg, align_method)
        else:
            # Run single evaluation for non-finetune models
            result = run_embedding_baseline_eval(
                cfg, train_ds_clf, train_ds_reg, test_ds_clf, test_ds_reg, align_method)

    # Print and save results
    print_results_table(result)
    save_results(results_dir, cfg, result)

    if cfg.evaluation.cross_validation:
        print("\n=== Cross-Validation Evaluation Complete ===")
    else:
        print("\n=== Embedding Baseline Evaluation Complete ===")


if __name__ == "__main__":
    main()
