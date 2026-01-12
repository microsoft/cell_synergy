"""Path configuration module for cell_synergy package.

This module defines standard paths used throughout the package:
- ROOT: Package root directory
- PROJECT_DIR: Project folder containing data and models
- CONFIG_DIR: Configuration files directory
- UNI_EMBEDDINGS_DIR: Unimodal embeddings storage
- MODEL_DIR: Trained models storage
"""
from pathlib import Path

# Adjust for the new package structure
ROOT = Path(__file__).parent.parent.resolve()
PROJECT_DIR = ROOT.parent / "project_folder"
CONFIG_DIR = ROOT.parent / "configs"
UNI_EMBEDDINGS_DIR = PROJECT_DIR / "unimodal_embeddings" if PROJECT_DIR.exists() else None
MODEL_DIR = PROJECT_DIR / "trained_models" if PROJECT_DIR.exists() else None

# Only assert paths that should always exist
assert CONFIG_DIR.exists(), f"Config directory does not exist: {CONFIG_DIR}"
assert ROOT.exists(), f"Root directory does not exist: {ROOT}"
