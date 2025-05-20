from pathlib import Path

# Adjust for the new package structure
ROOT = Path(__file__).parent.parent.resolve()
PROJECT_DIR = ROOT / "project_folder" 
CONFIG_DIR = ROOT / "configs"
UNI_EMBEDDINGS_DIR = PROJECT_DIR / "unimodal_embeddings"
MODEL_DIR = PROJECT_DIR / "trained_models"