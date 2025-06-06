from pathlib import Path
import os

def get_project_dir():
    return Path(os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/Projects/till_richter/"))

def get_data_dir():
    return Path("nf_4000donors/")

PROJECT_DIR = get_project_dir() # Keep this for existing usages that expect a variable
print(f"--- [paths.py] PROJECT_DIR set to: {PROJECT_DIR}")
DATA_DIR = get_data_dir() # Keep this for existing usages that expect a variable
print(f"--- [paths.py] DATA_DIR set to: {DATA_DIR}") 
UNI_EMBEDDINGS_DIR = PROJECT_DIR / "unimodal_embeddings"
print(f"--- [paths.py] UNI_EMBEDDINGS_DIR set to: {UNI_EMBEDDINGS_DIR}")
MODEL_DIR = PROJECT_DIR / "trained_models"
print(f"--- [paths.py] MODEL_DIR set to: {MODEL_DIR}")