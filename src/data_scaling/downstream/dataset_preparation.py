# prepare_tensor_datasets.py

import os
from pathlib import Path
import torch
from datasets import load_from_disk
from torch.utils.data import TensorDataset
import pickle
import numpy as np
from typing import Dict, List

def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/Projects/till_richter/")
    return Path(os.path.expandvars(raw))

def donor_from_name(name: str) -> str:
    return name.split("_")[0]

def convert_to_tensor_dataset(samples: List[Dict]) -> TensorDataset:
    # Check embedding dimensions and log them
    img_embeds = [s["img_embed"] for s in samples]
    gex_embeds = [s["gex_embed"] for s in samples]
    
    # Check for dimension consistency and log
    img_dims = [len(embed) for embed in img_embeds]
    gex_dims = [len(embed) for embed in gex_embeds]
    
    unique_img_dims = set(img_dims)
    unique_gex_dims = set(gex_dims)
    
    print(f"📊 Image embedding dimensions found: {unique_img_dims}")
    print(f"📊 Gene expression embedding dimensions found: {unique_gex_dims}")
    
    # Handle inconsistent dimensions by padding/truncating to the most common size
    if len(unique_img_dims) > 1:
        most_common_img_dim = max(unique_img_dims, key=img_dims.count)
        print(f"⚠️  Multiple image embedding dimensions detected. Using most common: {most_common_img_dim}")
        
        # Pad or truncate to the most common dimension
        processed_img_embeds = []
        for embed in img_embeds:
            if len(embed) > most_common_img_dim:
                # Truncate
                processed_img_embeds.append(embed[:most_common_img_dim])
            elif len(embed) < most_common_img_dim:
                # Pad with zeros
                padded = embed + [0.0] * (most_common_img_dim - len(embed))
                processed_img_embeds.append(padded)
            else:
                processed_img_embeds.append(embed)
        img_embeds = processed_img_embeds
    
    if len(unique_gex_dims) > 1:
        most_common_gex_dim = max(unique_gex_dims, key=gex_dims.count)
        print(f"⚠️  Multiple gene expression embedding dimensions detected. Using most common: {most_common_gex_dim}")
        
        # Pad or truncate to the most common dimension
        processed_gex_embeds = []
        for embed in gex_embeds:
            if len(embed) > most_common_gex_dim:
                # Truncate
                processed_gex_embeds.append(embed[:most_common_gex_dim])
            elif len(embed) < most_common_gex_dim:
                # Pad with zeros
                padded = embed + [0.0] * (most_common_gex_dim - len(embed))
                processed_gex_embeds.append(padded)
            else:
                processed_gex_embeds.append(embed)
        gex_embeds = processed_gex_embeds
    
    x_img = torch.tensor(img_embeds)
    x_gex = torch.tensor(gex_embeds)
    y = torch.tensor([s["label"] for s in samples])
    
    print(f"✅ Created tensors with shapes: img={x_img.shape}, gex={x_gex.shape}, labels={y.shape}")
    
    return TensorDataset(x_img, x_gex, y)

def save_tensor_dataset(ds: TensorDataset, path: Path):
    with open(path, "wb") as f:
        pickle.dump(ds, f)

def load_tensor_dataset(path: Path) -> TensorDataset:
    with open(path, "rb") as f:
        ds = pickle.load(f)
    print(f"📁 Loaded TensorDataset from {path.name} with {len(ds)} samples")
    if len(ds) > 0:
        sample_tensors = ds[0]
        print(f"   - Image tensor shape: {sample_tensors[0].shape}")
        print(f"   - Gene expression tensor shape: {sample_tensors[1].shape}")
    return ds

def prepare_donor_tensor_datasets(
    dataset_name: str,
    test_donors: List[str],
    embed_keys: Dict[str, str],
    label_key: str,
    num_proc: int = 4
) -> Dict[str, TensorDataset]:

    project_dir = get_project_dir()
    tensor_dir = project_dir / f"{dataset_name}_test_tds"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    donor_paths = {donor: tensor_dir / f"{donor}.pkl" for donor in test_donors}

    # Check if already done
    if all(p.exists() for p in donor_paths.values()):
        print("✅ Found existing TensorDatasets. Loading...")
        return {d: load_tensor_dataset(p) for d, p in donor_paths.items()}

    print("⚙️ Loading full HF dataset and filtering for test donors...")
    ds = load_from_disk(str(project_dir / f"{dataset_name}_hf"))

    def donor_filter_fn(example):
        return donor_from_name(example["name"]) in test_donors

    filtered = ds.filter(donor_filter_fn, num_proc=num_proc)
    print(f"✅ Filtered dataset to {len(filtered)} samples")

    donor_data = {donor: [] for donor in test_donors}
    for sample in filtered:
        donor = donor_from_name(sample["name"])
        donor_data[donor].append({
            "img_embed": sample[embed_keys["img"]],
            "gex_embed": sample[embed_keys["gex"]],
            "label": sample[label_key],
        })

    result = {}
    for donor, samples in donor_data.items():
        print(f"📦 Saving donor {donor} with {len(samples)} samples")
        ds_tensor = convert_to_tensor_dataset(samples)
        save_tensor_dataset(ds_tensor, donor_paths[donor])
        result[donor] = ds_tensor

    return result
