"""
Step 1: One-time HuggingFace dataset download and cache creation
Run this script once on a machine with internet and HF token to create the dataset cache.
"""
import os
import tempfile
from datasets import load_dataset, Dataset
from pathlib import Path

def download_and_cache_dataset():
    """Download the full HuggingFace dataset and create local cache."""
    
    # Configuration
    dataset_name = "microsoft/multimodal_img_st"
    local_cache_dir = "/tmp/hf_cache_lung"  # Local cache directory
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set!")
    
    # Create cache directory
    os.makedirs(local_cache_dir, exist_ok=True)
    
    # Set environment variables for HuggingFace
    os.environ["HF_DATASETS_CACHE"] = local_cache_dir
    os.environ["HF_HOME"] = local_cache_dir
    
    # Create temp directory
    temp_dir = "/tmp/hf_cache_temp"
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["TMPDIR"] = temp_dir
    tempfile.tempdir = temp_dir
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Cache directory: {local_cache_dir}")
    print(f"HF Token: {'✓ Available' if hf_token else '✗ Not found'}")
    
    # Download the full dataset with streaming=False to populate cache
    dataset = load_dataset(
        path=dataset_name,
        name="default",
        split="train",
        streaming=False,  # THIS IS IMPORTANT - forces local caching
        cache_dir=local_cache_dir,
        token=hf_token
    )
    
    print(f"✓ Dataset downloaded successfully!")
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    
    # Print some sample data info
    first_item = dataset[0]
    print(f"Sample item keys: {list(first_item.keys())}")
    if "name" in first_item:
        print(f"Sample name: {first_item['name']}")
    
    # Optional: Create filtered subsets for S, M, L
    create_filtered_subsets = True
    if create_filtered_subsets:
        print("\n📦 Creating filtered subsets...")
        
        for scale in ["S", "M", "L"]:
            print(f"Filtering {scale} subset...")
            filtered = dataset.filter(lambda x: x["name"].startswith(scale + "_"))
            subset_dir = f"{local_cache_dir}/{scale.lower()}_subset"
            
            print(f"Saving {scale} subset to {subset_dir}")
            print(f"{scale} subset length: {len(filtered)}")
            filtered.save_to_disk(subset_dir)
            print(f"✓ {scale} subset saved")
    
    print(f"\n🎉 Download complete!")
    print(f"Cache structure:")
    for root, dirs, files in os.walk(local_cache_dir):
        level = root.replace(local_cache_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Upload cache to Azure: azcopy copy '{local_cache_dir}/*' 'https://...your-blob-url...' --recursive=true")
    print(f"2. In your training job, download cache from Azure")
    print(f"3. Use streaming=False in your training code")
    
    return local_cache_dir

if __name__ == "__main__":
    download_and_cache_dataset()