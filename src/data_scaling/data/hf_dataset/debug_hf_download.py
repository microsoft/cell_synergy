"""
Debug script to test HF cache download and dataset loading.
This runs before the main training script in your YAML.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from datasets import load_dataset, load_from_disk

def run_azcopy_download():
    """Download HF cache from Azure Blob Storage using azcopy."""
    
    # Configuration from environment
    azcopy_link = os.getenv('AZCOPY_LINK', '')
    azure_container_url = "https://exvivohoteastus.blob.core.windows.net/projects/Projects/till_richter/hf_cache_lung"
    local_cache_dir = "/mnt/projects/hf_cache/lung"
    
    print(f"🔽 Downloading HF cache from Azure...")
    print(f"Azure URL: {azure_container_url}")
    print(f"Local cache dir: {local_cache_dir}")
    
    # Create local cache directory
    os.makedirs(local_cache_dir, exist_ok=True)
    
    # Build azcopy command
    azcopy_cmd = [
        "azcopy", "copy", 
        f"{azure_container_url}/*?{azcopy_link}",
        local_cache_dir,
        "--recursive=true",
        "--overwrite=true"
    ]
    
    try:
        print(f"Running: {' '.join(azcopy_cmd[:3])} [URL_WITH_SAS] {azcopy_cmd[4:]}")
        result = subprocess.run(azcopy_cmd, capture_output=True, text=True, check=True)
        print("✅ Download successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        return False
    
    return True

def verify_cache_structure():
    """Verify that the downloaded cache has the expected structure."""
    
    cache_dir = "/mnt/projects/hf_cache/lung"
    
    print(f"\n🔍 Verifying cache structure...")
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return False
    
    print(f"✅ Cache directory exists: {cache_dir}")
    
    # Check for main dataset cache
    main_cache = f"{cache_dir}/microsoft___multimodal_img_st"
    if os.path.exists(main_cache):
        print(f"✅ Main dataset cache found: {main_cache}")
    else:
        print(f"⚠️  Main dataset cache not found: {main_cache}")
    
    # Check for subset caches
    for scale in ["s", "m", "l"]:
        subset_dir = f"{cache_dir}/{scale}_subset"
        if os.path.exists(subset_dir):
            print(f"✅ {scale.upper()} subset found: {subset_dir}")
        else:
            print(f"⚠️  {scale.upper()} subset not found: {subset_dir}")
    
    # Show directory structure
    print(f"\n📁 Cache directory contents:")
    for root, dirs, files in os.walk(cache_dir):
        level = root.replace(cache_dir, '').count(os.sep)
        if level <= 2:  # Don't go too deep
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
    
    return True

def test_dataset_loading():
    """Test loading the dataset to ensure everything works."""
    
    print(f"\n🧪 Testing dataset loading...")
    
    # Set environment variables
    cache_dir = "/mnt/projects/hf_cache/lung"
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    
    temp_dir = "/mnt/projects/hf_cache/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["TMPDIR"] = temp_dir
    tempfile.tempdir = temp_dir
    
    hf_token = os.getenv('HF_TOKEN')
    
    # Test loading each subset
    for scale in ["S", "M", "L"]:
        print(f"\n🔄 Testing {scale} subset...")
        
        subset_dir = f"{cache_dir}/{scale.lower()}_subset"
        
        try:
            if os.path.exists(subset_dir):
                print(f"Loading pre-filtered {scale} subset...")
                dataset = load_from_disk(subset_dir)
            else:
                print(f"Loading full dataset and filtering for {scale}...")
                dataset = load_dataset(
                    "microsoft/multimodal_img_st",
                    name="default",
                    split="train",
                    streaming=False,
                    cache_dir=cache_dir,
                    token=hf_token
                )
                dataset = dataset.filter(lambda x: x["name"].startswith(scale))
            
            print(f"✅ {scale} dataset loaded successfully!")
            print(f"   Length: {len(dataset)}")
            
            # Test first item
            first_item = dataset[0]
            print(f"   Sample keys: {list(first_item.keys())}")
            print(f"   Sample name: {first_item.get('name', 'N/A')}")
            
            if 'img_uni_pool' in first_item:
                img_shape = len(first_item['img_uni_pool']) if isinstance(first_item['img_uni_pool'], list) else "Unknown"
                print(f"   img_uni_pool shape: {img_shape}")
            
            if 'nicheformer_pool' in first_item:
                gex_shape = len(first_item['nicheformer_pool']) if isinstance(first_item['nicheformer_pool'], list) else "Unknown"
                print(f"   nicheformer_pool shape: {gex_shape}")
                
        except Exception as e:
            print(f"❌ Failed to load {scale} dataset: {e}")
            continue

def main():
    """Main debug function."""
    
    print("🐛 Debug: Testing HF cache setup...")
    print("="*50)
    
    # Step 1: Download cache from Azure
    download_success = run_azcopy_download()
    if not download_success:
        print("❌ Download failed, cannot proceed with testing")
        return
    
    # Step 2: Verify cache structure
    verify_cache_structure()
    
    # Step 3: Test dataset loading
    test_dataset_loading()
    
    print("\n🎉 Debug complete!")
    print("="*50)

if __name__ == "__main__":
    main()