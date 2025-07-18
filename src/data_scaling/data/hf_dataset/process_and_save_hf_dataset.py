import os
import time
from datasets import load_dataset, Dataset as HFDataset

def process_and_save_large_hf_dataset():
    dataset_name = "microsoft/multimodal_img_st"
    hf_token = os.getenv("HF_TOKEN")
    output_root = "/mnt/projects/Projects/till_richter/hf_dataset"
    max_samples_per_scale = 10000
    batch_size = 1000

    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not set!")

    os.makedirs(output_root, exist_ok=True)

    print(f"📥 Streaming dataset: {dataset_name}")
    print("⏳ Loading dataset iterator (this may take a while for large datasets)...")
    
    start_time = time.time()
    dataset = load_dataset(
        path=dataset_name,
        name="default",
        split="train",
        streaming=True,
        token=hf_token
    )
    
    load_time = time.time() - start_time
    print(f"✓ Dataset iterator loaded in {load_time:.1f}s")

    scale_buffers = {"S": [], "M": [], "L": []}
    scale_counts = {"S": 0, "M": 0, "L": 0}
    scale_dirs = {s: os.path.join(output_root, f"{s.lower()}_subset") for s in "SML"}

    for s in scale_dirs:
        os.makedirs(scale_dirs[s], exist_ok=True)

    print("🚀 Processing and saving in batches...")
    print("⏳ Starting to iterate through dataset (first item may take time)...")
    
    iteration_start = time.time()
    last_progress_time = time.time()
    
    for i, item in enumerate(dataset):
        # Progress reporting
        current_time = time.time()
        if i == 0:
            first_item_time = current_time - iteration_start
            print(f"✓ First item received after {first_item_time:.1f}s")
        
        # Progress every 1000 items or every 30 seconds
        if i % 1000 == 0 or (current_time - last_progress_time) > 30:
            elapsed = current_time - iteration_start
            items_per_sec = i / elapsed if elapsed > 0 else 0
            print(f"📊 Progress: {i} items processed ({items_per_sec:.1f} items/sec)")
            print(f"   Current counts - S: {scale_counts['S']}, M: {scale_counts['M']}, L: {scale_counts['L']}")
            last_progress_time = current_time
        
        name = item.get("name", "")
        
        # Debug: Print some sample names to see the pattern
        if i < 10:
            print(f"   Sample {i}: name='{name}'")
        
        for scale in "SML":
            if name.startswith(f"{scale}_") and scale_counts[scale] < max_samples_per_scale:
                scale_buffers[scale].append(item)
                
                if len(scale_buffers[scale]) >= batch_size:
                    batch_idx = scale_counts[scale] // batch_size
                    batch_path = os.path.join(scale_dirs[scale], f"batch_{batch_idx}")
                    
                    print(f"💾 Saving batch {batch_idx} for scale {scale}...")
                    save_start = time.time()
                    HFDataset.from_list(scale_buffers[scale]).save_to_disk(batch_path)
                    save_time = time.time() - save_start
                    
                    print(f"✓ Saved {len(scale_buffers[scale])} items to {batch_path} in {save_time:.1f}s")
                    scale_counts[scale] += len(scale_buffers[scale])
                    scale_buffers[scale] = []

        # Check if we're done
        if all(scale_counts[s] >= max_samples_per_scale for s in "SML"):
            print("✅ Reached sample limits for all scales.")
            break
        
        # Safety check - if we've processed a lot without finding any samples
        if i > 50000 and all(scale_counts[s] == 0 for s in "SML"):
            print("⚠️  Warning: Processed 50k items but found no S/M/L samples")
            print("   This might indicate the naming pattern is different than expected")
            # Show some recent names for debugging
            print(f"   Recent sample name: '{name}'")

    # Save any remaining buffers
    print("💾 Saving remaining buffers...")
    for scale in "SML":
        if scale_buffers[scale]:
            batch_idx = scale_counts[scale] // batch_size
            batch_path = os.path.join(scale_dirs[scale], f"batch_{batch_idx}")
            
            print(f"💾 Saving final batch for scale {scale}...")
            HFDataset.from_list(scale_buffers[scale]).save_to_disk(batch_path)
            print(f"✓ Saved final {len(scale_buffers[scale])} items to {batch_path}")
            scale_counts[scale] += len(scale_buffers[scale])

    total_time = time.time() - start_time
    print(f"\n🎉 Done in {total_time:.1f}s. Summary:")
    for scale in "SML":
        print(f"  {scale}: {scale_counts[scale]} samples in {scale_dirs[scale]}")

if __name__ == "__main__":
    process_and_save_large_hf_dataset()
