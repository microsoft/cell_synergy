import os
import argparse
import numpy as np
import h5py
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from data_scaling.paths import UNI_EMBEDDINGS_DIR, PROJECT_DIR
from data_scaling.config import load_data_splits

HF_CACHE_DIR = PROJECT_DIR / "hf_cache"
HF_DATASET_ID = "theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6"

IMG_MODELS = ["1M", "22M", "100M", "200M"]

VALID_COMBOS = [
    ("full", "pretrain", "S"),
    ("full", "pretrain", "M"),
    ("full", "pretrain", "L"),
    ("full", "finetune", "S"),
    ("full", "finetune", "M"),
    ("full", "finetune", "L"),
    ("full", "test", "S"),
    ("full", "test", "M"),
    ("full", "test", "L"),
    ("finetune_S", "finetune", "S"),
    ("finetune_M", "finetune", "M"),
    ("finetune_L", "finetune", "L"),
    ("finetune_S", "test", "S"),
    ("finetune_M", "test", "M"),
    ("finetune_L", "test", "L"),
]

def build_gex_path(model, split, scale):
    # For unimodal_gex mode, we need to specify which GEX model to use
    # The model parameter should be the actual GEX model (full, finetune_S, etc.)
    if split == "test":
        return UNI_EMBEDDINGS_DIR / "lung" / "gex" / f"nicheformer_{model}_test.npy"
    return UNI_EMBEDDINGS_DIR / "lung" / "gex" / f"nicheformer_{model}_{split}_{scale}.npy"


def build_gex_h5_path(model, split, scale):
    """Build path for GEX embeddings in H5 format (with sample names)."""
    if split == "test":
        return UNI_EMBEDDINGS_DIR / "lung" / "gex" / f"nicheformer_{model}_test.h5"
    return UNI_EMBEDDINGS_DIR / "lung" / "gex" / f"nicheformer_{model}_{split}_{scale}.h5"


def load_gex_embeddings_with_names(gex_model, split, scale):
    """Load GEX embeddings, preferring H5 format if available, falling back to NPY."""
    # Try H5 format first (with sample names)
    h5_path = build_gex_h5_path(gex_model, split, scale)
    if h5_path.exists():
        print(f"📊 Loading GEX embeddings from H5 file: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            embeddings = f["embeddings"][:]
            sample_names = [x.decode() for x in f["sample_names"][:]]
        return dict(zip(sample_names, embeddings))
    
    # Fall back to NPY format (without sample names)
    npy_path = build_gex_path(gex_model, split, scale)
    if npy_path.exists():
        print(f"📊 Loading GEX embeddings from NPY file: {npy_path}")
        embeddings = np.load(npy_path)
        # For NPY format, we don't have sample names, so we'll use sequential indices
        # This maintains backward compatibility but is less robust
        print(f"⚠️  Warning: Using NPY format without sample names - less robust matching")
        return embeddings
    else:
        raise FileNotFoundError(f"Neither H5 file {h5_path} nor NPY file {npy_path} found")


def build_img_path(img_model, split, scale):
    # For unimodal_img mode, use the actual model name
    if img_model == "unimodal_img":
        raise ValueError("img_model should be the actual model name (1M, 22M, etc.), not 'unimodal_img'")
    
    if split == "test":
        return UNI_EMBEDDINGS_DIR / "lung" / "img" / f"img_{img_model}_test.h5"
    return UNI_EMBEDDINGS_DIR / "lung" / "img" / f"img_{img_model}_{split}_{scale}.h5"


def load_h5_embeddings_with_names(path):
    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:]
        sample_names = [x.decode() for x in f["sample_names"][:]]
    return dict(zip(sample_names, embeddings))

def enrich_and_save(dataset, gex_data, img_dict, out_path):
    """Enrich dataset with embeddings and save."""
    # Create a mapping from unique identifiers to embeddings
    filtered = dataset.filter(lambda x: x["name"] in [name.split('_')[0] for name in img_dict.keys()]) if img_dict else dataset
    print(f"📦 Filtered to {len(filtered)} samples")

    # Determine GEX data format
    gex_is_dict = isinstance(gex_data, dict)
    gex_arr = gex_data if not gex_is_dict else None

    def add_columns(batch):
        out_gex, out_img = [], []
        for i, (name, coords) in enumerate(zip(batch["name"], batch["cell_coords"])):
            # Create the same unique identifier as in generate_img_embeddings.py
            coord_str = f"{coords[0][0]}_{coords[0][1]}"
            unique_name = f"{name}_{coord_str}"
            
            # Handle image embeddings if present
            if img_dict:
                if unique_name in img_dict:
                    out_img.append(img_dict[unique_name].tolist())
                else:
                    print(f"⚠️  Warning: Could not find image embedding for {unique_name}")
                    return None
            
            # Handle GEX embeddings if present
            if gex_data is not None and len(gex_data) > 0:
                if gex_is_dict:
                    # H5 format with sample names
                    if unique_name in gex_data:
                        out_gex.append(gex_data[unique_name].tolist())
                    else:
                        print(f"⚠️  Warning: Could not find GEX embedding for {unique_name}")
                        return None
                else:
                    # NPY format - use sequential index (less robust)
                    if img_dict:
                        # For multimodal, use the same index as image
                        idx = list(img_dict.keys()).index(unique_name)
                    else:
                        # For unimodal GEX, use sequential index
                        # CRITICAL FIX: Ensure we don't go out of bounds
                        if i >= len(gex_arr):
                            print(f"⚠️  Warning: GEX embedding index {i} out of bounds (max: {len(gex_arr)-1})")
                            return None
                        idx = i
                    out_gex.append(gex_arr[idx].tolist())
        
        # Return only the relevant columns
        result = {}
        if gex_data is not None and len(gex_data) > 0:
            result["nicheformer_pool"] = out_gex
        if img_dict:
            result["img_uni_pool"] = out_img
        return result

    # CRITICAL FIX: For unimodal GEX with NPY format, ensure the dataset is processed in the same order as embeddings were generated
    if gex_data is not None and len(gex_data) > 0 and not gex_is_dict and not img_dict:
        print(f"🔧 Unimodal GEX mode (NPY format): Ensuring dataset order matches embedding order")
        print(f"   Dataset has {len(filtered)} samples, GEX array has {len(gex_arr)} embeddings")
        
        # Get the sample names in the order they appear in the dataset
        dataset_sample_names = [sample["name"] for sample in filtered]
        print(f"   First 5 dataset sample names: {dataset_sample_names[:5]}")
        
        # Verify we have the right number of samples
        if len(dataset_sample_names) != len(gex_arr):
            print(f"⚠️  WARNING: Dataset has {len(dataset_sample_names)} samples but GEX array has {len(gex_arr)} embeddings!")
            print(f"   This suggests a mismatch in the data filtering process.")
            print(f"   Dataset sample names: {dataset_sample_names}")
            return

    enriched = filtered.map(
        add_columns,
        batched=True,
        batch_size=1024,
        desc=f"Adding embeddings"
    )

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"💾 Saving enriched dataset to: {out_path}")
    enriched.save_to_disk(str(out_path))

def check_all_required_files(mode=None):
    """Check that required embedding files exist for the given mode."""
    print("🔎 Checking that required embedding files exist...\n")
    missing = []
    
    if mode == "unimodal_gex":
        # Only check GEX files for unimodal_gex mode
        valid_combinations = [
            ("full", "pretrain", "S"), ("full", "pretrain", "M"), ("full", "pretrain", "L"),
            ("full", "finetune", "S"), ("full", "finetune", "M"), ("full", "finetune", "L"),
            ("full", "test", "S"),
            ("finetune_S", "finetune", "S"), ("finetune_S", "test", "S"),
            ("finetune_M", "finetune", "M"), ("finetune_M", "test", "S"),
            ("finetune_L", "finetune", "L"), ("finetune_L", "test", "S"),
        ]
        
        for model, split, scale in valid_combinations:
            # Check GEX files - either NPY or H5 format is acceptable
            gex_npy_path = build_gex_path(model, split, scale)
            gex_h5_path = build_gex_h5_path(model, split, scale)
            if not gex_npy_path.exists() and not gex_h5_path.exists():
                missing.append(f"GEX: {gex_npy_path} or {gex_h5_path}")
    
    elif mode == "unimodal_img":
        # Only check IMG files for unimodal_img mode
        for split in ["pretrain", "finetune", "test"]:
            for scale in ["S", "M", "L"] if split != "test" else ["S"]:
                for img_model in IMG_MODELS:
                    img_path = build_img_path(img_model, split, scale)
                    if not img_path.exists():
                        missing.append(str(img_path))
    
    elif mode == "multimodal":
        # Check both GEX and IMG files for multimodal mode
        for model, split, scale in VALID_COMBOS:
            # Check GEX files - either NPY or H5 format is acceptable
            gex_npy_path = build_gex_path(model, split, scale)
            gex_h5_path = build_gex_h5_path(model, split, scale)
            if not gex_npy_path.exists() and not gex_h5_path.exists():
                missing.append(f"GEX: {gex_npy_path} or {gex_h5_path}")
            
            # Check IMG files
            for img_model in IMG_MODELS:
                img_path = build_img_path(img_model, split, scale)
                if not img_path.exists():
                    missing.append(str(img_path))
    
    else:
        # Default: check all files (for backward compatibility)
        for model, split, scale in VALID_COMBOS:
            # Check GEX files - either NPY or H5 format is acceptable
            gex_npy_path = build_gex_path(model, split, scale)
            gex_h5_path = build_gex_h5_path(model, split, scale)
            if not gex_npy_path.exists() and not gex_h5_path.exists():
                missing.append(f"GEX: {gex_npy_path} or {gex_h5_path}")
            
            # Check IMG files
            for img_model in IMG_MODELS:
                img_path = build_img_path(img_model, split, scale)
                if not img_path.exists():
                    missing.append(str(img_path))

    if missing:
        print("⚠️  Missing the following embedding files:\n")
        for m in missing:
            print(f"   - {m}")
        print(f"\n⚠️  Found {len(missing)} missing files. Continuing with available files...\n")
        return False
    else:
        print("✅ All required embedding files are present.\n")
        return True

def run_single(gex_model, dataset_name, img_model, split, scale):
    """Run a single configuration to generate embeddings."""
    # Determine the mode based on what models are provided
    is_multimodal = (gex_model in ["full", "finetune_S", "finetune_M", "finetune_L"] and img_model in IMG_MODELS)
    is_unimodal_gex = (gex_model in ["full", "finetune_S", "finetune_M", "finetune_L"] and (img_model is None or img_model not in IMG_MODELS))
    is_unimodal_img = ((gex_model is None or gex_model not in ["full", "finetune_S", "finetune_M", "finetune_L"]) and img_model in IMG_MODELS)
    
    # For test split, we don't need scale
    if split == "test":
        out_dir = PROJECT_DIR / dataset_name / "hf_datasets"
        if is_unimodal_gex:
            out_dir = out_dir / f"gex_only_{gex_model}_test.{scale}"
        elif is_unimodal_img:
            out_dir = out_dir / f"img_only_{img_model}_test.{scale}"
        else:  # multimodal
            out_dir = out_dir / f"{gex_model}_{img_model}_test.{scale}"
    else:
        out_dir = PROJECT_DIR / dataset_name / "hf_datasets"
        if is_unimodal_gex:
            out_dir = out_dir / f"gex_only_{gex_model}_{split}.{scale}"
        elif is_unimodal_img:
            out_dir = out_dir / f"img_only_{img_model}_{split}.{scale}"
        else:  # multimodal
            out_dir = out_dir / f"{gex_model}_{img_model}_{split}.{scale}"

    if out_dir.exists():
        print(f"⏭️  Skipping {out_dir}, already exists.")
        return

    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    print(f"\n🔄 Loading HF dataset for split={split}, scale={scale}...")
    dataset = load_dataset(
        HF_DATASET_ID,
        split="train",
        cache_dir=str(HF_CACHE_DIR),
        use_auth_token=hf_token,
    )

    cfg = load_data_splits(dataset_name=dataset_name)
    try:
        if split == "test":
            sample_names = set(cfg.multimodal[split])
        else:
            sample_names = set(cfg.multimodal[split][scale])
    except KeyError:
        raise ValueError(f"No samples found for split={split}, scale={scale}")

    dataset = dataset.filter(lambda x: x["name"] in sample_names)
    
    # Load embeddings based on mode
    if is_unimodal_gex:
        img_dict = {}  # Empty dict for img embeddings
        gex_data = load_gex_embeddings_with_names(gex_model, split, scale)
    elif is_unimodal_img:
        img_path = build_img_path(img_model, split, scale)
        if not img_path.exists():
            raise FileNotFoundError(f"IMG embedding file not found: {img_path}")
        print(f"IMG embedding file: {img_path}")
        img_dict = load_h5_embeddings_with_names(img_path)
        gex_data = None  # Empty for gex embeddings
    else:  # multimodal case
        img_path = build_img_path(img_model, split, scale)
        if not img_path.exists():
            raise FileNotFoundError(f"IMG embedding file not found: {img_path}")
        print(f"IMG embedding file: {img_path}")
        img_dict = load_h5_embeddings_with_names(img_path)
        gex_data = load_gex_embeddings_with_names(gex_model, split, scale)

    enrich_and_save(dataset, gex_data, img_dict, out_dir)

def run_all():
    check_all_required_files()
    
    # Start with test splits for all combinations (most important for evaluation)
    print("Starting with test splits for all model combinations...")
    for model in ["full", "finetune_S", "finetune_M", "finetune_L"]:
        for img_model in IMG_MODELS:
            try:
                run_single(model, "lung", img_model, "test", "S")
            except FileNotFoundError as e:
                print(f"⚠️  Skipping {model}_{img_model}_test - file not found: {e}")
                continue
    
    # Then handle other splits
    print("Processing other splits...")
    for model, split, scale in VALID_COMBOS:
        if split == "test":  # Skip test splits as we already processed them
            continue
        for img_model in IMG_MODELS:
            try:
                run_single(model, "lung", img_model, split, scale)
            except FileNotFoundError as e:
                print(f"⚠️  Skipping {model}_{img_model}_{split}_{scale} - file not found: {e}")
                continue

# Example usage. Add all combinations to the hf datasets
# python add_embeddings_to_hf_dataset.py --mode multimodal --gex_model_scale full --img_model_scale 200M --split pretrain --data_scale L

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["unimodal_img", "unimodal_gex", "multimodal"], required=True,
                      help="Mode to run in (unimodal_img, unimodal_gex, or multimodal)")
    parser.add_argument("--gex_model_scale", type=str, help="e.g. full, finetune_S, finetune_M, finetune_L")
    parser.add_argument("--img_model_scale", type=str, help="e.g. 1M, 22M, 100M, 200M")
    parser.add_argument("--data_scale", type=str, help="S, M, or L")
    parser.add_argument("--split", type=str, help="pretrain, finetune, or test")
    parser.add_argument("--dataset", type=str, default="lung", help="Dataset name (default: lung)")
    parser.add_argument("--all", action="store_true", help="Run all predefined combinations")
    args = parser.parse_args()

    if args.all:
        check_all_required_files(args.mode)
        
        if args.mode == "unimodal_img":
            # For unimodal image, start with test splits first, then other splits
            print("Starting with test splits for all image models...")
            for img_model in IMG_MODELS:
                run_single(None, "lung", img_model, "test", "S")  # Scale doesn't matter for test
            
            print("Processing other splits...")
            for img_model in IMG_MODELS:
                for split in ["pretrain", "finetune"]:
                    for scale in ["S", "M", "L"]:
                        try:
                            run_single(None, "lung", img_model, split, scale)
                        except FileNotFoundError as e:
                            print(f"⚠️  Skipping img_only_{img_model}_{split}_{scale} - file not found: {e}")
                            continue
        elif args.mode == "unimodal_gex":
            # For unimodal GEX, start with full model + test split, then handle other combinations
            print("Starting with full model + test split...")
            run_single("full", "lung", None, "test", "S")
            
            # Then handle other combinations that actually exist
            valid_combinations = [
                # full model with all splits
                ("full", "pretrain", "S"), ("full", "pretrain", "M"), ("full", "pretrain", "L"),
                ("full", "finetune", "S"), ("full", "finetune", "M"), ("full", "finetune", "L"),
                # finetune models with their corresponding splits only
                ("finetune_S", "finetune", "S"), ("finetune_S", "test", "S"),
                ("finetune_M", "finetune", "M"), ("finetune_M", "test", "S"),
                ("finetune_L", "finetune", "L"), ("finetune_L", "test", "S"),
            ]
            
            for gex_model, split, scale in valid_combinations:
                try:
                    run_single(gex_model, "lung", None, split, scale)
                except FileNotFoundError as e:
                    print(f"⚠️  Skipping {gex_model}_{split}_{scale} - file not found: {e}")
                    continue
        else:  # multimodal
            run_all()
    else:
        # Validate arguments based on mode
        if args.mode == "unimodal_img":
            if not args.img_model_scale or not args.split:
                raise ValueError("unimodal_img mode requires --img_model_scale and --split")
            if args.split != "test" and not args.data_scale:
                raise ValueError("Non-test splits require --data_scale")
        elif args.mode == "unimodal_gex":
            if not args.gex_model_scale or not args.split:
                raise ValueError("unimodal_gex mode requires --gex_model_scale and --split")
            if args.split != "test" and not args.data_scale:
                raise ValueError("Non-test splits require --data_scale")
            if args.gex_model_scale not in ["full", "finetune_S", "finetune_M", "finetune_L"]:
                raise ValueError("gex_model_scale must be one of: full, finetune_S, finetune_M, finetune_L")
        else:  # multimodal
            if not (args.gex_model_scale and args.img_model_scale and args.split):
                raise ValueError("multimodal mode requires --gex_model_scale, --img_model_scale, and --split")
            if args.split != "test" and not args.data_scale:
                raise ValueError("Non-test splits require --data_scale")
        
        if args.mode == "unimodal_img":
            run_single(None, args.dataset, args.img_model_scale, args.split, args.data_scale)
        elif args.mode == "unimodal_gex":
            run_single(args.gex_model_scale, args.dataset, None, args.split, args.data_scale)
        else:  # multimodal
            run_single(args.gex_model_scale, args.dataset, args.img_model_scale, args.split, args.data_scale)
