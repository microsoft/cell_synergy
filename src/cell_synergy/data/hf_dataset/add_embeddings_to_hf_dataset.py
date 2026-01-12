import os
import argparse
import numpy as np
import h5py
import tempfile
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from cell_synergy.paths import UNI_EMBEDDINGS_DIR, PROJECT_DIR
from cell_synergy.config import load_data_splits
# Set environment variables to use lustre for all temporary files
os.environ['TMPDIR'] = '/lustre/groups/ml01/workspace/till.richter/tmp'
os.environ['TEMP'] = '/lustre/groups/ml01/workspace/till.richter/tmp'
os.environ['TMP'] = '/lustre/groups/ml01/workspace/till.richter/tmp'
tempfile.tempdir = '/lustre/groups/ml01/workspace/till.richter/tmp'

HF_CACHE_DIR = PROJECT_DIR / "hf_cache"

# Map dataset names to HF dataset identifiers
HF_DATASET_IDS = {
    "lung": "theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6",
    "thymus": "theislab-multimodal-ssl/thymus_ts2",
    "breast": "theislab-multimodal-ssl/breast_embeddings",
}
HF_DATASET_ID = HF_DATASET_IDS["lung"]  # Default for backward compatibility

# Set up custom temp directory on lustre to avoid quota issues
CUSTOM_TMP_DIR = Path("/lustre/groups/ml01/workspace/till.richter/tmp")
CUSTOM_TMP_DIR.mkdir(exist_ok=True)

# Set up output directory on lustre to avoid quota issues
LUSTRE_OUTPUT_DIR = Path("/lustre/groups/ml01/workspace/till.richter/hf_datasets")
LUSTRE_OUTPUT_DIR.mkdir(exist_ok=True)

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
    ("finetune_1pct_cfg1", "test", "S"),
    ("finetune_1pct_cfg2", "test", "S"),
    ("finetune_1pct_cfg3", "test", "S"),
    ("finetune_3.16pct_cfg1", "test", "S"),
    ("finetune_3.16pct_cfg2", "test", "S"),
    ("finetune_3.16pct_cfg3", "test", "S"),
    ("finetune_10pct_cfg1", "test", "S"),
    ("finetune_10pct_cfg2", "test", "S"),
    ("finetune_10pct_cfg3", "test", "S"),
    ("finetune_31.6pct_cfg1", "test", "S"),
    ("finetune_31.6pct_cfg2", "test", "S"),
    ("finetune_31.6pct_cfg3", "test", "S"),
    ("finetune_100pct_cfg1", "test", "S"),
    ("finetune_100pct_cfg2", "test", "S"),
    ("finetune_100pct_cfg3", "test", "S"),
]

def build_gex_path(model, split, scale, dataset_name="lung"):
    # For unimodal_gex mode, we need to specify which GEX model to use
    # Handle the correct naming pattern for finetuned models
    if model.startswith("finetune_"):
        # For finetuned models, try multiple naming patterns
        pct_name = model.replace('finetune_', '')
        gex_dir = UNI_EMBEDDINGS_DIR / dataset_name / "gex"

        # First try the new naming pattern: nicheformer_finetune_{pct}_{cfg}_full_dataset.npy
        finetune_path = gex_dir / f"nicheformer_{model}_full_dataset.npy"
        if finetune_path.exists():
            return finetune_path

        # Then try the finetune_S pattern: nicheformer_finetune_{pct}_{cfg}_finetune_{scale}.npy
        finetune_s_path = gex_dir / f"nicheformer_{model}_finetune_{scale}.npy"
        if finetune_s_path.exists():
            return finetune_s_path

        # Then try the comm naming pattern (old style)
        comm_path = gex_dir / f"nicheformer_comm_{pct_name}_full_dataset.npy"
        if comm_path.exists():
            return comm_path

        # Then try checkpoint-based naming pattern (new style: nicheformer_merlin_{pct}_epoch=..._full_dataset.npy)
        # Find the most recent checkpoint-based file matching the pattern
        import glob
        pattern = str(gex_dir / f"nicheformer_merlin_{pct_name}_*_full_dataset.npy")
        matching_files = glob.glob(pattern)
        if matching_files:
            # Return the most recently modified file
            return Path(max(matching_files, key=lambda p: Path(p).stat().st_mtime))

        # Fallback to finetune pattern (will be checked for existence by caller)
        return finetune_path
    else:
        # For full model, use the original pattern
        if split == "test":
            return UNI_EMBEDDINGS_DIR / dataset_name / "gex" / f"nicheformer_{model}_test.npy"
        return UNI_EMBEDDINGS_DIR / dataset_name / "gex" / f"nicheformer_{model}_{split}_{scale}.npy"

def build_gex_h5_path(model, split, scale, dataset_name="lung"):
    """Build path for GEX embeddings in H5 format (with sample names)."""
    if model.startswith("finetune_"):
        # For finetuned models, try multiple naming patterns
        pct_name = model.replace('finetune_', '')
        gex_dir = UNI_EMBEDDINGS_DIR / dataset_name / "gex"

        # First try the new naming pattern: nicheformer_finetune_{pct}_{cfg}_full_dataset.h5
        finetune_path = gex_dir / f"nicheformer_{model}_full_dataset.h5"
        if finetune_path.exists():
            return finetune_path

        # Then try the finetune_S pattern: nicheformer_finetune_{pct}_{cfg}_finetune_S.h5
        finetune_s_path = gex_dir / f"nicheformer_{model}_finetune_{scale}.h5"
        if finetune_s_path.exists():
            return finetune_s_path

        # Then try the comm naming pattern (old style)
        comm_path = gex_dir / f"nicheformer_comm_{pct_name}_full_dataset.h5"
        if comm_path.exists():
            return comm_path

        # Then try checkpoint-based naming pattern (new style: nicheformer_merlin_{pct}_epoch=..._full_dataset.h5)
        import glob
        pattern = str(gex_dir / f"nicheformer_merlin_{pct_name}_*_full_dataset.h5")
        matching_files = glob.glob(pattern)
        if matching_files:
            # Return the most recently modified file
            return Path(max(matching_files, key=lambda p: Path(p).stat().st_mtime))

        # Fallback to finetune pattern (will be checked for existence by caller)
        return finetune_path
    else:
        # For full model, use the original pattern
        if split == "test":
            return UNI_EMBEDDINGS_DIR / dataset_name / "gex" / f"nicheformer_{model}_test.h5"
        return UNI_EMBEDDINGS_DIR / dataset_name / "gex" / f"nicheformer_{model}_{split}_{scale}.h5"

def load_gex_embeddings_with_names(gex_model, split, scale, dataset_name="lung"):
    """
    Load GEX embeddings, preferring H5 format if available, falling back to NPY.

    For finetuned models, filters embeddings by split (train/test) using donor IDs from config.

    Data format notes:
    - H5 files contain full dataset (train + test) with sample names like:
      * Thymus: WSSS_F_IMMsp11765870_6200_3462 (full donor ID + coordinates)
      * Breast: TENX97_vectorized_annotated_allpolys_noCT_209_3391 (full name + coordinates)
    - Config test donors format:
      * Thymus: ['WSSS_F_IMMsp11604686', 'WSSS_F_IMMsp11604687', ...] (full donor IDs)
      * Breast: ['TENX97', 'TENX96', ...] (short donor IDs)
    - Matching strategy: Use prefix matching (name.startswith(donor + '_')) to handle
      both formats correctly.

    Expected output:
    - Dictionary mapping sample names to embeddings
    - For test split: ~6k-9k samples (thymus), ~25k samples (breast)
    - For train split: ~38k samples (thymus), ~100k samples (breast)
    """
    # Try H5 format first (with sample names)
    h5_path = build_gex_h5_path(gex_model, split, scale, dataset_name)
    if h5_path.exists():
        print(f"Loading GEX embeddings from H5 file: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            embeddings = f["embeddings"][:]
            sample_names = [x.decode() for x in f["sample_names"][:]]

        # For finetuned models, we have the full dataset (train + test) in the h5 file
        # For CoMM training, we need ALL samples (train + test), not just a split
        # The split parameter is only used for output directory naming
        if gex_model.startswith("finetune_"):
            # Don't filter - use ALL samples from the H5 file
            # The main dataset filtering (in run_single) already handles train+test
            print(f"  Using ALL samples from H5 file: {len(sample_names)} samples (train + test)")
            # Keep all embeddings as-is

        return dict(zip(sample_names, embeddings))

    # Fall back to NPY format (without sample names)
    npy_path = build_gex_path(gex_model, split, scale, dataset_name)
    if npy_path.exists():
        print(f"Loading GEX embeddings from NPY file: {npy_path}")
        embeddings = np.load(npy_path)
        # For NPY format, we don't have sample names, so we'll use sequential indices
        # This maintains backward compatibility but is less robust
        print("Warning: Using NPY format without sample names - less robust matching")
        return embeddings
    else:
        raise FileNotFoundError(f"Neither H5 file {h5_path} nor NPY file {npy_path} found")

def build_img_path(img_model, split, scale, dataset_name="lung"):
    # For unimodal_img mode, use the actual model name
    if img_model == "unimodal_img":
        raise ValueError("img_model should be the actual model name (1M, 22M, etc.), not 'unimodal_img'")

    if split == "test":
        return UNI_EMBEDDINGS_DIR / dataset_name / "img" / f"img_{img_model}_test.h5"
    return UNI_EMBEDDINGS_DIR / dataset_name / "img" / f"img_{img_model}_{split}_{scale}.h5"

def load_h5_embeddings_with_names(path):
    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:]
        sample_names = [x.decode() for x in f["sample_names"][:]]
    return dict(zip(sample_names, embeddings))

def load_img_from_hf_dataset(img_model, split, scale, dataset_name="lung"):
    """Load image embeddings from HF dataset as fallback when h5 file doesn't exist."""
    try:
        from datasets import load_from_disk
        from cell_synergy.paths import PROJECT_DIR
        # Use LUSTRE_OUTPUT_DIR (defined at top of file) for HF datasets
        # Try to load img_only dataset
        img_pattern = f"img_only_{img_model}.{scale}"
        img_dir = LUSTRE_OUTPUT_DIR / dataset_name / img_pattern

        if not img_dir.exists():
            print(f"   HF dataset not found: {img_dir}")
            return None

        print(f"   Loading from HF dataset: {img_dir}")
        img_ds = load_from_disk(str(img_dir))

        # Extract image embeddings and create mapping
        # The dataset should have 'name' and 'img_uni_pool' columns
        if 'img_uni_pool' not in img_ds.column_names:
            print(f"   Dataset missing 'img_uni_pool' column. Available columns: {img_ds.column_names}")
            return None

        # Create mapping: for multimodal datasets, we need to match by name + coords
        # The HF dataset has img_uni_pool which is already aggregated per sample
        # But enrich_and_save expects unique_name format: "{name}_{coord_x}_{coord_y}"
        # Since img_uni_pool is per-sample, we'll create entries for all possible coords
        # Actually, let's check if the dataset has cell_coords - if not, it's already aggregated
        img_dict = {}
        has_cell_coords = 'cell_coords' in img_ds.column_names

        if has_cell_coords:
            # Dataset has cell-level data, create unique_name entries
            for i, example in enumerate(img_ds):
                name = example.get('name', f'sample_{i}')
                coords = example.get('cell_coords', [])
                if coords and len(coords) > 0:
                    coord_str = f"{coords[0][0]}_{coords[0][1]}"
                    unique_name = f"{name}_{coord_str}"
                    img_dict[unique_name] = example['img_uni_pool']
                else:
                    # Fallback: use name only
                    img_dict[name] = example['img_uni_pool']
        else:
            # Dataset is already aggregated per sample, store by name only
            # enrich_and_save will handle matching by name
            for i, example in enumerate(img_ds):
                name = example.get('name', f'sample_{i}')
                img_dict[name] = example['img_uni_pool']

        print(f"   Loaded {len(img_dict)} image embeddings from HF dataset")
        return img_dict
    except Exception as e:
        print(f"   Error loading from HF dataset: {e}")
        return None

def enrich_and_save(dataset, gex_data, img_dict, out_path):
    """Enrich dataset with embeddings and save."""
    # Set custom temp directory for datasets library to avoid quota issues
    os.environ['TMPDIR'] = str(CUSTOM_TMP_DIR)
    tempfile.tempdir = str(CUSTOM_TMP_DIR)

    # img_dict is a mapping from unique identifiers to embeddings (created in load_img_from_hf_dataset or load_h5_embeddings_with_names)
    # Don't pre-filter the dataset - let add_columns handle matching (it has fallback logic for name-only matching)
    # The image dataset may have aggregated embeddings (one per sample name) while the main dataset
    # has cell-level data (multiple cells per sample name), so pre-filtering would incorrectly remove samples
    filtered = dataset

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
            img_embed = None
            if img_dict:
                # Image embeddings can be stored either:
                # 1. By unique_name (name + coords) - for cell-level data
                # 2. By name only - for aggregated per-sample data
                # Try unique_name first, then fall back to name
                if unique_name in img_dict:
                    img_embed = img_dict[unique_name]
                elif name in img_dict:
                    img_embed = img_dict[name]
                else:
                    # ROOT CAUSE: Image embedding not found - this should not happen
                    available_keys_sample = [k for k in list(img_dict.keys())[:20] if k.startswith(f"{name}_")]
                    raise ValueError(
                        f"ERROR: Could not find image embedding for sample '{name}' (unique_name='{unique_name}'). "
                        "Tried both unique_name and name matching. "
                        "This indicates a mismatch between the dataset and image embeddings. "
                        f"Available image keys starting with '{name}_' (first 20): {available_keys_sample}"
                    )

            # Handle GEX embeddings if present
            gex_embed = None
            if gex_data is not None and len(gex_data) > 0:
                if gex_is_dict:
                    # H5 format with sample names
                    # The H5 sample names have format: {donor}_{middle_parts}_{coord_x}_{coord_y}
                    # But unique_name is: {name}_{coord_x}_{coord_y} where name is just the donor
                    # So we need to find a key in gex_data that:
                    #   1. Starts with the donor name (name)
                    #   2. Ends with the coord_str (coord_x_coord_y)
                    gex_key = None
                    if unique_name in gex_data:
                        gex_key = unique_name
                    else:
                        # Try to find a key that matches the pattern
                        # H5 format: TENX97_vectorized_annotated_allpolys_noCT_209_3391
                        # unique_name: TENX97_209_3391
                        # So we need to find keys that start with name and end with coord_str
                        coord_suffix = f"_{coord_str}"
                        for key in gex_data.keys():
                            if key.startswith(f"{name}_") and key.endswith(coord_suffix):
                                gex_key = key
                                break

                    if gex_key:
                        gex_embed = gex_data[gex_key]
                    else:
                        # ROOT CAUSE: GEX embedding not found - this should not happen
                        # Check what keys are actually available to help debug
                        available_keys_sample = [k for k in list(gex_data.keys())[:20] if k.startswith(f"{name}_")]
                        raise ValueError(
                            f"ERROR: Could not find GEX embedding for sample '{unique_name}'. "
                            f"Tried exact match and pattern match (name='{name}', coord_str='{coord_str}'). "
                            "This indicates a mismatch between the dataset and GEX embeddings. "
                            f"Sample keys starting with '{name}_' (first 20): {available_keys_sample}"
                        )
                else:
                    # NPY format - use sequential index (less robust)
                    if img_dict:
                        # For multimodal, use the same index as image
                        try:
                            idx = list(img_dict.keys()).index(unique_name)
                        except ValueError:
                            # ROOT CAUSE: unique_name not found in img_dict - this should not happen
                            raise ValueError(
                                f"ERROR: Could not find '{unique_name}' in img_dict. "
                                "This indicates a mismatch between the dataset and image embeddings. "
                                f"Available img_dict keys (first 10): {list(img_dict.keys())[:10] if img_dict else 'N/A'}"
                            )
                    else:
                        # For unimodal GEX, use sequential index
                        # Ensure we don't go out of bounds
                        if i >= len(gex_arr):
                            # ROOT CAUSE: Index out of bounds - dataset and embeddings don't match
                            raise ValueError(
                                f"ERROR: GEX embedding index {i} out of bounds. "
                                f"Dataset has more samples ({i+1}) than GEX embeddings ({len(gex_arr)}). "
                                "This indicates a mismatch between the dataset and GEX embeddings."
                            )
                        idx = i
                        gex_embed = gex_arr[idx]

            # Convert embeddings to lists
            # At this point, both embeddings should be found (errors raised above if not)
            if img_embed is not None:
                if hasattr(img_embed, 'tolist'):
                    out_img.append(img_embed.tolist())
                elif isinstance(img_embed, list):
                    out_img.append(img_embed)
                else:
                    out_img.append(list(img_embed))
            else:
                # This should never happen if img_dict is provided (error raised above)
                if img_dict:
                    raise ValueError(f"ERROR: img_embed is None but img_dict was provided for sample '{name}'")

            if gex_embed is not None:
                if hasattr(gex_embed, 'tolist'):
                    out_gex.append(gex_embed.tolist())
                elif isinstance(gex_embed, list):
                    out_gex.append(gex_embed)
                else:
                    out_gex.append(list(gex_embed))
            else:
                # This should never happen if gex_data is provided (error raised above)
                if gex_data is not None and len(gex_data) > 0:
                    raise ValueError(f"ERROR: gex_embed is None but gex_data was provided for sample '{unique_name}'")

        # Return columns - use None for missing embeddings, will filter after
        result = {}
        if gex_data is not None and len(gex_data) > 0:
            result["nicheformer_pool"] = out_gex
        if img_dict:
            result["img_uni_pool"] = out_img
        return result

    # For unimodal GEX with NPY format, ensure the dataset is processed in the same order as embeddings were generated
    if gex_data is not None and len(gex_data) > 0 and not gex_is_dict and not img_dict:
        print("Unimodal GEX mode (NPY format): Ensuring dataset order matches embedding order")
        print(f"   Dataset has {len(filtered)} samples, GEX array has {len(gex_arr)} embeddings")

        # Get the sample names in the order they appear in the dataset
        dataset_sample_names = [sample["name"] for sample in filtered]
        print(f"   First 5 dataset sample names: {dataset_sample_names[:5]}")

        # Verify we have the right number of samples
        if len(dataset_sample_names) != len(gex_arr):
            print(f"WARNING: Dataset has {len(dataset_sample_names)} samples but GEX array has {len(gex_arr)} embeddings!")
            print("   This suggests a mismatch in the data filtering process.")
            print(f"   Dataset sample names: {dataset_sample_names}")
            return

    enriched = filtered.map(
        add_columns,
        batched=True,
        batch_size=1024,
        desc="Adding embeddings"
    )

    # No filtering needed - all samples should have embeddings (errors raised if not)
    # If we get here, all embeddings were successfully matched
    print(f"All {len(enriched)} samples have embeddings")

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving enriched dataset to: {out_path}")
    enriched.save_to_disk(str(out_path))

    # Clean up temporary files
    print(f"🧹 Cleaning up temporary files in {CUSTOM_TMP_DIR}")
    for tmp_file in CUSTOM_TMP_DIR.glob("*"):
        try:
            if tmp_file.is_file():
                tmp_file.unlink()
            elif tmp_file.is_dir():
                import shutil
                shutil.rmtree(tmp_file)
        except Exception as e:
            print(f"Could not clean up {tmp_file}: {e}")

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
            ("finetune_1_pct", "finetune", "S"), ("finetune_1_pct", "test", "S"),
            ("finetune_3_16_pct", "finetune", "M"), ("finetune_3_16_pct", "test", "S"),
            ("finetune_10_pct", "finetune", "L"), ("finetune_10_pct", "test", "S"),
            ("finetune_31_6_pct", "finetune", "L"), ("finetune_31_6_pct", "test", "S"),
            ("finetune_100_pct", "finetune", "L"), ("finetune_100_pct", "test", "S"),
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
        print("Missing the following embedding files:\n")
        for m in missing:
            print(f"   - {m}")
        print(f"\nFound {len(missing)} missing files. Continuing with available files...\n")
        return False
    else:
        print("All required embedding files are present.\n")
        return True

def run_single(gex_model, dataset_name, img_model, split, scale):
    """Run a single configuration to generate embeddings."""
    # Determine the mode based on what models are provided
    finetune_models = ["full"] + [f"finetune_{size}_{cfg}" for size in ["1pct", "3.16pct", "10pct", "31.6pct", "100pct"] for cfg in ["cfg1", "cfg2", "cfg3"]]
    is_multimodal = (gex_model in finetune_models and img_model in IMG_MODELS)
    is_unimodal_gex = (gex_model in finetune_models and (img_model is None or img_model not in IMG_MODELS))
    is_unimodal_img = ((gex_model is None or gex_model not in finetune_models) and img_model in IMG_MODELS)

    # For test split, we don't need scale
    if split == "test":
        out_dir = LUSTRE_OUTPUT_DIR / dataset_name
        if is_unimodal_gex:
            out_dir = out_dir / f"gex_only_{gex_model}_test.{scale}"
        elif is_unimodal_img:
            out_dir = out_dir / f"img_only_{img_model}_test.{scale}"
        else:  # multimodal
            out_dir = out_dir / f"{gex_model}_{img_model}_test.{scale}"
    else:
        out_dir = LUSTRE_OUTPUT_DIR / dataset_name
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

    # Get the correct HF dataset ID for this dataset
    if dataset_name not in HF_DATASET_IDS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of: {list(HF_DATASET_IDS.keys())}")
    hf_dataset_id = HF_DATASET_IDS[dataset_name]

    dataset = load_dataset(
        hf_dataset_id,
        split="train",
        cache_dir=str(HF_CACHE_DIR),
        use_auth_token=hf_token,
    )

    cfg = load_data_splits(dataset_name=dataset_name)
    try:
        if gex_model and gex_model.startswith("finetune_"):
            # For finetune models (CoMM training), use ALL samples (train + test)
            # This ensures we have the full dataset for subset filtering during training
            sample_names = set()
            if "train" in cfg.multimodal and "L" in cfg.multimodal.train:
                sample_names.update(cfg.multimodal.train.L)
            elif "train" in cfg.multimodal:
                # Fallback: use the largest available scale
                available_scales = list(cfg.multimodal.train.keys())
                if available_scales:
                    largest_scale = max(available_scales, key=lambda x: len(cfg.multimodal.train[x]))
                    sample_names.update(cfg.multimodal.train[largest_scale])
                else:
                    raise ValueError(f"No train scales found in config for {dataset_name}")
            else:
                raise ValueError(f"No train split found in config for {dataset_name}")
            # Also add test samples if available
            if "test" in cfg.multimodal:
                sample_names.update(cfg.multimodal.test)
            print(f"Using ALL samples for finetune model: {len(sample_names)} unique sample names (train + test)")
        elif split == "test":
            sample_names = set(cfg.multimodal[split])
        else:
            sample_names = set(cfg.multimodal[split][scale])
    except KeyError as e:
        raise ValueError(f"No samples found for split={split}, scale={scale}, dataset={dataset_name}. Error: {e}")

    # Filter dataset by sample names from config
    # Dataset 'name' field format varies by dataset:
    #   - Thymus: Full donor ID (e.g., "WSSS_F_IMMsp11604686")
    #   - Breast: Full name (e.g., "TENX97_vectorized_annotated_allpolys_noCT")
    # Config sample_names format:
    #   - Thymus: Full donor IDs (e.g., ["WSSS_F_IMMsp11604686", ...])
    #   - Breast: Short donor IDs (e.g., ["TENX97", "TENX96", ...])
    # Matching strategy: Use prefix matching to handle both cases
    # Expected: Should match most/all samples (e.g., ~38k train, ~6k test for thymus)
    def matches_sample(x):
        name = x["name"]
        # Check exact match first
        if name in sample_names:
            return True
        # Check if any config name is a prefix of the dataset name
        # This handles breast case: "TENX97_vectorized_..." matches "TENX97"
        for config_name in sample_names:
            if name.startswith(config_name + "_") or name == config_name:
                return True
        return False

    dataset = dataset.filter(matches_sample)
    print(f"📦 Filtered to {len(dataset)} samples")
    if len(dataset) == 0:
        print("WARNING: Dataset is empty after filtering!")
        print(f"   Config sample_names (first 5): {list(sample_names)[:5]}")
        print(f"   Dataset name examples: {dataset['name'][:5] if len(dataset) > 0 else 'N/A'}")

    # Load embeddings based on mode
    if is_unimodal_gex:
        img_dict = {}  # Empty dict for img embeddings
        gex_data = load_gex_embeddings_with_names(gex_model, split, scale, dataset_name)
    elif is_unimodal_img:
        img_path = build_img_path(img_model, split, scale, dataset_name)
        if not img_path.exists():
            # Try loading from HF dataset as fallback
            print(f"IMG embedding file not found: {img_path}")
            print("   Attempting to load from HF dataset...")
            img_dict = load_img_from_hf_dataset(img_model, split, scale, dataset_name)
            if img_dict is None:
                raise FileNotFoundError(f"IMG embedding file not found: {img_path} and could not load from HF dataset")
        else:
            print(f"IMG embedding file: {img_path}")
            img_dict = load_h5_embeddings_with_names(img_path)
        gex_data = None  # Empty for gex embeddings
    else:  # multimodal case
        # For finetune models, need to load image embeddings from ALL sources (finetune + pretrain + test)
        # because the dataset includes train + test samples, but image embeddings are split across files
        if gex_model and gex_model.startswith("finetune_"):
            # Load from all three sources and combine
            img_dict = {}
            img_paths_to_try = [
                build_img_path(img_model, "finetune", scale, dataset_name),
                build_img_path(img_model, "pretrain", scale, dataset_name),
                build_img_path(img_model, "test", scale, dataset_name),
            ]

            for img_path in img_paths_to_try:
                if img_path.exists():
                    print(f"Loading image embeddings from: {img_path}")
                    partial_dict = load_h5_embeddings_with_names(img_path)
                    # Merge into main dict (later files override earlier ones if there are duplicates)
                    img_dict.update(partial_dict)
                    print(f"  Loaded {len(partial_dict)} embeddings (total: {len(img_dict)})")

            if len(img_dict) == 0:
                # Try loading from HF dataset as fallback
                print("No IMG embedding files found, attempting to load from HF dataset...")
                img_dict = load_img_from_hf_dataset(img_model, split, scale, dataset_name)
                if img_dict is None:
                    raise FileNotFoundError(
                        "IMG embedding files not found for finetune model. "
                        f"Tried: {[str(p) for p in img_paths_to_try]}"
                    )
            else:
                print(f"Loaded {len(img_dict)} total image embeddings from multiple sources")
        else:
            img_path = build_img_path(img_model, split, scale, dataset_name)
            if not img_path.exists():
                # Try loading from HF dataset as fallback
                print(f"IMG embedding file not found: {img_path}")
                print("   Attempting to load from HF dataset...")
                img_dict = load_img_from_hf_dataset(img_model, split, scale, dataset_name)
                if img_dict is None:
                    raise FileNotFoundError(f"IMG embedding file not found: {img_path} and could not load from HF dataset")
            else:
                print(f"IMG embedding file: {img_path}")
                img_dict = load_h5_embeddings_with_names(img_path)
        gex_data = load_gex_embeddings_with_names(gex_model, split, scale, dataset_name)

    enrich_and_save(dataset, gex_data, img_dict, out_dir)

def run_all():
    check_all_required_files()

    # Start with test splits for all combinations (most important for evaluation)
    print("Starting with test splits for all model combinations...")
    for model in ["full", "finetune_1_pct", "finetune_3_16_pct", "finetune_10_pct", "finetune_31_6_pct", "finetune_100_pct"]:
        for img_model in IMG_MODELS:
            try:
                run_single(model, "lung", img_model, "test", "S")
            except FileNotFoundError as e:
                print(f"Skipping {model}_{img_model}_test - file not found: {e}")
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
                print(f"Skipping {model}_{img_model}_{split}_{scale} - file not found: {e}")
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
                            print(f"Skipping img_only_{img_model}_{split}_{scale} - file not found: {e}")
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
                ("finetune_1_pct", "finetune", "S"), ("finetune_1_pct", "test", "S"),
                ("finetune_3_16_pct", "finetune", "M"), ("finetune_3_16_pct", "test", "S"),
                ("finetune_10_pct", "finetune", "L"), ("finetune_10_pct", "test", "S"),
                ("finetune_31_6_pct", "finetune", "L"), ("finetune_31_6_pct", "test", "S"),
                ("finetune_100_pct", "finetune", "L"), ("finetune_100_pct", "test", "S"),
            ]

            for gex_model, split, scale in valid_combinations:
                try:
                    run_single(gex_model, "lung", None, split, scale)
                except FileNotFoundError as e:
                    print(f"Skipping {gex_model}_{split}_{scale} - file not found: {e}")
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
            valid_models = ["full"] + [f"finetune_{size}_{cfg}" for size in ["1pct", "3.16pct", "10pct", "31.6pct", "100pct"] for cfg in ["cfg1", "cfg2", "cfg3"]]
            if args.gex_model_scale not in valid_models:
                raise ValueError(f"gex_model_scale must be one of: {', '.join(valid_models)}")
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
