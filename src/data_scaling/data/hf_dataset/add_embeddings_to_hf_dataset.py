import os
import numpy as np
import h5py
from datasets import load_dataset, Dataset
from tqdm import tqdm
from data_scaling.paths import UNI_EMBEDDINGS_DIR, PROJECT_DIR
from data_scaling.config import load_data_splits

# --- CONFIG ---
HF_DATASET_DIR = PROJECT_DIR / "hf"
HF_CACHE_DIR = PROJECT_DIR / "hf_cache"
SPLITS = ["S", "M", "L"]
GEX_PATHS = {
    "S": UNI_EMBEDDINGS_DIR / "gex" / "nicheformer_pool_S.npy",
    "M": UNI_EMBEDDINGS_DIR / "gex" / "nicheformer_pool_M.npy",
    "L": UNI_EMBEDDINGS_DIR / "gex" / "nicheformer_pool_L.npy",
}
IMG_PATHS = {
    "S": UNI_EMBEDDINGS_DIR / "img" / "img_200M_S_pretrain.h5",
    "M": UNI_EMBEDDINGS_DIR / "img" / "img_200M_M_pretrain.h5",
    "L": UNI_EMBEDDINGS_DIR / "img" / "img_200M_L_pretrain.h5",
}

HF_DATASET_ID = "theislab-multimodal-ssl/paired-image-gexp-xenium-lungmed55-broadCT6"

def load_nicheformer_embeddings(npy_path):
    arr = np.load(npy_path)
    return arr

def load_img_embeddings_with_names(h5_path):
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        sample_names = [x.decode() for x in f["sample_names"][:]]
    return dict(zip(sample_names, embeddings)), sample_names

def check_correspondence(hf_dataset, img_sample_names, gex_sample_names=None):
    hf_names = hf_dataset["name"]
    if len(hf_names) != len(img_sample_names):
        raise ValueError(f"HF dataset and image embedding sample name count mismatch: {len(hf_names)} vs {len(img_sample_names)}")
    for i, (hf_name, img_name) in enumerate(zip(hf_names, img_sample_names)):
        if hf_name != img_name:
            raise ValueError(f"[IMG] Mismatch at index {i}: HF name={hf_name}, IMG name={img_name}")
    if gex_sample_names is not None:
        if len(hf_names) != len(gex_sample_names):
            raise ValueError(f"HF dataset and GEX embedding sample name count mismatch: {len(hf_names)} vs {len(gex_sample_names)}")
        for i, (hf_name, gex_name) in enumerate(zip(hf_names, gex_sample_names)):
            if hf_name != gex_name:
                raise ValueError(f"[GEX] Mismatch at index {i}: HF name={hf_name}, GEX name={gex_name}")
    print("All sample names match between HF dataset, image embeddings, and GEX embeddings.")

def main():
    # Load HF dataset from the Hub
    hf_token = os.environ.get("HF_DATASETS_TOKEN")
    print("Loading HF dataset from the Hub...")
    dataset = load_dataset(
        HF_DATASET_ID,
        split="train",
        cache_dir=str(HF_CACHE_DIR),
        use_auth_token=hf_token,
    )

    # Load splits config
    splits_cfg = load_data_splits(dataset_name="lung")
    split_sample_names = splits_cfg.multimodal["pretrain"]

    for split in SPLITS:
        print(f"\nProcessing split: {split}")
        sample_names = set(split_sample_names[split])
        # Filter dataset to only relevant samples
        split_dataset = dataset.filter(lambda x: x["name"] in sample_names)
        print(f"Filtered to {len(split_dataset)} samples for split {split}.")
        # Load embeddings
        nicheformer_arr = load_nicheformer_embeddings(GEX_PATHS[split])
        img_dict, img_sample_names = load_img_embeddings_with_names(IMG_PATHS[split])
        # Build mapping from sample name to embedding index for GEX
        gex_name_to_idx = {name: i for i, name in enumerate(img_sample_names)}
        # Check correspondence for only the split
        for name in sample_names:
            if name not in img_sample_names:
                raise ValueError(f"Sample {name} in split {split} not found in image embeddings!")
        print("All split sample names found in image embeddings.")

        def add_columns(batch):
            out_gex = []
            out_img = []
            for name in batch["name"]:
                idx = gex_name_to_idx[name]
                out_gex.append(nicheformer_arr[idx].tolist())
                out_img.append(img_dict[name].tolist())
            return {
                "nicheformer_pool": out_gex,
                "img_uni_pool": out_img,
            }

        split_dataset = split_dataset.map(
            add_columns,
            batched=True,
            batch_size=1024,
            desc=f"Adding embeddings for split {split}"
        )

        # Save updated split dataset
        split_dir = PROJECT_DIR / f"hf_{split}"
        print(f"Saving split {split} dataset to {split_dir}")
        split_dataset.save_to_disk(str(split_dir))
        print(f"Done with split {split}.")

    print("All splits processed and saved.")

    # --- Direct upload (optional) ---
    # from huggingface_hub import HfApi
    # api = HfApi(token=hf_token)
    # api.upload_folder(
    #     repo_id="your-username/your-dataset-name",
    #     repo_type="dataset",
    #     folder_path=str(OUTPUT_PATH),
    #     ignore_patterns=["*.txt"],
    # )
    # print("Uploaded to Hugging Face Hub.")

if __name__ == "__main__":
    main()