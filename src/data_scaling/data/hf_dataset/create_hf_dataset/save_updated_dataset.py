import os
import numpy as np
from datasets import load_dataset

token = os.getenv("HF_TOKEN")

organ = "breast"

if organ == "lung":
    dataset_name = "theislab-multimodal-ssl/lung_GAT_scvi_PCA_UNI_CONCH_CTtranspath"
elif organ == "breast":
    dataset_name = (
        "theislab-multimodal-ssl/breast_7samples_GAT_scvi_PCA_UNI_CONCH_CTtranspath"
    )
elif organ == "thymus":
    dataset_name = "theislab-multimodal-ssl/thymus_ts2_conch_ctranspath_uni_scvi_pca"
else:
    raise ValueError(f"Unknown organ: {organ}")

cache_dir = (
    "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0003/merel.kuijs/multimodal-ssl/data/raw"
)

ds = load_dataset(
    dataset_name,
    token=token,
    cache_dir=cache_dir,
)

full_path = f"/dss/dssfs03/tumdss/pn36po/pn36po-dss-0003/merel.kuijs/create_hf_dataset/data/{organ}/paired_ov0_ps224_ts1.dataset"
niche_pool_arr = np.load(os.path.join(full_path, "nicheformer", "niche_pool.npy"))
scgpt_pool_arr = np.load(os.path.join(full_path, "scgpt", "scgpt_pool.npy"))

data_split = ds["train"] if "train" in ds else ds
data_split = data_split.add_column("nicheformer_pool", niche_pool_arr.tolist())
data_split = data_split.add_column("scgpt_pool", scgpt_pool_arr.tolist())

data_split.save_to_disk(full_path)
print(f"Saved updated dataset to {full_path}")
