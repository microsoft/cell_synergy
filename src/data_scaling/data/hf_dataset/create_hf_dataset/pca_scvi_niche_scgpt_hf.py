#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from process.loader.get_dataloader import ExpressionDataLoader

# --------------------------------------------------------------------------- #
# 1.  Embedding helpers                                                      #
# --------------------------------------------------------------------------- #


def load_scvi_model(scvi_model_path, train_adata_path):
    """
    Load the pretrained scVI model using the AnnData that was used for training.
    """
    import anndata
    import scvi

    train_adata = anndata.read_h5ad(train_adata_path)
    scvi.model.SCVI.setup_anndata(train_adata, layer="counts")
    model = scvi.model.SCVI.load(scvi_model_path, adata=train_adata)
    return model


def fit_pca_model(dataset, n_components=128):
    """
    Fit an IncrementalPCA model on the entire dataset.
    """
    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=n_components)
    for row in tqdm(dataset, desc="Fitting PCA on dataset"):
        X_patch = np.array(row["gexp"])
        ipca.partial_fit(X_patch)
    return ipca


def compute_pca_embeddings(gexp, cell_mask, pca_model):
    X = np.array(gexp)
    if cell_mask is not None:
        mask = np.array(cell_mask, dtype=bool)
        X_filtered = X[mask]
    else:
        X_filtered = X[np.sum(X, axis=1) > 0]
    if X_filtered.shape[0] == 0:
        print("Warning: No valid cells in this patch; using unfiltered data.")
        X_filtered = X
    cell_embeddings = pca_model.transform(X_filtered)
    pca_pool = cell_embeddings.mean(axis=0)
    pseudobulk_counts = X_filtered.mean(axis=0)
    pca_pseudobulk = pca_model.transform(pseudobulk_counts.reshape(1, -1))[0]
    return pca_pool, pca_pseudobulk


def compute_scvi_embeddings(gexp, cell_mask, scvi_model):
    import anndata

    X = np.array(gexp)
    if cell_mask is not None:
        mask = np.array(cell_mask, dtype=bool)
        X_filtered = X[mask]
    else:
        X_filtered = X[np.sum(X, axis=1) > 0]
    if X_filtered.shape[0] == 0:
        print("Warning: No valid cells in this patch; using unfiltered data.")
        X_filtered = X
    # scVI pool
    adata_pool = anndata.AnnData(X=X_filtered)
    adata_pool.layers["counts"] = X_filtered.copy()
    scvi_model.setup_anndata(adata_pool, layer="counts")
    cell_latents = scvi_model.get_latent_representation(adata_pool)
    scvi_pool = cell_latents.mean(axis=0)
    # scVI pseudobulk
    pseudobulk_counts = X_filtered.mean(axis=0)
    adata_pseudo = anndata.AnnData(X=pseudobulk_counts.reshape(1, -1))
    adata_pseudo.layers["counts"] = adata_pseudo.X.copy()
    scvi_model.setup_anndata(adata_pseudo, layer="counts")
    scvi_pseudobulk = scvi_model.get_latent_representation(adata_pseudo)[0]
    return scvi_pool, scvi_pseudobulk


# --------------------------------------------------------------------------- #
# 2.  Main pipeline                                                           #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Compute missing PCA, scVI, Nicheformer, and scGPT embeddings."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset_name", type=str, help="HF dataset ID (e.g. org/dataset)."
    )
    group.add_argument(
        "--dataset_path",
        type=str,
        help="Path to a local dataset saved with `Dataset.save_to_disk`.",
    )
    parser.add_argument(
        "--scvi_model_path",
        type=str,
        default="/lustre/groups/shared/users/multimodal-ssl/model_weights/scVI/lung/model_dir",
        help="Path to the pretrained scVI model file.",
    )
    parser.add_argument(
        "--scvi_train_adata_path",
        type=str,
        default="/lustre/groups/shared/users/multimodal-ssl/model_weights/scVI/lung/lung_train_median55.h5ad",
        help="Path to the AnnData file used for training the scVI model.",
    )
    parser.add_argument(
        "--n_components", type=int, default=128, help="Number of PCA components."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the updated dataset will be saved.",
    )
    # Optional flags to indicate which embeddings to compute
    parser.add_argument("--pca", action="store_true", help="Compute PCA embedding")
    parser.add_argument("--scvi", action="store_true", help="Compute scVI embedding")
    parser.add_argument("--niche", action="store_true", help="Compute NF embedding")
    parser.add_argument("--scgpt", action="store_true", help="Compute scGPT embedding")
    args = parser.parse_args()

    nicheformer_dir = "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0003/merel.kuijs/multimodal-ssl/assets/ckpts/Nicheformer"
    scgpt_dir = "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0003/merel.kuijs/multimodal-ssl/assets/ckpts/scGPT_human"
    cache_dir = (
        "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0003/merel.kuijs/multimodal-ssl/data/raw"
    )

    # Load dataset (HF or local)
    if args.dataset_path:
        ds = load_from_disk(args.dataset_path)
    else:
        token = os.getenv("HF_TOKEN")
        ds = load_dataset(
            args.dataset_name,
            token=token,
            cache_dir=cache_dir,
        )

    data_split = ds["train"] if "train" in ds else ds

    # Determine which embeddings to compute
    calc_pca = args.pca
    calc_scvi = args.scvi
    calc_niche = args.niche
    calc_scgpt = args.scgpt

    # Import embedder functions only when needed
    if calc_niche:
        from process.nicheformer.embedder import (
            compute_nicheformer_embeddings,
            compute_nicheformer_embeddings_batched,
            NicheformerEmbedder,
        )
    if calc_scgpt:
        from process.scGPT.embedder import (
            compute_scgpt_embeddings,
            compute_scgpt_embeddings_batched,
            ScGPTEmbedder,
        )

    if not any([calc_pca, calc_scvi, calc_niche, calc_scgpt]):
        print(
            "No embedding flags were provided (e.g. --pca, --scvi, --niche, --scgpt). Skipping computation and saving unchanged dataset."
        )
        data_split.save_to_disk(args.output_dir)
        return

    # Print selected embeddings
    to_compute = []
    if calc_pca:
        to_compute.append("PCA")
    if calc_scvi:
        to_compute.append("scVI")
    if calc_niche:
        to_compute.append("Nicheformer")
    if calc_scgpt:
        to_compute.append("scGPT")

    print(f"Computing the following embeddings: {', '.join(to_compute)}")

    # Fit/load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if calc_pca:
        pca_model = fit_pca_model(data_split, n_components=args.n_components)
    if calc_scvi:
        scvi_model = load_scvi_model(args.scvi_model_path, args.scvi_train_adata_path)
    if calc_niche:
        nicheformer_model = NicheformerEmbedder(nicheformer_dir, device=device)
        nicheformer_model = nicheformer_model.to(device)
        nicheformer_model.eval()
    if calc_scgpt:
        scgpt_model = ScGPTEmbedder(scgpt_dir, device=device)
        scgpt_model = scgpt_model.to(device)
        scgpt_model.eval()

    # Prepare result lists
    if calc_pca:
        pca_pool_list, pca_pseudo_list = [], []
    if calc_scvi:
        scvi_pool_list, scvi_pseudo_list = [], []
    if calc_niche:
        niche_pool_list, niche_pseudo_list = [], []
    if calc_scgpt:
        scgpt_pool_list, scgpt_pseudo_list = [], []

    dataloader = ExpressionDataLoader(data_split).get_loader(
        batch_size=64, num_workers=8
    )

    for batch in tqdm(dataloader, desc="Computing embeddings"):
        gexps = batch["gexp"]
        masks = batch.get("mask")

        if calc_niche:
            pool = compute_nicheformer_embeddings_batched(
                gexps, masks, nicheformer_model, device="cuda"
            )
            niche_pool_list.extend(pool)

        if calc_scgpt:
            pool = compute_scgpt_embeddings_batched(
                gexps, masks, scgpt_model, device="cuda"
            )
            scgpt_pool_list.extend(pool)

        # Models that require per-sample input
        for gexp, mask in zip(gexps, masks):
            if calc_pca:
                pool, pseudo = compute_pca_embeddings(
                    gexp, cell_mask=mask, pca_model=pca_model
                )
                pca_pool_list.append(pool.tolist())
                pca_pseudo_list.append(pseudo.tolist())
            if calc_scvi:
                pool, pseudo = compute_scvi_embeddings(
                    gexp, cell_mask=mask, scvi_model=scvi_model
                )
                scvi_pool_list.append(pool.tolist())
                scvi_pseudo_list.append(pseudo.tolist())

    # Save embedding arrays separately
    subdir = "paired_ov0_ps224_ts1.dataset"
    full_path = os.path.join(args.output_dir, subdir)
    os.makedirs(full_path, exist_ok=True)
    if calc_pca:
        np.save(os.path.join(full_path, "pca_pool.npy"), np.array(pca_pool_list))
        np.save(
            os.path.join(full_path, "pca_pseudobulk.npy"), np.array(pca_pseudo_list)
        )
    if calc_scvi:
        np.save(os.path.join(full_path, "scvi_pool.npy"), np.array(scvi_pool_list))
        np.save(
            os.path.join(full_path, "scvi_pseudobulk.npy"), np.array(scvi_pseudo_list)
        )
    if calc_niche:
        niche_pool_list = [np.array(x, dtype=np.float32) for x in niche_pool_list]
        np.save(os.path.join(full_path, "niche_pool.npy"), np.array(niche_pool_list))
        # np.save(
        #     os.path.join(full_path, "nicheformer_pseudobulk.npy"),
        #     np.array(niche_pseudo_list),
        # )
    if calc_scgpt:
        scgpt_pool_list = [np.array(x, dtype=np.float32) for x in scgpt_pool_list]
        np.save(os.path.join(full_path, "scgpt_pool.npy"), np.array(scgpt_pool_list))
        # np.save(
        #     os.path.join(full_path, "scgpt_pseudobulk.npy"), np.array(scgpt_pseudo_list)
        # )

    print(f"Saved embedding arrays to {full_path}")

    # Add new columns
    if calc_pca:
        data_split = data_split.add_column("pca_pool", pca_pool_list)
        data_split = data_split.add_column("pca_pseudobulk", pca_pseudo_list)
    if calc_scvi:
        data_split = data_split.add_column("scvi_pool", scvi_pool_list)
        data_split = data_split.add_column("scvi_pseudobulk", scvi_pseudo_list)
    if calc_niche:
        data_split = data_split.add_column("nicheformer_pool", niche_pool_list)
        # data_split = data_split.add_column("nicheformer_pseudobulk", niche_pseudo_list)
    if calc_scgpt:
        data_split = data_split.add_column("scgpt_pool", scgpt_pool_list)
        # data_split = data_split.add_column("scgpt_pseudobulk", scgpt_pseudo_list)

    # Save updated dataset
    data_split.save_to_disk(full_path)
    print(f"Saved updated dataset to {full_path}")


if __name__ == "__main__":
    main()
