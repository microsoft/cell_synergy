import os
import numpy as np
import anndata
import mygene
import torch
import torch.nn.functional as F
import warnings
from data_scaling.paths import MODEL_DIR

from ._nicheformer import Nicheformer
from ..utils import LUNG_GENE_NAMES, BREAST_GENE_NAMES, THYMUS_GENE_NAMES


class NicheformerEmbedder(torch.nn.Module):
    def __init__(
        self,
        local_dir,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        technology_mean="xenium_mean_script.npy",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        # Load technology-specific mean vector
        technology_mean = np.load(os.path.join(local_dir, technology_mean))
        self.technology_mean = torch.from_numpy(technology_mean).to(self.device)

        # Load reference AnnData for gene-to-token mapping
        self.nicheformer_reference = anndata.read_h5ad(
            os.path.join(local_dir, "nicheformer_reference.h5ad")
        )
        gene_names = self.nicheformer_reference.var_names
        self.gene_to_token_mapping = {gene: idx for idx, gene in enumerate(gene_names)}

        # Model
        self.max_seq_len = 1500
        self.aux_tokens = 30
        self.gene_mask = None
        self.token_ids = None

        self.model = Nicheformer.load_from_checkpoint(
            MODEL_DIR / "uni_gex" / "nicheformer.ckpt", strict=False
        )

    def _init_gene_mapping_from_adata(self, adata, gene_to_token_mapping):
        mg = mygene.MyGeneInfo()
        gene_symbols = adata.var_names.tolist()

        # Query mygene.info to convert gene symbols to Ensembl IDs
        results = mg.querymany(
            gene_symbols, scopes="symbol", fields="ensembl.gene", species="human"
        )

        symbol_to_ensembl = {}
        ambiguous_count = 0
        for r in results:
            gene_symbol = r["query"]
            ensembl_field = r.get("ensembl")
            if isinstance(ensembl_field, list):
                all_ids = [
                    entry.get("gene") for entry in ensembl_field if "gene" in entry
                ]
                matched_ids = [
                    eid
                    for eid in all_ids
                    if eid in self.nicheformer_reference.var_names
                ]
                if len(matched_ids) > 1:
                    warnings.warn(
                        f"Gene symbol '{gene_symbol}' has multiple matching Ensembl IDs in reference: {matched_ids}"
                    )
                selected_id = next(
                    (
                        eid
                        for eid in all_ids
                        if eid in self.nicheformer_reference.var_names
                    ),
                    None,
                )
                if selected_id is None and all_ids:
                    selected_id = all_ids[0]
                symbol_to_ensembl[gene_symbol] = selected_id
                ambiguous_count += 1
            elif isinstance(ensembl_field, dict):
                symbol_to_ensembl[gene_symbol] = ensembl_field.get("gene")
            else:
                symbol_to_ensembl[gene_symbol] = None

        print(
            f"\nTotal genes with multiple Ensembl hits: {ambiguous_count} of {len(results)}"
        )

        adata.var["gene_symbol"] = adata.var_names
        adata.var_names = adata.var_names.map(symbol_to_ensembl)

        adata_ensembls = set(adata.var_names)
        reference_ensembls = set(self.nicheformer_reference.var_names)
        missing_in_reference = {
            eid for eid in (adata_ensembls - reference_ensembls) if eid is not None
        }

        missing_symbols = adata.var.loc[
            adata.var_names.isin(missing_in_reference), "gene_symbol"
        ]
        print(
            f"\n{len(missing_in_reference)} Ensembl IDs in adata not found in nicheformer_reference."
        )
        print("Corresponding gene symbols:")
        print(missing_symbols.tolist())

        token_ids = [gene_to_token_mapping.get(gene, -1) for gene in adata.var_names]
        adata.var["token_id"] = token_ids

        self.gene_mask = adata.var["token_id"] != -1
        adata = adata[:, self.gene_mask]
        self.token_ids = torch.tensor(adata.var["token_id"].values).to(self.device)

    def tokenize_genes(self, X: torch.Tensor):
        exp_matrix = X[:, self.gene_mask]
        counts_per_cell = torch.mean(exp_matrix, dim=1)
        counts_per_cell[counts_per_cell == 0] = 1
        scaling_factor = 10_000 / counts_per_cell
        exp_matrix = exp_matrix * scaling_factor[:, None]

        tech_mean = torch.nan_to_num(self.technology_mean)
        tech_mean[tech_mean == 0] = 1
        exp_matrix = exp_matrix / tech_mean[self.token_ids][None, :]

        sorted_ids = torch.zeros_like(exp_matrix, device=self.device, dtype=int)
        sorted_idx = torch.argsort(-exp_matrix, dim=1)
        for idx in range(exp_matrix.shape[0]):
            sorted_ids[idx, :] = self.token_ids[sorted_idx[idx, :]] + self.aux_tokens
        return sorted_ids

    def prepare_tokens(self, sorted_ids: torch.Tensor):
        tokens_final = torch.zeros(
            (sorted_ids.shape[0], self.max_seq_len),
            device=self.device,
            dtype=torch.long,
        )
        for idx in range(sorted_ids.shape[0]):
            tokens = torch.tensor(sorted_ids[idx][: self.max_seq_len]).unsqueeze(0)
            padding = self.max_seq_len - tokens.shape[1]
            tokens = F.pad(tokens, (0, padding)).to(torch.int)
            tokens_final[idx, :] = tokens
        return tokens_final

    def forward(self, x_real):
        if x_real.size(0) == 0:
            raise ValueError(
                "No real cells found in the input data. Check your mask or input data."
            )

        if self.token_ids is None:
            adata = anndata.AnnData(X=x_real.cpu().numpy())
            num_genes = adata.shape[1]
            if num_genes == 343:
                adata.var.index = LUNG_GENE_NAMES
            elif num_genes == 280:
                adata.var.index = BREAST_GENE_NAMES
            elif num_genes == 2000:
                adata.var.index = THYMUS_GENE_NAMES
            else:
                raise NotImplementedError(
                    f"Gene set with {num_genes} genes is not supported."
                )
            self._init_gene_mapping_from_adata(adata, self.gene_to_token_mapping)

        n_cells = x_real.size(0)
        embeddings = []
        modality = torch.tensor(4).to(self.device)
        specie = torch.tensor(5).to(self.device)
        assay = torch.tensor(9).to(self.device)

        with torch.no_grad():
            for i in range(0, n_cells, self.batch_size):
                xb = x_real[i : i + self.batch_size]
                sorted_ids = self.tokenize_genes(xb)
                tokens_final = self.prepare_tokens(sorted_ids)
                b = {
                    "X": tokens_final,
                    "modality": modality.expand(tokens_final.size(0)),
                    "specie": specie.expand(tokens_final.size(0)),
                    "assay": assay.expand(tokens_final.size(0)),
                }
                emb = self.model.get_embeddings(batch=b, layer=-1)
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)


def compute_nicheformer_embeddings(
    gexp,
    cell_mask,
    nicheformer_model,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    X = np.array(gexp)
    mask = (
        np.array(cell_mask, dtype=bool)
        if cell_mask is not None
        else (X.sum(axis=1) > 0)
    )
    X_filtered = X[mask]
    if X_filtered.shape[0] == 0:
        print("Warning: No valid cells; using all data.")
        X_filtered = X

    pool_tensor = torch.tensor(X_filtered, dtype=torch.float32)
    with torch.no_grad():
        emb_pool = nicheformer_model(pool_tensor.to(device))
    nicheformer_pool = np.mean(emb_pool.cpu().numpy(), axis=0)

    pseudobulk_counts = np.mean(X_filtered, axis=0)
    pseudobulk_tensor = torch.tensor(
        pseudobulk_counts.reshape(1, -1), dtype=torch.float32
    )
    with torch.no_grad():
        emb_pseudobulk = nicheformer_model(pseudobulk_tensor.to(device))
    nicheformer_pseudobulk = emb_pseudobulk.cpu().numpy().squeeze()

    return [nicheformer_pool], [nicheformer_pseudobulk]


def compute_nicheformer_embeddings_batched(
    gexp,
    cell_mask,
    nicheformer_model,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Flatten data and mask
    batch_size, num_cells, num_genes = gexp.shape
    x_flat = gexp.view(-1, num_genes)  # Shape: (batch_size * num_cells, num_genes)

    # If no mask is provided, consider all cells as valid
    if cell_mask is None:
        mask_flat = torch.ones(batch_size * num_cells, dtype=torch.bool, device=device)
    else:
        mask_flat = cell_mask.view(-1)  # Shape: (batch_size * num_cells,)

    # Filter real cells using the mask
    x_real = x_flat[mask_flat]

    # Remove empty rows (cells with all-zero values) from x_real
    non_empty_mask = x_real.sum(dim=1) > 0
    x_real = x_real[non_empty_mask]  # Shape: (num_real_cells, num_genes)

    # Ensure x_real has at least one cell
    if x_real.size(0) == 0:
        raise ValueError(
            "No real cells found in the input data. Check your mask or input data."
        )

    # Compute batch indices for real cells
    real_cell_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
    batch_indices = real_cell_indices // num_cells
    batch_indices = batch_indices[non_empty_mask]  # Shape: (num_real_cells,)

    with torch.no_grad():
        emb_pool = nicheformer_model(
            x_real.to(device)
        )  # Shape: (num_real_cells, emb_dim)

    emb_pool = emb_pool.cpu().numpy()
    batch_indices = batch_indices.cpu().numpy()

    # Initialize list of pooled embeddings
    nicheformer_pool = []
    for i in range(batch_size):
        indices = np.where(batch_indices == i)[0]
        if len(indices) == 0:
            nicheformer_pool.append(np.zeros(emb_pool.shape[1], dtype=np.float32))
        else:
            nicheformer_pool.append(
                np.mean(emb_pool[indices], axis=0).astype(np.float32)
            )

    return nicheformer_pool
