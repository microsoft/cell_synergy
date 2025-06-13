I use `pca_scvi_niche_scgpt.sh` to run `pca_scvi_niche_scgpt_hf.py`, which computes embeddings and adds them to an existing Hugging Face dataset. You can ignore `save_updated_dataset.py` (it's similar but only used for adding precomputed embeddings).

Once all embeddings are added, I upload the dataset to HF using `upload.py`.

Just a heads-up: there are some private tokens in the repo history, so please be cautious when making anything public :slightly_smiling_face: The current code is clean and doesn't contain any secrets.
