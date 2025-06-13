from huggingface_hub import HfApi

token = os.getenv("HF_TOKEN")

# Initialize the API
api = HfApi(token=token)

# Define your repository and local folder paths
repo_id = "theislab-multimodal-ssl/breast_embeddings"  # Your dataset repository
repo_type = "dataset"  # Set the repository type to 'dataset'
folder_path = "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0003/merel.kuijs/create_hf_dataset/data/breast/paired_ov0_ps224_ts1.dataset/breast_7samples_embeddings"  # Path to your local folder

# Upload the entire folder to Hugging Face
api.upload_large_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=folder_path,
    num_workers=18,
    ignore_patterns=["*.txt"],  # ignore log files
)

print("Folder uploaded successfully!")
