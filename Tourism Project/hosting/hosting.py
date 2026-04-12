import os
from huggingface_hub import HfApi

token = os.getenv("HF_TOURISM_PACKAGE_PREDICTION_TOKEN")
api = HfApi(token=token)
repo_id = "Thaseen75/Tourism_Package_Prediction"

try:
    # Check if exists
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space '{repo_id}' already exists.")
except Exception:
    print(f"Creating new Streamlit space...")
    # Use api.create_repo directly instead of the standalone create_repo function
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="streamlit",  # Ensure lowercase
        private=False,
        exist_ok=True # This prevents errors if it was created a microsecond ago
    )

# Upload logic
api.upload_folder(
    folder_path="Tourism Project/deployment", 
    repo_id=repo_id,
    repo_type="space",
    path_in_repo=".", 
)
