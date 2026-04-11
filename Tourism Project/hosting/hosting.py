from huggingface_hub import HfApi, create_repo
import os

# Use the token passed from GitHub Actions
token = os.getenv("HF_TOURISM_PACKAGE_PREDICTION_TOKEN")
api = HfApi(token=token)
repo_id = "Thaseen75/Tourism_Package_Prediction"

# 1. Check if the Space exists; if not, create it
try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space '{repo_id}' exists. Proceeding with upload...")
except Exception:
    print(f"Space '{repo_id}' not found. Creating new Streamlit space...")
    create_repo(
        repo_id=repo_id, 
        token=token, 
        repo_type="space", 
        space_sdk="streamlit",  # Change to "gradio" if using Gradio
        private=False
    )

# 2. Upload the deployment folder
api.upload_folder(
    folder_path="Tourism Project/deployment", 
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
)
print("Upload successful!")
