import os
from huggingface_hub import HfApi

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
    api.create_repo(
        repo_id=repo_id, 
        repo_type="space", 
        space_sdk="streamlit", 
        private=False
    )

# 2. Upload the deployment folder
# Note: Ensure app.py and requirements.txt are in this folder!
api.upload_folder(
    folder_path="Tourism Project/deployment", 
    repo_id=repo_id,
    repo_type="space",
    path_in_repo=".", # "." ensures it goes to the root
)

print(f"Upload successful! View your app at: https://huggingface.co/spaces/{repo_id}")
