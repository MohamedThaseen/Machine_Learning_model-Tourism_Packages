from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOURISM_PACKAGE_PREDICTION_TOKEN"))
api.upload_folder(
    folder_path="Tourism Project/deployment",     # the local folder containing your files
    repo_id="Thaseen75/Tourism_Package_Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
