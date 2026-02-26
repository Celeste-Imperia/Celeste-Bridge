from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
# The New Branded Repository Name
repo_id = "CelesteImperia/Celeste-Bridge"

# 1. Create the repository
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"--- Repository Ready: {repo_id} ---")
except Exception as e:
    print(f"Repo creation or access failed: {e}")

# 2. Upload the Hybrid Efficiency Log from your metadata folder
try:
    api.upload_file(
        path_or_fileobj=r"F:\Porting_Project\metadata\hybrid_efficiency_report.json",
        path_in_repo="metadata/hybrid_efficiency_report.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial Release: Hybrid CPU/GPU Efficiency Log"
    )
    print("--- Success: Hybrid Log Synced to Hugging Face ---")
    print(f"Your first asset is live: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Upload failed: {e}")