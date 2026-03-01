import os
from huggingface_hub import snapshot_download

LOCAL_DIR = os.path.expanduser("~/datasets/indicvoices_parquet")
os.makedirs(LOCAL_DIR, exist_ok=True)

path = snapshot_download(
    repo_id="ai4bharat/IndicVoices",
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=["**/tamil/*.parquet"],
)
print("✅ downloaded to:", path)
