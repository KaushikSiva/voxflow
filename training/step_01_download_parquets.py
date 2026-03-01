#!/usr/bin/env python3
"""Step 01: Download Tamil parquet shards from IndicVoices."""

from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_ID = "ai4bharat/IndicVoices"
LOCAL_DIR = Path("~/datasets/indicvoices_parquet").expanduser()
ALLOW_PATTERNS = ["**/tamil/*.parquet"]


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
    )
    print("Downloaded to:", path)


if __name__ == "__main__":
    main()
