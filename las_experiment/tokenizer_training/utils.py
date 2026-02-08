"""
Utility functions for LAS tokenizer training.

Includes HuggingFace Hub upload functionality.
"""

import logging
from pathlib import Path


def push_to_hf(output_dir: Path, hf_username: str, repo_name: str) -> None:
    """Create/update a public HF repo and upload tokenizer JSON artifacts."""
    from huggingface_hub import HfApi, HfFolder

    api = HfApi()
    token = HfFolder.get_token()
    repo_id = f"{hf_username}/{repo_name}"
    api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        token=token,
        allow_patterns=["*.json"],  # save only .json tokenizer files.
        ignore_patterns=[".git/*"],
    )
    logging.info("Pushed to HF repo: https://huggingface.co/%s", repo_id)
