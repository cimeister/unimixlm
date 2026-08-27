"""
Utility functions for LAS tokenizer training.

Includes HuggingFace Hub upload functionality.
"""

import logging
from pathlib import Path


def push_to_hf(output_dir: Path, hf_username: str, repo_name: str) -> None:
    """Create/update a public HF repo and upload tokenizer JSON artifacts.

    Token resolution: HF_TOKEN env var > HUGGING_FACE_HUB_TOKEN >
    HUGGINGFACE_TOKEN > HfFolder. The orchestrator hardcodes HF_TOKEN in
    .env so this should always have a token without `huggingface-cli login`.
    """
    import os
    from huggingface_hub import HfApi
    # HfFolder was removed from the top-level export in huggingface_hub >=1.0;
    # try the legacy import path, fall through to None if unavailable.
    try:
        from huggingface_hub import HfFolder
        _folder_token = HfFolder.get_token()
    except (ImportError, AttributeError):
        _folder_token = None

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or _folder_token
    )
    if not token:
        raise RuntimeError(
            "No HF token found. Set HF_TOKEN in env (or run "
            "`huggingface-cli login`) before pushing."
        )

    api = HfApi(token=token)
    repo_id = f"{hf_username}/{repo_name}"
    api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        token=token,
        allow_patterns=["*.json"],
        ignore_patterns=[".git/*"],
    )
    logging.info("Pushed to HF repo: https://huggingface.co/%s", repo_id)
