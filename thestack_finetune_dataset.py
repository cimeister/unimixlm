"""Create and push per-language code finetuning datasets to HuggingFace Hub.

Streams source code from bigcode/the-stack for specified programming languages,
collects up to a target token count per language (counted with the OLMo-3 tokenizer),
sanitises the code, saves as parquet, and pushes each language to its own HF repo.

HF repo naming: {hf_user}/LangMap-TheStack-{lang}-{size_label}
  e.g. MultilingualUnigramLM/LangMap-TheStack-rust-100M

Usage examples:
    # Build and push 100M-token datasets for Tier 1 languages
    python thestack_finetune_dataset.py \\
        --hf-user MultilingualUnigramLM \\
        --target-tokens 100000000 \\
        --languages python javascript java c cpp csharp go typescript

    # Build a 5M-token dataset for a single language
    python thestack_finetune_dataset.py \\
        --hf-user MultilingualUnigramLM \\
        --target-tokens 5000000 \\
        --languages rust

    # Verify a pushed dataset by printing samples
    python thestack_finetune_dataset.py \\
        --hf-user MultilingualUnigramLM \\
        --target-tokens 100000000 \\
        --verify rust
"""

import argparse
import logging
import os
import tempfile

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 30 programming languages supported by The Stack / code tokenizers.
CODE_LANGUAGES = [
    "assembly", "batchfile", "c++", "c", "c-sharp", "cmake", "css",
    "dockerfile", "fortran", "go", "haskell", "html", "java",
    "javascript", "julia", "lua", "makefile", "markdown", "perl",
    "php", "powershell", "python", "ruby", "rust",
    "scala", "shell", "sql", "tex", "typescript", "visual-basic",
]

# Filesystem / HF-safe name mapping (matches code_tokenizer_training.py).
SAFE_LANG_NAME: dict[str, str] = {
    "c++":          "cpp",
    "c-sharp":      "csharp",
    "visual-basic": "visual_basic",
}

SOURCE_DATASET = "bigcode/the-stack"
TOKENIZER_NAME = "allenai/OLMo-3-1025-7B"

# Sanitisation limits (same as code_tokenizer_training.py).
MAX_LINE_LENGTH: int = 2_000


def safe_name(lang: str) -> str:
    """Return the filesystem / HF-safe stem for a language name."""
    return SAFE_LANG_NAME.get(lang, lang)


# Reverse mapping: safe name -> The Stack directory name.
# Needed when the user passes safe names (e.g. "cpp") on the CLI but
# The Stack uses the original names (e.g. "c++") as data_dir.
UNSAFE_LANG_NAME: dict[str, str] = {v: k for k, v in SAFE_LANG_NAME.items()}


def stack_dir_name(lang: str) -> str:
    """Return the data_dir name used by The Stack for a language.

    Accepts both original names ("c++") and safe names ("cpp").
    """
    return UNSAFE_LANG_NAME.get(lang, lang)


def _human_size(n_tokens: int) -> str:
    """Return a short human label like '5M' or '100M' for the target token count."""
    if n_tokens >= 1_000_000_000:
        return f"{n_tokens // 1_000_000_000}B"
    elif n_tokens >= 1_000_000:
        return f"{n_tokens // 1_000_000}M"
    elif n_tokens >= 1_000:
        return f"{n_tokens // 1_000}K"
    return str(n_tokens)


def _sanitise(text: str) -> str:
    """Light sanitisation of source code before tokenising.

    Removes null bytes, lone surrogates, and lines > MAX_LINE_LENGTH
    (minified / auto-generated junk with no useful tokenisation signal).
    """
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.replace("\x00", "")
    lines = [ln for ln in text.splitlines() if len(ln) <= MAX_LINE_LENGTH]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core streaming
# ---------------------------------------------------------------------------

def stream_language_data(
    lang: str,
    target_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[str], int]:
    """Stream code from The Stack for a language until the target token count.

    Tokens are counted using the provided tokenizer. If adding the last file
    would exceed the target, it is truncated at the token boundary.
    """
    # The Stack uses original names as data directories (e.g. "c++", "c-sharp")
    # but the user may pass safe names (e.g. "cpp", "csharp") on the CLI.
    dir_name = stack_dir_name(lang)
    ds_stream = load_dataset(
        SOURCE_DATASET,
        data_dir=f"data/{dir_name}",
        split="train",
        streaming=True,
    )

    texts: list[str] = []
    token_count = 0

    pbar = tqdm(ds_stream, desc=f"Streaming {lang}")
    for example in pbar:
        text = _sanitise(example.get("content", "").strip())
        if not text:
            continue

        token_ids = tokenizer.encode(text)
        num_tokens = len(token_ids)

        if token_count + num_tokens > target_tokens:
            remaining = target_tokens - token_count
            truncated = tokenizer.decode(token_ids[:remaining])
            texts.append(truncated)
            token_count += remaining
            break

        texts.append(text)
        token_count += num_tokens
        pbar.set_postfix(tokens=f"{token_count:,}/{target_tokens:,}")

    pbar.close()

    if token_count < target_tokens:
        logger.warning(
            "%s: only %d tokens available (target was %d)",
            lang,
            token_count,
            target_tokens,
        )
    logger.info(
        "%s: collected %d tokens in %d documents", lang, token_count, len(texts)
    )

    return texts, token_count


def _build_readme(lang: str, safe: str, token_count: int, target: int) -> str:
    """Build README.md content with HuggingFace dataset config YAML."""
    return (
        f"---\n"
        f"configs:\n"
        f"- config_name: {safe}\n"
        f"  data_files:\n"
        f"    - split: train\n"
        f"      path: data/{safe}-*.parquet\n"
        f"---\n\n"
        f"# LangMap-TheStack-{safe}-{_human_size(target)}\n\n"
        f"Code finetuning dataset for **{lang}** streamed from "
        f"[bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack).\n\n"
        f"- **Tokens collected**: {token_count:,} (target: {target:,})\n"
        f"- **Tokenizer**: `{TOKENIZER_NAME}`\n"
        f"- **Schema**: `{{\"text\": [...]}}` (sanitised source code)\n"
    )


def _upload_readme(
    api: HfApi, hf_repo: str, lang: str, safe: str,
    token_count: int, target: int,
) -> None:
    """Generate and upload a README.md for a per-language repo."""
    readme_content = _build_readme(lang, safe, token_count, target)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme_content)
        readme_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Add README for {lang} code dataset",
        )
    finally:
        os.unlink(readme_path)

    logger.info("README.md uploaded to %s", hf_repo)


# ---------------------------------------------------------------------------
# Build and push
# ---------------------------------------------------------------------------

def build_and_push_dataset(
    languages: list[str],
    target_tokens: int,
    hf_user: str,
    tmp_dir: str = "tmp_thestack",
    private: bool = False,
) -> None:
    """Stream code data, save as parquet, push each language to its own HF repo."""
    os.makedirs(tmp_dir, exist_ok=True)
    api = HfApi()
    size_label = _human_size(target_tokens)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for lang in languages:
        safe = safe_name(lang)
        hf_repo = f"{hf_user}/LangMap-TheStack-{safe}-{size_label}"

        try:
            api.create_repo(
                repo_id=hf_repo, exist_ok=True, private=private, repo_type="dataset"
            )

            texts, token_count = stream_language_data(lang, target_tokens, tokenizer)
            if not texts:
                logger.warning("Skipped %s: no valid code found", lang)
                continue

            # Store as "text" field for consistency with fineweb2_finetune_dataset.py
            ds = Dataset.from_dict({"text": texts})
            local_path = os.path.join(tmp_dir, f"{safe}.parquet")
            ds.to_parquet(local_path)
            logger.info(
                "Saved %s with ~%d tokens to %s", lang, token_count, local_path
            )

            ds.push_to_hub(hf_repo, split="train", private=private)
            logger.info("Pushed %s to %s", lang, hf_repo)

            _upload_readme(api, hf_repo, lang, safe, token_count, target_tokens)
        except Exception:
            logger.exception("Skipped %s due to error", lang)


def verify_dataset(hf_user: str, lang: str, target_tokens: int, num_samples: int = 5) -> None:
    """Load and print sample entries from a pushed dataset."""
    safe = safe_name(lang)
    size_label = _human_size(target_tokens)
    hf_repo = f"{hf_user}/LangMap-TheStack-{safe}-{size_label}"

    ds = load_dataset(hf_repo, split="train")
    logger.info("Loaded %s from %s (%d total examples)", lang, hf_repo, len(ds))

    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        print(f"--- Document {i + 1} ---")
        print(f"{example['text'][:500]}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and push per-language code finetuning datasets to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-user",
        default="MultilingualUnigramLM",
        help="HuggingFace user/org (default: MultilingualUnigramLM)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=100_000_000,
        help="Target number of tokens per language (default: 100000000)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=CODE_LANGUAGES,
        help="Programming language names to process (default: all 30 languages)",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp_thestack",
        help="Directory for intermediate parquet files (default: tmp_thestack)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repositories private",
    )
    parser.add_argument(
        "--verify",
        metavar="LANG",
        help="After pushing, verify by printing samples for this language",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    build_and_push_dataset(
        languages=args.languages,
        target_tokens=args.target_tokens,
        hf_user=args.hf_user,
        tmp_dir=args.tmp_dir,
        private=args.private,
    )

    if args.verify:
        verify_dataset(args.hf_user, args.verify, args.target_tokens)


if __name__ == "__main__":
    main()
