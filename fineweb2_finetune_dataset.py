"""Create and push FineWeb2 finetuning datasets to HuggingFace Hub.

Streams data from HuggingFaceFW/fineweb-2 for specified languages, collects
up to a target token count per language, saves as parquet, and pushes to a
HuggingFace dataset repository.

Usage examples:
    # Build and push a 5M-token-per-language dataset
    python fineweb2_finetune_dataset.py --hf-repo MultilingualUnigramLM/FineWeb2-5M --target-tokens 5000000

    # Build a 10M dataset with a subset of languages
    python fineweb2_finetune_dataset.py --hf-repo MultilingualUnigramLM/FineWeb2-10M --target-tokens 10000000 \
        --languages fil_Latn ind_Latn lit_Latn

    # Verify a pushed dataset by printing samples
    python fineweb2_finetune_dataset.py --hf-repo MultilingualUnigramLM/FineWeb2-5M --verify fil_Latn
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

DEFAULT_LANGUAGES = [
    "eng_Latn",
    "zsm_Latn",
    "sna_Latn",
    "nya_Latn",
    "som_Latn",
    "fil_Latn",
    "ind_Latn",
    "lit_Latn",
    "ron_Latn",
    "hau_Latn",
    "ibo_Latn",
    "swh_Latn",
    "yor_Latn",
]

SOURCE_DATASET = "HuggingFaceFW/fineweb-2"
SOURCE_DATASET_EN = "HuggingFaceFW/fineweb"
ENGLISH_LANG = "eng_Latn"
TOKENIZER_NAME = "allenai/OLMo-3-1025-7B"


def stream_language_data(
    lang: str,
    target_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[str], int]:
    """Stream FineWeb-2 data for a language until the target token count is reached.

    Tokens are counted using the provided tokenizer. If adding the last document
    would exceed the target, it is truncated at the token boundary.
    """
    if lang == ENGLISH_LANG:
        ds_stream = load_dataset(
            SOURCE_DATASET_EN,
            split="train",
            streaming=True,
        )
    else:
        ds_stream = load_dataset(
            SOURCE_DATASET,
            name=lang,
            split="train",
            streaming=True,
        )

    texts: list[str] = []
    token_count = 0

    pbar = tqdm(ds_stream, desc=f"Streaming {lang}")
    for example in pbar:
        text = example.get("text", "").strip()
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


def _build_readme(languages: list[str]) -> str:
    """Build README.md content with HuggingFace dataset config YAML."""
    configs_yaml = "\n".join(
        f"- config_name: {lang}\n"
        f"  data_files:\n"
        f"    - split: train\n"
        f"      path: data/{lang}-*.parquet"
        for lang in sorted(languages)
    )
    return f"---\nconfigs:\n{configs_yaml}\n---\n"


def _upload_readme(api: HfApi, hf_repo: str, languages: list[str]) -> None:
    """Generate and upload a README.md with per-language dataset configs."""
    readme_content = _build_readme(languages)
    repo_name = hf_repo.split("/")[-1]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme_content)
        readme_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Add dataset configs for {repo_name} multilingual dataset",
        )
    finally:
        os.unlink(readme_path)

    logger.info("README.md updated and pushed to %s", hf_repo)


def build_and_push_dataset(
    languages: list[str],
    target_tokens: int,
    hf_repo: str,
    tmp_dir: str = "tmp_fineweb2",
    private: bool = False,
) -> None:
    """Stream FineWeb-2 data, save as parquet, push to Hub, and upload README."""
    os.makedirs(tmp_dir, exist_ok=True)
    api = HfApi()
    api.create_repo(
        repo_id=hf_repo, exist_ok=True, private=private, repo_type="dataset"
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    pushed_languages: list[str] = []

    for lang in languages:
        try:
            texts, token_count = stream_language_data(lang, target_tokens, tokenizer)
            if not texts:
                logger.warning("Skipped %s: no valid text found", lang)
                continue

            ds = Dataset.from_dict({"text": texts})
            local_path = os.path.join(tmp_dir, f"{lang}.parquet")
            ds.to_parquet(local_path)
            logger.info("Saved %s with ~%d tokens to %s", lang, token_count, local_path)

            ds.push_to_hub(hf_repo, split=lang, private=private)
            logger.info("Pushed %s to %s", lang, hf_repo)
            pushed_languages.append(lang)
        except Exception:
            logger.exception("Skipped %s due to error", lang)

    if pushed_languages:
        _upload_readme(api, hf_repo, pushed_languages)


def verify_dataset(hf_repo: str, lang: str, num_samples: int = 5) -> None:
    """Load and print sample entries from a pushed dataset."""
    ds = load_dataset(hf_repo, lang, split="train")
    logger.info("Loaded %s from %s (%d total examples)", lang, hf_repo, len(ds))

    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        print(f"{i + 1}: {example['text']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and push FineWeb2 finetuning datasets to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-repo",
        required=True,
        help="HuggingFace dataset repository ID (e.g. MultilingualUnigramLM/FineWeb2-5M)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=5_000_000,
        help="Target number of tokens per language (default: 5000000)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Language codes to process (default: 12 predefined languages)",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp_fineweb2",
        help="Directory for intermediate parquet files (default: tmp_fineweb2)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace repository private",
    )
    parser.add_argument(
        "--verify",
        metavar="LANG",
        help="After pushing, verify by printing samples for this language (e.g. fil_Latn)",
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
        hf_repo=args.hf_repo,
        tmp_dir=args.tmp_dir,
        private=args.private,
    )

    if args.verify:
        verify_dataset(args.hf_repo, args.verify)


if __name__ == "__main__":
    main()
