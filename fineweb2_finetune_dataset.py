"""Create and push FineWeb2 finetuning datasets to HuggingFace Hub.

Streams data from HuggingFaceFW/fineweb-2 for specified languages, collects
up to a target token count per language, saves as parquet, and pushes to a
HuggingFace dataset repository.

Usage examples:
    # Build and push a 5M-token-per-language dataset (single multi-split repo)
    python fineweb2_finetune_dataset.py --hf-repo MultilingualUnigramLM/FineWeb2-5M --target-tokens 5000000

    # Build a 10M dataset with a subset of languages
    python fineweb2_finetune_dataset.py --hf-repo MultilingualUnigramLM/FineWeb2-10M --target-tokens 10000000 \
        --languages fil_Latn ind_Latn lit_Latn

    # Large-scale: 500M tokens PER language, one HF repo per language.
    # Uses batched tokenization and incremental sharded parquet writes so
    # memory stays bounded regardless of the token target.
    python fineweb2_finetune_dataset.py --per-language-repos \
        --hf-repo-template "MultilingualUnigramLM/FineWeb2-500M-{lang}" \
        --target-tokens 500000000 --languages fin_Latn hun_Latn tur_Latn

    # Verify a pushed per-language repo by printing samples
    python fineweb2_finetune_dataset.py --hf-repo MultilingualUnigramLM/FineWeb2-500M-fin_Latn --verify fin_Latn
"""

import argparse
import logging
import os
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
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
    "deu_Latn",
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


def stream_language_to_shards(
    lang: str,
    target_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    out_dir: str,
    shard_tokens: int = 25_000_000,
    encode_batch: int = 512,
) -> tuple[int, int, int]:
    """Stream FineWeb-2 for a language, writing bounded parquet shards to disk.

    Unlike ``stream_language_data``, this never holds the full corpus in memory:
    documents are tokenized in batches (fast-tokenizer parallelism) purely to
    count tokens, and texts are flushed to ``out_dir/<lang>-NNNNN.parquet`` every
    ``shard_tokens`` tokens. This keeps RAM flat for arbitrarily large targets
    (e.g. 500M tokens/language).

    Returns ``(num_shards, total_tokens, num_docs)``.
    """
    if lang == ENGLISH_LANG:
        ds_stream = load_dataset(SOURCE_DATASET_EN, split="train", streaming=True)
    else:
        ds_stream = load_dataset(
            SOURCE_DATASET, name=lang, split="train", streaming=True
        )

    os.makedirs(out_dir, exist_ok=True)

    total_tokens = 0
    total_docs = 0
    shard_idx = 0
    shard_texts: list[str] = []
    shard_token_count = 0
    text_buffer: list[str] = []

    def flush_shard() -> None:
        nonlocal shard_idx, shard_texts, shard_token_count
        if not shard_texts:
            return
        path = os.path.join(out_dir, f"{lang}-{shard_idx:05d}.parquet")
        pq.write_table(pa.table({"text": shard_texts}), path)
        logger.info(
            "%s: wrote shard %d (%d docs, ~%d tokens) -> %s",
            lang,
            shard_idx,
            len(shard_texts),
            shard_token_count,
            path,
        )
        shard_idx += 1
        shard_texts = []
        shard_token_count = 0

    pbar = tqdm(ds_stream, desc=f"Streaming {lang}")

    def drain_buffer(stop_at_target: bool) -> bool:
        """Tokenize buffered texts in one batch; append until target. Returns
        True if the target has been reached (caller should stop)."""
        nonlocal total_tokens, total_docs, shard_token_count
        if not text_buffer:
            return False
        encodings = tokenizer(
            text_buffer, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        for text, ids in zip(text_buffer, encodings):
            num_tokens = len(ids)
            if total_tokens + num_tokens >= target_tokens:
                remaining = target_tokens - total_tokens
                if remaining > 0:
                    shard_texts.append(tokenizer.decode(ids[:remaining]))
                    total_tokens += remaining
                    shard_token_count += remaining
                    total_docs += 1
                text_buffer.clear()
                return True
            shard_texts.append(text)
            total_tokens += num_tokens
            shard_token_count += num_tokens
            total_docs += 1
            if shard_token_count >= shard_tokens:
                flush_shard()
        text_buffer.clear()
        pbar.set_postfix(tokens=f"{total_tokens:,}/{target_tokens:,}")
        return stop_at_target and total_tokens >= target_tokens

    for example in pbar:
        text = example.get("text", "").strip()
        if not text:
            continue
        text_buffer.append(text)
        if len(text_buffer) >= encode_batch:
            if drain_buffer(stop_at_target=True):
                break

    drain_buffer(stop_at_target=False)
    flush_shard()
    pbar.close()

    if total_tokens < target_tokens:
        logger.warning(
            "%s: only %d tokens available (target was %d)",
            lang,
            total_tokens,
            target_tokens,
        )
    logger.info(
        "%s: collected %d tokens, %d docs across %d shard(s)",
        lang,
        total_tokens,
        total_docs,
        shard_idx,
    )
    return shard_idx, total_tokens, total_docs


def _single_config_readme(lang: str, total_tokens: int) -> str:
    """README for a one-language-per-repo dataset (default config)."""
    return (
        "---\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "    - split: train\n"
        "      path: data/*.parquet\n"
        "---\n\n"
        f"# FineWeb-2 finetuning subset: {lang}\n\n"
        f"~{total_tokens:,} tokens streamed from `{SOURCE_DATASET}` "
        f"(config `{lang}`), token-counted with `{TOKENIZER_NAME}`.\n"
    )


def build_and_push_per_language(
    languages: list[str],
    target_tokens: int,
    repo_template: str,
    tmp_dir: str = "tmp_fineweb2",
    shard_tokens: int = 25_000_000,
    encode_batch: int = 512,
    private: bool = False,
) -> None:
    """Stream each language into its own HF dataset repo via sharded uploads.

    ``repo_template`` must contain ``{lang}`` (e.g.
    ``MultilingualUnigramLM/FineWeb2-500M-{lang}``).
    """
    if "{lang}" not in repo_template:
        raise ValueError("--hf-repo-template must contain the '{lang}' placeholder")

    api = HfApi()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for lang in languages:
        repo_id = repo_template.format(lang=lang)
        lang_dir = os.path.join(tmp_dir, lang)
        if os.path.isdir(lang_dir):
            shutil.rmtree(lang_dir)
        try:
            api.create_repo(
                repo_id=repo_id,
                exist_ok=True,
                private=private,
                repo_type="dataset",
            )
            num_shards, total_tokens, _ = stream_language_to_shards(
                lang,
                target_tokens,
                tokenizer,
                lang_dir,
                shard_tokens=shard_tokens,
                encode_batch=encode_batch,
            )
            if num_shards == 0:
                logger.warning("Skipped %s: no valid text found", lang)
                continue

            api.upload_folder(
                folder_path=lang_dir,
                path_in_repo="data",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=(
                    f"Add {lang}: ~{total_tokens:,} tokens "
                    f"streamed from {SOURCE_DATASET}"
                ),
            )
            readme_content = _single_config_readme(lang, total_tokens)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            ) as f:
                f.write(readme_content)
                readme_path = f.name
            try:
                api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Add README for {lang}",
                )
            finally:
                os.unlink(readme_path)

            logger.info("Pushed %s (~%d tokens) to %s", lang, total_tokens, repo_id)
        except Exception:
            logger.exception("Skipped %s due to error", lang)
        finally:
            if os.path.isdir(lang_dir):
                shutil.rmtree(lang_dir)


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
        help="HuggingFace dataset repository ID for single multi-split mode "
        "(e.g. MultilingualUnigramLM/FineWeb2-5M). Required unless "
        "--per-language-repos is set.",
    )
    parser.add_argument(
        "--per-language-repos",
        action="store_true",
        help="Push each language to its own repo using --hf-repo-template, "
        "with batched tokenization and incremental sharded parquet writes "
        "(memory-bounded; use for large targets like 500M tokens/language).",
    )
    parser.add_argument(
        "--hf-repo-template",
        default="MultilingualUnigramLM/FineWeb2-500M-{lang}",
        help="Repo template for --per-language-repos; must contain '{lang}' "
        "(default: MultilingualUnigramLM/FineWeb2-500M-{lang})",
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=25_000_000,
        help="Flush a parquet shard every N tokens in --per-language-repos "
        "mode (default: 25000000)",
    )
    parser.add_argument(
        "--encode-batch",
        type=int,
        default=512,
        help="Documents per batched tokenizer call in --per-language-repos "
        "mode (default: 512)",
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

    if args.per_language_repos:
        build_and_push_per_language(
            languages=args.languages,
            target_tokens=args.target_tokens,
            repo_template=args.hf_repo_template,
            tmp_dir=args.tmp_dir,
            shard_tokens=args.shard_tokens,
            encode_batch=args.encode_batch,
            private=args.private,
        )
    else:
        if not args.hf_repo:
            raise SystemExit(
                "--hf-repo is required unless --per-language-repos is set"
            )
        build_and_push_dataset(
            languages=args.languages,
            target_tokens=args.target_tokens,
            hf_repo=args.hf_repo,
            tmp_dir=args.tmp_dir,
            private=args.private,
        )

    if args.verify:
        verify_dataset(args.hf_repo or args.hf_repo_template, args.verify)


if __name__ == "__main__":
    main()
    # Native extensions (torch / pyarrow) can crash during interpreter
    # finalization with a GIL error. All uploads are complete by this point,
    # so flush logs and hard-exit to keep long unattended runs clean.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
