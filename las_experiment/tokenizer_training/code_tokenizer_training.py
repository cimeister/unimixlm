#!/usr/bin/env python3
"""
las_experiment/tokenizer_training/code_tokenizer_training.py

End-to-end pipeline for training a family of programming-language-specific
LAS UnigramLM tokenizers from The-Stack-10K.

Fixes vs previous version
--------------------------
1. Removed manual merge step.
   StandardUnigramLMTokenizer.train() already accepts a list of files and
   handles merging internally (same path used by the FineWeb2 pipeline).
   Pre-merging caused a double-processing pass through tokenizer_formatting_utils
   and was the root cause of spm_train exit-code-1 failures.

2. Added corpus sanitisation in prepare_code_corpora.
   Source-code files can contain null bytes, lone surrogates, and lines that
   are tens of thousands of characters long (e.g. minified JS).  These are
   silently dropped / truncated before writing so they cannot corrupt the
   corpus or trip up spm_train's text parser.

3. Raised default char_budget to 10 M (from 5 M).
   After sanitisation a language's effective corpus size can shrink
   significantly; starting with more raw chars ensures spm_train always
   receives a corpus large enough for a 100 K vocabulary.
"""

import logging
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer
from code.standard_unigramlm import StandardUnigramLMTokenizer
from las_experiment.tokenizer_training.validation import validate_tokenizers
from las_experiment.tokenizer_training.utils import push_to_hf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_LANGUAGES = [
    "assembly", "batchfile", "c++", "c", "c-sharp", "cmake", "css",
    "dockerfile", "fortran", "go", "haskell", "html", "java",
    "javascript", "julia", "lua", "makefile", "markdown", "perl",
    "php", "powershell", "python", "ruby", "rust",
    "scala", "shell", "sql", "tex", "typescript", "visual-basic",
]

# spm_train uses gflags, which treats "++" as a flag-override prefix and "-"
# as a flag separator.  Any language name that reaches a filename or the
# comma-separated --input argument must be filesystem- and gflags-safe.
#
# Mapping: dataset config name  →  safe filename stem
# Only languages whose names contain [+] or [-] need an entry; all others
# are already safe.
SAFE_LANG_NAME: dict[str, str] = {
    "c++":          "cpp",
    "c-sharp":      "csharp",
    "visual-basic": "visual_basic",
}


def safe_name(lang: str) -> str:
    """Return the filesystem / gflags-safe stem for a language name."""
    return SAFE_LANG_NAME.get(lang, lang)


# Raised from 5 M → 10 M so that sanitisation losses still leave enough text
# for spm_train to build a 100 K vocabulary.
DEFAULT_CHAR_BUDGET: int = 10_000_000

# Lines longer than this are almost certainly minified / generated code with
# no useful tokenisation signal – drop them entirely.
MAX_LINE_LENGTH: int = 2_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise(text: str) -> str:
    """
    Light sanitisation of a source-code snippet before writing to corpus.

    Removes:
    - Null bytes (cause spm_train to silently truncate or crash).
    - Lone UTF-16 surrogates that survive JSON decode with errors='replace'.
    - Lines longer than MAX_LINE_LENGTH (minified / auto-generated junk).

    Keeps all valid Unicode, whitespace structure, and indentation.
    """
    # Drop null bytes and lone surrogates in one pass.
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.replace("\x00", "")

    # Drop extremely long lines.
    lines = [ln for ln in text.splitlines() if len(ln) <= MAX_LINE_LENGTH]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------

def prepare_code_corpora(
    langs: list[str],
    char_budget: int = DEFAULT_CHAR_BUDGET,
    output_dir: str | Path = "./data/code_parallel",
) -> dict[str, str]:
    """
    Stream source-code files from MultilingualUnigramLM/The-Stack-10K and
    write one sanitised plain-text corpus file per programming language.

    Each source file becomes one document separated by a double-newline so
    the Unigram trainer sees clean boundaries without splitting on internal
    newlines (critical for indentation-sensitive languages like Python).

    Args:
        langs:       Programming-language names matching The-Stack-10K config
                     names, e.g. ["python", "rust", "java"].
        char_budget: Max UTF-8 characters to collect *before* sanitisation.
                     Sampling stops once this budget is reached.
        output_dir:  Directory for per-language .txt corpus files.

    Returns:
        corpus_info: {lang: absolute_path_to_txt_file}
                     Languages that fail to stream are omitted.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_info: dict[str, str] = {}

    for lang in langs:
        # Use the safe stem for the filename; keep `lang` for HF config lookup.
        # "c++" -> "cpp", "c-sharp" -> "csharp", "visual-basic" -> "visual_basic"
        # spm_train uses gflags which treats "++" as a flag-override prefix and
        # "-" as a flag separator inside comma-separated --input file lists.
        out_file = output_dir / f"{safe_name(lang)}.txt"

        logging.info(
            f"[{lang}] Streaming from The-Stack-10K "
            f"(budget={char_budget:,} chars)..."
        )

        try:
            ds_stream = load_dataset(
                "MultilingualUnigramLM/The-Stack-10K",
                name=lang,
                split="train",
                streaming=True,
            )
        except Exception as exc:
            logging.warning(f"[{lang}] Skipping – could not load dataset: {exc}")
            continue

        snippets: list[str] = []
        raw_chars = 0
        clean_chars = 0

        for example in ds_stream:
            raw = (
                example.get("content")
                or example.get("text")
                or ""
            ).strip()

            if not raw:
                continue

            raw_chars += len(raw)
            clean = _sanitise(raw)

            if not clean.strip():
                continue

            snippets.append(clean)
            clean_chars += len(clean)

            if raw_chars >= char_budget:
                break

        if not snippets:
            logging.warning(f"[{lang}] No usable content after sanitisation – skipping.")
            continue

        # Write corpus: double-newline between documents.
        with open(out_file, "w", encoding="utf-8") as fh:
            for snippet in snippets:
                fh.write(snippet)
                fh.write("\n\n")

        logging.info(
            f"[{lang}] {len(snippets):,} files | "
            f"{raw_chars:,} raw chars → {clean_chars:,} clean chars | "
            f"{out_file}"
        )
        corpus_info[safe_name(lang)] = str(out_file)

    if not corpus_info:
        logging.error("prepare_code_corpora: no corpora produced.")

    return corpus_info


# ---------------------------------------------------------------------------
# spm_train stderr capture
# ---------------------------------------------------------------------------

def _install_spm_stderr_shim() -> tuple[str, str]:
    """
    Create a temporary directory containing a thin 'spm_train' wrapper script
    that tees stderr to a log file AND still passes it through to the terminal.

    standard_unigramlm.py calls spm_train via subprocess.run without capturing
    stderr, so any failure message is silently lost.  This shim restores
    visibility without touching that file.

    Returns:
        (shim_dir, log_path)  – shim_dir should be prepended to PATH before
        the training call and removed afterwards.
    """
    import shutil, stat, tempfile

    shim_dir = tempfile.mkdtemp(prefix="spm_shim_")
    log_path = os.path.join(shim_dir, "spm_train_stderr.log")
    shim_path = os.path.join(shim_dir, "spm_train")

    real_spm = shutil.which("spm_train") or "/usr/bin/spm_train"

    # Use process substitution to tee stderr to the log file while keeping
    # it visible on the terminal.  The wrapper preserves the exit code.
    script = (
        "#!/bin/bash\n"
        f'"{real_spm}" "$@" 2> >(tee -a "{log_path}" >&2)\n'
        "exit ${PIPESTATUS[0]}\n"
    )
    with open(shim_path, "w") as fh:
        fh.write(script)
    os.chmod(shim_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)

    return shim_dir, log_path


# ---------------------------------------------------------------------------
# Base tokenizer training
# ---------------------------------------------------------------------------

def train_base_tokenizer(
    *,
    corpus_info: dict[str, str],
    model_tokenizer_path: str | Path,
    base_output_path: str | Path,
    em_mode: str = "soft",
    unk_token: str | None = None,
) -> None:
    """
    Train a single shared base Unigram tokenizer from all code corpora.

    Installs a spm_train stderr-capture shim so that failure messages are
    logged (standard_unigramlm.py discards spm_train's stderr by default).
    Passes corpus files as a list directly to StandardUnigramLMTokenizer –
    same pattern as the FineWeb2 pipeline, no manual pre-merge.
    """
    train_files = list(corpus_info.values())

    if not train_files:
        logging.error("train_base_tokenizer: no training files available.")
        return

    logging.info(
        f"Training base Unigram tokenizer on {len(train_files)} "
        f"language corpus file(s)..."
    )

    # --- install stderr shim so spm_train failures are visible --------------
    shim_dir, spm_log = _install_spm_stderr_shim()
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = shim_dir + ":" + old_path
    logging.info(f"spm_train stderr will be logged to: {spm_log}")

    base_tokenizer = StandardUnigramLMTokenizer(
        em_mode=em_mode,
        from_base_tokenizer=str(model_tokenizer_path),
        unk_token=unk_token,
    )

    try:
        base_tokenizer.train(
            train_files,
            str(base_output_path),
            corpus_info=corpus_info,
        )
    except Exception as exc:
        # Surface the real spm_train error before re-raising.
        if os.path.exists(spm_log):
            with open(spm_log) as fh:
                stderr_text = fh.read().strip()
            if stderr_text:
                logging.error(
                    "spm_train stderr (full output):\n%s", stderr_text
                )
            else:
                logging.error(
                    "spm_train log is empty – failure may be before stderr "
                    "was written, or the shim did not intercept the call."
                )
        raise
    finally:
        os.environ["PATH"] = old_path
        import shutil as _shutil
        _shutil.rmtree(shim_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_code_tokenizer_training(
    langs: list[str],
    output_dir: str | Path,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    train_new_base_tokenizer: bool = True,
    hf_username: str = "MultilingualUnigramLM",
    model_name: str = "allenai/OLMo-2-1124-7B",
    training_cfg: dict | None = None,
    is_push_to_hf: bool = True,
) -> None:
    """
    End-to-end pipeline: stream → sanitise → train base → re-estimate per
    language → validate → push to Hub.

    Output layout::

        {output_dir}/las-code-tokenizers-{model}-{lang_codes}/
            ├── base_tokenizer.json
            ├── langspec_python.json
            ├── langspec_rust.json
            └── …
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = training_cfg or {}
    em_mode = str(training_cfg.get("em_mode", "soft"))

    # ------------------------------------------------------------------
    # 1. Load seed model tokenizer
    # ------------------------------------------------------------------
    logging.info(f"Loading seed model tokenizer from {model_name}…")
    model_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    seed_dir = output_dir / "model_tokenizer"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_tokenizer.save_pretrained(seed_dir)
    model_tokenizer_path = seed_dir / "tokenizer.json"

    # Warn if not byte-level – code corpora are full of non-ASCII content.
    pretokenizer = getattr(
        model_tokenizer.backend_tokenizer, "pre_tokenizer", None
    )
    is_byte_level = (
        pretokenizer is not None
        and "ByteLevel" in type(pretokenizer).__name__
    )
    if not is_byte_level:
        logging.warning(
            "Seed tokenizer does not appear to be byte-level. "
            "Code corpora may contain non-ASCII identifiers / string literals. "
            "Consider a byte-level seed (GPT-2, StarCoder) for better coverage."
        )

    unk_token: str = (
        model_tokenizer.unk_token or model_tokenizer.eos_token
    )
    logging.info(f"unk_token={unk_token!r}  byte_level={is_byte_level}")

    # ------------------------------------------------------------------
    # 2. Stream + sanitise corpora
    # ------------------------------------------------------------------
    logging.info(f"Preparing code corpora for {len(langs)} language(s)…")
    corpus_info = prepare_code_corpora(
        langs,
        char_budget=char_budget,
        output_dir=output_dir / "corpora",
    )

    if not corpus_info:
        logging.error("No corpora prepared – aborting.")
        return

    # ------------------------------------------------------------------
    # 3. Output directory (named after the future HF repo)
    # ------------------------------------------------------------------
    truncated = [lang[:3] for lang in sorted(corpus_info.keys())[:3]]
    lang_codes = "_".join(truncated)
    if len(corpus_info) > 3:
        lang_codes += f"_plus{len(corpus_info) - 3}"

    model_short = model_name.split("/")[-1]
    repo_name = f"las-code-tokenizers-{model_short}-{lang_codes}"
    tokenizers_dir = output_dir / repo_name
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Tokenizer artefacts → {tokenizers_dir}")

    # ------------------------------------------------------------------
    # 4. Train shared base Unigram tokenizer
    # ------------------------------------------------------------------
    if train_new_base_tokenizer:
        base_tokenizer_path = tokenizers_dir / "base_tokenizer.json"
        train_base_tokenizer(
            corpus_info=corpus_info,
            model_tokenizer_path=model_tokenizer_path,
            base_output_path=base_tokenizer_path,
            em_mode=em_mode,
            unk_token=unk_token,
        )
        validate_tokenizers(model_tokenizer_path, base_tokenizer_path)
    else:
        logging.info(
            "Skipping base training; using seed tokenizer as base."
        )
        base_tokenizer_path = model_tokenizer_path

    # ------------------------------------------------------------------
    # 5. Per-language re-estimation
    # ------------------------------------------------------------------
    logging.info(
        f"Training {len(corpus_info)} language-specific tokenizers…"
    )
    las_tokenizer = LanguageSpecificUnigramLMTokenizer(
        reestimation_em_mode=em_mode,
        unk_token=unk_token,
    )
    las_tokenizer.train(
        corpus_info=corpus_info,
        base_tokenizer_path=str(base_tokenizer_path),
        tokenizer_path_format=str(
            tokenizers_dir / "langspec_{lang_code}.json"
        ),
        load_base_if_exists=True,
        load_tokenizers_if_exists=False,
    )

    logging.info("All code tokenizers trained successfully.")

    # ------------------------------------------------------------------
    # 6. Validate
    # ------------------------------------------------------------------
    lang_tokenizer_paths = {
        lang: tokenizers_dir / f"langspec_{lang}.json"
        for lang in corpus_info  # keys are already safe_name() strings
    }
    validate_tokenizers(
        model_tokenizer_path,
        base_tokenizer_path,
        lang_tokenizer_paths,
    )

    # ------------------------------------------------------------------
    # 7. Push to HuggingFace Hub
    # ------------------------------------------------------------------
    if is_push_to_hf:
        logging.info("Pushing tokenizer family to HuggingFace Hub…")
        push_to_hf(tokenizers_dir, hf_username, repo_name)
        logging.info(
            f"Done → https://huggingface.co/{hf_username}/{repo_name}"
        )


# ---------------------------------------------------------------------------
# CLI entry point (no Hydra dependency)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train code-language-specific LAS UnigramLM tokenizers "
            "from MultilingualUnigramLM/The-Stack-10K."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--langs", nargs="+", default=CODE_LANGUAGES,
        help="Programming languages to train on (default: all 30).",
    )
    parser.add_argument(
        "--char_budget", type=int, default=DEFAULT_CHAR_BUDGET,
        help="Max characters to stream per language before sanitisation.",
    )
    parser.add_argument(
        "--seed_model", default="allenai/OLMo-2-1124-7B",
        help="HF path of the seed model tokenizer.",
    )
    parser.add_argument(
        "--em_mode", default="soft", choices=["soft", "hard"],
    )
    parser.add_argument(
        "--train_new_base", action="store_true",
        help="Train a new base tokenizer (default: False → use seed).",
    )
    parser.add_argument(
        "--push_to_hf", action="store_true",
        help="Push resulting tokenizers to HuggingFace Hub.",
    )
    args = parser.parse_args()

    run_code_tokenizer_training(
        langs=args.langs,
        output_dir=args.output_dir,
        char_budget=args.char_budget,
        train_new_base_tokenizer=args.train_new_base,
        model_name=args.seed_model,
        training_cfg={"em_mode": args.em_mode},
        is_push_to_hf=args.push_to_hf,
    )