#!/usr/bin/env python3
"""
run_las_tokenizer.py
Train LAS-UnigramLM tokenizers and optionally run v* inference.

Example usage:
python3 run_las_tokenizer.py --high en --low yo --model FacebookAI/xlm-roberta-base --inference_string 'Hello world!' --length_prior_weights '{"en": 0.1, "yo": 0.3}'
"""

import subprocess, json, logging
from pathlib import Path
from huggingface_hub import HfFolder
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer


# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------- LAS → HF tokenizer export ----------------------
def export_las_tokenizer_hf_single_repo(
    las_tokenizer,
    output_dir,
    push_to_hub=True,
    repo_id=None,
    hf_username="suchirsalhan",
):
    logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Shared base tokenizer ----------------
    base_path = output_dir / "shared_base"
    base_path.mkdir(exist_ok=True, parents=True)
    base_tok_json = base_path / "tokenizer.json"
    las_tokenizer.base_tokenizer.save(str(base_tok_json))

    # ---------------- Per-language tokenizers + metadata ----------------
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for lang, info in las_tokenizer.per_lang_tok.items():
        lang_dir = output_dir / f"langspec_soft_{lang}"
        lang_dir.mkdir(exist_ok=True, parents=True)

        lang_tok_json = lang_dir / "tokenizer.json"
        info["tokenizer"].save(str(lang_tok_json))

        # Save metadata
        meta_path = metadata_dir / f"langspec_soft_{lang}.metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)  # ensure parent exists
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "training_config": info.get("training_config", {}),
                    "corpus_info": info.get("corpus_info", {}),
                    "path": str(lang_tok_json),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ---------------- HF-compatible root tokenizer ----------------
    special_tokens = {}
    base_vocab = las_tokenizer.base_tokenizer.get_vocab()
    for tok in ["<unk>", "<pad>", "<s>", "</s>", "<mask>"]:
        if tok in base_vocab:
            special_tokens[tok] = tok

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(base_tok_json),
        unk_token=special_tokens.get("<unk>"),
        pad_token=special_tokens.get("<pad>"),
        bos_token=special_tokens.get("<s>"),
        eos_token=special_tokens.get("</s>"),
        mask_token=special_tokens.get("<mask>"),
    )
    hf_tokenizer.save_pretrained(output_dir)

    # ---------------- README.md ----------------
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        readme_content = f"""# LAS-UnigramLM tokenizer for {", ".join(las_tokenizer.per_lang_tok.keys())}

This repository contains the LAS-UnigramLM tokenizer trained on languages: {", ".join(las_tokenizer.per_lang_tok.keys())}.

- Tokenizer type: Unigram LM
- Special tokens: {list(special_tokens.keys())}
"""
        readme_path.write_text(readme_content, encoding="utf-8")

    # ---------------- Commit & push to HF ----------------
    if push_to_hub:
        from huggingface_hub import HfApi

        if repo_id is None:
            repo_id = output_dir.name
        hf_url = f"https://huggingface.co/{hf_username}/{repo_id}"

        # Create repo if missing
        api = HfApi()
        token = HfFolder.get_token()
        try:
            api.repo_info(f"{hf_username}/{repo_id}", token=token)
        except:
            logger.info(f"Creating HF repo: {hf_url}")
            api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)

        # ---------------- Git operations manually ----------------
        if not (output_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=str(output_dir), check=True)
            subprocess.run(
                ["git", "checkout", "-b", "main"], cwd=str(output_dir), check=True
            )
            subprocess.run(
                ["git", "remote", "add", "origin", hf_url],
                cwd=str(output_dir),
                check=True,
            )

        # Stage all files
        subprocess.run(["git", "add", "."], cwd=str(output_dir), check=True)

        # Commit (ignore if nothing to commit)
        try:
            subprocess.run(
                ["git", "commit", "-m", "Add LAS tokenizers + metadata + README"],
                cwd=str(output_dir),
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.warning("Nothing to commit, skipping git commit.")

        # Push to main branch
        subprocess.run(
            ["git", "push", "-u", "origin", "main", "--force"],
            cwd=str(output_dir),
            check=True,
        )

        logger.info(f"Pushed LAS tokenizers to HF repo: {hf_url}")

    return output_dir


# ---------------------- v* inference ----------------------
def v_star_tokenize(s, las_tokenizer, length_prior_weights=None):
    best_lang, best_tokens = las_tokenizer.best_language_encode(s)
    log_prob = sum(
        las_tokenizer.per_lang_tok[best_lang]["scores"].get(t, -13.0)
        for t in best_tokens
    )
    if length_prior_weights and best_lang in length_prior_weights:
        weight = length_prior_weights[best_lang]
        log_prob += -weight * torch.log(
            torch.tensor(len(best_tokens), dtype=torch.float32)
        )
    return best_lang, best_tokens, log_prob.item()


def tokenize_file(file_path, las_tokenizer, length_prior_weights=None):
    results, log_probs, langs = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lang, toks, logp = v_star_tokenize(
                line, las_tokenizer, length_prior_weights
            )
            results.append(toks)
            log_probs.append(logp)
            langs.append(lang)
    return results, log_probs, langs


def run_tokenizer(cfg):
    high = cfg.high
    low = cfg.low
    model = cfg.model
    inference_string = cfg.inference_string
    inference_file = cfg.inference_file
    length_prior_weights = cfg.length_prior_weights or {}

    LANGUAGES = [high, low]

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    preprocess_fn = (
        lambda x: x if getattr(tokenizer, "byte_level", False) else x.strip().lower()
    )

    CORPUS_DIR = Path("data/parallel/dev")
    corpus_paths = {lang: CORPUS_DIR / f"{lang}.txt" for lang in LANGUAGES}
    for lang, p in corpus_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Corpus {lang} not found: {p}")

    safe_model_name = model.split("/")[-1]
    output_dir = Path(f"multilingual/tokenizers/unimix-{high}_{low}-{safe_model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_las = LanguageSpecificUnigramLMTokenizer(
        vocab_size=len(tokenizer), byte_level=True
    )
    tokenizer_paths = shared_las.train(
        corpus_info={lang: str(corpus_paths[lang]) for lang in LANGUAGES},
        base_tokenizer_path=str(output_dir / "shared_base/tokenizer.json"),
        output_dir=str(output_dir),
        load_base_if_exists=False,
        load_tokenizers_if_exists=False,
    )
    logger.info(f"LAS-UnigramLM saved: {tokenizer_paths}")

    # ---- v* inference ----
    if inference_string:
        lang, toks, logp = v_star_tokenize(
            preprocess_fn(inference_string), shared_las, length_prior_weights
        )
        logger.info(f"Input string: {inference_string}")
        logger.info(f"Best lang: {lang}, tokens: {toks}, adjusted log-prob: {logp:.6f}")
    if inference_file:
        file_path = Path(inference_file)
        results, log_probs, langs = tokenize_file(
            file_path, shared_las, length_prior_weights
        )
        logger.info(f"v* tokenization results for {file_path}:")
        for toks, logp in zip(results, log_probs):
            print(f"{' '.join(toks)} | adjusted log-prob: {logp:.6f}")

    return output_dir
