#!/usr/bin/env python3
"""
run_las_experiments.py
Full LAS-UnigramLM experiment pipeline for a high/low resource language pair.
Supports multiple pretrained LMs with automatic tokenizer adjustment, memory-efficient large model loading,
automatic embedding alignment, and single HF repo organization.

Example usage:
python3 run_las_experiments.py --high en --low yo --model FacebookAI/xlm-roberta-base --inference_string 'Hello world!' --length_prior_weights '{"en": 0.1, "yo": 0.3}'
"""

import sys, subprocess, json, logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from transformers import PreTrainedTokenizerFast

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Ensure dependencies ----------------------
required_packages = [
    "torch",
    "numpy",
    "sentencepiece",
    "datasets",
    "transformers",
    "tokenizers",
    "accelerate",
]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", pkg]
        )

# ---------------------- Add code/ to sys.path ----------------------
sys.path.insert(0, str(Path(__file__).parent / "code"))
from language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer


# ---------------------- LAS → HF tokenizer export ----------------------
def export_las_tokenizer_hf_single_repo(
    las_tokenizer,
    output_dir,
    push_to_hub=True,
    repo_id=None,
    hf_username="suchirsalhan",
):
    import os, json
    from pathlib import Path
    from transformers import PreTrainedTokenizerFast
    from huggingface_hub import HfApi, HfFolder
    import subprocess
    import logging

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


# ---------------------- Dataset helper ----------------------
class TokenizedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, preprocess_fn):
        self.examples = [
            tokenizer(preprocess_fn(t), return_tensors="pt")["input_ids"].squeeze(0)
            for t in texts
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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


# ---------------------- Perplexity computation ----------------------
def compute_perplexity(model, tokenizer, texts, preprocess_fn, device="cuda"):
    total_loss, total_tokens = 0.0, 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(
                preprocess_fn(t), return_tensors="pt", truncation=True, padding=False
            )
            ids = enc["input_ids"].to(device)
            outputs = model(ids, labels=ids)
            total_loss += outputs.loss.item() * ids.numel()
            total_tokens += ids.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


# ---------------------- Main pipeline ----------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LAS-UnigramLM experiments.")
    parser.add_argument("--high", type=str, required=True)
    parser.add_argument("--low", type=str, required=True)
    parser.add_argument("--model", type=str, default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--inference_string", type=str, default=None)
    parser.add_argument("--inference_file", type=str, default=None)
    parser.add_argument("--length_prior_weights", type=str, default="{}")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--ft_epochs", type=int, default=1)
    parser.add_argument("--ft_batch_size", type=int, default=2)
    parser.add_argument(
        "--push-hf", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    LANGUAGES = [args.high, args.low]
    length_prior_weights = json.loads(args.length_prior_weights)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    preprocess_fn = (
        lambda x: x if getattr(tokenizer, "byte_level", False) else x.strip().lower()
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    CORPUS_DIR = Path("data/parallel/dev")
    corpus_paths = {lang: CORPUS_DIR / f"{lang}.txt" for lang in LANGUAGES}
    for lang, p in corpus_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Corpus {lang} not found: {p}")

    safe_model_name = args.model.split("/")[-1]
    OUTPUT_DIR = Path(
        f"multilingual/tokenizers/unimix-{args.high}_{args.low}-{safe_model_name}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shared_las = LanguageSpecificUnigramLMTokenizer(
        vocab_size=len(tokenizer), byte_level=True
    )
    tokenizer_paths = shared_las.train(
        corpus_info={lang: str(corpus_paths[lang]) for lang in LANGUAGES},
        base_tokenizer_path=str(OUTPUT_DIR / "shared_base/tokenizer.json"),
        output_dir=str(OUTPUT_DIR),
        load_base_if_exists=False,
        load_tokenizers_if_exists=False,
    )
    logger.info(f"LAS-UnigramLM saved: {tokenizer_paths}")

    # # ---- Export LAS tokenizer + metadata in single HF repo and push ----
    # las_hf_path = export_las_tokenizer_hf_single_repo(
    #     shared_las,
    #     OUTPUT_DIR,
    #     push_to_hub=args.push_hf,
    #     repo_id=f"unimix-{args.high}_{args.low}-{safe_model_name}"
    # )
    # logger.info(f"HF LAS tokenizer exported to {las_hf_path}")

    model.resize_token_embeddings(len(shared_las.base_tokenizer.get_vocab()))
    logger.info(
        f"Resized LM embeddings to {len(shared_las.base_tokenizer.get_vocab())} tokens"
    )

    # ---- v* inference ----
    if args.inference_string:
        lang, toks, logp = v_star_tokenize(
            preprocess_fn(args.inference_string), shared_las, length_prior_weights
        )
        logger.info(f"Input string: {args.inference_string}")
        logger.info(f"Best lang: {lang}, tokens: {toks}, adjusted log-prob: {logp:.6f}")
    if args.inference_file:
        file_path = Path(args.inference_file)
        results, log_probs, langs = tokenize_file(
            file_path, shared_las, length_prior_weights
        )
        logger.info(f"v* tokenization results for {file_path}:")
        for toks, logp in zip(results, log_probs):
            print(f"{' '.join(toks)} | adjusted log-prob: {logp:.6f}")

    # ---- Optional fine-tuning ----
    if args.fine_tune:
        all_texts, lang_texts = [], {}
        for lang, path in corpus_paths.items():
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
                tok_lines = [
                    " ".join(
                        v_star_tokenize(
                            preprocess_fn(l), shared_las, length_prior_weights
                        )[1]
                    )
                    for l in lines
                ]
                all_texts.extend(tok_lines)
                lang_texts[lang] = tok_lines

        dataset = TokenizedTextDataset(all_texts, tokenizer, preprocess_fn)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir="las_finetuned_model",
            overwrite_output_dir=True,
            num_train_epochs=args.ft_epochs,
            per_device_train_batch_size=args.ft_batch_size,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            report_to=[],
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args,
        )
        trainer.train()
        model.save_pretrained("las_finetuned_model")
        logger.info("Fine-tuning complete.")

        # ---- Automatic perplexity evaluation per language ----
        model_orig = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="auto"
        )
        model_ft = AutoModelForCausalLM.from_pretrained(
            "las_finetuned_model", trust_remote_code=True, device_map="auto"
        )
        logger.info("\nPerplexity evaluation (original vs LAS pre/post fine-tuning):")
        table_header = f"{'Lang':<5} | {'#Lines':>5} | {'Orig PPL':>10} | {'LAS-pre PPL':>12} | {'LAS-post PPL':>13}"
        print(table_header)
        print("-" * len(table_header))
        for lang in LANGUAGES:
            texts = lang_texts[lang][: min(20, len(lang_texts[lang]))]
            ppl_orig = compute_perplexity(model_orig, tokenizer, texts, preprocess_fn)
            ppl_las_pre = compute_perplexity(
                model_orig, tokenizer, texts, preprocess_fn
            )
            ppl_las_post = compute_perplexity(model_ft, tokenizer, texts, preprocess_fn)
            print(
                f"{lang:<5} | {len(texts):>5} | {ppl_orig:>10.4f} | {ppl_las_pre:>12.4f} | {ppl_las_post:>13.4f}"
            )

        all_sample = sum(
            [lang_texts[l][: min(20, len(lang_texts[l]))] for l in LANGUAGES], []
        )
        ppl_orig_all = compute_perplexity(
            model_orig, tokenizer, all_sample, preprocess_fn
        )
        ppl_las_pre_all = compute_perplexity(
            model_orig, tokenizer, all_sample, preprocess_fn
        )
        ppl_las_post_all = compute_perplexity(
            model_ft, tokenizer, all_sample, preprocess_fn
        )
        print(
            f"{'ALL':<5} | {len(all_sample):>5} | {ppl_orig_all:>10.4f} | {ppl_las_pre_all:>12.4f} | {ppl_las_post_all:>13.4f}"
        )

    logger.info("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
