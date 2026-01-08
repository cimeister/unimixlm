#!/usr/bin/env python3
"""
run_las_experiments.py
Full LAS-UnigramLM experiment pipeline for a high/low resource language pair.
Supports multiple pretrained LMs with automatic tokenizer adjustment and memory-efficient large model loading.

Example usage:
python3 run_las_experiments.py --high en --low yo --model FacebookAI/xlm-roberta-base --inference_string 'Hello world!' --length_prior_weights '{"en": 0.1, "yo": 0.3}'
"""

import sys, subprocess, json, logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Ensure dependencies ----------------------
required_packages = ["torch","numpy","sentencepiece","datasets","transformers","tokenizers","accelerate"]
for pkg in required_packages:
    try: __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

# ---------------------- Add code/ to sys.path ----------------------
sys.path.insert(0, str(Path(__file__).parent / "code"))
from language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

# ---------------------- Dataset helper ----------------------
class TokenizedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, preprocess_fn):
        self.examples = [
            tokenizer(preprocess_fn(t), return_tensors="pt")["input_ids"].squeeze(0)
            for t in texts
        ]
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

# ---------------------- v* inference ----------------------
def v_star_tokenize(s, las_tokenizer, length_prior_weights=None):
    best_lang, best_tokens = las_tokenizer.best_language_encode(s)
    log_prob = sum(las_tokenizer.per_lang_tok[best_lang]["scores"].get(t,-13.0) for t in best_tokens)
    if length_prior_weights and best_lang in length_prior_weights:
        weight = length_prior_weights[best_lang]
        log_prob += -weight * torch.log(torch.tensor(len(best_tokens), dtype=torch.float32))
    return best_lang, best_tokens, log_prob.item()

def tokenize_file(file_path, las_tokenizer, length_prior_weights=None):
    results, log_probs, langs = [], [], []
    with open(file_path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            lang, toks, logp = v_star_tokenize(line, las_tokenizer, length_prior_weights)
            results.append(toks)
            log_probs.append(logp)
            langs.append(lang)
    return results, log_probs, langs

# ---------------------- Perplexity computation ----------------------
def compute_perplexity(model, tokenizer, texts, preprocess_fn, device="cuda"):
    total_loss, total_tokens = 0.0, 0
    model.to(device); model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(preprocess_fn(t), return_tensors="pt", truncation=True, padding=False)
            ids = enc["input_ids"].to(device)
            outputs = model(ids, labels=ids)
            total_loss += outputs.loss.item() * ids.numel()
            total_tokens += ids.numel()
    return torch.exp(torch.tensor(total_loss/total_tokens)).item()

# ---------------------- Main pipeline ----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LAS-UnigramLM experiments.")
    parser.add_argument("--high", type=str, required=True, help="High-resource language code (e.g., 'en')")
    parser.add_argument("--low", type=str, required=True, help="Low-resource language code (e.g., 'yo')")
    parser.add_argument("--model", type=str, default="allenai/Olmo-3-1025-7B", choices=[
        "allenai/Olmo-3-1025-7B",
        "FacebookAI/xlm-roberta-base",
        "CohereLabs/aya-expanse-8b"
    ], help="Pretrained LM to use")
    parser.add_argument("--inference_string", type=str, default=None)
    parser.add_argument("--inference_file", type=str, default=None)
    parser.add_argument("--length_prior_weights", type=str, default="{}", help='JSON: {"en":0.1,"yo":0.3}')
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune LLM on LAS-tokenized text")
    parser.add_argument("--ft_epochs", type=int, default=1)
    parser.add_argument("--ft_batch_size", type=int, default=2)
    args = parser.parse_args()

    LANGUAGES = [args.high, args.low]
    length_prior_weights = json.loads(args.length_prior_weights)

    # ---- Load tokenizer with automatic adjustments ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(tokenizer, "byte_level") and tokenizer.byte_level:
        logger.info("Byte-level tokenizer detected; using raw text for preprocessing.")
        preprocess_fn = lambda x: x
    else:
        logger.info("Non-byte-level tokenizer detected; using normalized lowercase text for preprocessing.")
        preprocess_fn = lambda x: x.strip().lower()

    # ---- Efficient model loading for large models ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",  # auto device placement
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    logger.info(f"Loaded model {args.model} with memory-efficient settings.")

    # ---- Check corpora ----
    CORPUS_DIR = Path("data/parallel/dev")
    corpus_paths = {lang: CORPUS_DIR/f"{lang}.txt" for lang in LANGUAGES}
    for lang,p in corpus_paths.items():
        if not p.exists(): raise FileNotFoundError(f"Corpus {lang} not found: {p}")

    # ---- Train LAS-UnigramLM (model-specific folder) ----
    safe_model_name = args.model.replace("/", "_")
    OUTPUT_DIR = Path(f"multilingual/tokenizers/{safe_model_name}_{args.high}_{args.low}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shared_las = LanguageSpecificUnigramLMTokenizer(vocab_size=len(tokenizer), byte_level=True)
    tokenizer_paths = shared_las.train(
        corpus_info={lang:str(corpus_paths[lang]) for lang in LANGUAGES},
        base_tokenizer_path=str(OUTPUT_DIR/"shared_base.json"),
        output_dir=str(OUTPUT_DIR),
        load_base_if_exists=False,
        load_tokenizers_if_exists=False
    )
    logger.info(f"LAS-UnigramLM saved: {tokenizer_paths}")

    # ---- v* inference ----
    if args.inference_string:
        lang, toks, logp = v_star_tokenize(preprocess_fn(args.inference_string), shared_las, length_prior_weights)
        logger.info(f"Input string: {args.inference_string}")
        logger.info(f"Best lang: {lang}, tokens: {toks}, adjusted log-prob: {logp:.6f}")
    if args.inference_file:
        file_path = Path(args.inference_file)
        results, log_probs, langs = tokenize_file(file_path, shared_las, length_prior_weights)
        logger.info(f"v* tokenization results for {file_path}:")
        for toks, logp in zip(results, log_probs):
            print(f"{' '.join(toks)} | adjusted log-prob: {logp:.6f}")

    # ---- Optional fine-tuning ----
    if args.fine_tune:
        all_texts, lang_texts = [], {}
        for lang,path in corpus_paths.items():
            with open(path,"r",encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
                tok_lines = [" ".join(v_star_tokenize(preprocess_fn(l), shared_las, length_prior_weights)[1]) for l in lines]
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
            fp16=torch.cuda.is_available()
        )
        trainer = Trainer(model=model, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator, args=training_args)
        trainer.train()
        model.save_pretrained("las_finetuned_model")
        logger.info("Fine-tuning complete.")

        # ---- Automatic perplexity evaluation per language ----
        model_orig = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, device_map="auto")
        model_ft = AutoModelForCausalLM.from_pretrained("las_finetuned_model", trust_remote_code=True, device_map="auto")
        logger.info("\nPerplexity evaluation (original vs LAS pre/post fine-tuning):")
        table_header = f"{'Lang':<5} | {'#Lines':>5} | {'Orig PPL':>10} | {'LAS-pre PPL':>12} | {'LAS-post PPL':>13}"
        print(table_header)
        print("-"*len(table_header))
        for lang in LANGUAGES:
            texts = lang_texts[lang][:min(20,len(lang_texts[lang]))]
            ppl_orig = compute_perplexity(model_orig, tokenizer, texts, preprocess_fn)
            ppl_las_pre = compute_perplexity(model_orig, tokenizer, texts, preprocess_fn)
            ppl_las_post = compute_perplexity(model_ft, tokenizer, texts, preprocess_fn)
            print(f"{lang:<5} | {len(texts):>5} | {ppl_orig:>10.4f} | {ppl_las_pre:>12.4f} | {ppl_las_post:>13.4f}")

        all_sample = sum([lang_texts[l][:min(20,len(lang_texts[l]))] for l in LANGUAGES],[])
        ppl_orig_all = compute_perplexity(model_orig, tokenizer, all_sample, preprocess_fn)
        ppl_las_pre_all = compute_perplexity(model_orig, tokenizer, all_sample, preprocess_fn)
        ppl_las_post_all = compute_perplexity(model_ft, tokenizer, all_sample, preprocess_fn)
        print(f"{'ALL':<5} | {len(all_sample):>5} | {ppl_orig_all:>10.4f} | {ppl_las_pre_all:>12.4f} | {ppl_las_post_all:>13.4f}")

    logger.info("\nAll steps completed successfully.")

if __name__=="__main__":
    main()
