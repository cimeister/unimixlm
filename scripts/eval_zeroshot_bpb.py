#!/usr/bin/env python3
"""
Zero-shot BpB evaluation: compare tokenizer strategies on the base model.

Tests whether LAS tokenization improves a pretrained model's bits-per-byte
WITHOUT any finetuning. This isolates the tokenization effect from the
training adaptation effect.

For each language, computes BpB using:
  - original (BPE): model's native tokenizer
  - base (Unigram): shared Unigram tokenizer
  - pathpiece: shortest-segmentation Unigram
  - langspec: language-specific Unigram

Usage:
  python scripts/eval_zeroshot_bpb.py --lang afr
  python scripts/eval_zeroshot_bpb.py --lang afr,swh,ron,ind
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def compute_bpb(model, token_ids_list, num_bytes_list, device, batch_size=4, max_seq_len=1024):
    """Compute bits-per-byte on a list of tokenized sequences."""
    total_nll = 0.0
    total_bytes = 0

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(token_ids_list), batch_size), desc="Computing BpB"):
            batch_ids = token_ids_list[i : i + batch_size]
            batch_bytes = num_bytes_list[i : i + batch_size]

            # Pad to max length in batch
            max_len = min(max(len(ids) for ids in batch_ids), max_seq_len)
            padded = []
            masks = []
            for ids in batch_ids:
                ids = ids[:max_len]
                pad_len = max_len - len(ids)
                padded.append(ids + [0] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)

            input_ids = torch.tensor(padded, device=device)
            attention_mask = torch.tensor(masks, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute per-token NLL (shifted: predict position i+1 from position i)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.shape)

            # Mask padding
            masked_losses = token_losses * shift_mask
            batch_nll = masked_losses.sum().item()

            total_nll += batch_nll
            total_bytes += sum(batch_bytes)

    bpb = total_nll / (total_bytes * math.log(2)) if total_bytes > 0 else float("inf")
    return bpb, total_nll, total_bytes


def tokenize_texts(texts, tokenizer, max_seq_len=1024):
    """Tokenize texts and compute byte counts."""
    token_ids_list = []
    num_bytes_list = []

    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        ids = enc.ids[:max_seq_len]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        num_bytes = len(decoded.encode("utf-8"))

        if len(ids) > 0 and num_bytes > 0:
            token_ids_list.append(ids)
            num_bytes_list.append(num_bytes)

    return token_ids_list, num_bytes_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="afr,swh,ron,ind")
    parser.add_argument("--model", type=str, default="allenai/Olmo-3-1025-7B")
    parser.add_argument("--tokenizer_dir", type=str, default="./las_tokenizers/olmo3-7b_v100k_fineweb2sample_10klines_afr-ind-ron-swh_softem")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="FineWeb2-sample directory "
                             "(tokenizer_training/downstream_lm_training_data/fineweb2_sample); "
                             "defaults to $FINEWEB2_SAMPLE_DIR if unset")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--peft_path", type=str, default=None,
                        help="Optional: path to LoRA adapter for finetuned model evaluation")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name for text data (text for FineWeb2, content for StarCoder)")
    parser.add_argument("--lang_suffix", type=str, default="_Latn",
                        help="Suffix appended to lang code for data dirs (empty for code)")
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = os.environ.get("FINEWEB2_SAMPLE_DIR")
        if not args.data_dir:
            raise RuntimeError(
                "Neither --data_dir nor FINEWEB2_SAMPLE_DIR is set; pass "
                "--data_dir or set FINEWEB2_SAMPLE_DIR to the FineWeb2-sample "
                "directory."
            )

    langs = [l.strip() for l in args.lang.split(",")]
    tok_dir = Path(args.tokenizer_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logging.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    if args.peft_path:
        from peft import PeftModel
        logging.info(f"Loading LoRA adapter: {args.peft_path}")
        model = PeftModel.from_pretrained(model, args.peft_path)

    model.eval()

    # Load model tokenizer (BPE)
    model_tokenizer = AutoTokenizer.from_pretrained(args.model)
    bpe_tokenizer = model_tokenizer.backend_tokenizer

    # Load alternative tokenizers
    tokenizers = {"original": bpe_tokenizer}

    base_path = tok_dir / "base_tokenizer.json"
    if base_path.exists():
        tokenizers["base"] = Tokenizer.from_file(str(base_path))

    pp_path = tok_dir / "pathpiece_tokenizer.json"
    if pp_path.exists():
        tokenizers["pathpiece"] = Tokenizer.from_file(str(pp_path))

    # Results
    print()
    print("=" * 75)
    print(f"  {'Lang':<6} {'Tokenizer':<14} {'Eval BpB':>10} {'Δ vs BPE':>10} {'Tokens':>10} {'Bytes':>10}")
    print("=" * 75)

    for lang in langs:
        # Load langspec tokenizer for this language
        lang_path = tok_dir / f"langspec_{lang}{args.lang_suffix}.json"
        lang_tokenizers = dict(tokenizers)
        if lang_path.exists():
            lang_tokenizers["langspec"] = Tokenizer.from_file(str(lang_path))

        # Load eval data
        data_path = Path(args.data_dir) / f"{lang}{args.lang_suffix}"
        logging.info(f"Loading data: {data_path}")
        ds = load_dataset("parquet", data_dir=str(data_path), split="train")

        # Use last N samples as eval (not first, to avoid overlap with training)
        texts = [x[args.text_column] for x in ds]
        eval_texts = texts[-args.max_samples:]
        logging.info(f"{lang}: using {len(eval_texts)} eval samples (from end of dataset)")

        bpe_bpb = None
        for tok_name, tok in lang_tokenizers.items():
            token_ids_list, num_bytes_list = tokenize_texts(eval_texts, tok, args.max_seq_len)

            bpb, total_nll, total_bytes = compute_bpb(
                model, token_ids_list, num_bytes_list, device,
                batch_size=args.batch_size, max_seq_len=args.max_seq_len,
            )

            if tok_name == "original":
                bpe_bpb = bpb

            delta = f"{bpb - bpe_bpb:+.4f}" if bpe_bpb is not None else "—"
            total_tokens = sum(len(ids) for ids in token_ids_list)

            print(f"  {lang:<6} {tok_name:<14} {bpb:>10.4f} {delta:>10} {total_tokens:>10,} {total_bytes:>10,}")

        print()

    print("=" * 75)


if __name__ == "__main__":
    main()
