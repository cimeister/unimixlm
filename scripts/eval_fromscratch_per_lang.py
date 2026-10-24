#!/usr/bin/env python3
"""Per-language BpB evaluation for from-scratch GPT-2 125M models.

Evaluates each (model, tokenizer) on a *fixed* held-out slice (the last
VAL_DOCS_PER_LANG documents) of each training language's parquet, computing
bits-per-byte using the same convention as BpBTrainer in
`las_experiment/finetune.py`:

    BpB = Σ NLL / (Σ bytes × ln 2)

with NLL = mean_loss × num-non-pad-shifted-positions, and "bytes" = the
UTF-8 byte length of the *truncated* token sequence decoded back to string.

For LangSpec models, encoding uses LanguageSpecificUnigramLMTokenizer's
``best_language_encode_batch`` — i.e. each text is segmented with all 8
per-language tokenizers and the segmentation with the highest log-prob is
selected. Token IDs are shared across the per-lang tokenizers.

The held-out slice is the same set of documents for every model, so per-lang
BpB numbers are directly comparable across the four tokenizer variants and
both vocab sizes.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow `import code.*` from project root regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — match training-time settings exactly
# ─────────────────────────────────────────────────────────────────────────────
def _require_env(name: str, purpose: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Point it at {purpose}.")
    return value


DATA_DIR = _require_env(
    "FINEWEB2_SAMPLE_DIR",
    "the FineWeb2-sample directory "
    "(tokenizer_training/downstream_lm_training_data/fineweb2_sample)",
)
LANGUAGES = [
    "afr_Latn", "swh_Latn", "ron_Latn", "ind_Latn",
    "fil_Latn", "cat_Latn", "por_Latn", "dan_Latn",
]
FROMSCRATCH_ROOT = _require_env(
    "FROMSCRATCH_ROOT",
    "the scratch directory holding the from-scratch model/tokenizer runs "
    "(unimix_fromscratch)",
)
MAX_SEQ_LEN = 512                # matches training
VAL_DOCS_PER_LANG = 1000         # fixed held-out slice (last N docs of parquet)
EVAL_BATCH_SIZE = 64             # examples per forward pass


def list_runs() -> List[dict]:
    runs = []
    for vs in (32000, 100000):
        tok_dir = f"{FROMSCRATCH_ROOT}/tokenizers_{vs}"
        for tok_type in ("bpe", "unigram_base", "pathpiece", "langspec"):
            model_dir = f"{FROMSCRATCH_ROOT}/model_{tok_type}_v{vs // 1000}k"
            runs.append(
                {
                    "model_dir": model_dir,
                    "tok_type": tok_type,
                    "vocab_size": vs,
                    "tok_dir": tok_dir,
                }
            )
    return runs


def load_val_texts() -> Dict[str, List[str]]:
    """Last `VAL_DOCS_PER_LANG` non-empty documents per language."""
    out: Dict[str, List[str]] = {}
    for lang in LANGUAGES:
        ds = load_dataset("parquet", data_dir=f"{DATA_DIR}/{lang}", split="train")
        n = len(ds)
        start = max(0, n - VAL_DOCS_PER_LANG)
        texts: List[str] = []
        for i in range(start, n):
            t = ds[i]["text"]
            if t:
                t = t.strip()
            if t:
                texts.append(t)
        out[lang] = texts
        log.info(
            f"  {lang}: {len(texts)} non-empty docs from indices "
            f"[{start}, {n}) of {n}-doc parquet"
        )
    return out


def load_tokenizer_for_run(run: dict):
    """Return either a single Tokenizer or a LanguageSpecificUnigramLMTokenizer."""
    tt = run["tok_type"]
    tdir = run["tok_dir"]
    if tt == "bpe":
        return Tokenizer.from_file(f"{tdir}/bpe_tokenizer.json")
    if tt == "unigram_base":
        return Tokenizer.from_file(f"{tdir}/unigram_base.json")
    if tt == "pathpiece":
        return Tokenizer.from_file(f"{tdir}/pathpiece.json")
    if tt == "langspec":
        # WORKAROUND for from-scratch race-condition bug (see investigation
        # in conversation log): the on-disk `unigram_base.json` was overwritten
        # by a parallel base-training job AFTER the per-lang files were
        # generated, so its vocab no longer matches the per-lang files
        # (3-token symmetric difference, ~21k position swaps). The per-lang
        # files are mutually consistent, so we use one of them as the "base"
        # for the load() call. The assertion in
        # LanguageSpecificUnigramLMTokenizer.load() then trivially passes
        # (per-lang vs same per-lang). best_language_encode_batch only uses
        # self.per_lang_tok, so loading one per-lang as the base is purely a
        # mechanism to satisfy the assertion. The proper fix is to re-train
        # the from-scratch tokenizers with a deterministic build order so the
        # base file is not overwritten (see notes added to the conversation).
        las = LanguageSpecificUnigramLMTokenizer(byte_level=True)
        first_lang = LANGUAGES[0]
        las.load(
            base_path=f"{tdir}/langspec_{first_lang}.json",
            language_paths={l: f"{tdir}/langspec_{l}.json" for l in LANGUAGES},
        )
        return las
    raise ValueError(f"unknown tok_type: {tt}")


LANGSPEC_ENCODE_CHUNK = 100  # texts per call to best_language_encode_batch


def encode_texts(tok_obj, texts: List[str], tok_type: str):
    """Return (list_of_id_lists, decoder_for_byte_count, optional list_of_chosen_langs)."""
    if tok_type != "langspec":
        encs = tok_obj.encode_batch(texts, add_special_tokens=False)
        return [e.ids for e in encs], tok_obj, None

    # LangSpec: cross-language max-log-prob segmentation. Chunk the call so
    # we don't hold 1000-text × 8-tokenizer worth of Encoding objects in
    # memory at once — that combination OOMs the login node for 100k-vocab
    # Unigram models.
    ids_list: List[List[int]] = []
    best_langs: List[str] = []
    for i in range(0, len(texts), LANGSPEC_ENCODE_CHUNK):
        chunk = texts[i : i + LANGSPEC_ENCODE_CHUNK]
        bl, encs = tok_obj.best_language_encode_batch(chunk)
        ids_list.extend(e.ids for e in encs)
        best_langs.extend(bl)
        del encs, bl
    decoder = next(iter(tok_obj.per_lang_tok.values()))["tokenizer"]
    return ids_list, decoder, best_langs


def build_examples(ids_list, decoder, pad_id, eos_id):
    """Mirror TokenizedTextDataset's per-document chunking exactly."""
    examples = []
    for ids in ids_list:
        ids = ids[:MAX_SEQ_LEN]
        # bytes from decoded truncated tokens (matches BpBTrainer convention)
        decoded = decoder.decode(ids, skip_special_tokens=True)
        num_bytes = len(decoded.encode("utf-8"))
        if len(ids) < MAX_SEQ_LEN:
            ids = ids + [eos_id]
        pad_len = MAX_SEQ_LEN - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        labels = list(ids) + [-100] * pad_len
        ids_padded = ids + [pad_id] * pad_len
        examples.append(
            {
                "input_ids": ids_padded,
                "attention_mask": attention_mask,
                "labels": labels,
                "num_bytes": num_bytes,
            }
        )
    return examples


@torch.no_grad()
def compute_bpb(model, examples, device) -> dict:
    """Mirror BpBTrainer.evaluation_loop accumulator math."""
    model.eval()
    total_nll = 0.0
    total_loss_tokens = 0
    total_bytes = 0
    n = len(examples)
    for i in range(0, n, EVAL_BATCH_SIZE):
        batch = examples[i : i + EVAL_BATCH_SIZE]
        input_ids = torch.tensor(
            [e["input_ids"] for e in batch], dtype=torch.long, device=device
        )
        attention_mask = torch.tensor(
            [e["attention_mask"] for e in batch], dtype=torch.long, device=device
        )
        labels = torch.tensor(
            [e["labels"] for e in batch], dtype=torch.long, device=device
        )
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        # `loss` is mean CE over (labels[:, 1:] != -100); reproduce BpBTrainer's
        # accumulator: NLL contribution = mean_loss × num_loss_tokens
        n_loss_tok = (labels[:, 1:] != -100).sum().item()
        total_nll += out.loss.item() * n_loss_tok
        total_loss_tokens += n_loss_tok
        for e in batch:
            total_bytes += e["num_bytes"]
    if total_bytes == 0:
        return {
            "bpb": float("nan"),
            "n_examples": n,
            "n_loss_tokens": total_loss_tokens,
            "n_bytes": 0,
            "total_nll": total_nll,
        }
    bpb = total_nll / (total_bytes * math.log(2))
    return {
        "bpb": bpb,
        "n_examples": n,
        "n_loss_tokens": total_loss_tokens,
        "n_bytes": total_bytes,
        "total_nll": total_nll,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results/fromscratch_per_lang.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info(
        f"Loading val sets ({VAL_DOCS_PER_LANG} docs/lang from end of parquet) ..."
    )
    val_by_lang = load_val_texts()

    git_commit = os.popen("git rev-parse HEAD 2>/dev/null").read().strip()
    results = {
        "_meta": {
            "max_seq_len": MAX_SEQ_LEN,
            "val_docs_per_lang": VAL_DOCS_PER_LANG,
            "val_slice": "last N docs of parquet (deterministic, identical across models)",
            "data_dir": DATA_DIR,
            "languages": LANGUAGES,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "git_commit": git_commit,
            "encoding_method": {
                "non_langspec": "Tokenizer.encode_batch (single shared tokenizer)",
                "langspec": (
                    "LanguageSpecificUnigramLMTokenizer.best_language_encode_batch "
                    "(arg-max log-prob across all 8 per-language tokenizers)"
                ),
            },
            "byte_count_convention": (
                "len(decode(truncated_ids, skip_special_tokens=True).encode('utf-8'))"
            ),
            "loss_convention": (
                "BpBTrainer: NLL = out.loss.item() × (labels[:,1:] != -100).sum(); "
                "BpB = Σ NLL / (Σ bytes × ln 2)"
            ),
        },
        "runs": {},
    }

    # Resume support: if output JSON already exists, preserve completed runs
    out_path = Path(args.output)
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            results["runs"] = existing.get("runs", {})
            log.info(f"Resuming: found {len(results['runs'])} completed runs in {out_path}")
        except Exception as e:
            log.warning(f"Could not read existing {out_path}: {e}")

    for run in list_runs():
        mdir = run["model_dir"]
        if not Path(mdir, "model.safetensors").exists():
            log.warning(f"Skipping {mdir} (no model.safetensors)")
            continue
        run_name = Path(mdir).name
        if run_name in results["runs"]:
            log.info(f"=== {run_name} === [SKIP — already in output JSON]")
            continue
        log.info(f"\n=== {run_name} ===")
        log.info(f"  loading model from {mdir}")
        model = GPT2LMHeadModel.from_pretrained(
            mdir, torch_dtype=torch.bfloat16
        ).to(device)
        model.eval()

        log.info(f"  loading tokenizer ({run['tok_type']})")
        tok_obj = load_tokenizer_for_run(run)

        # All from-scratch tokenizers share <pad>=3, </s>=2, <unk>=0 (verified
        # offline), but pull from a real tokenizer to be safe.
        ref_tok = (
            tok_obj
            if run["tok_type"] != "langspec"
            else next(iter(tok_obj.per_lang_tok.values()))["tokenizer"]
        )
        pad_id = ref_tok.token_to_id("<pad>")
        eos_id = ref_tok.token_to_id("</s>")
        if pad_id is None or eos_id is None:
            raise RuntimeError(f"missing pad/eos in {run_name}")

        run_results = {
            "tok_type": run["tok_type"],
            "vocab_size": run["vocab_size"],
            "model_dir": mdir,
            "per_lang": {},
        }
        for lang in LANGUAGES:
            texts = val_by_lang[lang]
            log.info(f"  [{lang}] encoding {len(texts)} docs")
            ids_list, decoder, best_langs = encode_texts(
                tok_obj, texts, run["tok_type"]
            )
            examples = build_examples(ids_list, decoder, pad_id, eos_id)
            log.info(f"  [{lang}] computing BpB on {len(examples)} examples")
            res = compute_bpb(model, examples, device)
            res["chosen_lang_hist"] = (
                {l: best_langs.count(l) for l in LANGUAGES}
                if best_langs is not None
                else None
            )
            log.info(
                f"  [{lang}] bpb={res['bpb']:.4f}  bytes={res['n_bytes']:,} "
                f"loss_toks={res['n_loss_tokens']:,}"
            )
            if best_langs is not None:
                top = sorted(
                    res["chosen_lang_hist"].items(), key=lambda kv: -kv[1]
                )[:3]
                log.info(f"  [{lang}] top picked-langs: {top}")
            run_results["per_lang"][lang] = res
        results["runs"][run_name] = run_results

        # Persist after each model so crashes don't lose progress
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  [checkpoint] wrote {out_path}")

        del model
        torch.cuda.empty_cache()

    log.info(f"\nFinal output: {out_path}")

    # ─── summary table ──────────────────────────────────────────────────────
    print("\n=== Per-language BpB ===")
    header = f"{'run':30s}" + "".join(f"  {l:>10s}" for l in LANGUAGES) + "    avg"
    print(header)
    for run_name, rd in results["runs"].items():
        row = [f"{run_name:30s}"]
        bpbs = []
        for lang in LANGUAGES:
            b = rd["per_lang"][lang]["bpb"]
            row.append(f"  {b:>10.4f}")
            bpbs.append(b)
        row.append(f"  {sum(bpbs) / len(bpbs):.4f}")
        print("".join(row))


if __name__ == "__main__":
    main()
