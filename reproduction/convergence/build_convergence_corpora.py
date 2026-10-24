#!/usr/bin/env python3
"""Build the per-(language, N) estimation corpora for the convergence table.

Grid: N in {125, 500, 1000, 4000, 16000, 32000} documents (rows of the
FineWeb2-<lang>-100M dataset, the unit that ``scripts/prep_f15_tokenizers.py``
and ``LanguageSpecificUnigramLMTokenizer`` count), plus one production-scale
point: the first document index at which the cumulative token count, measured
with the shared Qwen2.5-0.5B base tokenizer, reaches PROD_TOKEN_TARGET
(25,000,000 tokens, the budget used for every LangMAP fit in the paper).

Unit caveat. The paper's convergence table describes its ~32k-example
reference corpus as about 5M characters, i.e. about 156 characters per
example, which corresponds to sentence- or line-level examples. This script
uses FineWeb2 documents, the only estimation-corpus protocol in this
repository, and FineWeb2 documents are about 2,900 to 3,600 characters long
(see the corpora manifests). "125 training examples" here therefore covers
roughly 19 to 25 times more text than 125 examples in the paper's original
protocol. Under the document protocol the 25M-token threshold is crossed at
18k to 26k documents for all three languages, below the N=32000 grid point.

Corpora sourcing:
  - fin, tur: reconstructed from the Hub revision recorded in the F15 corpus
    provenance files written by ``scripts/prep_f15_tokenizers.py`` under
    ``$LAS_TOK_CORPORA_ROOT/f15_corpora/``. The reconstruction is verified
    byte-for-byte against the recorded sha256 before use (see
    verify_reconstruction()); on mismatch the script aborts. This guarantees
    the small-N prefixes are genuine prefixes of the document stream the F15
    fin/tur overlays were fit on.
  - deu: downloaded from the Hub; the revision is resolved at build time and
    recorded in the manifest.

Environment:
  CONVERGENCE_WORK_DIR   directory for the corpora and manifests this script
                         writes (required)
  LAS_TOK_CORPORA_ROOT   root holding f15_corpora/<lang>.provenance.json
                         (required for fin and tur)

Run one language at a time:
    python build_convergence_corpora.py --lang fin
    python build_convergence_corpora.py --lang tur
    python build_convergence_corpora.py --lang deu
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer  # noqa: E402

from las_experiment.tokenizer_training.training import filter_line_by_script  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_convergence_corpora")


def _env_dir(var: str, what: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"{var} is not set; point it at {what}.")
    return Path(value)


OUT_DIR = _env_dir("CONVERGENCE_WORK_DIR", "the directory for convergence corpora and manifests")
CORPORA_SCRATCH = OUT_DIR / "corpora"
F15_CORPORA_ROOT = _env_dir(
    "LAS_TOK_CORPORA_ROOT", "the corpora root that holds f15_corpora/<lang>.provenance.json"
) / "f15_corpora"
BASE_TOKENIZER_PATH = REPO_ROOT / "las_tokenizers/f15/qwen2_5-0_5b/base_tokenizer.json"

LANG_CODES = {"deu": "deu_Latn", "fin": "fin_Latn", "tur": "tur_Latn"}
GRID_DOCS = [125, 500, 1000, 4000, 16000, 32000]  # "training examples" grid (documents)
PROD_TOKEN_TARGET = 25_000_000  # token budget of every LangMAP fit in the paper
# Safety cap on documents scanned per language before aborting rather than
# silently shipping a corpus smaller than requested.
SAFETY_CAP_DOCS = 70_000


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_path(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def load_kept_documents(lang_code: str, revision: str) -> tuple[list[str], int, int, int]:
    """Walk the pinned-revision FineWeb2-<lang_code>-100M dataset in file
    order, applying the same filter_line_by_script gate as
    scripts/prep_f15_tokenizers.py's build_corpus(), collecting up to
    SAFETY_CAP_DOCS kept documents. Returns (kept, scanned, filtered_empty,
    filtered_script).
    """
    from datasets import load_dataset

    repo = f"MultilingualUnigramLM/FineWeb2-{lang_code}-100M"
    t0 = time.time()
    ds = load_dataset(repo, split="train", revision=revision)
    logger.info(f"[{lang_code}] loaded dataset ({len(ds)} rows) in {time.time()-t0:.1f}s")

    kept, scanned, filtered_empty, filtered_script = [], 0, 0, 0
    for ex in ds:
        scanned += 1
        text = (ex.get("text") or "").strip()
        if not text:
            filtered_empty += 1
            continue
        if not filter_line_by_script(text, lang_code):
            filtered_script += 1
            continue
        kept.append(text)
        if len(kept) >= SAFETY_CAP_DOCS:
            break

    if len(kept) < max(GRID_DOCS):
        raise RuntimeError(
            f"[{lang_code}] Only collected {len(kept)} documents after scanning "
            f"{scanned} rows (filtered_empty={filtered_empty}, "
            f"filtered_script={filtered_script}); need at least {max(GRID_DOCS)} "
            f"for the convergence grid. Aborting rather than silently using a "
            f"short corpus."
        )
    return kept, scanned, filtered_empty, filtered_script


def verify_reconstruction(lang_code: str, kept: list[str]) -> dict:
    """For fin/tur: verify the reconstructed document list, joined exactly as
    build_corpus() wrote it, reproduces the existing F15 corpus file's
    recorded sha256, i.e. this really is the same document stream (same
    order, same content) that las_tokenizers/f15/qwen2_5-0_5b/{lang}/
    langspec_{lang_code}.json was fit on. Aborts on any mismatch.
    """
    prov_path = F15_CORPORA_ROOT / f"{lang_code}.provenance.json"
    if not prov_path.exists():
        raise RuntimeError(f"Expected existing F15 provenance file not found: {prov_path}")
    prov = json.loads(prov_path.read_text())
    expected_n = prov["lines_written"]
    expected_sha = prov["sha256"]

    prefix = kept[:expected_n]
    joined = ("\n".join(prefix) + "\n").encode("utf-8")
    got_sha = sha256_bytes(joined)
    if got_sha != expected_sha or len(prefix) != expected_n:
        raise RuntimeError(
            f"[{lang_code}] Reconstruction of the existing F15 corpus FAILED "
            f"to reproduce it byte-for-byte (expected {expected_n} docs, "
            f"sha256={expected_sha}; got {len(prefix)} docs, sha256={got_sha}). "
            f"Refusing to use this document stream for the convergence grid: "
            f"it would not be a genuine prefix relationship with the real F15 "
            f"fin/tur overlays. Aborting rather than silently substituting a "
            f"divergent corpus."
        )
    logger.info(f"[{lang_code}] Reconstruction verified byte-identical to existing F15 corpus "
                f"({expected_n} docs, sha256 match).")
    return prov


def build_lang(lang: str) -> dict:
    lang_code = LANG_CODES[lang]
    base_tok = Tokenizer.from_file(str(BASE_TOKENIZER_PATH))
    CORPORA_SCRATCH.mkdir(parents=True, exist_ok=True)

    if lang in ("fin", "tur"):
        prov = json.loads((F15_CORPORA_ROOT / f"{lang_code}.provenance.json").read_text())
        revision = prov["revision"]
        repo = prov["repo"]
        kept, scanned, filt_empty, filt_script = load_kept_documents(lang_code, revision)
        verify_reconstruction(lang_code, kept)
        source_note = (
            "Reconstructed (and sha256-verified byte-identical) from the exact "
            "HF revision recorded in the existing F15 corpus provenance."
        )
    elif lang == "deu":
        from huggingface_hub import HfApi

        repo = f"MultilingualUnigramLM/FineWeb2-{lang_code}-100M"
        api = HfApi()
        info = api.dataset_info(repo)
        revision = info.sha
        kept, scanned, filt_empty, filt_script = load_kept_documents(lang_code, revision)
        source_note = (
            "Downloaded from the Hub (no pre-existing F15 corpus for deu). "
            "Revision resolved at build time."
        )
    else:
        raise ValueError(f"Unknown lang {lang!r}")

    # Cumulative token/char counts over the ordered kept-document stream via
    # the SAME base tokenizer used to seed the per-language EM re-estimation
    # (las_tokenizers/f15/qwen2_5-0_5b/base_tokenizer.json).
    encs = base_tok.encode_batch(kept)
    cum_tok, cum_chr = 0, 0
    per_doc_cum_tok = []
    for enc, doc in zip(encs, kept):
        cum_tok += len(enc.ids)
        cum_chr += len(doc)
        per_doc_cum_tok.append((cum_tok, cum_chr))

    prod_doc_count = None
    for i, (ct, cc) in enumerate(per_doc_cum_tok, start=1):
        if ct >= PROD_TOKEN_TARGET:
            prod_doc_count = i
            break
    if prod_doc_count is None:
        raise RuntimeError(
            f"[{lang_code}] Never reached PROD_TOKEN_TARGET={PROD_TOKEN_TARGET} "
            f"tokens within {len(kept)} documents (cumulative tokens at end: "
            f"{per_doc_cum_tok[-1][0]}). SAFETY_CAP_DOCS={SAFETY_CAP_DOCS} too "
            f"low; aborting rather than silently using a corpus below the "
            f"stated production scale."
        )

    points = {f"N{n}": n for n in GRID_DOCS}
    points["prod25M"] = prod_doc_count

    lang_manifest = {
        "lang": lang,
        "lang_code": lang_code,
        "repo": repo,
        "revision": revision,
        "source_note": source_note,
        "docs_scanned": scanned,
        "docs_filtered_empty": filt_empty,
        "docs_filtered_script": filt_script,
        "docs_collected": len(kept),
        "base_tokenizer_used_for_sizing": str(BASE_TOKENIZER_PATH),
        "prod_token_target": PROD_TOKEN_TARGET,
        "points": {},
    }

    for label, n_docs in points.items():
        out_path = CORPORA_SCRATCH / f"{lang_code}_{label}.txt"
        prefix = kept[:n_docs]
        out_path.write_text("\n".join(prefix) + "\n", encoding="utf-8")
        cum_tok_here, cum_chr_here = per_doc_cum_tok[n_docs - 1]
        lang_manifest["points"][label] = {
            "n_docs": n_docs,
            "n_tokens": cum_tok_here,
            "n_chars": cum_chr_here,
            "tokens_per_char": cum_tok_here / cum_chr_here,
            "path": str(out_path),
            "sha256": sha256_path(out_path),
        }
        logger.info(
            f"[{lang_code}] {label}: n_docs={n_docs} n_tokens={cum_tok_here} "
            f"n_chars={cum_chr_here} -> {out_path}"
        )

    return lang_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(LANG_CODES.keys()))
    args = ap.parse_args()

    manifest = build_lang(args.lang)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / f"corpora_manifest_{args.lang}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
