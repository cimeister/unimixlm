#!/usr/bin/env python3
"""Driver for the 8-family x 9-programming-language AST-leaf-boundary-alignment
grid that replaces the paper's overall code alignment table and its values table
with artifact-backed measurements.

WHY THIS EXISTS
---------------
The published code tables report a Base (base BPE) and an LM (LangMAP) column
for 9 model families x 9 programming languages. Neither column is backed by an
artifact:

  * ZERO LangMAP code overlays exist anywhere in the repo, so the LM column
    cannot be checked against anything.
  * The Base column's snippet selection is unrecoverable. The earlier
    AST re-verification (`eval_results/f15_intrinsic/ast_base_reverify.json`)
    reproduced the *phenomenon* the table shows (at reproducible corpus scales
    the Base value is essentially tokenizer-invariant across large-vocab
    families -- bit-identical for Llama-3.2-1B / Phi-4 / Qwen-2.5-0.5B) but
    could NOT reproduce the printed per-cell `n`, and the recomputed Base levels
    differ from the printed ones by 0.004-0.056 absolute.

This driver rebuilds BOTH columns from scratch, on a DOCUMENTED snippet set with
a fixed file list and per-file sha256s -- the reproducibility the old table
lacked.

ONE INVOCATION = ONE (family, programming-language) CELL
--------------------------------------------------------
A cell is:

  1. resolve the family's model tokenizer and its BASE Unigram conversion --
     REUSED READ-ONLY from the natural-language sweep
     (`scripts/sweep_intrinsic_v2.py`). Nothing is refit here; a family whose
     base is missing aborts and names the command that would produce it.
  2. ensure the (family, code-language) LangMAP overlay -- per-language
     re-estimation seeded with the BASE tokenizer's own trained log-scores
     (`prep_f15_tokenizers.BaseScoreSeededLASTokenizer`), soft EM,
     NUM_FINETUNE_EM_ITERATIONS sub-iterations, on that language's
     eval-disjoint StarCoder corpus (see CORPUS POLICY below).
  3. materialise the fixed evaluation snippet set for the language (see
     EVALUATION SET below) and hard-verify it round-trips through the eval
     tool's own loader.
  4. score three arms in ONE `ASTBoundaryMetrics.compute()` call --
     Original(BPE) / base_tokenizer.json / LangMAP(langspec) -- with the paper's
     own tool (`tokenizer_analysis/metrics/code_ast.py`, driven through the
     `_AstOnlyProvider` that `scripts/run_ast_only.py` defines, imported from
     that file so it cannot drift). Phase 1 of `compute()` parses each language
     once and shares the spans across tokenizers, so three arms cost one parse.
  5. write one JSON per cell carrying the metrics AND the sha256 of every input
     that determined them.

  Arm -> published column:  Original(BPE) -> "Base",  LangMAP(langspec) -> "LM".
  base_tokenizer.json is recorded as a third arm (it costs one extra encode and
  separates "Unigram re-scoring of the same vocabulary" from "language
  adaptation"), but it is NOT one of the table's two columns.

CORPUS POLICY (fitting) -- and the disjointness guarantee
---------------------------------------------------------
Established empirically: for all 9 languages the AST evaluation
parquet (`tokenizer-intrinsic-evals/starcoder/<lang>.parquet`, the file
`starcoder_ast_config.json` names) is an exact `id`-SUBSET of the repo's own
code-tokenizer-training sample
(`.../downstream_lm_training_data/starcoder_sample/<lang>/data.parquet`).
Overlap is 100% of eval rows, every language. So the sample CANNOT be used
as-is: it contains every evaluation snippet.

The fitting corpus is therefore:

    every row of starcoder_sample/<lang>/data.parquet whose `id` is NOT in the
    eval parquet's `id` set.

That is a structural disjointness guarantee at the level of the whole eval
parquet -- not merely the 100 snippets actually scored. It is verified twice:
by `id` (set difference, recorded) and by content sha256 (no normalised fitting
document may hash-collide with any evaluation snippet). Either check failing
aborts the build.

No cap is applied: the whole disjoint pool is used. That introduces no tuned
constant, and mirrors the natural-language protocol, which likewise fixes an
extraction rule rather than a corpus size (its realised per-language corpora
run 154-472 MB; the code pools run 140-242 M characters).

FILTER POLICY (stated, not improvised)
--------------------------------------
`las_experiment/tokenizer_training/training.py::filter_line_by_script` is a
NATURAL-LANGUAGE filter. Its own documented behaviour for code is to keep every
line: `filter_line_by_script` returns True immediately for any lang in
`_CODE_LANGS`. This driver therefore applies NO script filter to code text, which
is that function's code path, not a bypass of it.

One asymmetry is recorded rather than papered over: `_CODE_LANGS` lists
{python, javascript, java, go, c, cpp, rust, typescript, ruby, php, shell, bash}
but NOT `csharp`. Calling the repo's filter with "csharp" would raise
(`_resolve_target_script` cannot read a script suffix off it). The driver
declares the exemption per language in `CODE_LANGS` below and writes
`script_filter` into every corpus provenance record, so a reader can see which
languages the repo's own function would have covered and which one is covered by
this driver's declaration instead.

What the corpus files look like, and the one known consequence
-------------------------------------------------------------
Each document's source is written with its newlines intact, so one physical line
of code is one line of the corpus file. This is byte-for-byte the format the
repo's own code LAS tokenizers use (e.g.
`las_tokenizers/qwen3-0.6b_v151k_starcoder2gb_50klines_go-java-js-py_softem_20260503/corpora/python.txt`:
50,000 documents, 5,843,235 physical lines).

CONSEQUENCE, MEASURED AND REPORTED, NOT SILENTLY ACCEPTED:
`code/tokenizer_formatting_utils.py::write_hf_bytelevel_corpus` reads the corpus
line by line and does `line.rstrip("\\n")` before pre-tokenising, so the newline
character itself never appears inside a training sentence. Vocabulary pieces that
CONTAIN a newline (ByteLevel "Ċ") therefore get zero expected count during the
overlay's EM and their log-scores collapse, while at evaluation time whole
multi-line files are encoded. A collapsed "Ċ}"-style piece splits where the base
BPE would not, which tends to ADD token boundaries and can inflate LangMAP's
alignment for reasons that have nothing to do with language adaptation.

`newline_piece_diagnostic()` measures exactly this for every cell (mean base ->
langspec log-score change for Ċ-bearing vs other pieces, and the share of the
vocabulary affected) and the number is written into the cell JSON. It is
reported, not gated: see the smoke-cell findings.

Two things this format does NOT cost, both measured:
  * spm's default `--max_sentence_length` (4192 bytes; `SP_DEFAULT_ARGS` does not
    override it) drops nothing: 0 of 692,844 lines across python/javascript/
    typescript exceed it, longest line 995 bytes. The count is recorded per
    language anyway.
  * StarCoder metadata tags (`<reponame>`, `<filename>`, `<gh_stars>`) are
    stripped from the fitting corpus with the EVAL TOOL'S OWN regex
    (`CodeDataLoader._strip_starcoder_metadata`), and the same normalisation
    (`_normalize_source`: BOM strip, CRLF -> LF) is applied, so fitting text and
    evaluation text are drawn from one distribution. This DIFFERS from the repo's
    older code corpora, which left the tags in (their python.txt line 1 is
    `<reponame>Phaeton-lang/baselines<filename>...`). Declared here because it
    changes the artifact.

EVALUATION SET
--------------
`EVAL_SNIPPETS_PER_LANG` files per language, taken as the FIRST N snippets in
eval-parquet row order after the loader's own normalisation -- i.e. exactly what
`CodeDataLoader(max_snippets_per_lang=N)` yields from that parquet, which is what
the earlier AST re-verification's `cap100` configuration measured, so the Base column
here is directly comparable with `ast_base_reverify_raw_cap100.json`.

They are MATERIALISED to `eval_results/code_ast_v1/evalsets/<lang>/` as
zero-padded, correctly-extensioned files, with a manifest recording, per file:
its sha256, its source row's `id`, and its `max_stars_repo_path` /
`max_stars_repo_name`. The metric is then pointed at that DIRECTORY. The
materialised set is hard-verified to reload byte-identically through
`CodeDataLoader`, so pointing at the directory measures the same text the
parquet would have supplied -- while giving the fixed, hashed, citable file list
the published table never had.

ARTIFACT LOCATIONS
------------------
Overlays are written into the family's EXISTING artifact root, beside the base
they were seeded from -- `las_tokenizers/f15/<family>/<lang>/` for the two F15
families and
`las_tokenizers/intrinsic_sweep_v2/<NL_PROTOCOL_TAG>/<family>/<lang>/` for the
rest. That is `sweep_intrinsic_v2.py`'s own rule ("an overlay is only meaningful
next to the base it was seeded from"). Code language names cannot collide with
the natural-language sweep's (deu/fin/hun/tur/hin/arb/tam).

USAGE
-----
  # one cell (fit-if-missing, then score)
  python scripts/sweep_code_ast_v1.py cell --family qwen2_5-0_5b --lang python \\
      --base-langs tur,fin,hun,arb,hin

  # build just the eval-disjoint fitting corpus for one language
  python scripts/sweep_code_ast_v1.py corpus --lang python

  # materialise just the fixed evaluation snippet set for one language
  python scripts/sweep_code_ast_v1.py evalset --lang python

  # aggregate every finished cell into one manifest + the two paper tables' data
  python scripts/sweep_code_ast_v1.py manifest

`--base-langs` has NO default. It selects which natural-language base
conversion is reused, it changes every number, and it is part of the artifact
path -- so it is stated at every invocation, exactly as in the sweep this driver
reuses.

NO SILENT FALLBACKS. A missing base conversion, corpus, parquet or manifest field
aborts and names the artifact. Nothing is substituted.
"""

import os

# Must precede any tokenizers/transformers import -- Rayon reads these once, at
# first use. Same note as scripts/prep_f15_tokenizers.py and
# scripts/sweep_intrinsic_v2.py.
os.environ.setdefault("RAYON_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sweep_code_ast_v1")

# --- the paper's own intrinsic-eval package ---------------------------------
# The AST metric, its loader and the driver's provider all come from there. The
# path is overridable but never guessed: a missing checkout aborts by name.
TOKEVAL_REPO = Path(os.environ.get(
    "TOKEVAL_REPO", str(Path.home() / "tokenizer-intrinsic-evals")))
if not (TOKEVAL_REPO / "tokenizer_analysis" / "metrics" / "code_ast.py").is_file():
    raise RuntimeError(
        f"The paper's intrinsic-eval package is not at {TOKEVAL_REPO} "
        f"(expected tokenizer_analysis/metrics/code_ast.py under it). Set "
        f"TOKEVAL_REPO to the checkout that holds the AST metric; this driver "
        f"will not substitute another implementation.")
if str(TOKEVAL_REPO) not in sys.path:
    sys.path.insert(0, str(TOKEVAL_REPO))

import sweep_intrinsic_v2 as sw  # noqa: E402  (the NL sweep: families, gates, paths)
import prep_f15_tokenizers as prep  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from code.constants import NUM_FINETUNE_EM_ITERATIONS  # noqa: E402
from code.tokenizer_formatting_utils import _load_vocab_from_hf_tokenizer  # noqa: E402
from las_experiment.tokenizer_training.training import _CODE_LANGS  # noqa: E402
from las_experiment.tokenizer_training.validation import validate_tokenizers  # noqa: E402
from tokenizer_analysis.loaders.code_data import CodeDataLoader  # noqa: E402
from tokenizer_analysis.metrics.code_ast import ASTBoundaryMetrics  # noqa: E402


def _load_run_ast_only():
    """Import `_AstOnlyProvider` from the eval repo's own run_ast_only.py.

    Not reimplemented here: that file is the entry point the earlier
    AST re-verification used, and a local copy of its provider could drift from it
    silently."""
    path = TOKEVAL_REPO / "scripts" / "run_ast_only.py"
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing. This driver drives the AST metric through that "
            f"file's `_AstOnlyProvider` so the two cannot diverge; it will not "
            f"substitute a local copy.")
    spec = importlib.util.spec_from_file_location("_tokeval_run_ast_only", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RUN_AST_ONLY = _load_run_ast_only()
_AstOnlyProvider = _RUN_AST_ONLY._AstOnlyProvider


# ---------------------------------------------------------------------------
# Constants. Every one of these changes the numbers; none is buried.
# ---------------------------------------------------------------------------

# NEW CONSTANT. Files per language in the evaluation snippet set. 100 is the
# documented PRE-1.0.0 CodeDataLoader.DEFAULT_MAX_SNIPPETS_PER_LANG, and the cap
# the earlier AST re-verification measured its `cap100` column at, so this grid's
# Base column is directly comparable with
# eval_results/f15_intrinsic/ast_base_reverify_raw_cap100.json. It is also what
# makes 72 cells tractable: the uncapped corpus is 2.2-3.7M leaf spans per
# language.
EVAL_SNIPPETS_PER_LANG = 100

# Inherited, not redeclared: the overlay protocol is the natural-language
# sweep's, unchanged, so a code overlay and a language overlay of the same
# family differ only in their corpus.
EM_MODE = sw.EM_MODE                       # "soft"
PER_LANG_EM_SEED = sw.PER_LANG_EM_SEED     # "base_log_scores"
EM_ITERATIONS = NUM_FINETUNE_EM_ITERATIONS  # 10

# spm's own default; SP_DEFAULT_ARGS does not override it. Restated so the
# per-language over-long-line count recorded in each corpus provenance has a
# stated threshold rather than an implied one. Measured 0/692,844 lines over it.
MAX_SPM_SENTENCE_BYTES = 4192

EVAL_PARQUET_ROOT = Path(os.environ.get(
    "CODE_AST_EVAL_PARQUET_ROOT", str(TOKEVAL_REPO / "starcoder")))
_CODE_AST_FIT_PARQUET_ROOT = os.environ.get("CODE_AST_FIT_PARQUET_ROOT")
if not _CODE_AST_FIT_PARQUET_ROOT:
    raise RuntimeError(
        "CODE_AST_FIT_PARQUET_ROOT is not set. Point it at the "
        "starcoder_sample fitting-corpus directory "
        "(tokenizer_training/downstream_lm_training_data/starcoder_sample)."
    )
FIT_PARQUET_ROOT = Path(_CODE_AST_FIT_PARQUET_ROOT)

_LAS_TOK_CORPORA_ROOT = os.environ.get("LAS_TOK_CORPORA_ROOT")
if not _LAS_TOK_CORPORA_ROOT:
    raise RuntimeError(
        "LAS_TOK_CORPORA_ROOT is not set. Point it at the scratch directory "
        "holding the LAS tokenizer-training corpora."
    )
CORPORA_ROOT = Path(_LAS_TOK_CORPORA_ROOT) / "code_ast_v1_corpora"

OUT_ROOT = REPO_ROOT / "eval_results" / "code_ast_v1"
CELL_DIR = OUT_ROOT / "cells"
EVALSET_DIR = OUT_ROOT / "evalsets"

# The paper's nine programming languages (column order of the paper's
# overall code alignment table), each mapped to the file the AST config names, the fitting-sample
# directory, and the script-filter policy that applies to it.
#
# `in_repo_code_langs` records whether
# las_experiment/tokenizer_training/training.py::_CODE_LANGS covers the name.
# csharp is the one that it does not; its exemption is declared here so it is
# an artifact-recorded decision rather than an accident of a bypassed call.
CODE_LANGS = {
    "c":          {"label": "C",       "eval_parquet": "c.parquet",          "fit_dir": "c"},
    "cpp":        {"label": "C++",     "eval_parquet": "cpp.parquet",        "fit_dir": "cpp"},
    "csharp":     {"label": "C#",      "eval_parquet": "c-sharp.parquet",    "fit_dir": "c-sharp",
                   "script_filter_exemption": (
                       "training.py::_CODE_LANGS does not list 'csharp', so the repo's "
                       "filter_line_by_script would raise on it rather than pass it "
                       "through. No script filter is applied to code text by this "
                       "driver; declared per language so the exemption is on the "
                       "record.")},
    "go":         {"label": "Go",      "eval_parquet": "go.parquet",         "fit_dir": "go"},
    "java":       {"label": "Java",    "eval_parquet": "java.parquet",       "fit_dir": "java"},
    "javascript": {"label": "JS",      "eval_parquet": "javascript.parquet", "fit_dir": "javascript"},
    "php":        {"label": "PHP",     "eval_parquet": "php.parquet",        "fit_dir": "php"},
    "python":     {"label": "Py",      "eval_parquet": "python.parquet",     "fit_dir": "python"},
    "typescript": {"label": "TS",      "eval_parquet": "typescript.parquet", "fit_dir": "typescript"},
}

# The paper's nine code model families, each
# mapped to its key in scripts/sweep_intrinsic_v2.FAMILIES.
#
# Phi-3-mini is EXCLUDED: microsoft/Phi-3-mini-4k-instruct
# and microsoft/Phi-3-mini-128k-instruct both ship "pre_tokenizer": null, and unlike
# Yi-1.5-6B transformers' save_pretrained does NOT synthesise one
# (the saved tokenizer.json is still null, its space handling staying in a
# Prepend+Replace normalizer). The base-Unigram conversion pre-tokenises the corpus
# with the pre-tokenizer and validate_tokenizers requires it non-null, so the family
# cannot be converted by this pipeline. Microsoft published no non-instruct Phi-3-mini:
# a Hub search returns only -4k-instruct, -128k-instruct, Phi-3.5-mini-instruct and
# gguf/onnx repackagings of those. No substitute was used; the grid is 8 x 9 = 72.
CODE_FAMILIES = {
    "bloom-3b":         {"paper_row": "BLOOM-3B"},
    "gemma2-2b":        {"paper_row": "Gemma 2-2B",
                         # ASYMMETRY, declared: the natural-language sweep excluded
                         # gemma2-2b because the paper's tokenizer table row is Gemma 3
                         # (established by reproducing the paper's Devanagari
                         # subword count). The paper's code family list names "Gemma 2-2b"
                         # explicitly, so the code grid uses gemma2-2b. Its base
                         # conversion already exists -- it was built during the NL
                         # sweep and then dropped from that grid -- and is reused
                         # read-only here, so the two grids legitimately stand on
                         # different Gemma tokenizers.
                         "note": (
                             "google/gemma-2-2b. The NL sweep's Gemma row is "
                             "google/gemma-3-1b-pt; the paper's CODE family list says "
                             "Gemma 2-2b. Different tokenizers (256,000 vs 262,144 "
                             "entries); the two grids' Gemma rows are not the same "
                             "family and must not be pooled.")},
    "granite3.0-8b":    {"paper_row": "Granite 3.0-8B"},
    "llama3.2-1b":      {"paper_row": "Llama 3.2-1B"},
    "mistral-nemo-12b": {"paper_row": "Mistral-NeMo-12B"},
    "phi-4":            {"paper_row": "Phi-4"},
    "qwen2_5-0_5b":     {"paper_row": "Qwen 2.5-0.5B"},
    "xglm-2.9b":        {"paper_row": "XGLM-2.9B"},
}

PHI3_MINI_EXCLUSION = {
    "paper_row": "Phi-3-mini",
    "model_ids_checked": ["microsoft/Phi-3-mini-4k-instruct",
                          "microsoft/Phi-3-mini-128k-instruct"],
    "hub_pre_tokenizer": None,
    "save_pretrained_synthesises_pre_tokenizer": False,
    "base_variant_exists": False,
    "hub_search": ("microsoft/Phi-3-mini* returns only -4k-instruct, -128k-instruct, "
                   "Phi-3.5-mini-instruct and gguf/onnx repackagings; no non-instruct "
                   "base variant is published."),
    "consequence": "grid is 8 families x 9 languages = 72 cells, not 81",
    "verified": "re-measured by this driver",
}

# The three scored arms, in the order they are reported.
ARM_ORIGINAL = "Original(BPE)"
ARM_BASE_UNIGRAM = "base_tokenizer.json"
ARM_LANGMAP = "LangMAP(langspec)"
ARMS = (ARM_ORIGINAL, ARM_BASE_UNIGRAM, ARM_LANGMAP)

# Published-table column semantics, recorded so a reader of a cell JSON cannot
# mistake which arm became which column.
ARM_TO_PAPER_COLUMN = {ARM_ORIGINAL: "Base", ARM_LANGMAP: "LM",
                       ARM_BASE_UNIGRAM: None}


sha256 = sw.sha256
_Lock = sw._Lock


# `gemma2-2b` has artifacts under the natural-language sweep's protocol tag --
# its base conversion was built during that sweep and then dropped from that
# grid, because the paper's tokenizer table Gemma row is Gemma 3 -- but it is NOT
# a key in sweep_intrinsic_v2.FAMILIES (only a comment there). The paper's code
# family list names "Gemma 2-2b", so this grid needs it, and sw.family_root /
# sw.ensure_model_tokenizer both read sw.FAMILIES.
#
# Registered here rather than by editing the natural-language sweep: explicit,
# checked, reported, never overwriting an existing entry, and carrying
# paper_row=None so it can never enter that sweep's PAPER_GRID. Same pattern as
# sweep_intrinsic_v2.register_sweep_langs.
_GEMMA2_REGISTRATION = {
    "model_id": "google/gemma-2-2b",
    "paper_row": None,
    "paper_V": 256000,
    "provenance": "new",
    "registered_by": "scripts/sweep_code_ast_v1.py",
    "why": ("The paper's code family list names Gemma 2-2b; the NL sweep's Gemma "
            "row is gemma3-1b. Different tokenizers, so the two grids' Gemma rows "
            "are not the same family."),
}


def register_code_families() -> None:
    """Make gemma2-2b resolvable without touching the natural-language sweep."""
    existing = sw.FAMILIES.get("gemma2-2b")
    if existing is None:
        sw.FAMILIES["gemma2-2b"] = dict(_GEMMA2_REGISTRATION)
        logger.info("registered family gemma2-2b -> google/gemma-2-2b "
                    "(absent from sweep_intrinsic_v2.FAMILIES; paper_row=None so it "
                    "cannot enter that sweep's PAPER_GRID)")
    elif existing.get("model_id") != _GEMMA2_REGISTRATION["model_id"]:
        raise RuntimeError(
            f"Family id conflict for 'gemma2-2b': this grid expects "
            f"{_GEMMA2_REGISTRATION['model_id']!r} but "
            f"scripts/sweep_intrinsic_v2.py declares {existing.get('model_id')!r}. "
            f"Refusing to measure against an ambiguous family.")

    # The base conversion must already record the same model id, or the reused
    # artifact is not the tokenizer this grid claims to be measuring.
    root = sw.SWEEP_TOK_ROOT / sw.protocol_tag(["tur", "fin", "hun", "arb", "hin"])
    prov = root / "gemma2-2b" / "base_tokenizer.json.provenance.json"
    if prov.is_file():
        rec = json.loads(prov.read_text()).get("model_id")
        if rec != _GEMMA2_REGISTRATION["model_id"]:
            raise RuntimeError(
                f"{prov} records model_id={rec!r}, but this grid registers "
                f"gemma2-2b as {_GEMMA2_REGISTRATION['model_id']!r}. The reused "
                f"base is a different tokenizer than the one being claimed.")


register_code_families()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_lang(lang: str) -> dict:
    if lang not in CODE_LANGS:
        raise RuntimeError(
            f"{lang!r} is not one of the paper's nine code languages "
            f"{sorted(CODE_LANGS)}")
    if lang not in CodeDataLoader._LANG_EXTENSIONS:
        raise RuntimeError(
            f"{lang!r} has no entry in CodeDataLoader._LANG_EXTENSIONS, so the "
            f"eval tool cannot load it. Known: "
            f"{sorted(CodeDataLoader._LANG_EXTENSIONS)}")
    return CODE_LANGS[lang]


def check_family(family: str) -> dict:
    if family not in CODE_FAMILIES:
        raise RuntimeError(
            f"{family!r} is not one of the code grid's eight families "
            f"{sorted(CODE_FAMILIES)}. Phi-3-mini is excluded: "
            f"{PHI3_MINI_EXCLUSION['consequence']} "
            f"({PHI3_MINI_EXCLUSION['hub_search']})")
    if family not in sw.FAMILIES:
        raise RuntimeError(
            f"{family!r} has no entry in scripts/sweep_intrinsic_v2.FAMILIES, so "
            f"its model id and artifact root are undefined here. This driver "
            f"never invents one.")
    return CODE_FAMILIES[family]


def script_filter_policy(lang: str) -> dict:
    """What was (not) applied to this language's text, as a recorded fact."""
    cfg = CODE_LANGS[lang]
    covered = lang in _CODE_LANGS
    rec = {
        "applied": False,
        "rule": ("none: las_experiment/tokenizer_training/training.py::"
                 "filter_line_by_script returns True for every line of a code "
                 "language, so keeping every line IS that function's code path"),
        "in_repo_CODE_LANGS": covered,
    }
    if not covered:
        exemption = cfg.get("script_filter_exemption")
        if exemption is None:
            raise RuntimeError(
                f"{lang!r} is not in training.py::_CODE_LANGS and CODE_LANGS[{lang!r}] "
                f"declares no 'script_filter_exemption'. The repo's own filter would "
                f"raise on this name, so the policy for it must be declared "
                f"explicitly rather than assumed.")
        rec["exemption"] = exemption
    return rec


# ---------------------------------------------------------------------------
# Evaluation snippet set
# ---------------------------------------------------------------------------
def _normalise_like_loader(raw: str):
    """Exactly what CodeDataLoader._read_parquet does to one row's content."""
    text = CodeDataLoader._normalize_source(
        CodeDataLoader._strip_starcoder_metadata(raw))
    text = text.rstrip()
    if not text.strip():
        return None
    return text


def ensure_evalset(lang: str) -> dict:
    """Materialise the fixed evaluation snippet set and its manifest.

    The set is the first EVAL_SNIPPETS_PER_LANG snippets in eval-parquet row
    order after the loader's own normalisation -- what
    CodeDataLoader(max_snippets_per_lang=N) yields from that parquet. It is then
    hard-verified to reload byte-identically from the materialised directory, so
    the directory the metric reads is provably the same text.
    """
    cfg = check_lang(lang)
    import pandas as pd

    parquet = EVAL_PARQUET_ROOT / cfg["eval_parquet"]
    if not parquet.is_file():
        raise RuntimeError(
            f"Evaluation parquet for {lang!r} is missing: {parquet}. It is the "
            f"file starcoder_ast_config.json names; no substitute is used.")

    out_dir = EVALSET_DIR / lang
    man_path = out_dir / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        ok = (man.get("n_snippets") == EVAL_SNIPPETS_PER_LANG
              and man.get("eval_parquet_sha256") == sha256(parquet))
        if ok:
            for rec in man["files"]:
                p = out_dir / rec["file"]
                if not p.is_file() or sha256(p) != rec["sha256"]:
                    ok = False
                    break
        if ok:
            logger.info(f"[{lang}] reusing verified evalset at {out_dir}")
            return man
        logger.warning(f"[{lang}] evalset at {out_dir} does not verify -- rebuilding")

    ext = CodeDataLoader._LANG_EXTENSIONS[lang][0]
    df = pd.read_parquet(parquet)
    for col in ("content", "id"):
        if col not in df.columns:
            raise RuntimeError(
                f"{parquet} has no {col!r} column (columns: {list(df.columns)}); "
                f"the snippet set could not be given reproducible provenance.")

    files, texts = [], []
    scanned = 0
    for row in df.itertuples(index=False):
        scanned += 1
        text = _normalise_like_loader(getattr(row, "content"))
        if text is None:
            continue
        idx = len(texts)
        name = f"snippet_{idx:04d}{ext}"
        files.append({
            "file": name,
            "sha256": sha256_text(text),
            "chars": len(text),
            "source_row_id": str(getattr(row, "id")),
            "source_repo_path": str(getattr(row, "max_stars_repo_path", "")),
            "source_repo_name": str(getattr(row, "max_stars_repo_name", "")),
            "parquet_row_index": scanned - 1,
        })
        texts.append(text)
        if len(texts) >= EVAL_SNIPPETS_PER_LANG:
            break

    if len(texts) < EVAL_SNIPPETS_PER_LANG:
        raise RuntimeError(
            f"[{lang}] only {len(texts)} non-blank snippets available in "
            f"{parquet} (wanted {EVAL_SNIPPETS_PER_LANG}); refusing to score a "
            f"cell on a short evaluation set.")

    with _Lock(man_path):
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec, text in zip(files, texts):
            (out_dir / rec["file"]).write_text(text, encoding="utf-8")

        # Round-trip gate: the metric will read this DIRECTORY, so the directory
        # must hand back exactly the strings the parquet did. _read_file decodes
        # per line and rstrips, and _normalize_source is idempotent, so this
        # should hold -- it is checked rather than assumed.
        probe = CodeDataLoader({lang: str(out_dir)},
                               max_snippets_per_lang=EVAL_SNIPPETS_PER_LANG)
        probe.load_all()
        reloaded = probe.get_code_snippets(lang)
        if len(reloaded) != len(texts):
            raise RuntimeError(
                f"[{lang}] materialised evalset reloads as {len(reloaded)} "
                f"snippets, not {len(texts)}. The directory the metric reads is "
                f"not the snippet set that was selected; aborting.")
        for i, (a, b) in enumerate(zip(texts, reloaded)):
            if a != b:
                raise RuntimeError(
                    f"[{lang}] materialised evalset snippet {i} does not "
                    f"round-trip through CodeDataLoader (len {len(a)} -> "
                    f"{len(b)}). Aborting rather than measuring different text "
                    f"than was selected.")

        man = {
            "lang": lang,
            "paper_label": cfg["label"],
            "built": now(),
            "n_snippets": len(texts),
            "eval_snippets_per_lang_constant": EVAL_SNIPPETS_PER_LANG,
            "selection_rule": (
                "first N non-blank snippets in eval-parquet row order after "
                "CodeDataLoader's own normalisation (_strip_starcoder_metadata, "
                "_normalize_source, rstrip) -- identical to "
                "CodeDataLoader(max_snippets_per_lang=N) on that parquet"),
            "deterministic": True,
            "seed": None,
            "seed_note": ("selection is by parquet row order, not sampling, so no "
                          "seed exists to record; the file list below IS the "
                          "selection"),
            "eval_parquet": str(parquet),
            "eval_parquet_sha256": sha256(parquet),
            "eval_parquet_rows": int(len(df)),
            "rows_scanned_to_fill": scanned,
            "extension": ext,
            "dir": str(out_dir),
            "total_chars": sum(r["chars"] for r in files),
            "roundtrip_verified_through": "CodeDataLoader(dir).get_code_snippets",
            "files": files,
        }
        man_path.write_text(json.dumps(man, indent=2))
    logger.info(f"[{lang}] evalset: {len(texts)} files, "
                f"{man['total_chars']:,} chars -> {out_dir}")
    return man


# ---------------------------------------------------------------------------
# Fitting corpus (eval-disjoint)
# ---------------------------------------------------------------------------
def ensure_fit_corpus(lang: str) -> dict:
    """Build (or verify) the eval-disjoint fitting corpus for one language."""
    cfg = check_lang(lang)
    import pandas as pd

    eval_parquet = EVAL_PARQUET_ROOT / cfg["eval_parquet"]
    fit_parquet = FIT_PARQUET_ROOT / cfg["fit_dir"] / "data.parquet"
    for p, what in ((eval_parquet, "evaluation parquet"),
                    (fit_parquet, "fitting-sample parquet")):
        if not p.is_file():
            raise RuntimeError(f"{what} for {lang!r} is missing: {p}")

    CORPORA_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CORPORA_ROOT / f"{lang}.txt"
    prov_path = CORPORA_ROOT / f"{lang}.provenance.json"

    if out_path.is_file() and prov_path.is_file():
        prov = json.loads(prov_path.read_text())
        if (prov.get("eval_parquet_sha256") == sha256(eval_parquet)
                and prov.get("fit_parquet_sha256") == sha256(fit_parquet)
                and prov.get("sha256") == sha256(out_path)
                and prov.get("eval_snippets_per_lang") == EVAL_SNIPPETS_PER_LANG):
            logger.info(f"[{lang}] reusing verified fit corpus at {out_path}")
            return prov
        logger.warning(f"[{lang}] fit corpus/provenance stale -- rebuilding")

    ev = pd.read_parquet(eval_parquet, columns=["id", "content"])
    eval_ids = set(str(x) for x in ev["id"])

    # Content hashes of the WHOLE eval parquet, normalised the way the loader
    # does. The id filter below is the guarantee; this is the independent second
    # check that catches identical content republished under a different id.
    eval_hashes = set()
    for raw in ev["content"]:
        t = _normalise_like_loader(raw)
        if t is not None:
            eval_hashes.add(sha256_text(t))

    ft = pd.read_parquet(fit_parquet, columns=["id", "content"])
    fit_rows_total = len(ft)

    kept_docs = 0
    dropped_eval_id = 0
    dropped_blank = 0
    dropped_hash_collision = 0
    lines_written = 0
    chars_written = 0
    overlong_lines = 0
    overlong_chars = 0
    max_line_bytes = 0

    with _Lock(out_path):
        with open(out_path, "w", encoding="utf-8") as fout:
            for row in ft.itertuples(index=False):
                if str(getattr(row, "id")) in eval_ids:
                    dropped_eval_id += 1
                    continue
                text = _normalise_like_loader(getattr(row, "content"))
                if text is None:
                    dropped_blank += 1
                    continue
                if sha256_text(text) in eval_hashes:
                    # Same content, different id. Excluded so the disjointness
                    # claim holds on content as well as on id.
                    dropped_hash_collision += 1
                    continue
                for ln in text.split("\n"):
                    b = len(ln.encode("utf-8"))
                    if b > max_line_bytes:
                        max_line_bytes = b
                    if b > MAX_SPM_SENTENCE_BYTES:
                        overlong_lines += 1
                        overlong_chars += b
                    lines_written += 1
                fout.write(text + "\n")
                kept_docs += 1
                chars_written += len(text) + 1

        if kept_docs == 0:
            raise RuntimeError(
                f"[{lang}] fitting corpus is EMPTY after excluding the evaluation "
                f"rows ({dropped_eval_id} dropped by id, {dropped_hash_collision} "
                f"by content hash, {dropped_blank} blank of {fit_rows_total}). "
                f"Aborting rather than fitting on nothing.")

        if dropped_eval_id != len(eval_ids):
            # Recorded, not fatal: the eval parquet was an exact id-subset of the
            # sample for all nine languages when this was established. If that
            # ever stops holding, the number says so instead of the claim
            # quietly weakening.
            logger.warning(
                f"[{lang}] {dropped_eval_id} of {len(eval_ids)} evaluation ids "
                f"were found in {fit_parquet}. The eval parquet is no longer a "
                f"strict subset of the fitting sample; disjointness still holds "
                f"(nothing with an eval id or an eval content hash was written) "
                f"but the relationship between the two files has changed.")

        prov = {
            "lang": lang,
            "paper_label": cfg["label"],
            "built": now(),
            "path": str(out_path),
            "sha256": sha256(out_path),
            "policy": (
                "every row of the repo's code-tokenizer-training StarCoder sample "
                "whose `id` is NOT in the AST evaluation parquet's id set, and whose "
                "normalised content does not hash-collide with any evaluation "
                "snippet; no cap"),
            "eval_parquet": str(eval_parquet),
            "eval_parquet_sha256": sha256(eval_parquet),
            "eval_parquet_rows": int(len(ev)),
            "eval_snippets_per_lang": EVAL_SNIPPETS_PER_LANG,
            "fit_parquet": str(fit_parquet),
            "fit_parquet_sha256": sha256(fit_parquet),
            "fit_parquet_rows": int(fit_rows_total),
            "disjointness": {
                "by_id": True,
                "by_content_sha256": True,
                "eval_ids_total": len(eval_ids),
                "eval_ids_found_in_fit_sample": dropped_eval_id,
                "eval_parquet_is_id_subset_of_fit_sample":
                    dropped_eval_id == len(eval_ids),
                "docs_dropped_by_eval_id": dropped_eval_id,
                "docs_dropped_by_content_hash": dropped_hash_collision,
                "guarantee": (
                    "no document in this corpus shares an id with, or has the same "
                    "normalised content as, ANY row of the evaluation parquet -- not "
                    f"merely the {EVAL_SNIPPETS_PER_LANG} snippets actually scored"),
            },
            "docs_written": kept_docs,
            "docs_dropped_blank": dropped_blank,
            "lines_written": lines_written,
            "chars_written": chars_written,
            "line_format": (
                "one document per record, newlines intact, so one physical line of "
                "code is one spm sentence -- the format the repo's own code LAS "
                "corpora use"),
            "text_normalisation": (
                "CodeDataLoader._strip_starcoder_metadata + _normalize_source + "
                "rstrip: the eval tool's own functions, so fitting and evaluation "
                "text come from one distribution. NOTE this differs from the repo's "
                "older code corpora, which left the <reponame>/<filename>/<gh_stars> "
                "tags in."),
            "script_filter": script_filter_policy(lang),
            "spm_sentence_length": {
                "max_sentence_length_bytes": MAX_SPM_SENTENCE_BYTES,
                "source": "spm_train default; code/constants.SP_DEFAULT_ARGS does not override it",
                "lines_over_limit": overlong_lines,
                "chars_in_over_limit_lines": overlong_chars,
                "max_line_bytes": max_line_bytes,
                "note": "lines over the limit are dropped by spm_train, not by this driver",
            },
        }
        prov_path.write_text(json.dumps(prov, indent=2))
    logger.info(
        f"[{lang}] fit corpus: {kept_docs:,} docs / {lines_written:,} lines / "
        f"{chars_written:,} chars (excluded {dropped_eval_id} eval-id + "
        f"{dropped_hash_collision} hash-collision docs) -> {out_path}")
    return prov


# ---------------------------------------------------------------------------
# Family artifacts: base is REUSED READ-ONLY, overlay is fit here
# ---------------------------------------------------------------------------
def resolve_base_readonly(family: str, base_langs) -> tuple:
    """Resolve the family's model tokenizer and its EXISTING base conversion.

    Never fits. A missing base aborts and names the command that produces it --
    refitting it here would produce a second base under the same protocol tag
    and silently split the grid across two artifacts.
    """
    root = sw.family_root(family, base_langs)
    base_path = root / "base_tokenizer.json"
    model_tok = root / "model_tokenizer" / "tokenizer.json"
    if not base_path.is_file():
        raise RuntimeError(
            f"{family}: base Unigram conversion {base_path} does not exist. This "
            f"driver reuses the natural-language sweep's conversions read-only and "
            f"will not refit one. Produce it with:\n"
            f"  python scripts/sweep_intrinsic_v2.py convert --family {family} "
            f"--base-langs {','.join(base_langs)}")
    if not model_tok.is_file():
        raise RuntimeError(
            f"{family}: model tokenizer {model_tok} is missing, so the "
            f"Original(BPE) arm has no file to score and the base cannot be "
            f"validated against it.")

    mt = sw.ensure_model_tokenizer(family, root)
    sw._log_paper_V(family, mt)

    # The NL sweep's own gates, on the reuse path: pre/post-tokenizer and token
    # identity against the model tokenizer, then the read-only specials check.
    validate_tokenizers(model_tok, base_path)
    specials = sw.verify_specials_readonly(
        family, base_path, mt["_tok"], mt["embed_rows"], f"{family} base_tokenizer")

    prov_file = Path(str(base_path) + ".provenance.json")
    base_prov = json.loads(prov_file.read_text()) if prov_file.is_file() else None
    f15_manifest = None
    if sw.FAMILIES[family]["provenance"] == "f15":
        mpath = sw.F15_ROOT / family / "manifest.json"
        if not mpath.is_file():
            raise RuntimeError(
                f"{family}: reuse of F15 artifacts requires {mpath}, which is "
                f"missing; the protocol behind {root} cannot be verified.")
        f15_manifest = json.loads(mpath.read_text())
        rec = (f15_manifest.get("base_tokenizer") or {}).get("sha256")
        cur = sha256(base_path)
        if rec != cur:
            raise RuntimeError(
                f"{family}: base_tokenizer.json sha256 {cur} != F15 manifest {rec}; "
                f"the file changed after the manifest was written.")
    elif base_prov is None:
        raise RuntimeError(
            f"{family}: {base_path} exists but {prov_file} does not, so the corpora "
            f"and settings behind it cannot be verified. Refusing to build a "
            f"published table on an artifact of unknown provenance.")

    base_rec = {
        "path": str(base_path),
        "sha256": sha256(base_path),
        "reused_read_only": True,
        "refit_by_this_driver": False,
        "vocab_size_trainable": len(_load_vocab_from_hf_tokenizer(
            Tokenizer.from_file(str(base_path)))),
        "nl_protocol_tag": (sw.protocol_tag(base_langs)
                            if sw.FAMILIES[family]["provenance"] != "f15" else "f15"),
        "base_langs": list(base_langs),
        "provenance_file": str(prov_file) if base_prov is not None else None,
        "provenance": base_prov,
        "f15_manifest_seeding": (f15_manifest or {}).get("seeding"),
        **specials,
    }
    return root, mt, base_rec


def ensure_code_overlay(family: str, root: Path, lang: str, corpus: dict,
                        mt: dict) -> dict:
    """Ensure <root>/<lang>/langspec_<lang>.json for a code language.

    Same protocol as the natural-language overlays -- seeded with the base
    tokenizer's own trained log-scores, soft EM, EM_ITERATIONS sub-iterations --
    differing only in the corpus.
    """
    lang_dir = root / lang
    ls_path = lang_dir / f"langspec_{lang}.json"
    base_path = root / "base_tokenizer.json"

    expected_prov = {
        "kind": "langspec_overlay_code",
        "family": family,
        "model_id": mt["model_id"],
        "model_tokenizer_sha256": mt["sha256"],
        "base_tokenizer_sha256": sha256(base_path),
        "lang_code": lang,
        "domain": "code",
        "corpus": {
            "path": corpus["path"],
            "sha256": corpus["sha256"],
            "docs_written": corpus["docs_written"],
            "lines_written": corpus["lines_written"],
            "eval_parquet_sha256": corpus["eval_parquet_sha256"],
            "fit_parquet_sha256": corpus["fit_parquet_sha256"],
        },
        "em_mode": EM_MODE,
        "em_seed": PER_LANG_EM_SEED,
        "em_iterations": EM_ITERATIONS,
        "unk_token_internal": mt["unk_token"] or mt["eos_token"],
    }

    fitted = False
    t0 = time.time()
    if not ls_path.is_file():
        with _Lock(ls_path):
            lang_dir.mkdir(parents=True, exist_ok=True)
            trainer = prep.BaseScoreSeededLASTokenizer(
                reestimation_em_mode=EM_MODE,
                unk_token=mt["unk_token"] or mt["eos_token"],
            )
            if trainer.num_reestimation_iterations != EM_ITERATIONS:
                raise RuntimeError(
                    f"num_reestimation_iterations="
                    f"{trainer.num_reestimation_iterations}, expected "
                    f"{EM_ITERATIONS}")
            trainer.train(
                corpus_info={lang: corpus["path"]},
                base_tokenizer_path=str(base_path),
                tokenizer_path_format=str(lang_dir / "langspec_{lang_code}.json"),
                load_base_if_exists=True,
                load_tokenizers_if_exists=False,
                use_sentencepiece=True,
                max_workers=1,
            )
            fitted = True
            if not ls_path.is_file():
                raise RuntimeError(
                    f"{family}/{lang}: the overlay trainer returned without "
                    f"writing {ls_path}.")
            validate_tokenizers(root / "model_tokenizer" / "tokenizer.json", ls_path)
            override, _ = sw.embed_rows_policy(family)
            specials = prep.verify_and_patch_specials(
                ls_path, mt["_tok"], mt["_tok_json"],
                mt["embed_rows"] if override is None else override)
            specials.update(sw.apply_embed_rows_gate(
                family, ls_path, mt["embed_rows"], specials, f"{family}/{lang} overlay"))
            sw.write_provenance(ls_path, expected_prov)
    else:
        sw.check_provenance(ls_path, expected_prov, f"{family}/{lang} overlay")
        validate_tokenizers(root / "model_tokenizer" / "tokenizer.json", ls_path)
        specials = sw.verify_specials_readonly(
            family, ls_path, mt["_tok"], mt["embed_rows"], f"{family}/{lang} overlay")
    fit_seconds = round(time.time() - t0, 2) if fitted else None

    return {
        "path": str(ls_path),
        "sha256": sha256(ls_path),
        "fitted_now": fitted,
        "fit_seconds": fit_seconds,
        "em_mode": EM_MODE,
        "em_seed": PER_LANG_EM_SEED,
        "em_iterations": EM_ITERATIONS,
        "provenance_file": str(ls_path) + ".provenance.json",
        **specials,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _unigram_scores(path: Path) -> dict:
    """{piece: log_score} for a Unigram tokenizer.json."""
    j = json.loads(Path(path).read_text())
    model = j.get("model") or {}
    if model.get("type") != "Unigram":
        raise RuntimeError(f"{path} is a {model.get('type')!r} model, not Unigram; "
                           f"the score diagnostic does not apply to it.")
    return {p: float(s) for p, s in model["vocab"]}


def newline_piece_diagnostic(base_path: Path, langspec_path: Path) -> dict:
    """Quantify the one known consequence of the line-per-sentence corpus format.

    `write_hf_bytelevel_corpus` strips the newline before pre-tokenising, so
    vocabulary pieces containing a newline get zero expected count during the
    overlay's EM. Evaluation encodes whole multi-line files, so such a piece
    collapsing changes segmentation at line boundaries. This measures how far the
    newline-bearing pieces moved relative to the rest.

    "Ċ" (U+010A) is GPT-2 ByteLevel's rendering of 0x0A. Metaspace-family
    tokenizers keep a literal newline instead, so both are counted.
    """
    base, ls = _unigram_scores(base_path), _unigram_scores(langspec_path)
    shared = [p for p in base if p in ls]
    if not shared:
        raise RuntimeError(
            f"{base_path} and {langspec_path} share no vocabulary pieces; the "
            f"overlay is not a re-scoring of that base.")

    def has_nl(p: str) -> bool:
        return "Ċ" in p or "\n" in p

    nl = [p for p in shared if has_nl(p)]
    other = [p for p in shared if not has_nl(p)]

    def stats(keys):
        if not keys:
            return None
        d = [ls[p] - base[p] for p in keys]
        d.sort()
        return {
            "n": len(keys),
            "mean_delta_log_score": sum(d) / len(d),
            "median_delta_log_score": d[len(d) // 2],
            "p05_delta_log_score": d[max(0, int(0.05 * len(d)) - 1)],
            "min_delta_log_score": d[0],
            "max_delta_log_score": d[-1],
        }

    return {
        "what": ("mean base -> langspec log-score change for vocabulary pieces that "
                 "contain a newline vs those that do not. The fitting corpus's "
                 "line-per-sentence format hides newlines from EM, so a large "
                 "negative gap here means newline-bearing pieces collapsed and "
                 "LangMAP will split at line boundaries where the base BPE does "
                 "not -- a segmentation change that is an artifact of the corpus "
                 "format, not of language adaptation"),
        "shared_pieces": len(shared),
        "newline_pieces": stats(nl),
        "other_pieces": stats(other),
        "newline_piece_share": len(nl) / len(shared),
    }


def example_segmentations(arm_paths: dict, lang: str, evalset: dict,
                          n_lines: int = 6) -> list:
    """A few identifier-bearing lines segmented under every arm, for eyeballing."""
    toks = {a: Tokenizer.from_file(str(p)) for a, p in arm_paths.items()}
    src = (EVALSET_DIR / lang / evalset["files"][0]["file"]).read_text(encoding="utf-8")
    cands = [ln for ln in src.split("\n")
             if 20 <= len(ln) <= 90 and any(c.isalpha() for c in ln)
             and any(c in ln for c in "._(")]
    out = []
    for ln in cands[:n_lines]:
        rec = {"line": ln}
        for arm, t in toks.items():
            enc = t.encode(ln)
            rec[arm] = {"n_tokens": len(enc.tokens), "tokens": enc.tokens}
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_arms(lang: str, arm_paths: dict, evalset: dict) -> dict:
    """Run ASTBoundaryMetrics once over all arms for one language."""
    # class "huggingface" -> _load_huggingface_tokenizer -> its strategy 1, which
    # is `Tokenizer.from_file` for a path ending in .json. All three arms are
    # raw tokenizer.json files, so all three load through that one path and are
    # scored by one wrapper class -- no arm gets different offsets semantics from
    # the others.
    #
    # NOT "custom_bpe": despite CustomBPETokenizer._rust's docstring claiming a
    # .json path yields a raw Tokenizer, its from_config calls
    # _load_custom_bpe_from_directory unconditionally, which joins "vocab.json"
    # onto the path and dies with NotADirectoryError on a file. The docstring and
    # the code disagree; the code is what runs.
    tok_cfg = {arm: {"class": "huggingface", "path": str(p)}
               for arm, p in arm_paths.items()}
    code_cfg = {lang: str(EVALSET_DIR / lang)}

    provider = _AstOnlyProvider(tok_cfg)
    metric = ASTBoundaryMetrics(provider, code_config=code_cfg,
                                max_snippets_per_lang=EVAL_SNIPPETS_PER_LANG)
    t0 = time.time()
    out = metric.compute()
    seconds = round(time.time() - t0, 2)

    ast = out.get("ast_boundary_alignment", {})
    if "error" in ast:
        raise RuntimeError(f"[{lang}] AST metric returned an error: {ast['error']}")
    per_tok = ast.get("per_tokenizer", {})
    missing = [a for a in arm_paths if a not in per_tok]
    if missing:
        raise RuntimeError(
            f"[{lang}] the AST metric produced no result for arm(s) {missing}; "
            f"present: {sorted(per_tok)}")

    arms = {}
    for arm in arm_paths:
        entry = per_tok[arm]
        overall = entry.get("overall") or {}
        if not overall or overall.get("count", 0) == 0:
            raise RuntimeError(
                f"[{lang}] arm {arm!r} scored 0 leaf spans; refusing to record a "
                f"cell with nothing measured.")
        arms[arm] = {
            "paper_column": ARM_TO_PAPER_COLUMN[arm],
            "overall": overall,
            "per_language": entry.get("per_language"),
            "by_category": entry.get("by_category"),
        }

    # Fragmentation control. Boundary alignment is monotone in splitting: a
    # tokenizer that cuts more often puts a boundary at more span edges and
    # scores higher for that reason alone. `compute()` already returns
    # identifier fragmentation in the same call, so it costs nothing to keep,
    # and without it a positive Delta cannot be told apart from an arm that
    # simply segments more finely. Read alongside the alignment rates, never
    # instead of them.
    for arm in arm_paths:
        # Both blocks are shaped like ast_boundary_alignment: a per_tokenizer
        # level, then the arm, then overall/by_language.
        frag = ((out.get("identifier_fragmentation") or {})
                .get("per_tokenizer") or {}).get(arm)
        indent = ((out.get("indentation_consistency") or {})
                  .get("per_tokenizer") or {}).get(arm)
        if frag is None:
            raise RuntimeError(
                f"[{lang}] identifier_fragmentation holds no entry for arm {arm!r} "
                f"(present: {sorted(((out.get('identifier_fragmentation') or {}).get('per_tokenizer') or {}))}). "
                f"The fragmentation control is what separates a real alignment gain "
                f"from an arm that simply splits more, so a cell is not recorded "
                f"without it.")
        arms[arm]["identifier_fragmentation"] = frag
        arms[arm]["indentation_consistency"] = indent

    def _avg_tokens(arm):
        return ((arms[arm].get("identifier_fragmentation") or {})
                .get("overall") or {}).get("avg_tokens_per_identifier")

    base_r = arms[ARM_ORIGINAL]["overall"]["full_alignment_rate"]
    lm_r = arms[ARM_LANGMAP]["overall"]["full_alignment_rate"]
    return {
        "arms": arms,
        "summary": ast.get("summary"),
        "fragmentation_control": {
            "what": ("avg tokens per identifier for each arm. AST boundary "
                     "alignment rises with splitting, so a LangMAP gain that comes "
                     "with a higher token count per identifier is at least partly a "
                     "fragmentation effect rather than better structural agreement."),
            "avg_tokens_per_identifier": {a: _avg_tokens(a) for a in arm_paths},
        },
        "dropped_languages": getattr(metric, "_dropped_languages", {}),
        "seconds": seconds,
        "paper_cell": {
            "Base": base_r,
            "LM": lm_r,
            "Delta": lm_r - base_r,
            "n": arms[ARM_ORIGINAL]["overall"].get("count"),
            "n_langmap": arms[ARM_LANGMAP]["overall"].get("count"),
        },
    }


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------
def run_cell(family: str, lang: str, base_langs, force: bool = False) -> dict:
    check_family(family)
    check_lang(lang)
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    cell_path = CELL_DIR / f"{family}__{lang}.json"

    if cell_path.is_file() and not force:
        rec = json.loads(cell_path.read_text())
        stale = []
        for key, path in (("base_tokenizer", rec.get("base_tokenizer", {}).get("path")),
                          ("overlay", rec.get("overlay", {}).get("path"))):
            recorded = (rec.get(key) or {}).get("sha256")
            if path and Path(path).is_file() and sha256(Path(path)) != recorded:
                stale.append(key)
        if not stale:
            logger.info(f"[{family}/{lang}] cell already present and verified: "
                        f"{cell_path}")
            return rec
        logger.warning(f"[{family}/{lang}] recorded {stale} hashes no longer match "
                       f"-- recomputing")

    t_cell = time.time()
    evalset = ensure_evalset(lang)
    corpus = ensure_fit_corpus(lang)
    root, mt, base_rec = resolve_base_readonly(family, base_langs)
    overlay = ensure_code_overlay(family, root, lang, corpus, mt)

    arm_paths = {
        ARM_ORIGINAL: Path(mt["path"]),
        ARM_BASE_UNIGRAM: Path(base_rec["path"]),
        ARM_LANGMAP: Path(overlay["path"]),
    }
    scored = score_arms(lang, arm_paths, evalset)

    rec = {
        "schema": "code_ast_v1/cell/1",
        "written": now(),
        "family": family,
        "paper_row": CODE_FAMILIES[family]["paper_row"],
        "family_note": CODE_FAMILIES[family].get("note"),
        "lang": lang,
        "paper_col": CODE_LANGS[lang]["label"],
        "grid": {"families": len(CODE_FAMILIES), "languages": len(CODE_LANGS),
                 "cells": len(CODE_FAMILIES) * len(CODE_LANGS),
                 "phi_3_mini_exclusion": PHI3_MINI_EXCLUSION},
        "protocol": {
            "em_mode": EM_MODE,
            "em_seed": PER_LANG_EM_SEED,
            "em_iterations": EM_ITERATIONS,
            "base_langs": list(base_langs),
            "base_reused_read_only": True,
            "eval_snippets_per_lang": EVAL_SNIPPETS_PER_LANG,
            "script_filter": script_filter_policy(lang),
            "arm_to_paper_column": ARM_TO_PAPER_COLUMN,
        },
        "model_tokenizer": {k: v for k, v in mt.items() if not k.startswith("_")},
        "base_tokenizer": base_rec,
        "overlay": overlay,
        "fit_corpus": corpus,
        "evalset": {k: v for k, v in evalset.items() if k != "files"},
        "evalset_files": evalset["files"],
        "results": scored,
        "diagnostics": {
            "newline_pieces": newline_piece_diagnostic(
                Path(base_rec["path"]), Path(overlay["path"])),
            "example_segmentations": example_segmentations(arm_paths, lang, evalset),
        },
        "timing": {
            "overlay_fit_seconds": overlay["fit_seconds"],
            "ast_score_seconds": scored["seconds"],
            "cell_seconds": round(time.time() - t_cell, 2),
        },
        "code_fingerprint": {
            **sw.code_fingerprint(),
            "sweep_code_ast_v1.py": sha256(Path(__file__).resolve()),
            "code_ast.py": sha256(TOKEVAL_REPO / "tokenizer_analysis" / "metrics" / "code_ast.py"),
            "code_data.py": sha256(TOKEVAL_REPO / "tokenizer_analysis" / "loaders" / "code_data.py"),
            "run_ast_only.py": sha256(TOKEVAL_REPO / "scripts" / "run_ast_only.py"),
        },
        "env": sw.env_versions(),
        "git_commit": sw.git_commit(),
    }
    cell_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    p = scored["paper_cell"]
    logger.info(
        f"[{family}/{lang}] Base={p['Base']:.4f} LM={p['LM']:.4f} "
        f"Delta={p['Delta']:+.4f} n={p['n']} "
        f"({rec['timing']['cell_seconds']:.0f}s) -> {cell_path}")
    return rec


def build_manifest() -> dict:
    cells = sorted(CELL_DIR.glob("*.json")) if CELL_DIR.is_dir() else []
    rows, missing = [], []
    for fam in CODE_FAMILIES:
        for lang in CODE_LANGS:
            p = CELL_DIR / f"{fam}__{lang}.json"
            if not p.is_file():
                missing.append(f"{fam}/{lang}")
                continue
            r = json.loads(p.read_text())
            pc = r["results"]["paper_cell"]
            rows.append({
                "family": fam, "paper_row": r["paper_row"],
                "lang": lang, "paper_col": r["paper_col"],
                "Base": pc["Base"], "LM": pc["LM"], "Delta": pc["Delta"],
                "n": pc["n"],
                "base_unigram": r["results"]["arms"][ARM_BASE_UNIGRAM]["overall"]["full_alignment_rate"],
            })
    man = {
        "schema": "code_ast_v1/manifest/1",
        "written": now(),
        "grid": {"families": sorted(CODE_FAMILIES), "languages": list(CODE_LANGS),
                 "cells_expected": len(CODE_FAMILIES) * len(CODE_LANGS),
                 "cells_present": len(rows), "cells_missing": missing},
        "phi_3_mini_exclusion": PHI3_MINI_EXCLUSION,
        "eval_snippets_per_lang": EVAL_SNIPPETS_PER_LANG,
        "rows": rows,
        "cell_files": [str(c) for c in cells],
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(man, indent=2))
    logger.info(f"manifest: {len(rows)}/{len(CODE_FAMILIES) * len(CODE_LANGS)} cells "
                f"-> {OUT_ROOT / 'manifest.json'}")
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cell", help="fit-if-missing then score one (family, language)")
    c.add_argument("--family", required=True)
    c.add_argument("--lang", required=True)
    c.add_argument("--base-langs", required=True,
                   help="which natural-language base conversion to reuse; no default")
    c.add_argument("--force", action="store_true")

    co = sub.add_parser("corpus", help="build the eval-disjoint fitting corpus")
    co.add_argument("--lang", required=True)

    ev = sub.add_parser("evalset", help="materialise the fixed evaluation snippet set")
    ev.add_argument("--lang", required=True)

    sub.add_parser("manifest", help="aggregate every finished cell")

    args = ap.parse_args()
    if args.cmd == "cell":
        run_cell(args.family, args.lang, sw.parse_base_langs(args.base_langs),
                 force=args.force)
    elif args.cmd == "corpus":
        print(json.dumps(ensure_fit_corpus(args.lang), indent=2))
    elif args.cmd == "evalset":
        m = ensure_evalset(args.lang)
        print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=2))
    elif args.cmd == "manifest":
        build_manifest()


if __name__ == "__main__":
    main()
