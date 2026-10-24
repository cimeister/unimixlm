"""F15 overlay MorphScore table, v2: offsets-patched harness + repaired Arabic gold.

Differences from f15_final.py (v1):
  1. harness = eval_results/f15_intrinsic/harness_patched/morphscore_patched.py
     (predicted boundaries and per-token strings taken from
     tokenizers.Encoding.offsets instead of per-id decoding);
  2. Arabic is measured on eval_results/f15_intrinsic/arabic_data_repaired.csv
     (lemma undiacritized and re-located inside the wordform) as the primary
     row, with the original broken gold kept alongside, clearly labelled, for
     the record.

Everything else -- arms, configs, metric list, key schema -- matches v1.

Output key schema (unchanged from v1):
    "stem_eq_lemma={True|False}|{family}|{flores}|{arm}"
plus, for Arabic only, a parallel set of record-only keys
    "stem_eq_lemma={True|False}|{family}|arb_Arab|{arm}|gold=original-broken"

Usage:
    python f15_final_v2.py                 # all 10 cells, writes the merged JSON
    python f15_final_v2.py --cell qwen2_5-0_5b:tur   # one shard, writes a part file
    python f15_final_v2.py --merge         # merge part files into the final JSON
"""
import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness_patched"))
from morphscore_patched import MorphScore  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import os

def _env_dir(var, what):
    v = os.environ.get(var)
    if v is None:
        raise RuntimeError(f"{var} is not set; point it at {what}.")
    return v


DATA_DIR = _env_dir("MORPHSCORE_DATA_DIR", "the MorphScore per-language *_data.csv files")
F15 = str(HERE.parents[1] / "las_tokenizers" / "f15")
OUT = HERE / "f15_morphscore_final_v2.json"
PARTS = Path(_env_dir("MORPHSCORE_PARTS_DIR", "a scratch directory for per-(family,lang) part files"))

LANGS = {"tur": "tur_Latn", "fin": "fin_Latn", "hun": "hun_Latn",
         "arb": "arb_Arab", "hin": "hin_Deva"}
FAMS = ["qwen2_5-0_5b", "llama3.2-1b"]
M = ["micro_f1", "macro_f1", "morphscore_recall", "morphscore_precision",
     "micro_precision", "micro_recall", "mean_token_char_ratio"]

# Arabic gold variants: (label, data_dir, dataset name passed to MorphScore)
ARB_GOLDS = [
    ("repaired", str(HERE), "arabic_data_repaired.csv"),
    ("original-broken", DATA_DIR, "arb_Arab"),
]


def gold_variants(lang, flores):
    """Yield (gold_label, data_dir, dataset_key) for one language."""
    if lang == "arb":
        for label, ddir, name in ARB_GOLDS:
            yield label, ddir, name
    else:
        yield None, DATA_DIR, flores


def run_cell(fam, lang):
    flores = LANGS[lang]
    arms = {
        "Original(BPE)": f"{F15}/{fam}/model_tokenizer/tokenizer.json",
        "base_tokenizer.json": f"{F15}/{fam}/base_tokenizer.json",
        "LangMAP(langspec)": f"{F15}/{fam}/{lang}/langspec_{flores}.json",
    }
    for p in arms.values():
        if not Path(p).is_file():
            raise FileNotFoundError(f"missing tokenizer: {p}")

    res = {}
    for sel in (True, False):
        tag = f"stem_eq_lemma={sel}"
        for gold, ddir, dsname in gold_variants(lang, flores):
            for arm, path in arms.items():
                ms = MorphScore(data_dir=ddir, language_subset=[dsname],
                                by_split=False, freq_scale=True,
                                exclude_single_tok=False, stem_eq_lemma=sel)
                out = ms.eval(Tokenizer.from_file(path))
                if dsname not in out:
                    raise KeyError(f"MorphScore returned no entry for {dsname!r} "
                                   f"({fam}/{lang}/{arm}); got {list(out)}")
                r = out[dsname]
                if "error" in r:
                    raise RuntimeError(f"{fam}/{lang}/{arm}/{gold}: {r['error']}")
                key = f"{tag}|{fam}|{flores}|{arm}"
                if gold == "original-broken":
                    key += "|gold=original-broken"
                rec = {k: float(r[k]) for k in M}
                rec["num_samples"] = int(r["num_samples"])
                if gold is not None:
                    rec["gold"] = gold
                res[key] = rec
                print(f"{tag:<22}{fam:<14}{flores:<10}{str(gold):<17}{arm:<22}"
                      f"n={r['num_samples']:>6} "
                      + "".join(f"{r[k]:>10.4f}" for k in M), flush=True)
    return res


def metadata():
    return {
        "harness": "offsets-patched",
        "harness_path": str(HERE / "harness_patched" / "morphscore_patched.py"),
        "harness_upstream": "<local>/tokenizer-intrinsic-evals/morphscore/morphscore.py",
        "harness_upstream_md5": "232725beb78ad195d2a64f3ae5651681",
        "patch": ("predicted boundaries and per-token strings taken from "
                  "tokenizers.Encoding.offsets (character offsets into the "
                  "ORIGINAL, pre-normalization wordform) instead of per-id "
                  "decoding, which turns partial multi-byte tokens into U+FFFD "
                  "and drifts the cumulative boundary offsets"),
        "config": ("exclude_single_tok=False, freq_scale=True, by_split=False, "
                   "unique_only=True, exclude_numbers=True, splits=train/dev/test, "
                   "exclude_single_morpheme=True"),
        "arb_gold_primary": "repaired",
        "arb_gold_repaired_csv": str(HERE / "arabic_data_repaired.csv"),
        "arb_gold_original": f"{DATA_DIR}/arabic_data.csv",
        "arb_gold_note": ("keys carrying the suffix '|gold=original-broken' are the "
                          "original CSV's segmentation, kept for the record only; "
                          "its lemma column is vocalized while wordform is not, so "
                          "its prefix|stem|suffix split is a longest-common-substring "
                          "artefact (mean stem 1.77 chars, 26.5% one-character stems)"),
        "arb_repaired_subset": ("aligned subset only: 16014 of the 19126 filtered rows "
                                "(83.73%) where the undiacritized lemma occurs as a "
                                "contiguous substring of the wordform"),
        "source_v1": str(HERE / "f15_morphscore_final.json"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", help="fam:lang, e.g. qwen2_5-0_5b:tur")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        allres = {}
        parts = sorted(glob.glob(str(PARTS / "*.json")))
        expected = len(FAMS) * len(LANGS)
        if len(parts) != expected:
            raise RuntimeError(f"expected {expected} part files in {PARTS}, found {len(parts)}")
        for p in parts:
            allres.update(json.loads(Path(p).read_text()))
        allres["_metadata"] = metadata()
        OUT.write_text(json.dumps(allres, indent=2))
        print("wrote", OUT, f"({len(allres)-1} rows)")
        return

    if args.cell:
        fam, lang = args.cell.split(":")
        if fam not in FAMS or lang not in LANGS:
            raise ValueError(f"bad cell {args.cell!r}")
        res = run_cell(fam, lang)
        PARTS.mkdir(parents=True, exist_ok=True)
        (PARTS / f"{fam}__{lang}.json").write_text(json.dumps(res, indent=2))
        print("wrote part", PARTS / f"{fam}__{lang}.json")
        return

    allres = {}
    for fam in FAMS:
        for lang in LANGS:
            allres.update(run_cell(fam, lang))
    allres["_metadata"] = metadata()
    OUT.write_text(json.dumps(allres, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
