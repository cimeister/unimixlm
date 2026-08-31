"""Repair the Arabic MorphScore gold segmentation.

Problem: in arabic_data.csv the `lemma` column is fully vocalized (harakat)
while `wordform` is (essentially) unvocalized, so whatever produced
preceding_part|stem|following_part could not match the lemma inside the
wordform.  Its output has the signature of a longest-common-SUBSTRING fallback:
mean stem length 1.77 chars against mean wordform length 5.73, and 26.5% of
stems are a single character.  (prefix+stem+suffix == wordform holds for 100%
of rows, but that is trivially true of ANY index split of the wordform and does
not validate the boundaries.)

Repair: undiacritize the lemma, locate it inside the wordform, and re-derive the
split from that span.

Normalization is applied ONLY for matching.  The strings written out are always
slices of the ORIGINAL wordform, so prefix+stem+suffix == wordform still holds
exactly and the tokenizer still sees the untouched wordform.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
_data_dir = os.environ.get("MORPHSCORE_DATA_DIR")
if _data_dir is None:
    raise RuntimeError("MORPHSCORE_DATA_DIR is not set; point it at the MorphScore data (needs arabic_data.csv).")
SRC = str(Path(_data_dir) / "arabic_data.csv")
OUT_CSV = Path(__file__).resolve().parent / "arabic_data_repaired.csv"
OUT_STATS = OUT_CSV.parent / "arabic_repair_stats.json"

# --- normalization tables ---------------------------------------------------
# Deleted (combining marks + the pure-presentation tatweel). These never appear
# in unvocalized orthography, so they cannot be part of a matchable stem.
DELETE = set(chr(c) for c in range(0x064B, 0x0653))  # fathatan..sukun
DELETE |= {"ٰ"}   # superscript alef
DELETE |= {"ـ"}   # tatweel
# 1:1 letter substitutions (position-preserving).
ALEF = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
YA = {"ى": "ي"}
TA = {"ة": "ه"}
STAGE = 2  # 0 = delete only; 1 = +ALEF; 2 = +YA; 3 = +TA


def normalize(s, stage=STAGE):
    """(normalized string, index map into `s`)."""
    out, idx = [], []
    for i, ch in enumerate(s):
        if ch in DELETE:
            continue
        if stage >= 1 and ch in ALEF:
            ch = ALEF[ch]
        if stage >= 2 and ch in YA:
            ch = YA[ch]
        if stage >= 3 and ch in TA:
            ch = TA[ch]
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def strip_only(s):
    """Undiacritized lemma orthography (stage 0), used for the `lemma` column."""
    return "".join(ch for ch in s if ch not in DELETE)


def nan_str(x):
    return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def main():
    df = pd.read_csv(SRC, encoding="utf-8")
    f = pd.concat([df[df["data_split"].astype(str).str.startswith(s)]
                   for s in ["train", "dev", "test"]])
    f = f[f["unique"] == "unique"]
    f = f[~f["wordform"].astype(str).str.contains(r"\d")]
    n_total = len(f)

    rows = []
    n_aligned = 0
    n_multi = 0
    n_trailing_mark = 0
    unaligned_examples = []
    disagree = []
    agree_full = 0
    agree_partial = 0
    compared = 0

    for _, r in f.iterrows():
        wf = str(r["wordform"])
        lem_voc = str(r["lemma"])
        wfn, imap = normalize(wf)
        lemn, _ = normalize(lem_voc)
        if not lemn:
            unaligned_examples.append((wf, lem_voc, "empty lemma after normalization"))
            continue
        p = wfn.find(lemn)
        if p < 0:
            if len(unaligned_examples) < 400:
                unaligned_examples.append((wf, lem_voc, "lemma not a substring"))
            continue
        n_aligned += 1
        if wfn.count(lemn) > 1:
            n_multi += 1
        a = imap[p]
        b = imap[p + len(lemn) - 1] + 1
        if b < len(wf) and wf[b] in DELETE:
            n_trailing_mark += 1
        prefix, stem, suffix = wf[:a], wf[a:b], wf[b:]
        assert prefix + stem + suffix == wf

        # --- agreement vs the existing gold split -------------------------
        old_p, old_s, old_f = (nan_str(r["preceding_part"]), nan_str(r["stem"]),
                               nan_str(r["following_part"]))
        old_b = set()
        if old_p:
            old_b.add(len(old_p))
        if old_f:
            old_b.add(len(old_p) + len(old_s))
        new_b = set()
        if prefix:
            new_b.add(len(prefix))
        if suffix:
            new_b.add(len(prefix) + len(stem))
        compared += 1
        if old_b == new_b:
            agree_full += 1
        elif old_b & new_b:
            agree_partial += 1
            if len(disagree) < 200:
                disagree.append((wf, lem_voc, old_p, old_s, old_f, prefix, stem, suffix, "partial"))
        else:
            if len(disagree) < 200:
                disagree.append((wf, lem_voc, old_p, old_s, old_f, prefix, stem, suffix, "none"))

        row = r.to_dict()
        row["lemma_vocalized"] = lem_voc
        row["lemma"] = strip_only(lem_voc)
        row["stem"] = stem
        row["preceding_part"] = prefix if prefix else np.nan
        row["following_part"] = suffix if suffix else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = list(df.columns) + ["lemma_vocalized"]
    out = out[cols]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # ---------------- statistics ----------------
    print(f"total scored-subset rows            : {n_total}")
    print(f"aligned (lemma contiguous in word)  : {n_aligned}  = {n_aligned/n_total:.2%}")
    print(f"multi-occurrence (ambiguous) rows   : {n_multi}")
    print(f"stem span followed by a combining mark (mark assigned to suffix): {n_trailing_mark}")

    stem_len = out["stem"].astype(str).str.len()
    print(f"\nrepaired stem length dist: {collections.Counter(stem_len).most_common(10)}")
    print(f"repaired 1-char stem rate: {(stem_len==1).mean():.4f}   mean stem len {stem_len.mean():.2f}")

    def nmorph(row):
        n = 1
        if isinstance(row["preceding_part"], str) and row["preceding_part"]:
            n += 1
        if isinstance(row["following_part"], str) and row["following_part"]:
            n += 1
        return n
    nm = out.apply(nmorph, axis=1)
    print(f"repaired n_morphemes dist: {collections.Counter(nm).most_common()}")
    print(f"rows that will be SCORED (n_morphemes>1, exclude_single_morpheme=True): {(nm>1).sum()}")
    print(f"repaired stem_eq_lemma=True rows: {(out['stem']==out['lemma']).sum()}")

    print(f"\nAGREEMENT vs existing gold (on the {compared} aligned rows):")
    print(f"  identical boundary set : {agree_full}  = {agree_full/compared:.2%}")
    print(f"  partial overlap        : {agree_partial}  = {agree_partial/compared:.2%}")
    print(f"  no overlap             : {compared-agree_full-agree_partial}"
          f"  = {(compared-agree_full-agree_partial)/compared:.2%}")

    stats = dict(n_total=n_total, n_aligned=n_aligned, aligned_frac=n_aligned/n_total,
                 n_multi=n_multi, n_trailing_mark=n_trailing_mark,
                 n_scored=int((nm > 1).sum()),
                 n_stem_eq_lemma=int((out["stem"] == out["lemma"]).sum()),
                 repaired_1char_stem_rate=float((stem_len == 1).mean()),
                 repaired_mean_stem_len=float(stem_len.mean()),
                 agree_full=agree_full, agree_partial=agree_partial,
                 agree_none=compared-agree_full-agree_partial, compared=compared,
                 normalization_stage=STAGE)
    OUT_STATS.write_text(json.dumps(stats, indent=2))

    print("\n--- 20 ALIGNED EXAMPLES (existing gold -> repaired) ---")
    for i in range(0, min(len(out), 4000), 197):
        r = out.iloc[i]
        orig = f.iloc[i] if False else None
        print(f"  wf={r['wordform']!r:16} lemma_voc={r['lemma_vocalized']!r:18} "
              f"-> {nan_str(r['preceding_part'])!r} | {r['stem']!r} | {nan_str(r['following_part'])!r}")

    print("\n--- 25 DISAGREEING ROWS (old split vs repaired split) ---")
    for row in disagree[:25]:
        wf, lv, op, os_, of, np_, ns, nf, kind = row
        print(f"  wf={wf!r:16} lemma_voc={lv!r:18} old={op!r}|{os_!r}|{of!r:8} "
              f"new={np_!r}|{ns!r}|{nf!r:8} [{kind}]")

    print("\n--- 25 UNALIGNED ROWS ---")
    for wf, lv, why in unaligned_examples[:25]:
        print(f"  wf={wf!r:16} lemma_voc={lv!r:18} norm_lemma={normalize(lv)[0]!r:16} "
              f"norm_wf={normalize(wf)[0]!r:16} ({why})")

    print("\nwrote", OUT_CSV, len(out), "rows")


if __name__ == "__main__":
    main()
