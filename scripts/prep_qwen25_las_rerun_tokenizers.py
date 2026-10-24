#!/usr/bin/env python3
"""Prepare langspec + placebo tokenizers for the PR #21 Qwen2.5-0.5B rerun.

Replicates _prep_tok3.sh (the code-tokenizer branch, PR #21) exactly, with added
verification, for the tie-preserving rerun of the writeup tab:extrinsic
cells:

  1. Download langspec_<lang>_Latn.json from
     MultilingualUnigramLM/las-tokenizers-Qwen2.5-1.5B-<lang>
     (Qwen2.5 shares one tokenizer across 0.5B/1.5B, so these are the
     exact files the branch's 0.5B Phase-4 runs used).
  2. If the model's eos does not resolve, merge the model tokenizer's
     added_tokens (native ids) into the LAS json — byte-identical to the
     branch patch.
  3. Verify: eos/pad resolve; ACTUAL assigned ids recorded (HF tokenizers
     may reassign added-token ids sequentially from vocab length — record
     what really happens, don't trust the JSON's id field); every id the
     tokenizer can emit is < the model's embedding rows.
  4. Generate placebo_<lang>_Latn.json via scripts/make_placebo.py logic
     (seeded score permutation, vocab/ids untouched).
  5. Write a provenance manifest next to the outputs.

Fails loudly on any missing artifact — no fallbacks.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoConfig

LANGS = ["tur", "fin", "hun"]
SRC_REPO_TMPL = "MultilingualUnigramLM/las-tokenizers-Qwen2.5-1.5B-{lang}"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUT_ROOT = Path(__file__).resolve().parent.parent / "las_tokenizers" / "qwen2_5-0_5b_pr21"
PLACEBO_SEED = 1234  # make_placebo.py default; recorded in manifest
# Compression-matched placebo: global score permutation changes compression
# by 20-29% on the real corpora (confounds the control); rank-bucketed
# permutation at bucket=500 keeps bytes/token within 0.23% of langspec on
# tur/fin/hun while changing 7-8% of token boundaries (calibration
# 2026-08-13, 2000 docs/lang).
PLACEBO_MODE = "bucketed"
PLACEBO_BUCKET = 500
MAKE_PLACEBO = Path(__file__).resolve().parent / "make_placebo.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main() -> None:
    model_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_cfg = AutoConfig.from_pretrained(MODEL_NAME)
    if model_tok.pad_token is None:
        model_tok.pad_token = model_tok.eos_token
    model_tok_json = Path(
        hf_hub_download(MODEL_NAME, "tokenizer.json")
    )
    embed_rows = int(model_cfg.vocab_size)
    print(f"model={MODEL_NAME} eos={model_tok.eos_token!r} "
          f"pad={model_tok.pad_token!r} embed_rows={embed_rows}")

    manifest = {
        "script": "scripts/prep_qwen25_las_rerun_tokenizers.py",
        "date": str(date.today()),
        "model": MODEL_NAME,
        "placebo_seed": PLACEBO_SEED,
        "placebo_mode": PLACEBO_MODE,
        "placebo_bucket_size": PLACEBO_BUCKET,
        "langs": {},
    }

    for lang in LANGS:
        repo = SRC_REPO_TMPL.format(lang=lang)
        fname = f"langspec_{lang}_Latn.json"
        out_dir = OUT_ROOT / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / fname

        src = Path(hf_hub_download(repo, fname))
        src_sha = sha256(src)
        shutil.copyfile(src, dest)

        tok = Tokenizer.from_file(str(dest))
        patched = 0
        if tok.token_to_id(model_tok.eos_token) is None:
            ls = json.loads(dest.read_text())
            m = json.loads(model_tok_json.read_text())
            vlen = len(ls["model"]["vocab"])
            have_id = {a["id"] for a in ls.get("added_tokens", []) or []}
            have_c = {a["content"] for a in ls.get("added_tokens", []) or []}
            add = [a for a in (m.get("added_tokens", []) or [])
                   if a["id"] not in have_id and a["content"] not in have_c]
            for a in add:
                assert a["id"] >= vlen, (
                    f"special id {a['id']} collides with vocab(len {vlen})"
                )
            ls.setdefault("added_tokens", []).extend(add)
            dest.write_text(json.dumps(ls, ensure_ascii=False))
            patched = len(add)
            tok = Tokenizer.from_file(str(dest))

        eos_id = tok.token_to_id(model_tok.eos_token)
        pad_id = tok.token_to_id(model_tok.pad_token)
        if eos_id is None or pad_id is None:
            raise RuntimeError(
                f"{dest}: eos/pad do not resolve after patch "
                f"(eos_id={eos_id}, pad_id={pad_id})"
            )
        vocab_size_total = tok.get_vocab_size()
        if vocab_size_total > embed_rows:
            raise RuntimeError(
                f"{dest}: tokenizer can emit ids up to {vocab_size_total - 1} "
                f">= model embedding rows {embed_rows}"
            )
        probe = tok.encode("Merhaba dünya, kirjasto että 42.", add_special_tokens=False)
        if not probe.ids or max(probe.ids) >= embed_rows:
            raise RuntimeError(f"{dest}: probe encode produced out-of-range ids")

        placebo = out_dir / f"placebo_{lang}_Latn.json"
        subprocess.run(
            [sys.executable, str(MAKE_PLACEBO),
             "--langspec", str(dest), "--out", str(placebo),
             "--seed", str(PLACEBO_SEED),
             "--mode", PLACEBO_MODE, "--bucket-size", str(PLACEBO_BUCKET)],
            check=True,
        )
        # Placebo invariant: identical piece list + ids, permuted scores.
        pl = json.loads(placebo.read_text())
        lsj = json.loads(dest.read_text())
        assert [p for p, _ in pl["model"]["vocab"]] == \
               [p for p, _ in lsj["model"]["vocab"]], "placebo piece drift"
        assert sorted(s for _, s in pl["model"]["vocab"]) == \
               sorted(s for _, s in lsj["model"]["vocab"]), "placebo score-multiset drift"
        ptok = Tokenizer.from_file(str(placebo))
        assert ptok.token_to_id(model_tok.eos_token) == eos_id, "placebo eos drift"

        manifest["langs"][lang] = {
            "source_repo": repo,
            "source_file": fname,
            "source_sha256": src_sha,
            "patched_added_tokens": patched,
            "eos_token": model_tok.eos_token,
            "eos_id_actual": eos_id,
            "pad_id_actual": pad_id,
            "vocab_size_total": vocab_size_total,
            "langspec_sha256": sha256(dest),
            "placebo_sha256": sha256(placebo),
        }
        print(f"[{lang}] OK: vocab={vocab_size_total} eos_id={eos_id} "
              f"pad_id={pad_id} patched=+{patched} placebo=seed{PLACEBO_SEED}")

    (OUT_ROOT / "prep_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {OUT_ROOT / 'prep_manifest.json'}")


if __name__ == "__main__":
    main()
