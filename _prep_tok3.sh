#!/usr/bin/env bash
# _prep_tok3.sh — one-time Phase-3 tokenizer prep (no GPU, deterministic,
# idempotent). For each (model,lang) it downloads langspec_<lang>_Latn.json
# and ensures the model's eos/pad token resolves via Tokenizer.token_to_id
# (finetune.py raises otherwise). If it does NOT resolve (Qwen LAS Unigram
# vocab lacks the special tokens), it merges the model tokenizer's own
# added_tokens (native ids, byte-for-byte from _modeltok_<model>/
# tokenizer.json) into the LAS json — exact embedding alignment, Unigram
# scores untouched, mirroring how the working Olmo LAS toks carry eos.
# gemma2-2b resolves out-of-box (<eos>=id 1) -> patch auto-skipped.
# Persistent (small, reboot-safe): /workspace/unimixlm/_las_tok3/...
set -euo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
source .venv/bin/activate

# model_key -> HF LAS-tokenizer repo prefix (repo = <prefix><lang>)
declare -A REPO_PREFIX=(
  [qwen2_5-1_5b]="MultilingualUnigramLM/las-tokenizers-Qwen2.5-1.5B-"
  [gemma2-2b]="MultilingualUnigramLM/las-tokenizers-gemma-2-2b-"
)
MODELS=(qwen2_5-1_5b gemma2-2b)
LANGS=(tur fin hun)

for MODEL in "${MODELS[@]}"; do
  PREFIX="${REPO_PREFIX[$MODEL]}"
  for LANG in "${LANGS[@]}"; do
    D="$REPO/_las_tok3/$MODEL/$LANG"; mkdir -p "$D"
    LS="$D/langspec_${LANG}_Latn.json"
    REPO_ID="${PREFIX}${LANG}"
    echo "[prep3] $MODEL/$LANG: fetch langspec from $REPO_ID"
    python3 - "$REPO_ID" "$LANG" "$LS" "$MODEL" <<'PY'
import os,sys,shutil,json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer
repo,lang,dest,model=sys.argv[1:5]
p=hf_hub_download(repo,f"langspec_{lang}_Latn.json",token=os.environ["HF_TOKEN"])
shutil.copyfile(p,dest)
mt=AutoTokenizer.from_pretrained(f"./_modeltok_{model}")
def eos_ok(path):
    t=Tokenizer.from_file(path)
    return t.token_to_id(mt.eos_token) is not None, t.get_vocab_size()
ok,vsz=eos_ok(dest)
if ok:
    print(f"  resolves out-of-box (vocab={vsz}) -> no patch")
else:
    ls=json.loads(open(dest).read())
    m=json.loads(open(f"_modeltok_{model}/tokenizer.json").read())
    vlen=len(ls["model"]["vocab"])
    have_id={a["id"] for a in ls.get("added_tokens",[]) or []}
    have_c={a["content"] for a in ls.get("added_tokens",[]) or []}
    add=[a for a in (m.get("added_tokens",[]) or [])
         if a["id"] not in have_id and a["content"] not in have_c]
    for a in add:
        assert a["id"]>=vlen, f"special id {a['id']} collides with vocab(len {vlen})"
    ls.setdefault("added_tokens",[]).extend(add)
    open(dest,"w").write(json.dumps(ls,ensure_ascii=False))
    ok2,vsz2=eos_ok(dest)
    assert ok2, f"{dest}: eos still unresolved after patch"
    print(f"  patched +{len(add)} special tokens (vocab_len={vlen}, now {vsz2})")
t=Tokenizer.from_file(dest)
print(f"  VERIFY eos_id={t.token_to_id(mt.eos_token)} "
      f"pad_id={t.token_to_id(mt.pad_token) if mt.pad_token else None}")
PY
  done
done
echo "[prep3] DONE -> $REPO/_las_tok3/{$(IFS=,;echo "${MODELS[*]}")}/{tur,fin,hun}/langspec_*_Latn.json"
