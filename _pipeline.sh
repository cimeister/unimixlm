#!/usr/bin/env bash
# _pipeline.sh — full LangMAP pipeline, runs inside tmux session "langmap".
#   1. setup.sh (idempotent; venv already built -> fast)
#   2. prep_model_tokenizers.sh (original-arm tokenizer snapshots)
#   3. PREFLIGHT: env imports + 100M dataset reachability  (GATE)
#   4. 4-GPU orchestrator (qwen/tur first; auto-push to HF; 100M tokens)
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
exec > >(tee -a "$SHM/logs/pipeline.log") 2>&1

echo "==== [$(date -u)] STAGE 1/4: setup.sh --create-venv ===="
bash setup.sh --create-venv || echo "WARN: setup.sh returned nonzero (venv may already be complete) — preflight will decide"
source .venv/bin/activate

echo "==== [$(date -u)] STAGE 2/4: prep_model_tokenizers.sh ===="
bash scripts/prep_model_tokenizers.sh || echo "WARN: some tokenizer snapshots failed (gated model?) — those cells will fail-soft"

echo "==== [$(date -u)] STAGE 3/4: PREFLIGHT (gate) ===="
if ! python3 - <<'PY'
import os, sys
import torch, transformers, peft, datasets   # noqa
from datasets import load_dataset
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "GPUs", torch.cuda.device_count())
# 100M data path the patch points at:
t=next(iter(load_dataset("MultilingualUnigramLM/FineWeb2-tur_Latn-100M",
                         split="train", streaming=True)))
assert "text" in t, t.keys()
e=next(iter(load_dataset("MultilingualUnigramLM/FineWeb2-100M-olmo3-7b-toks",
                         "eng_Latn", split="train", streaming=True)))
assert "text" in e, e.keys()
print("PREFLIGHT OK: tur+eng 100M reachable, text column present")
sys.stdout.flush()
# HF datasets/pyarrow streaming iterators abort in the CPython finalizer
# (PyGILState_Release at shutdown). Checks already passed — hard-exit 0
# before the buggy finalizer runs so the gate isn't a false negative.
os._exit(0)
PY
then
  echo "PREFLIGHT FAILED — NOT launching GPUs. Fix data/env and re-run _pipeline.sh."
  exit 1
fi

echo "==== [$(date -u)] STAGE 4/4: 4-GPU orchestrator (100M) ===="
bash _orchestrator.sh

echo "==== [$(date -u)] PIPELINE COMPLETE ===="
