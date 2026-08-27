#!/usr/bin/env bash
# ============================================================================
# _payload_ft.sh — Finetuning payload invoked by run_emnlp_sl{2,3}.sh.
#
# Reads experiment axes from EXP_* environment variables (set by the SLURM
# wrapper) and dispatches a single `python run_finetune.py` invocation with
# the right Hydra overrides.
#
# Inputs (all required, set by wrapper):
#   EXP_DOMAIN    : "code" | "nl"
#   EXP_MODEL     : "qwen2_5-0_5b" | "gemma3-1b" | "olmo3-1025-7b"
#   EXP_LANG      : target language (e.g. "python", "tur_Latn")
#   EXP_REHEARSAL : rehearsal language (e.g. "eng", "python")
#   EXP_LAS_MODE  : "original" | "langspec"
#   EXP_VOCAB     : "100k" | "150k" | "256k"   (ignored when las_mode=original)
#   EXP_UNFREEZE  : "0" | "1"
#   EXP_LR        : learning rate (e.g. "2e-4")
#   EXP_DATA_SIZE : "10M" | "50M" | "100M" | "500M"
#   EXP_LORA_PRESET : "r16-k32" | "r32-k64" | "r64-k128"
#   EXP_EPOCHS    : "1" | "5"
#
# Tokenizer path convention for vocab-swept tokenizers (produced by the
# tokenizer-training jobs):
#   ./tokenizers_emnlp/las-${DOMAIN_TAG}-${EXP_MODEL}-vocab${EXP_VOCAB}/
#     base_tokenizer.json
#     langspec_<lang>.json   (code: no _Latn suffix; NL: with _Latn suffix)
#
# The model tokenizer (BPE checkpoint) is expected at:
#   ./tokenizers_emnlp/${EXP_MODEL}/model_tokenizer/
# ============================================================================
set -euo pipefail

: "${EXP_DOMAIN:?}"
: "${EXP_MODEL:?}"
: "${EXP_LANG:?}"
: "${EXP_REHEARSAL:?}"
: "${EXP_LAS_MODE:?}"
: "${EXP_VOCAB:?}"
: "${EXP_UNFREEZE:?}"
: "${EXP_LR:?}"
: "${EXP_DATA_SIZE:?}"
: "${EXP_LORA_PRESET:?}"
: "${EXP_EPOCHS:?}"

# ── Resolve config name ─────────────────────────────────────────────────────
case "$EXP_DOMAIN" in
    code) CFG_NAME="code_default" ;;
    nl)   CFG_NAME="default"      ;;
    *) echo "ERROR: unknown EXP_DOMAIN=$EXP_DOMAIN" >&2; exit 2 ;;
esac

# ── Resolve unfreeze flag ───────────────────────────────────────────────────
case "$EXP_UNFREEZE" in
    1) UNFREEZE_BOOL="true"  ;;
    0) UNFREEZE_BOOL="false" ;;
    *) echo "ERROR: EXP_UNFREEZE must be 0 or 1, got $EXP_UNFREEZE" >&2; exit 2 ;;
esac

# ── Resolve tokenizer paths ─────────────────────────────────────────────────
DOMAIN_TAG="$EXP_DOMAIN"   # "code" | "nl"
TOK_ROOT="./tokenizers_emnlp"
LAS_TOK_DIR="${TOK_ROOT}/las-${DOMAIN_TAG}-${EXP_MODEL}-vocab${EXP_VOCAB}"
MODEL_TOK_DIR="${TOK_ROOT}/${EXP_MODEL}/model_tokenizer"

if [ "$EXP_LAS_MODE" = "langspec" ]; then
    if [ ! -d "$LAS_TOK_DIR" ]; then
        echo "ERROR: LAS tokenizer dir missing: ${LAS_TOK_DIR}" >&2
        echo "       Run the tokenizer-training job for this (model, vocab) first." >&2
        exit 3
    fi
    # Sanity: per-language tokenizer JSON must exist.
    LANGSPEC_JSON="${LAS_TOK_DIR}/langspec_${EXP_LANG}.json"
    if [ ! -f "$LANGSPEC_JSON" ]; then
        echo "ERROR: per-language tokenizer missing: ${LANGSPEC_JSON}" >&2
        exit 3
    fi
fi

# ── Resolve output dir (one per experimental cell) ──────────────────────────
RUN_TAG="${EXP_MODEL}_${EXP_DOMAIN}_${EXP_LAS_MODE}_v${EXP_VOCAB}_e${EXP_UNFREEZE}_${EXP_DATA_SIZE}_lr${EXP_LR}_lora${EXP_LORA_PRESET}_ep${EXP_EPOCHS}"
OUT_DIR="./finetuned_model_emnlp/${RUN_TAG}/${EXP_LANG}"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo "EMNLP finetuning payload"
echo "  cell        : ${RUN_TAG}"
echo "  target lang : ${EXP_LANG}  (rehearsal: ${EXP_REHEARSAL})"
echo "  output dir  : ${OUT_DIR}"
[ "$EXP_LAS_MODE" = "langspec" ] && echo "  LAS dir     : ${LAS_TOK_DIR}"
echo "=========================================================================="

# ── Compose Hydra overrides ─────────────────────────────────────────────────
# Common args.
HYDRA_ARGS=(
    --config-name="${CFG_NAME}"
    "model=${EXP_MODEL}"
    "lora=${EXP_LORA_PRESET}"
    "+langs={${EXP_LANG}: 0.8, ${EXP_REHEARSAL}: 0.2}"
    "las_mode=${EXP_LAS_MODE}"
    "model_tokenizer_path=${MODEL_TOK_DIR}"
    "dataset.size=${EXP_DATA_SIZE}"
    "learning_rate=${EXP_LR}"
    "epochs=${EXP_EPOCHS}"
    "unfreeze_embeddings=${UNFREEZE_BOOL}"
    "output_dir=${OUT_DIR}"
)

if [ "$EXP_LAS_MODE" = "langspec" ]; then
    HYDRA_ARGS+=("las_tokenizer_path=${LAS_TOK_DIR}")
fi

# Resume hint passed by the supervisor on retries.
if [ "${RESUME_FROM_CHECKPOINT:-0}" = "1" ]; then
    HYDRA_ARGS+=("+resume=true")
    echo "RESUME_FROM_CHECKPOINT=1 — passing +resume=true to run_finetune.py"
fi

echo "+ python3 run_finetune.py ${HYDRA_ARGS[*]}"
python3 run_finetune.py "${HYDRA_ARGS[@]}"

# ── HF push (idempotent) ────────────────────────────────────────────────────
HF_REPO="MultilingualUnigramLM/ft-${EXP_DOMAIN}-${EXP_MODEL}-${EXP_LAS_MODE}-v${EXP_VOCAB}-e${EXP_UNFREEZE}-${EXP_DATA_SIZE}-${EXP_LANG}"
echo "Pushing to HF: ${HF_REPO}"
hf repo create "${HF_REPO}" --private 2>/dev/null || true
hf upload "${HF_REPO}" "${OUT_DIR}" --repo-type model

echo "Finetuning payload complete: ${RUN_TAG} / ${EXP_LANG}"
