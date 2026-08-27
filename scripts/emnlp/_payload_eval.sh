#!/usr/bin/env bash
# ============================================================================
# _payload_eval.sh — Evaluation payload invoked by run_emnlp_sl{2,3}.sh.
#
# Dispatches:
#   - EXP_DOMAIN=code → las_experiment/code_benchmark_eval.py
#                       (HumanEval + MBPP + HumanEval+ for python; MultiPL-E
#                       for the rest, per language; bigcode-evaluation-harness)
#   - EXP_DOMAIN=nl   → run_eval.py with model_lang=$EXP_LANG
#                       (BPB, XNLI, MultiBLiMP, WikiANN — wired in
#                       las_experiment/downstream_eval.py)
#
# Required env (set by SLURM wrapper):
#   EXP_DOMAIN, EXP_MODEL, EXP_LANG, EXP_LAS_MODE, EXP_VOCAB,
#   EXP_UNFREEZE, EXP_DATA_SIZE, EXP_LR, EXP_LORA_PRESET, EXP_EPOCHS
#
# When EXP_LAS_MODE=base-only (the special "base" cell), the eval is
# performed on the un-finetuned base model — no adapter is loaded.
# ============================================================================
set -euo pipefail

: "${EXP_DOMAIN:?}"
: "${EXP_MODEL:?}"
: "${EXP_LANG:?}"
: "${EXP_LAS_MODE:?}"
: "${EXP_VOCAB:?}"
: "${EXP_UNFREEZE:?}"
: "${EXP_DATA_SIZE:?}"
: "${EXP_LR:?}"

# Shared run tag (must match _payload_ft.sh).
RUN_TAG="${EXP_MODEL}_${EXP_DOMAIN}_${EXP_LAS_MODE}_v${EXP_VOCAB}_e${EXP_UNFREEZE}_${EXP_DATA_SIZE}_lr${EXP_LR}_lora${EXP_LORA_PRESET:-r64-k128}_ep${EXP_EPOCHS:-5}"
FT_DIR="./finetuned_model_emnlp/${RUN_TAG}/${EXP_LANG}"
HF_REPO="MultilingualUnigramLM/ft-${EXP_DOMAIN}-${EXP_MODEL}-${EXP_LAS_MODE}-v${EXP_VOCAB}-e${EXP_UNFREEZE}-${EXP_DATA_SIZE}-${EXP_LANG}"

EVAL_OUT="./eval_results_emnlp/${RUN_TAG}/${EXP_LANG}"
mkdir -p "$EVAL_OUT"

SENTINEL="${EVAL_OUT}/.eval_complete"
if [ -f "$SENTINEL" ] && [ "${RESUME_FROM_CHECKPOINT:-0}" != "1" ]; then
    echo "Eval already complete: ${SENTINEL} — skipping (delete to re-run)"
    exit 0
fi

# Resolve base model HF id from EXP_MODEL.
case "$EXP_MODEL" in
    qwen2_5-0_5b)  BASE_MODEL="Qwen/Qwen2.5-0.5B"        ;;
    gemma3-1b)     BASE_MODEL="google/gemma-3-1b-pt"     ;;
    olmo3-1025-7b) BASE_MODEL="allenai/Olmo-3-1025-7B"   ;;
    *) echo "ERROR: unknown EXP_MODEL=$EXP_MODEL" >&2; exit 2 ;;
esac

# Resolve LAS tokenizer dir (mirrors _payload_ft.sh).
TOK_ROOT="./tokenizers_emnlp"
LAS_TOK_DIR="${TOK_ROOT}/las-${EXP_DOMAIN}-${EXP_MODEL}-vocab${EXP_VOCAB}"

echo "=========================================================================="
echo "EMNLP eval payload"
echo "  cell    : ${RUN_TAG}"
echo "  domain  : ${EXP_DOMAIN}  lang: ${EXP_LANG}"
echo "  base    : ${BASE_MODEL}"
echo "  ft dir  : ${FT_DIR}    (HF: ${HF_REPO})"
echo "  out dir : ${EVAL_OUT}"
echo "=========================================================================="

if [ "$EXP_DOMAIN" = "code" ]; then
    # ── Code eval via bigcode-evaluation-harness ────────────────────────────
    export HF_ALLOW_CODE_EVAL=1
    export TOKENIZERS_PARALLELISM=false
    export HF_DATASETS_TRUST_REMOTE_CODE=1

    ARGS=(
        --lang "$EXP_LANG"
        --base-model "$BASE_MODEL"
        --output-dir "$EVAL_OUT"
        --n-samples "${N_SAMPLES:-10}"
        --batch-size "${BATCH_SIZE:-1}"
    )
    # For the base-only sanity cell, EXP_LAS_MODE=base — no adapter / no LAS.
    if [ "$EXP_LAS_MODE" = "base" ]; then
        echo "(base-model eval — no adapter, no LAS tokenizer)"
    else
        # Prefer HF repo if it exists (lighter than syncing the local dir).
        ARGS+=(--adapter-repo "$HF_REPO")
        if [ "$EXP_LAS_MODE" = "langspec" ] && [ -d "$LAS_TOK_DIR" ]; then
            ARGS+=(--las-tokenizer-dir "$LAS_TOK_DIR")
        fi
    fi

    echo "+ python3 las_experiment/code_benchmark_eval.py ${ARGS[*]}"
    python3 las_experiment/code_benchmark_eval.py "${ARGS[@]}"

elif [ "$EXP_DOMAIN" = "nl" ]; then
    # ── NL eval via run_eval.py ─────────────────────────────────────────────
    HYDRA_ARGS=(
        "model_lang=${EXP_LANG}"
        "model_args.pretrained=${BASE_MODEL}"
        "output_path=${EVAL_OUT}"
    )
    if [ "$EXP_LAS_MODE" = "base" ]; then
        HYDRA_ARGS+=("eval_ft.enabled=false" "custom_tokenizer.enabled=false")
    else
        # Point eval at the local FT dir (or HF repo as fallback).
        if [ -d "$FT_DIR" ]; then
            HYDRA_ARGS+=("model_args.peft=${FT_DIR}")
        else
            HYDRA_ARGS+=("model_args.peft=${HF_REPO}")
        fi
        if [ "$EXP_LAS_MODE" = "langspec" ]; then
            HYDRA_ARGS+=(
                "custom_tokenizer.enabled=true"
                "custom_tokenizer.las_lang=${EXP_LANG}"
                "custom_tokenizer.las_mode=langspec"
                "custom_tokenizer.las_tokenizer_main_path=${LAS_TOK_DIR}"
            )
        fi
    fi

    echo "+ python3 run_eval.py ${HYDRA_ARGS[*]}"
    python3 run_eval.py "${HYDRA_ARGS[@]}"

else
    echo "ERROR: unknown EXP_DOMAIN=$EXP_DOMAIN" >&2
    exit 2
fi

# Mark complete (idempotent on retries).
touch "$SENTINEL"
echo "Eval payload complete: ${RUN_TAG} / ${EXP_LANG}"
