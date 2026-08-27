#!/usr/bin/env bash
# ============================================================================
# _payload_tok.sh — Tokenizer-training payload invoked by run_emnlp_sl{2,3}.sh.
#
# Trains a LAS tokenizer (shared base + per-language re-estimation) for one
# (EXP_MODEL, EXP_DOMAIN, EXP_VOCAB) cell. The set of per-language tokenizers
# trained is fixed by domain:
#   code    → python javascript java cpp go typescript
#   nl      → tur_Latn fin_Latn hun_Latn
#
# Required env (set by SLURM wrapper):
#   EXP_MODEL  : "qwen2_5-0_5b" | "gemma3-1b" | "olmo3-1025-7b"
#   EXP_DOMAIN : "code" | "nl"
#   EXP_VOCAB  : "100k" | "150k" | "256k"
# ============================================================================
set -euo pipefail

: "${EXP_MODEL:?}"
: "${EXP_DOMAIN:?}"
: "${EXP_VOCAB:?}"

case "$EXP_VOCAB" in
    100k) VOCAB_INT=100000 ;;
    150k) VOCAB_INT=150000 ;;
    256k) VOCAB_INT=256000 ;;
    *) echo "ERROR: EXP_VOCAB must be 100k|150k|256k, got $EXP_VOCAB" >&2; exit 2 ;;
esac

case "$EXP_DOMAIN" in
    code)
        ENTRY="run_las_code_tok.py"
        LANGS_LIST="[python,javascript,java,c++,go,typescript]"
        ;;
    nl)
        ENTRY="run_las_tok_training.py"
        LANGS_LIST="[tur_Latn,fin_Latn,hun_Latn]"
        ;;
    *) echo "ERROR: unknown EXP_DOMAIN=$EXP_DOMAIN" >&2; exit 2 ;;
esac

OUT_DIR="./tokenizers_emnlp/las-${EXP_DOMAIN}-${EXP_MODEL}-vocab${EXP_VOCAB}"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo "EMNLP tokenizer-training payload"
echo "  model       : ${EXP_MODEL}"
echo "  domain      : ${EXP_DOMAIN}"
echo "  vocab       : ${EXP_VOCAB}  (=${VOCAB_INT})"
echo "  langs       : ${LANGS_LIST}"
echo "  output dir  : ${OUT_DIR}"
echo "=========================================================================="

# Hydra overrides:
#   model.model_name → from the per-model config (registered under
#       config/finetune/model/${EXP_MODEL}.yaml; tokenizer_training also
#       reads the same model field).
#   languages.langs  → fixed per domain.
#   data.output_dir  → namespaced per cell.
#   training.vocab_size → vocab axis (REQUIRES the §1.5 vocab plumbing).
HYDRA_ARGS=(
    "model=${EXP_MODEL}"
    "languages.langs=${LANGS_LIST}"
    "data.output_dir=${OUT_DIR}"
    "training.vocab_size=${VOCAB_INT}"
    "hf.push=true"
)

if [ "${RESUME_FROM_CHECKPOINT:-0}" = "1" ]; then
    HYDRA_ARGS+=("+resume=true")
fi

echo "+ python3 ${ENTRY} ${HYDRA_ARGS[*]}"
python3 "${ENTRY}" "${HYDRA_ARGS[@]}"

echo "Tokenizer-training payload complete: ${OUT_DIR}"
