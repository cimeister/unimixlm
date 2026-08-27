#!/usr/bin/env bash
# ============================================================================
# build_emnlp_datasets.sh — produce and push the 50M / 500M dataset variants
# required by the EMNLP main paper experiments.
#
# Run on a CPU-heavy login node (or a single-GPU node); each call streams from
# the upstream HF source (FineWeb2 / The-Stack), tokenises with the OLMo-3
# tokenizer to count tokens, sanitises, and pushes a per-size repo to
# MultilingualUnigramLM/.
#
# Existing variants on HF (per the plan §1.6 inventory):
#   FineWeb2-{5M,10M,100M,100M-olmo3-7b-toks}
#   LangMap-TheStack-{lang}-100M (12 languages)
#
# Missing variants this script produces:
#   FineWeb2-{50M,500M}
#   LangMap-TheStack-{lang}-{50M,500M} for the 6 EMNLP code languages.
#
# Usage:
#   bash scripts/emnlp/build_emnlp_datasets.sh           # all of the above
#   bash scripts/emnlp/build_emnlp_datasets.sh --fineweb # NL only
#   bash scripts/emnlp/build_emnlp_datasets.sh --code    # code only
#   bash scripts/emnlp/build_emnlp_datasets.sh --verify  # smoke (1 lang each)
# ============================================================================
set -euo pipefail

DO_FINEWEB=1
DO_CODE=1
VERIFY_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --fineweb) DO_CODE=0 ;;
        --code)    DO_FINEWEB=0 ;;
        --verify)  VERIFY_ONLY=1 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

HF_USER="${HF_USER:-MultilingualUnigramLM}"
CODE_LANGS=(python javascript java cpp go typescript)

run() {
    echo "+ $*"
    "$@"
}

if [ "$DO_FINEWEB" = "1" ]; then
    if [ "$VERIFY_ONLY" = "1" ]; then
        run python fineweb2_finetune_dataset.py \
            --hf-repo "${HF_USER}/FineWeb2-50M" \
            --target-tokens 50000000 \
            --languages tur_Latn
    else
        run python fineweb2_finetune_dataset.py \
            --hf-repo "${HF_USER}/FineWeb2-50M" \
            --target-tokens 50000000

        run python fineweb2_finetune_dataset.py \
            --hf-repo "${HF_USER}/FineWeb2-500M" \
            --target-tokens 500000000
    fi
fi

if [ "$DO_CODE" = "1" ]; then
    if [ "$VERIFY_ONLY" = "1" ]; then
        run python thestack_finetune_dataset.py \
            --hf-user "$HF_USER" \
            --target-tokens 500000000 \
            --languages python
    else
        run python thestack_finetune_dataset.py \
            --hf-user "$HF_USER" \
            --target-tokens 50000000 \
            --languages "${CODE_LANGS[@]}"

        run python thestack_finetune_dataset.py \
            --hf-user "$HF_USER" \
            --target-tokens 500000000 \
            --languages "${CODE_LANGS[@]}"
    fi
fi

echo "All requested EMNLP dataset variants built and pushed."
