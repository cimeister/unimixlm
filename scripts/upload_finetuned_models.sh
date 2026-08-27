#!/usr/bin/env bash
# =============================================================================
# Upload finetuned models AND tokenizers to HuggingFace Hub.
#
# Scans ./finetuned_model/ for completed finetuning runs and uploads any
# that haven't been pushed yet. Also uploads LAS tokenizers from
# ./olmo3_las_tokenizers/ if present.
#
# Usage:
#   bash scripts/upload_finetuned_models.sh              # upload all found models + tokenizers
#   bash scripts/upload_finetuned_models.sh --dry-run    # just list what would be uploaded
#   bash scripts/upload_finetuned_models.sh python        # upload only items matching "python"
#   bash scripts/upload_finetuned_models.sh --models-only # skip tokenizer upload
#   bash scripts/upload_finetuned_models.sh --tokenizers-only # skip model upload
# =============================================================================

set -euo pipefail

MODEL_DIR="./finetuned_model"
TOK_DIR="./olmo3_las_tokenizers"
DRY_RUN=false
MODELS_ONLY=false
TOKENIZERS_ONLY=false
FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --models-only) MODELS_ONLY=true ;;
        --tokenizers-only) TOKENIZERS_ONLY=true ;;
        *) FILTER="$arg" ;;
    esac
done

count=0

# ─── Tokenizers ──────────────────────────────────────────────────────────────

if [ "$MODELS_ONLY" = false ] && [ -d "$TOK_DIR" ]; then
    echo "========================================"
    echo "Uploading tokenizers"
    echo "========================================"
    echo ""

    for dir in "$TOK_DIR"/las-*/; do
        [ -d "$dir" ] || continue

        name="$(basename "$dir")"

        # Apply filter if provided
        if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
            continue
        fi

        # Check it has tokenizer JSON files
        if ! ls "$dir"/langspec_*.json 2>/dev/null | head -1 > /dev/null 2>&1; then
            echo "  SKIP (no langspec files): $name"
            continue
        fi

        n_langspec=$(ls "$dir"/langspec_*.json 2>/dev/null | wc -l)
        size=$(du -sh "$dir" | cut -f1)

        # Derive HF repo name from directory name
        # las-code-tokenizers-OLMo-3-1025-7B  -> las-code-tokenizers-OLMo-3-1025-7B
        # las-tokenizers-Olmo-3-1025-7B-yor   -> las-tokenizers-Olmo-3-1025-7B-yor
        hf_repo="MultilingualUnigramLM/${name}"

        echo "  ${name}  (${size}, ${n_langspec} langspec tokenizers)"
        echo "    -> ${hf_repo}"

        if [ "$DRY_RUN" = true ]; then
            echo "    [dry-run] would upload"
        else
            hf repo create "$hf_repo" 2>/dev/null || true
            hf upload "$hf_repo" "$dir" --repo-type model
            echo "    UPLOADED"
        fi

        count=$((count + 1))
        echo ""
    done
fi

# ─── Finetuned models ────────────────────────────────────────────────────────

if [ "$TOKENIZERS_ONLY" = false ]; then

if [ ! -d "$MODEL_DIR" ]; then
    echo "WARNING: No finetuned_model/ directory found. Skipping model upload."
else

echo "========================================"
echo "Uploading finetuned models"
echo "========================================"
echo ""

for dir in "$MODEL_DIR"/*/; do
    [ -d "$dir" ] || continue

    name="$(basename "$dir")"

    # Apply filter if provided
    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
        continue
    fi

    # Check it has actual model files (not an empty dir)
    if ! ls "$dir"/*.safetensors "$dir"/*.bin "$dir"/adapter_config.json 2>/dev/null | head -1 > /dev/null 2>&1; then
        echo "  SKIP (no model files): $name"
        continue
    fi

    # Derive HF repo name from directory name
    # ft-code-las-langspec_python-javascript_100M  -> ft-code-las-olmo3-7b-thestack100M-python
    # ft-code_python-javascript_100M               -> ft-code-olmo3-7b-thestack100M-python
    # ft-las-langspec_yor-eng_100M                 -> ft-las-olmo3-7b-fineweb100M-yor
    # ft_yor-eng_100M                              -> ft-olmo3-7b-fineweb100M-yor

    # Extract the primary language (first lang before the dash in the langs part)
    lang=$(echo "$name" | sed -E 's/^(ft-code-las-langspec|ft-code-las-base|ft-code|ft-las-langspec|ft-las-base|ft)_//' | cut -d'-' -f1)

    if [[ "$name" == ft-code-las-langspec_* ]]; then
        hf_repo="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-${lang}"
    elif [[ "$name" == ft-code_* ]]; then
        hf_repo="MultilingualUnigramLM/ft-code-olmo3-7b-thestack100M-${lang}"
    elif [[ "$name" == ft-las-langspec_* ]]; then
        hf_repo="MultilingualUnigramLM/ft-las-olmo3-7b-fineweb100M-${lang}"
    elif [[ "$name" == ft_* ]]; then
        hf_repo="MultilingualUnigramLM/ft-olmo3-7b-fineweb100M-${lang}"
    else
        echo "  SKIP (unknown naming): $name"
        continue
    fi

    size=$(du -sh "$dir" | cut -f1)
    echo "  ${name}  (${size})"
    echo "    -> ${hf_repo}"

    if [ "$DRY_RUN" = true ]; then
        echo "    [dry-run] would upload"
    else
        hf repo create "$hf_repo" 2>/dev/null || true
        hf upload "$hf_repo" "$dir" --repo-type model
        echo "    UPLOADED"
    fi

    count=$((count + 1))
    echo ""
done

fi  # [ -d "$MODEL_DIR" ]
fi  # [ "$TOKENIZERS_ONLY" = false ]

# ─── Summary ─────────────────────────────────────────────────────────────────

if [ "$count" -eq 0 ]; then
    echo "No models or tokenizers found to upload."
    [ -n "$FILTER" ] && echo "Filter: '$FILTER'"
else
    echo "========================================"
    if [ "$DRY_RUN" = true ]; then
        echo "Dry run: $count item(s) would be uploaded."
    else
        echo "Done: $count item(s) uploaded."
    fi
    echo "========================================"
fi
