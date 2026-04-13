#!/usr/bin/env bash
# =============================================================================
# run_code_eval.sh — UniMixLM Code Benchmark Evaluation Pipeline
#
# Evaluates all 11 language-specific LAS finetuned models against base OLMo-3-7B.
# Resumable: already-completed languages are skipped automatically.
#
# Usage:
#   bash scripts/run_code_eval.sh                  # all languages
#   bash scripts/run_code_eval.sh go java          # specific languages
#   SKIP_PLOTTING=1 bash scripts/run_code_eval.sh  # skip final plotting step
#
# Prerequisites:
#   - HuggingFace authentication (run: huggingface-cli login)
#   - Python environment with requirements.txt installed
#   - CUDA GPU recommended (A100 80GB for bfloat16)
# =============================================================================
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_MODEL="allenai/OLMo-3-1025-7B"
HF_ADAPTER_PREFIX="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M"
RESULTS_BASE="./eval_results"
LOGS_DIR="./logs/code_eval"
DTYPE="bfloat16"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
SKIP_PLOTTING="${SKIP_PLOTTING:-0}"

# HumanEval requires executing generated code — must be explicitly enabled
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

# Languages with functional code benchmarks (lm-eval-harness)
DEFAULT_CODE_LANGS=(go typescript java javascript cpp csharp php c)

# Languages evaluated via perplexity (BpB) only — no standard benchmark exists
DEFAULT_PERPLEXITY_LANGS=(html css markdown)

# ─── Parse arguments ─────────────────────────────────────────────────────────
if [ $# -gt 0 ]; then
    # If arguments given, treat as specific languages to evaluate
    REQUESTED_LANGS=("$@")
    CODE_LANGS=()
    PERPLEXITY_LANGS=()
    for lang in "${REQUESTED_LANGS[@]}"; do
        case "$lang" in
            html|css|markdown) PERPLEXITY_LANGS+=("$lang") ;;
            *)                 CODE_LANGS+=("$lang") ;;
        esac
    done
else
    CODE_LANGS=("${DEFAULT_CODE_LANGS[@]}")
    PERPLEXITY_LANGS=("${DEFAULT_PERPLEXITY_LANGS[@]}")
fi

ALL_LANGS=("${CODE_LANGS[@]}" "${PERPLEXITY_LANGS[@]}")

# ─── Utility functions ────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
err() { log "ERROR: $*" >&2; exit 1; }

# ─── Pre-flight checks ───────────────────────────────────────────────────────
log "=== UniMixLM Code Benchmark Evaluation ==="
log "Base model:   $BASE_MODEL"
log "Code langs:   ${CODE_LANGS[*]:-none}"
log "Perplexity:   ${PERPLEXITY_LANGS[*]:-none}"
log "Batch size:   $BATCH_SIZE"
log "Results dir:  $RESULTS_BASE"

# Check HuggingFace authentication
if ! huggingface-cli whoami >/dev/null 2>&1; then
    err "Not logged in to HuggingFace. Run: huggingface-cli login"
fi
log "HuggingFace: authenticated as $(huggingface-cli whoami 2>/dev/null | head -1)"

# Check Python environment
if ! python3 -c "import lm_eval" 2>/dev/null; then
    err "lm_eval not found. Install via: pip install -r requirements.txt"
fi
if ! python3 -c "import peft" 2>/dev/null; then
    err "peft not found. Install via: pip install peft"
fi

mkdir -p "$RESULTS_BASE" "$LOGS_DIR"

# ─── Step 1: Download LAS tokenizers (skip if already present) ───────────────
TOK_DIR="./olmo3_las_tokenizers/las-code-tokenizers-OLMo-3-1025-7B"
if [ -d "$TOK_DIR" ] && [ "$(ls -A "$TOK_DIR" 2>/dev/null | wc -l)" -gt 5 ]; then
    log "Tokenizers already present at $TOK_DIR — skipping download."
else
    log "=== Downloading LAS code tokenizers ==="
    bash scripts/download_code_tokenizers.sh
fi

# ─── Step 2: Run lm-eval functional benchmarks for CODE_LANGS ────────────────
if [ "${#CODE_LANGS[@]}" -gt 0 ]; then
    log "=== Running functional code benchmarks (lm-eval-harness) ==="
    for lang in "${CODE_LANGS[@]}"; do
        OUT_DIR="${RESULTS_BASE}/code_${lang}"
        SENTINEL="${OUT_DIR}/.eval_complete"
        LOG_FILE="${LOGS_DIR}/code_eval_${lang}.log"

        if [ -f "$SENTINEL" ]; then
            log "[$lang] Already complete — skipping (delete ${SENTINEL} to re-run)."
            continue
        fi

        mkdir -p "$OUT_DIR"
        log "[$lang] Starting lm-eval evaluation..."
        log "[$lang] Adapter: ${HF_ADAPTER_PREFIX}-${lang}"
        log "[$lang] Output:  $OUT_DIR"
        log "[$lang] Log:     $LOG_FILE"

        python3 las_experiment/code_benchmark_eval.py \
            --lang "$lang" \
            --base-model "$BASE_MODEL" \
            --adapter-repo "${HF_ADAPTER_PREFIX}-${lang}" \
            --output-dir "$OUT_DIR" \
            --dtype "$DTYPE" \
            --batch-size "$BATCH_SIZE" \
            --eval-mode lmeval \
            2>&1 | tee "$LOG_FILE"

        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            log "[$lang] ERROR: Evaluation failed. Check $LOG_FILE"
        else
            touch "$SENTINEL"
            log "[$lang] DONE."
        fi
    done
fi

# ─── Step 3: Run perplexity (BpB) evaluations for PERPLEXITY_LANGS ───────────
if [ "${#PERPLEXITY_LANGS[@]}" -gt 0 ]; then
    log "=== Running perplexity (BpB) evaluations ==="
    for lang in "${PERPLEXITY_LANGS[@]}"; do
        OUT_DIR="${RESULTS_BASE}/code_${lang}"
        SENTINEL="${OUT_DIR}/.perplexity_complete"
        LOG_FILE="${LOGS_DIR}/code_perplexity_${lang}.log"

        if [ -f "$SENTINEL" ]; then
            log "[$lang] Already complete — skipping (delete ${SENTINEL} to re-run)."
            continue
        fi

        mkdir -p "$OUT_DIR"
        log "[$lang] Starting perplexity (BpB) evaluation..."

        python3 las_experiment/code_benchmark_eval.py \
            --lang "$lang" \
            --base-model "$BASE_MODEL" \
            --adapter-repo "${HF_ADAPTER_PREFIX}-${lang}" \
            --output-dir "$OUT_DIR" \
            --dtype "$DTYPE" \
            --eval-mode perplexity \
            2>&1 | tee "$LOG_FILE"

        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            log "[$lang] ERROR: Evaluation failed. Check $LOG_FILE"
        else
            touch "$SENTINEL"
            log "[$lang] DONE."
        fi
    done
fi

# ─── Step 4: Aggregate results and generate plots ────────────────────────────
if [ "$SKIP_PLOTTING" -eq 1 ]; then
    log "Skipping plotting (SKIP_PLOTTING=1)."
else
    log "=== Generating plots and summary ==="
    SUMMARY_DIR="${RESULTS_BASE}/code_summary"
    mkdir -p "$SUMMARY_DIR"

    python3 las_experiment/plot_code_eval_results.py \
        --results-dir "$RESULTS_BASE" \
        --output-dir "$SUMMARY_DIR" \
        --langs "${ALL_LANGS[@]}" \
        2>&1 | tee "${LOGS_DIR}/plotting.log"

    log "Plots saved to: $SUMMARY_DIR"
    log "  per_lang/bar_{lang}.png   — per-language grouped bar charts"
    log "  heatmap_ft_scores.png     — cross-language score heatmap"
    log "  heatmap_delta.png         — improvement delta heatmap"
    log "  summary_delta.png         — mean improvement per language"
    log "  results_summary.csv       — full numeric results table"
fi

# ─── Done ────────────────────────────────────────────────────────────────────
log "=== All evaluations complete ==="
log "Results:  $RESULTS_BASE"
log "Logs:     $LOGS_DIR"
