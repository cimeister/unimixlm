#!/usr/bin/env bash
# =============================================================================
# run_code_eval.sh — UniMixLM Eval System v4 (bigcode-evaluation-harness)
#
# Evaluates all 12 LAS finetuned models vs. base OLMo-3-7B using
# bigcode-evaluation-harness with language-specific benchmarks.
#
# Launches ALL jobs in parallel (one per GPU, round-robin assignment).
# Resumable: completed languages are skipped (delete sentinel to re-run).
#
# Usage:
#   bash scripts/run_code_eval.sh                          # all 12 langs
#   bash scripts/run_code_eval.sh python go java            # specific languages
#   GPUS="1 2 3 4 5 6 7" bash scripts/run_code_eval.sh    # skip GPU 0
#   N_SAMPLES=1 bash scripts/run_code_eval.sh              # pass@1 only (faster)
#
# Environment overrides:
#   GPUS="0 1 2 3 4 5 6 7"   space-separated GPU IDs (default: auto-detect)
#   N_SAMPLES=10               completions per problem (default: 10)
#   BATCH_SIZE=1               bigcode-eval batch size (default: 1)
#   SKIP_PLOTTING=0            set to 1 to skip plot generation
#   BIGCODE_EVAL_MAIN=<path>   path to bigcode-evaluation-harness main.py
#   LAS_TOK_DIR=<path>         path to LAS tokenizer dir (default: auto)
# =============================================================================
set -euo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_MODEL="allenai/OLMo-3-1025-7B"
OUT_DIR="./eval_results"
LOG_DIR="./logs/code_eval"
N_SAMPLES="${N_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SKIP_PLOTTING="${SKIP_PLOTTING:-0}"
BIGCODE_EVAL_MAIN="${BIGCODE_EVAL_MAIN:-}"
LAS_TOK_DIR="${LAS_TOK_DIR:-./olmo3_las_tokenizers/las-code-tokenizers-OLMo-3-1025-7B}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ─── Exact HF model names (NEVER inferred — hardcoded for safety) ───────────
declare -A MODEL_MAP=(
  [python]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-python"
  [go]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-go"
  [typescript]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-typescript"
  [java]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-java"
  [javascript]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-javascript"
  [cpp]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-cpp"
  [csharp]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-csharp"
  [php]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-php"
  [c]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-c"
  [html]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-html"
  [css]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-css"
  [markdown]="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-markdown"
)

# ─── Language list (all 12) ──────────────────────────────────────────────────
ALL_LANGS=(
  python go typescript java javascript cpp csharp php c
  html css markdown
)

# ─── GPU detection ───────────────────────────────────────────────────────────
if [ -z "${GPUS:-}" ]; then
    if command -v nvidia-smi &>/dev/null; then
        readarray -t GPU_IDS < <(nvidia-smi --list-gpus | awk '{print NR-1}')
    else
        GPU_IDS=(0)
    fi
else
    read -ra GPU_IDS <<< "$GPUS"
fi
NUM_GPUS=${#GPU_IDS[@]}

# ─── Argument parsing (optional language filter) ────────────────────────────
if [ "$#" -gt 0 ]; then
    LANGS=("$@")
else
    LANGS=("${ALL_LANGS[@]}")
fi

# ─── Utilities ───────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─── Pre-flight checks ───────────────────────────────────────────────────────
log "=== UniMixLM Eval System v4 (bigcode-evaluation-harness) ==="
log "Base model   : $BASE_MODEL"
log "Languages    : ${LANGS[*]}"
log "GPUs         : ${GPU_IDS[*]} ($NUM_GPUS total)"
log "N samples    : $N_SAMPLES"
log "Batch size   : $BATCH_SIZE"
log "LAS tok dir  : $LAS_TOK_DIR"
[ -n "$BIGCODE_EVAL_MAIN" ] && log "Bigcode main : $BIGCODE_EVAL_MAIN"

if ! python3 -c "import bigcode_eval" 2>/dev/null; then
    log "WARNING: bigcode_eval not importable. Bigcode-eval tasks may fail."
    log "Install: pip install -e git+https://github.com/bigcode-project/bigcode-evaluation-harness@main#egg=bigcode_eval"
fi

if ! python3 -c "import peft" 2>/dev/null; then
    log "ERROR: peft not found. Run: pip install peft" >&2
    exit 1
fi

# ─── Build Python args for each language ─────────────────────────────────────
build_args() {
    local lang="$1"
    local model="${MODEL_MAP[$lang]}"
    local out_dir="${OUT_DIR}/code_${lang}"

    local args=(
        --lang "$lang"
        --base-model "$BASE_MODEL"
        --adapter-repo "$model"
        --output-dir "$out_dir"
        --n-samples "$N_SAMPLES"
        --batch-size "$BATCH_SIZE"
    )

    # LAS tokenizer
    if [ -d "$LAS_TOK_DIR" ]; then
        args+=(--las-tokenizer-dir "$LAS_TOK_DIR")
    fi

    # Bigcode-eval main.py override
    if [ -n "$BIGCODE_EVAL_MAIN" ]; then
        args+=(--bigcode-eval-path "$BIGCODE_EVAL_MAIN")
    fi

    echo "${args[@]}"
}

# ─── Launch all jobs in parallel ─────────────────────────────────────────────
log "=== Launching ${#LANGS[@]} jobs across $NUM_GPUS GPUs ==="

PIDS=()
PID_LANGS=()

for i in "${!LANGS[@]}"; do
    LANG="${LANGS[$i]}"
    GPU="${GPU_IDS[$((i % NUM_GPUS))]}"

    # Validate model exists in map
    if [ -z "${MODEL_MAP[$LANG]+x}" ]; then
        log "SKIP [$LANG] — no model defined in MODEL_MAP"
        continue
    fi

    # Check sentinel
    SENTINEL="${OUT_DIR}/code_${LANG}/.eval_complete"
    if [ -f "$SENTINEL" ]; then
        log "SKIP [$LANG] — already complete (delete $SENTINEL to re-run)"
        continue
    fi

    MODEL="${MODEL_MAP[$LANG]}"
    LOG_FILE="${LOG_DIR}/${LANG}.log"
    LANG_OUT="${OUT_DIR}/code_${LANG}"
    mkdir -p "$LANG_OUT"

    PY_ARGS=$(build_args "$LANG")

    log "[RUN] $LANG on GPU $GPU (adapter: $MODEL)"

    # Launch in background — GPU pinned via CUDA_VISIBLE_DEVICES
    (
        set +e
        CUDA_VISIBLE_DEVICES="$GPU" \
            python3 las_experiment/code_benchmark_eval.py ${PY_ARGS} \
            > "$LOG_FILE" 2>&1
        EXIT_CODE=$?

        if [ "$EXIT_CODE" -eq 0 ]; then
            touch "$SENTINEL"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK   [$LANG] GPU $GPU — done"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL [$LANG] GPU $GPU — exit $EXIT_CODE (see $LOG_FILE)"
        fi
        exit "$EXIT_CODE"
    ) &

    PIDS+=($!)
    PID_LANGS+=("$LANG")
done

# ─── Wait for all jobs ───────────────────────────────────────────────────────
log "=== Waiting for ${#PIDS[@]} jobs to complete ==="

FAILED_LANGS=()
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    lang="${PID_LANGS[$i]}"
    set +e
    wait "$pid"
    status=$?
    set -e
    if [ "$status" -ne 0 ]; then
        FAILED_LANGS+=("$lang")
        log "FAIL [$lang] exited with status $status — see ${LOG_DIR}/${lang}.log"
    else
        log "OK   [$lang] finished successfully"
    fi
done

if [ "${#FAILED_LANGS[@]}" -gt 0 ]; then
    log "=== WARNING: ${#FAILED_LANGS[@]} language(s) failed: ${FAILED_LANGS[*]} ==="
    log "    Check logs: tail -f ${LOG_DIR}/{$(IFS=,; echo "${FAILED_LANGS[*]}")}.log"
    log "    Re-run to retry only failed languages (sentinels prevent re-running passed ones)."
fi

# ─── Plotting ────────────────────────────────────────────────────────────────
if [ "$SKIP_PLOTTING" -eq 0 ]; then
    log "=== Generating plots ==="
    SUMMARY_DIR="${OUT_DIR}/code_summary"
    mkdir -p "$SUMMARY_DIR"

    python3 las_experiment/plot_code_eval_results.py \
        --results-dir "$OUT_DIR" \
        --output-dir "$SUMMARY_DIR" \
        --langs "${ALL_LANGS[@]}" \
        2>&1 | tee "${LOG_DIR}/plotting.log"

    log "Plots saved to: $SUMMARY_DIR"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
log "=== Evaluation complete ==="
log "Results : $OUT_DIR"
log "Logs    : $LOG_DIR"
log "Passed  : $((${#PIDS[@]} - ${#FAILED_LANGS[@]}))/${#PIDS[@]}"

if [ "${#FAILED_LANGS[@]}" -gt 0 ]; then
    exit 1
fi
