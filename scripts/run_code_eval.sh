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

format_duration() {
    local secs=$1
    if [ "$secs" -ge 3600 ]; then
        printf "%dh %dm %ds" $((secs/3600)) $((secs%3600/60)) $((secs%60))
    elif [ "$secs" -ge 60 ]; then
        printf "%dm %ds" $((secs/60)) $((secs%60))
    else
        printf "%ds" "$secs"
    fi
}

EVAL_START_EPOCH=$(date +%s)

# ─── Install language runtimes needed by bigcode-eval (--allow_code_execution) ─
install_runtime_deps() {
    local missing=()

    command -v go       &>/dev/null || missing+=(go)
    command -v javac    &>/dev/null || missing+=(java)
    command -v node     &>/dev/null || missing+=(node)
    command -v tsc      &>/dev/null || missing+=(typescript)
    command -v g++      &>/dev/null || missing+=(cpp)
    command -v dotnet   &>/dev/null || missing+=(csharp)
    command -v php      &>/dev/null || missing+=(php)

    [ ${#missing[@]} -eq 0 ] && return 0

    log "Installing missing language runtimes: ${missing[*]}"

    for dep in "${missing[@]}"; do
        case "$dep" in
            go)
                log "  -> Installing Go..."
                wget -q https://go.dev/dl/go1.22.2.linux-amd64.tar.gz -O /tmp/go.tar.gz
                sudo tar -C /usr/local -xzf /tmp/go.tar.gz
                rm /tmp/go.tar.gz
                export PATH=$PATH:/usr/local/go/bin
                ;;
            java)
                log "  -> Installing Java (default-jdk)..."
                sudo apt-get update -qq && sudo apt-get install -y -qq default-jdk
                ;;
            node)
                log "  -> Installing Node.js..."
                curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
                sudo apt-get install -y -qq nodejs
                ;;
            typescript)
                log "  -> Installing TypeScript..."
                command -v node &>/dev/null || {
                    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
                    sudo apt-get install -y -qq nodejs
                }
                sudo npm install -g typescript
                ;;
            cpp)
                log "  -> Installing g++..."
                sudo apt-get update -qq && sudo apt-get install -y -qq g++
                ;;
            csharp)
                log "  -> Installing .NET SDK..."
                sudo apt-get update -qq && sudo apt-get install -y -qq dotnet-sdk-8.0
                ;;
            php)
                log "  -> Installing PHP..."
                sudo apt-get update -qq && sudo apt-get install -y -qq php-cli
                ;;
        esac
    done

    log "Runtime dependency installation complete."

    # Verify all critical runtimes are now available
    local failed=()
    command -v go       &>/dev/null || failed+=(go)
    command -v javac    &>/dev/null || failed+=(javac)
    command -v node     &>/dev/null || failed+=(node)
    command -v tsc      &>/dev/null || failed+=(tsc)
    command -v g++      &>/dev/null || failed+=(g++)
    command -v dotnet   &>/dev/null || failed+=(dotnet)
    command -v php      &>/dev/null || failed+=(php)

    if [ ${#failed[@]} -gt 0 ]; then
        log "ERROR: The following runtimes are still missing after install: ${failed[*]}"
        log "       bigcode-eval --allow_code_execution will fail without these."
        log "       Please install them manually before running evaluation."
        exit 1
    fi

    log "All language runtimes verified: go javac node tsc g++ dotnet php"
}

install_runtime_deps

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
        JOB_START=$(date +%s)
        CUDA_VISIBLE_DEVICES="$GPU" \
            python3 las_experiment/code_benchmark_eval.py ${PY_ARGS} \
            > "$LOG_FILE" 2>&1
        EXIT_CODE=$?
        JOB_END=$(date +%s)
        JOB_ELAPSED=$((JOB_END - JOB_START))
        # Write timing for parent to read
        echo "$JOB_ELAPSED" > "${LANG_OUT}/.eval_timing"

        if [ "$EXIT_CODE" -eq 0 ]; then
            touch "$SENTINEL"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK   [$LANG] GPU $GPU — done in $(format_duration $JOB_ELAPSED)"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL [$LANG] GPU $GPU — exit $EXIT_CODE after $(format_duration $JOB_ELAPSED) (see $LOG_FILE)"
        fi
        exit "$EXIT_CODE"
    ) &

    PIDS+=($!)
    PID_LANGS+=("$LANG")
done

# ─── Background progress monitor (reports every 2 minutes) ─────────────────
TOTAL_JOBS=${#PIDS[@]}

if [ "$TOTAL_JOBS" -gt 0 ]; then
(
    while true; do
        sleep 120
        NOW=$(date +%s)
        ELAPSED=$((NOW - EVAL_START_EPOCH))
        DONE=0
        STILL_RUNNING=()
        for idx in "${!PID_LANGS[@]}"; do
            plang="${PID_LANGS[$idx]}"
            if [ -f "${OUT_DIR}/code_${plang}/.eval_complete" ]; then
                DONE=$((DONE + 1))
            else
                STILL_RUNNING+=("$plang")
            fi
        done
        PCT=$((DONE * 100 / TOTAL_JOBS))
        log "[MONITOR] Progress: ${DONE}/${TOTAL_JOBS} (${PCT}%) | Elapsed: $(format_duration $ELAPSED)"
        if [ "${#STILL_RUNNING[@]}" -gt 0 ]; then
            log "[MONITOR] Still running: ${STILL_RUNNING[*]}"
        fi
        if [ "$DONE" -gt 0 ] && [ "$DONE" -lt "$TOTAL_JOBS" ]; then
            REMAINING=$((TOTAL_JOBS - DONE))
            # Estimate based on parallel execution: assume max concurrency = NUM_GPUS
            WAVES_DONE=$(( (DONE + NUM_GPUS - 1) / NUM_GPUS ))
            if [ "$WAVES_DONE" -gt 0 ]; then
                AVG_WAVE=$((ELAPSED / WAVES_DONE))
                WAVES_LEFT=$(( (REMAINING + NUM_GPUS - 1) / NUM_GPUS ))
                ETA=$((AVG_WAVE * WAVES_LEFT))
                log "[MONITOR] Estimated time remaining: ~$(format_duration $ETA)"
            fi
        fi
        [ "$DONE" -ge "$TOTAL_JOBS" ] && break
    done
) &
MONITOR_PID=$!
fi

# ─── Wait for all jobs ───────────────────────────────────────────────────────
log "=== Waiting for ${TOTAL_JOBS} jobs to complete ==="

FAILED_LANGS=()
COMPLETED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    lang="${PID_LANGS[$i]}"
    set +e
    wait "$pid"
    status=$?
    set -e
    COMPLETED=$((COMPLETED + 1))
    NOW=$(date +%s)
    ELAPSED=$((NOW - EVAL_START_EPOCH))
    REMAINING_JOBS=$((TOTAL_JOBS - COMPLETED))

    # Read per-job timing if available
    TIMING_FILE="${OUT_DIR}/code_${lang}/.eval_timing"
    JOB_TIME_STR=""
    if [ -f "$TIMING_FILE" ]; then
        JOB_TIME=$(cat "$TIMING_FILE")
        JOB_TIME_STR=" (job time: $(format_duration $JOB_TIME))"
    fi

    if [ "$status" -ne 0 ]; then
        FAILED_LANGS+=("$lang")
        log "FAIL [$lang] exit $status${JOB_TIME_STR} | ${COMPLETED}/${TOTAL_JOBS} done | Elapsed: $(format_duration $ELAPSED) | See ${LOG_DIR}/${lang}.log"
    else
        log "OK   [$lang]${JOB_TIME_STR} | ${COMPLETED}/${TOTAL_JOBS} done | Elapsed: $(format_duration $ELAPSED)"
    fi

    # ETA for remaining jobs
    if [ "$REMAINING_JOBS" -gt 0 ] && [ "$COMPLETED" -gt 0 ]; then
        WAVES_DONE=$(( (COMPLETED + NUM_GPUS - 1) / NUM_GPUS ))
        if [ "$WAVES_DONE" -gt 0 ]; then
            AVG_WAVE=$((ELAPSED / WAVES_DONE))
            WAVES_LEFT=$(( (REMAINING_JOBS + NUM_GPUS - 1) / NUM_GPUS ))
            ETA=$((AVG_WAVE * WAVES_LEFT))
            log "     ${REMAINING_JOBS} jobs remaining | Est. time left: ~$(format_duration $ETA)"
        fi
    fi
done

# Kill the background monitor
if [ -n "${MONITOR_PID:-}" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
fi

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

# ─── Detailed timing summary ─────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - EVAL_START_EPOCH ))
log ""
log "======================================================================"
log "  EVALUATION SUMMARY"
log "======================================================================"
log "Results : $OUT_DIR"
log "Logs    : $LOG_DIR"
log "Passed  : $((${#PIDS[@]} - ${#FAILED_LANGS[@]}))/${#PIDS[@]}"
log "Total wall-clock time: $(format_duration $TOTAL_ELAPSED)"
log ""
log "  Per-Language Timing:"
log "  --------------------"
for lang_name in "${LANGS[@]}"; do
    TIMING_FILE="${OUT_DIR}/code_${lang_name}/.eval_timing"
    SENTINEL="${OUT_DIR}/code_${lang_name}/.eval_complete"
    if [ -f "$TIMING_FILE" ]; then
        JOB_TIME=$(cat "$TIMING_FILE")
        STATUS="OK"
        [ ! -f "$SENTINEL" ] && STATUS="FAIL"
        printf "  [%s] %-12s %s\n" "$STATUS" "$lang_name" "$(format_duration $JOB_TIME)"
    else
        printf "  [SKIP] %-12s (no timing data)\n" "$lang_name"
    fi
done
log ""
log "  GPU utilisation: ${NUM_GPUS} GPUs, ${#PIDS[@]} jobs"
if [ "${#PIDS[@]}" -gt 0 ]; then
    GPU_HOURS=$(( TOTAL_ELAPSED * NUM_GPUS / 3600 ))
    log "  Approx GPU-hours: ~${GPU_HOURS}"
fi
log "======================================================================"

if [ "${#FAILED_LANGS[@]}" -gt 0 ]; then
    exit 1
fi
