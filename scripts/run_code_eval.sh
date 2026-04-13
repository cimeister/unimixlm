#!/usr/bin/env bash
# =============================================================================
# run_code_eval.sh — UniMixLM Eval System v4 (bigcode-evaluation-harness)
#
# Evaluates all 12 LAS finetuned models vs. base OLMo-3-7B using
# bigcode-evaluation-harness with language-specific benchmarks:
#
#   Python:      HumanEval, MBPP, HumanEval+
#   Go:          MultiPL-E Go, HumanEvalPack Go
#   TypeScript:  MultiPL-E TS
#   Java:        MultiPL-E Java, HumanEvalPack Java
#   JavaScript:  MultiPL-E JS, HumanEvalPack JS
#   C++:         MultiPL-E C++, HumanEvalPack C++
#   C#:          MultiPL-E C#
#   PHP:         MultiPL-E PHP
#   C/HTML/CSS/Markdown: BpB perplexity (no functional benchmark)
#
# Parallelises across all available GPUs (one language per GPU).
# Resumable: completed languages are skipped (delete sentinel to re-run).
#
# Usage:
#   bash scripts/run_code_eval.sh                          # all langs, all GPUs
#   bash scripts/run_code_eval.sh go java python           # specific languages
#   GPUS="1 2 3 4 5 6 7" bash scripts/run_code_eval.sh    # skip GPU 0 (if busy)
#   bash scripts/run_code_eval.sh --slurm                  # SLURM array jobs
#   bash scripts/run_code_eval.sh --slurm go java          # SLURM, specific langs
#
# Environment overrides:
#   GPUS="0 1 2 3 4 5 6 7"   space-separated GPU IDs to use (default: all detected)
#   BATCH_SIZE=1               bigcode-eval batch size (integer)
#   N_SAMPLES=1                completions per problem (1=pass@1, 10=pass@1+10)
#   SKIP_PLOTTING=1            skip final plot generation
#   PUSH_TO_HUB=<repo>         upload leaderboard JSONs to HF dataset repo
#   BASELINE_DIR=<path>        directory with prior results for regression check
#   BIGCODE_EVAL_MAIN=<path>   path to bigcode-evaluation-harness main.py
#   SLURM_GPU=1                GPU count per SLURM job
#   SLURM_MEM=80G              memory per SLURM job
#   SLURM_TIME=04:00:00        wall time per SLURM job
# =============================================================================
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_MODEL="allenai/OLMo-3-1025-7B"
HF_ADAPTER_PREFIX="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M"
RESULTS_BASE="./eval_results"
LOGS_DIR="./logs/code_eval"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_SAMPLES="${N_SAMPLES:-1}"
SKIP_PLOTTING="${SKIP_PLOTTING:-0}"
PUSH_TO_HUB="${PUSH_TO_HUB:-}"
BASELINE_DIR="${BASELINE_DIR:-}"
BIGCODE_EVAL_MAIN="${BIGCODE_EVAL_MAIN:-}"

# SLURM settings
SLURM_MODE=0
SLURM_GPU="${SLURM_GPU:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"

# Required for HumanEval code execution
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

# Default language lists
# Code languages: evaluated with bigcode-evaluation-harness (language-specific benchmarks)
ALL_CODE_LANGS=(python go typescript java javascript cpp csharp php)
# Perplexity languages: evaluated with BpB (no functional benchmark available)
ALL_PERP_LANGS=(c html css markdown)

# ─── GPU detection ───────────────────────────────────────────────────────────
# Auto-detect available GPU IDs from nvidia-smi, or fall back to GPUS env var.
# Override: GPUS="1 2 3 4 5 6 7" to skip specific GPUs (e.g., one already in use).
if [ -z "${GPUS:-}" ]; then
    if command -v nvidia-smi &>/dev/null; then
        GPUS=$(nvidia-smi --list-gpus | awk '{print NR-1}' | tr '\n' ' ' | xargs)
    else
        GPUS="0"
    fi
fi
read -ra GPU_IDS <<< "$GPUS"
N_GPUS="${#GPU_IDS[@]}"

# ─── Argument parsing ────────────────────────────────────────────────────────
CODE_LANGS=()
PERP_LANGS=()
POSITIONAL=()

for arg in "$@"; do
    case "$arg" in
        --slurm) SLURM_MODE=1 ;;
        --*)     echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)       POSITIONAL+=("$arg") ;;
    esac
done

if [ "${#POSITIONAL[@]}" -gt 0 ]; then
    for lang in "${POSITIONAL[@]}"; do
        case "$lang" in
            c|html|css|markdown) PERP_LANGS+=("$lang") ;;
            *)                   CODE_LANGS+=("$lang") ;;
        esac
    done
else
    CODE_LANGS=("${ALL_CODE_LANGS[@]}")
    PERP_LANGS=("${ALL_PERP_LANGS[@]}")
fi

ALL_LANGS=("${CODE_LANGS[@]:-}" "${PERP_LANGS[@]:-}")

# ─── Utilities ───────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
err() { log "ERROR: $*" >&2; exit 1; }

# ─── Pre-flight checks ───────────────────────────────────────────────────────
log "=== UniMixLM Eval System v4 (bigcode-evaluation-harness) ==="
log "Base model  : $BASE_MODEL"
log "Code langs  : ${CODE_LANGS[*]:-none}"
log "Perp langs  : ${PERP_LANGS[*]:-none}"
log "GPUs        : ${GPUS} (${N_GPUS} total)"
log "Batch size  : $BATCH_SIZE"
log "N samples   : $N_SAMPLES"
log "SLURM mode  : $SLURM_MODE"
[ -n "$PUSH_TO_HUB" ]       && log "Hub sync    : $PUSH_TO_HUB"
[ -n "$BASELINE_DIR" ]      && log "Baseline    : $BASELINE_DIR"
[ -n "$BIGCODE_EVAL_MAIN" ] && log "Bigcode main: $BIGCODE_EVAL_MAIN"

if ! huggingface-cli whoami >/dev/null 2>&1; then
    err "Not logged in to HuggingFace. Run: huggingface-cli login"
fi
HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
log "HuggingFace : authenticated as $HF_USER"

if ! python3 -c "import bigcode_eval" 2>/dev/null; then
    err "bigcode_eval not found. Install: pip install -e git+https://github.com/bigcode-project/bigcode-evaluation-harness@main#egg=bigcode_eval"
fi
if ! python3 -c "import peft" 2>/dev/null; then
    err "peft not found. Run: pip install peft"
fi
if ! python3 -c "import accelerate" 2>/dev/null; then
    err "accelerate not found. Run: pip install accelerate"
fi

mkdir -p "$RESULTS_BASE" "$LOGS_DIR"

# ─── Tokenizer download ──────────────────────────────────────────────────────
TOK_DIR="./olmo3_las_tokenizers/las-code-tokenizers-OLMo-3-1025-7B"
N_TOK=$(ls -A "$TOK_DIR" 2>/dev/null | wc -l || echo 0)
if [ "$N_TOK" -lt 5 ]; then
    log "=== Downloading LAS code tokenizers ==="
    bash scripts/download_code_tokenizers.sh
else
    log "Tokenizers already present ($N_TOK files) — skipping download."
fi

# ─── Build common Python args ─────────────────────────────────────────────────
_python_args() {
    local lang="$1"
    local mode="$2"
    local out_dir="${RESULTS_BASE}/code_${lang}"

    local args=(
        --lang "$lang"
        --base-model "$BASE_MODEL"
        --adapter-repo "${HF_ADAPTER_PREFIX}-${lang}"
        --output-dir "$out_dir"
        --mode "$mode"
        --n-samples "$N_SAMPLES"
        --batch-size "$BATCH_SIZE"
    )
    [ -n "$PUSH_TO_HUB" ]       && args+=(--push-to-hub "$PUSH_TO_HUB")
    [ -n "$BASELINE_DIR" ]      && args+=(--baseline-dir "$BASELINE_DIR")
    [ -n "$BIGCODE_EVAL_MAIN" ] && args+=(--bigcode-eval-path "$BIGCODE_EVAL_MAIN")
    echo "${args[@]}"
}

# ─── GPU pool semaphore (FIFO-based) ─────────────────────────────────────────
# A named pipe acts as a blocking queue of GPU IDs.
# acquire_gpu: blocks until a GPU slot is free, prints the GPU ID.
# release_gpu: returns a GPU ID to the pool.

GPU_FIFO="/tmp/unimixlm_gpu_pool_$$"
mkfifo "$GPU_FIFO"
exec 9<>"$GPU_FIFO"   # open fd 9 for read+write so the pipe stays alive
rm -f "$GPU_FIFO"     # unlink name; fd 9 keeps the pipe open until script exits

# Populate the pool
for _gpu in "${GPU_IDS[@]}"; do
    printf '%s\n' "$_gpu" >&9
done

acquire_gpu() {
    local gpu_id
    read -r gpu_id <&9
    echo "$gpu_id"
}

release_gpu() {
    printf '%s\n' "$1" >&9
}

# ─── Parallel local execution ─────────────────────────────────────────────────
# PIDs and metadata for final wait/status reporting
PIDS=()
PID_LANGS=()
PID_GPUS=()

run_parallel() {
    local lang="$1"
    local mode="$2"
    local sentinel="$3"

    if [ -f "$sentinel" ]; then
        log "[$lang] Already complete — skipping (delete ${sentinel} to re-run)."
        return
    fi

    # Block here until a GPU slot is free
    local gpu_id
    gpu_id=$(acquire_gpu)
    log "[$lang] Acquired GPU $gpu_id — starting (mode=$mode)."

    local out_dir="${RESULTS_BASE}/code_${lang}"
    mkdir -p "$out_dir"
    local log_file="${LOGS_DIR}/${lang}.log"

    local py_args
    py_args=$(_python_args "$lang" "$mode")

    # Run in a background subshell pinned to the assigned GPU.
    # GPU is released back to the pool when the subshell exits (success or failure).
    (
        # Don't let set -e kill the subshell before we can release the GPU
        set +e
        CUDA_VISIBLE_DEVICES="$gpu_id" \
            python3 las_experiment/code_benchmark_eval.py ${py_args} \
            2>&1 | tee "$log_file"
        local exit_code="${PIPESTATUS[0]}"

        # Always release the GPU
        release_gpu "$gpu_id"

        if [ "$exit_code" -ne 0 ]; then
            log "[$lang] FAILED on GPU $gpu_id (exit $exit_code). See $log_file"
            exit "$exit_code"
        fi

        touch "$sentinel"
        log "[$lang] DONE on GPU $gpu_id."
    ) &

    PIDS+=($!)
    PID_LANGS+=("$lang")
    PID_GPUS+=("$gpu_id")
}

# ─── SLURM submission ─────────────────────────────────────────────────────────
submit_slurm() {
    local lang="$1"
    local mode="$2"
    local sentinel="$3"

    if [ -f "$sentinel" ]; then
        log "[$lang] Already complete — skipping."
        return
    fi

    local out_dir="${RESULTS_BASE}/code_${lang}"
    mkdir -p "$out_dir"

    local py_args
    py_args=$(_python_args "$lang" "$mode")

    # shellcheck disable=SC2086
    sbatch <<SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=unimixlm_${lang}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:${SLURM_GPU}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=${LOGS_DIR}/${lang}_%j.log
#SBATCH --error=${LOGS_DIR}/${lang}_%j.err

export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

python3 las_experiment/code_benchmark_eval.py ${py_args}
touch "${sentinel}"
SLURM_SCRIPT
    log "[$lang] SLURM job submitted."
}

# ─── Dispatch all languages ───────────────────────────────────────────────────
if [ "$SLURM_MODE" -eq 1 ]; then
    # SLURM: submit one job per language (cluster manages parallelism)
    if [ "${#CODE_LANGS[@]}" -gt 0 ]; then
        log "=== Submitting bigcode-eval benchmark jobs (SLURM) ==="
        for lang in "${CODE_LANGS[@]}"; do
            sentinel="${RESULTS_BASE}/code_${lang}/.eval_complete"
            submit_slurm "$lang" "bigcode" "$sentinel"
        done
    fi
    if [ "${#PERP_LANGS[@]}" -gt 0 ]; then
        log "=== Submitting perplexity benchmark jobs (SLURM) ==="
        for lang in "${PERP_LANGS[@]}"; do
            sentinel="${RESULTS_BASE}/code_${lang}/.perplexity_complete"
            submit_slurm "$lang" "perplexity" "$sentinel"
        done
    fi
else
    # Local: run up to N_GPUS languages in parallel
    log "=== Launching parallel evaluation (max ${N_GPUS} concurrent jobs) ==="

    if [ "${#CODE_LANGS[@]}" -gt 0 ]; then
        log "--- Functional benchmarks (bigcode-eval: HumanEval/MBPP/MultiPL-E/HumanEvalPack) ---"
        for lang in "${CODE_LANGS[@]}"; do
            sentinel="${RESULTS_BASE}/code_${lang}/.eval_complete"
            run_parallel "$lang" "bigcode" "$sentinel"
        done
    fi

    if [ "${#PERP_LANGS[@]}" -gt 0 ]; then
        log "--- Perplexity benchmarks (BpB on TheStack) ---"
        for lang in "${PERP_LANGS[@]}"; do
            sentinel="${RESULTS_BASE}/code_${lang}/.perplexity_complete"
            run_parallel "$lang" "perplexity" "$sentinel"
        done
    fi

    # ── Wait for all background jobs and collect exit codes ──────────────────
    log "=== Waiting for all jobs to complete ==="
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
            log "FAIL [$lang] exited with status $status"
        else
            log "OK   [$lang] finished successfully"
        fi
    done

    if [ "${#FAILED_LANGS[@]}" -gt 0 ]; then
        log "=== WARNING: ${#FAILED_LANGS[@]} language(s) failed: ${FAILED_LANGS[*]}"
        log "    Re-run without deleting sentinels to retry only failed languages."
    fi
fi

# ─── Plotting (local mode only — SLURM jobs run async) ────────────────────────
if [ "$SLURM_MODE" -eq 1 ]; then
    log "SLURM jobs submitted. Run plotting manually after jobs complete:"
    log "  python3 las_experiment/plot_code_eval_results.py \\"
    log "      --results-dir $RESULTS_BASE \\"
    log "      --output-dir ${RESULTS_BASE}/code_summary \\"
    log "      --langs ${ALL_LANGS[*]:-}"

elif [ "$SKIP_PLOTTING" -eq 0 ] && [ "${#FAILED_LANGS[@]:-0}" -eq 0 ]; then
    log "=== Generating plots ==="
    SUMMARY_DIR="${RESULTS_BASE}/code_summary"
    mkdir -p "$SUMMARY_DIR"

    python3 las_experiment/plot_code_eval_results.py \
        --results-dir "$RESULTS_BASE" \
        --output-dir "$SUMMARY_DIR" \
        --langs "${ALL_LANGS[@]:-}" \
        2>&1 | tee "${LOGS_DIR}/plotting.log"

    log "Plots saved to: $SUMMARY_DIR"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
log "=== Evaluation complete ==="
log "Results : $RESULTS_BASE"
log "Logs    : $LOGS_DIR"
