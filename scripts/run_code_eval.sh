#!/usr/bin/env bash
# =============================================================================
# run_code_eval.sh — UniMixLM Eval System v3
#
# Evaluates all 11 LAS finetuned models vs. base OLMo-3-7B.
# Resumable: completed languages are skipped (delete sentinel to re-run).
#
# Usage:
#   bash scripts/run_code_eval.sh                        # all languages
#   bash scripts/run_code_eval.sh go java typescript     # specific languages
#   bash scripts/run_code_eval.sh --slurm                # submit SLURM array jobs
#   bash scripts/run_code_eval.sh --slurm go java        # SLURM, specific langs
#
# Environment overrides:
#   BATCH_SIZE=auto:4         lm-eval batch size
#   N_SAMPLES=1               completions per problem (1=pass@1, 10=pass@1+10)
#   SKIP_PLOTTING=1           skip final plot generation
#   PUSH_TO_HUB=<repo>        upload leaderboard JSONs to HF dataset repo
#   BASELINE_DIR=<path>       directory with prior results for regression check
#   SLURM_GPU=1               GPU count per SLURM job
#   SLURM_MEM=80G             memory per SLURM job
#   SLURM_TIME=04:00:00       wall time per SLURM job
# =============================================================================
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_MODEL="allenai/OLMo-3-1025-7B"
HF_ADAPTER_PREFIX="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M"
RESULTS_BASE="./eval_results"
LOGS_DIR="./logs/code_eval"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
N_SAMPLES="${N_SAMPLES:-1}"
SKIP_PLOTTING="${SKIP_PLOTTING:-0}"
PUSH_TO_HUB="${PUSH_TO_HUB:-}"
BASELINE_DIR="${BASELINE_DIR:-}"

# SLURM settings
SLURM_MODE=0
SLURM_GPU="${SLURM_GPU:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"

# Required for HumanEval code execution
export HF_ALLOW_CODE_EVAL=1
export LM_EVAL_CONFIRM_RUN_UNSAFE_CODE=1
export TOKENIZERS_PARALLELISM=false

# Default language lists
ALL_CODE_LANGS=(go typescript java javascript cpp csharp php c)
ALL_PERP_LANGS=(html css markdown)

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
            html|css|markdown) PERP_LANGS+=("$lang") ;;
            *)                 CODE_LANGS+=("$lang") ;;
        esac
    done
else
    CODE_LANGS=("${ALL_CODE_LANGS[@]}")
    PERP_LANGS=("${ALL_PERP_LANGS[@]}")
fi

ALL_LANGS=("${CODE_LANGS[@]}" "${PERP_LANGS[@]}")

# ─── Utilities ───────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
err() { log "ERROR: $*" >&2; exit 1; }

# ─── Pre-flight checks ───────────────────────────────────────────────────────
log "=== UniMixLM Eval System v3 ==="
log "Base model  : $BASE_MODEL"
log "Code langs  : ${CODE_LANGS[*]:-none}"
log "Perp langs  : ${PERP_LANGS[*]:-none}"
log "Batch size  : $BATCH_SIZE"
log "N samples   : $N_SAMPLES"
log "SLURM mode  : $SLURM_MODE"
[ -n "$PUSH_TO_HUB" ]   && log "Hub sync    : $PUSH_TO_HUB"
[ -n "$BASELINE_DIR" ]  && log "Baseline    : $BASELINE_DIR"

if ! huggingface-cli whoami >/dev/null 2>&1; then
    err "Not logged in to HuggingFace. Run: huggingface-cli login"
fi
HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
log "HuggingFace : authenticated as $HF_USER"

if ! python3 -c "import lm_eval" 2>/dev/null; then
    err "lm_eval not found. Run: pip install -r requirements.txt"
fi
if ! python3 -c "import peft" 2>/dev/null; then
    err "peft not found. Run: pip install peft"
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
    local mode="$2"   # lmeval | perplexity
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
    [ -n "$PUSH_TO_HUB" ]  && args+=(--push-to-hub "$PUSH_TO_HUB")
    [ -n "$BASELINE_DIR" ] && args+=(--baseline-dir "$BASELINE_DIR")
    echo "${args[@]}"
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
export LM_EVAL_CONFIRM_RUN_UNSAFE_CODE=1
export TOKENIZERS_PARALLELISM=false

python3 las_experiment/code_benchmark_eval.py ${py_args}
touch "${sentinel}"
SLURM_SCRIPT
    log "[$lang] SLURM job submitted."
}

# ─── Local execution ──────────────────────────────────────────────────────────
run_local() {
    local lang="$1"
    local mode="$2"
    local sentinel="$3"

    if [ -f "$sentinel" ]; then
        log "[$lang] Already complete — skipping (delete ${sentinel} to re-run)."
        return
    fi

    local out_dir="${RESULTS_BASE}/code_${lang}"
    mkdir -p "$out_dir"
    local log_file="${LOGS_DIR}/${lang}.log"

    local py_args
    py_args=$(_python_args "$lang" "$mode")

    log "[$lang] Starting (mode=$mode)..."
    # shellcheck disable=SC2086
    python3 las_experiment/code_benchmark_eval.py ${py_args} \
        2>&1 | tee "$log_file"

    # Check exit status from the pipeline (not tee)
    local exit_code="${PIPESTATUS[0]}"
    if [ "$exit_code" -ne 0 ]; then
        log "[$lang] FAILED (exit $exit_code). See $log_file"
        return "$exit_code"
    fi

    touch "$sentinel"
    log "[$lang] DONE."
}

# ─── Run code langs (lm-eval) ─────────────────────────────────────────────────
if [ "${#CODE_LANGS[@]}" -gt 0 ]; then
    log "=== Functional benchmarks (HumanEval + MBPP + MultiPL-E) ==="
    for lang in "${CODE_LANGS[@]}"; do
        sentinel="${RESULTS_BASE}/code_${lang}/.eval_complete"
        if [ "$SLURM_MODE" -eq 1 ]; then
            submit_slurm "$lang" "lmeval" "$sentinel"
        else
            run_local "$lang" "lmeval" "$sentinel"
        fi
    done
fi

# ─── Run perplexity langs (BpB) ───────────────────────────────────────────────
if [ "${#PERP_LANGS[@]}" -gt 0 ]; then
    log "=== Perplexity benchmarks (BpB on TheStack) ==="
    for lang in "${PERP_LANGS[@]}"; do
        sentinel="${RESULTS_BASE}/code_${lang}/.perplexity_complete"
        if [ "$SLURM_MODE" -eq 1 ]; then
            submit_slurm "$lang" "perplexity" "$sentinel"
        else
            run_local "$lang" "perplexity" "$sentinel"
        fi
    done
fi

# ─── Plotting (local mode only — SLURM jobs run async) ────────────────────────
if [ "$SLURM_MODE" -eq 1 ]; then
    log "SLURM jobs submitted. Run plotting manually after jobs complete:"
    log "  python3 las_experiment/plot_code_eval_results.py \\"
    log "      --results-dir $RESULTS_BASE \\"
    log "      --output-dir ${RESULTS_BASE}/code_summary \\"
    log "      --langs ${ALL_LANGS[*]}"

elif [ "$SKIP_PLOTTING" -eq 0 ]; then
    log "=== Generating plots ==="
    SUMMARY_DIR="${RESULTS_BASE}/code_summary"
    mkdir -p "$SUMMARY_DIR"

    python3 las_experiment/plot_code_eval_results.py \
        --results-dir "$RESULTS_BASE" \
        --output-dir "$SUMMARY_DIR" \
        --langs "${ALL_LANGS[@]}" \
        2>&1 | tee "${LOGS_DIR}/plotting.log"

    log "Plots saved to: $SUMMARY_DIR"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
log "=== Evaluation complete ==="
log "Results : $RESULTS_BASE"
log "Logs    : $LOGS_DIR"
