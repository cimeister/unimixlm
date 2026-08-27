#!/usr/bin/env bash
# ============================================================================
# submit_all.sh — EMNLP main-paper job submission driver.
#
# Enumerates the Core (SL2) and Extension (SL3) grids defined in the EMNLP
# plan and emits one `sbatch` per cell, choosing the right wrapper:
#   Core      → scripts/emnlp/run_emnlp_sl2.sh   (36 h cap, 64 GPU cap)
#   Extension → scripts/emnlp/run_emnlp_sl3.sh   (12 h cap, 32 GPU cap)
#
# A single overflow cell — Olmo3-7B / unfrozen / 500M — is force-routed to
# SL2 even when it appears in an Extension sweep.
#
# Usage:
#   bash scripts/emnlp/submit_all.sh                       # submit everything
#   bash scripts/emnlp/submit_all.sh --dry-run             # print, don't submit
#   bash scripts/emnlp/submit_all.sh --filter suite=core   # only Core
#   bash scripts/emnlp/submit_all.sh --filter suite=extension
#   bash scripts/emnlp/submit_all.sh --filter kind=ft
#   bash scripts/emnlp/submit_all.sh --filter kind=eval
#   bash scripts/emnlp/submit_all.sh --filter kind=tok
#   bash scripts/emnlp/submit_all.sh --filter model=qwen2_5-0_5b
#   bash scripts/emnlp/submit_all.sh --filter domain=code
#   bash scripts/emnlp/submit_all.sh --filter section=4.2
#   bash scripts/emnlp/submit_all.sh --max-jobs 16         # safety throttle
#
# Filters are AND-combined; pass --filter multiple times.
#
# Required env: HF_TOKEN
# ============================================================================
set -euo pipefail

DRY_RUN=0
FILTERS=()
MAX_JOBS=""   # optional concurrency cap (use sbatch --dependency? simplest: stop after N)

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --filter)  FILTERS+=("$2"); shift 2 ;;
        --max-jobs) MAX_JOBS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

: "${HF_TOKEN:?HF_TOKEN must be set in env}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAP_SL2="${SCRIPT_DIR}/run_emnlp_sl2.sh"
WRAP_SL3="${SCRIPT_DIR}/run_emnlp_sl3.sh"

declare -i SUBMITTED=0
declare -i SKIPPED=0

# ── Filter helper ───────────────────────────────────────────────────────────
# matches_filters <suite> <kind> <model> <domain> <section>
matches_filters() {
    local suite="$1" kind="$2" model="$3" domain="$4" section="$5"
    [ "${#FILTERS[@]}" -eq 0 ] && return 0
    for f in "${FILTERS[@]}"; do
        local key="${f%%=*}" val="${f#*=}"
        case "$key" in
            suite)   [ "$suite"   = "$val" ] || return 1 ;;
            kind)    [ "$kind"    = "$val" ] || return 1 ;;
            model)   [ "$model"   = "$val" ] || return 1 ;;
            domain)  [ "$domain"  = "$val" ] || return 1 ;;
            section) [ "$section" = "$val" ] || return 1 ;;
            *) echo "Unknown filter key: $key" >&2; exit 2 ;;
        esac
    done
    return 0
}

# ── Submit one cell ─────────────────────────────────────────────────────────
# Args: suite kind domain model lang rehearsal las_mode vocab unfreeze lr
#       data_size lora_preset epochs section
submit_cell() {
    local suite="$1" kind="$2" domain="$3" model="$4" lang="$5" rehearsal="$6"
    local las_mode="$7" vocab="$8" unfreeze="$9" lr="${10}" data_size="${11}"
    local lora_preset="${12}" epochs="${13}" section="${14}"

    if ! matches_filters "$suite" "$kind" "$model" "$domain" "$section"; then
        SKIPPED+=1
        return 0
    fi

    # Force-reroute the SL3 overflow cell to SL2.
    local wrapper="$WRAP_SL3"
    [ "$suite" = "core" ] && wrapper="$WRAP_SL2"
    if [ "$kind" = "ft" ] && [ "$model" = "olmo3-1025-7b" ] \
       && [ "$unfreeze" = "1" ] && [ "$data_size" = "500M" ]; then
        wrapper="$WRAP_SL2"
    fi

    local jobname="emnlp-${kind}-${model}-${domain}-${lang}-${las_mode}-v${vocab}-e${unfreeze}-${data_size}"
    local export_str="HF_TOKEN=${HF_TOKEN}"
    export_str+=",EXP_KIND=${kind}"
    export_str+=",EXP_DOMAIN=${domain}"
    export_str+=",EXP_MODEL=${model}"
    export_str+=",EXP_LANG=${lang}"
    export_str+=",EXP_REHEARSAL=${rehearsal}"
    export_str+=",EXP_LAS_MODE=${las_mode}"
    export_str+=",EXP_VOCAB=${vocab}"
    export_str+=",EXP_UNFREEZE=${unfreeze}"
    export_str+=",EXP_LR=${lr}"
    export_str+=",EXP_DATA_SIZE=${data_size}"
    export_str+=",EXP_LORA_PRESET=${lora_preset}"
    export_str+=",EXP_EPOCHS=${epochs}"

    local cmd=(sbatch -J "$jobname" --export="ALL,${export_str}" "$wrapper")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[DRY] §${section} ${suite}/${kind} → $(basename "$wrapper") :: ${jobname}"
    else
        printf '[SUBMIT] §%s %s/%s → %s :: %s\n' "$section" "$suite" "$kind" "$(basename "$wrapper")" "$jobname"
        "${cmd[@]}"
    fi
    SUBMITTED+=1

    if [ -n "$MAX_JOBS" ] && [ "$SUBMITTED" -ge "$MAX_JOBS" ]; then
        echo "Hit --max-jobs=${MAX_JOBS}, stopping." >&2
        echo "Submitted=${SUBMITTED} Skipped=${SKIPPED}"
        exit 0
    fi
}

# ── Constants ───────────────────────────────────────────────────────────────
CODE_LANGS=(python javascript java cpp go typescript)
NL_LANGS=(tur_Latn fin_Latn hun_Latn)
ALL_MODELS=(qwen2_5-0_5b gemma3-1b olmo3-1025-7b)
SMALL_MODELS=(qwen2_5-0_5b gemma3-1b)

# ============================================================================
# CORE SUITE — SL2 (paper-critical)
# ============================================================================

# §4.1 Tokenizer training (vocab=256k, per model × domain).
for model in "${SMALL_MODELS[@]}"; do
    for domain in code nl; do
        submit_cell core tok "$domain" "$model" any none langspec 256k 0 \
                    2e-4 100M r64-k128 5 "4.1"
    done
done
# Olmo3 only NL (code tokenizer for olmo3 already exists).
submit_cell core tok nl olmo3-1025-7b any none langspec 256k 0 \
            2e-4 100M r64-k128 5 "4.1"

# §4.2 Code FT (Qwen + Gemma × 6 langs × {orig+frozen, ls256k+frozen, ls256k+unfrozen}).
for model in "${SMALL_MODELS[@]}"; do
    for lang in "${CODE_LANGS[@]}"; do
        # original / frozen — uses base BPE vocab (vocab axis ignored).
        submit_cell core ft code "$model" "$lang" eng original 256k 0 \
                    2e-4 100M r16-k32 1 "4.2"
        # langspec / 256k / frozen
        submit_cell core ft code "$model" "$lang" eng langspec 256k 0 \
                    2e-4 100M r16-k32 1 "4.2"
        # langspec / 256k / unfrozen
        submit_cell core ft code "$model" "$lang" eng langspec 256k 1 \
                    2e-4 100M r64-k128 5 "4.2"
    done
done

# §4.3 Code eval (one per FT in §4.2 + 2 base).
for model in "${SMALL_MODELS[@]}"; do
    for lang in "${CODE_LANGS[@]}"; do
        for cell in "original 256k 0 r16-k32 1" \
                    "langspec 256k 0 r16-k32 1" \
                    "langspec 256k 1 r64-k128 5"; do
            read -r las_mode vocab unfreeze lora_preset epochs <<<"$cell"
            submit_cell core eval code "$model" "$lang" eng \
                        "$las_mode" "$vocab" "$unfreeze" 2e-4 100M \
                        "$lora_preset" "$epochs" "4.3"
        done
    done
    # Base-model eval (no FT, no LAS) — pick one canonical lang per base eval.
    submit_cell core eval code "$model" python eng base 256k 0 \
                2e-4 100M r16-k32 1 "4.3"
done

# §4.4 Multi FT (3 models × 3 langs × 3 recipes).
for model in "${ALL_MODELS[@]}"; do
    for lang in "${NL_LANGS[@]}"; do
        submit_cell core ft nl "$model" "$lang" eng original 256k 0 \
                    2e-4 100M r16-k32 1 "4.4"
        submit_cell core ft nl "$model" "$lang" eng langspec 256k 0 \
                    2e-4 100M r16-k32 1 "4.4"
        submit_cell core ft nl "$model" "$lang" eng langspec 256k 1 \
                    2e-4 100M r64-k128 5 "4.4"
    done
done

# §4.5 Multi eval (one per FT in §4.4 + 3 base evals).
for model in "${ALL_MODELS[@]}"; do
    for lang in "${NL_LANGS[@]}"; do
        for cell in "original 256k 0 r16-k32 1" \
                    "langspec 256k 0 r16-k32 1" \
                    "langspec 256k 1 r64-k128 5"; do
            read -r las_mode vocab unfreeze lora_preset epochs <<<"$cell"
            submit_cell core eval nl "$model" "$lang" eng \
                        "$las_mode" "$vocab" "$unfreeze" 2e-4 100M \
                        "$lora_preset" "$epochs" "4.5"
        done
    done
    submit_cell core eval nl "$model" tur_Latn eng base 256k 0 \
                2e-4 100M r16-k32 1 "4.5"
done

# ============================================================================
# EXTENSION SUITE — SL3 (sweeps)
# ============================================================================

# §5.1 Vocab-size sweep (100k, 150k) — extra tok jobs + extra langspec FT/eval.
for model in "${SMALL_MODELS[@]}"; do
    for domain in code nl; do
        for vocab in 100k 150k; do
            submit_cell extension tok "$domain" "$model" any none langspec \
                        "$vocab" 0 2e-4 100M r64-k128 5 "5.1"
        done
    done
done
for vocab in 100k 150k; do
    submit_cell extension tok nl olmo3-1025-7b any none langspec \
                "$vocab" 0 2e-4 100M r64-k128 5 "5.1"
done

# Code langspec FT/eval at the extra vocabs.
for model in "${SMALL_MODELS[@]}"; do
    for lang in "${CODE_LANGS[@]}"; do
        for vocab in 100k 150k; do
            for unfreeze in 0 1; do
                if [ "$unfreeze" = "0" ]; then lp=r16-k32; ep=1; else lp=r64-k128; ep=5; fi
                submit_cell extension ft   code "$model" "$lang" eng langspec \
                            "$vocab" "$unfreeze" 2e-4 100M "$lp" "$ep" "5.1"
                submit_cell extension eval code "$model" "$lang" eng langspec \
                            "$vocab" "$unfreeze" 2e-4 100M "$lp" "$ep" "5.1"
            done
        done
    done
done
# Multi langspec FT/eval at the extra vocabs (all 3 models).
for model in "${ALL_MODELS[@]}"; do
    for lang in "${NL_LANGS[@]}"; do
        for vocab in 100k 150k; do
            for unfreeze in 0 1; do
                if [ "$unfreeze" = "0" ]; then lp=r16-k32; ep=1; else lp=r64-k128; ep=5; fi
                submit_cell extension ft   nl "$model" "$lang" eng langspec \
                            "$vocab" "$unfreeze" 2e-4 100M "$lp" "$ep" "5.1"
                submit_cell extension eval nl "$model" "$lang" eng langspec \
                            "$vocab" "$unfreeze" 2e-4 100M "$lp" "$ep" "5.1"
            done
        done
    done
done

# §5.2 Data-amount sweep (10M, 50M, 500M) — best vocab=256k, langspec, unfrozen.
for model in "${SMALL_MODELS[@]}"; do
    for lang in python javascript; do
        for d in 10M 50M 500M; do
            submit_cell extension ft   code "$model" "$lang" eng langspec \
                        256k 1 2e-4 "$d" r64-k128 5 "5.2"
            submit_cell extension eval code "$model" "$lang" eng langspec \
                        256k 1 2e-4 "$d" r64-k128 5 "5.2"
        done
    done
done
for model in "${ALL_MODELS[@]}"; do
    for d in 10M 50M 500M; do
        submit_cell extension ft   nl "$model" tur_Latn eng langspec \
                    256k 1 2e-4 "$d" r64-k128 5 "5.2"
        submit_cell extension eval nl "$model" tur_Latn eng langspec \
                    256k 1 2e-4 "$d" r64-k128 5 "5.2"
    done
done

# §5.3 LR sweep ({1e-4, 5e-4} on best config — langspec / unfrozen / 256k / 100M).
for model in "${SMALL_MODELS[@]}"; do
    for lang in python javascript; do
        for lr in 1e-4 5e-4; do
            submit_cell extension ft   code "$model" "$lang" eng langspec \
                        256k 1 "$lr" 100M r64-k128 5 "5.3"
            submit_cell extension eval code "$model" "$lang" eng langspec \
                        256k 1 "$lr" 100M r64-k128 5 "5.3"
        done
    done
done
for model in "${ALL_MODELS[@]}"; do
    for lr in 1e-4 5e-4; do
        submit_cell extension ft   nl "$model" tur_Latn eng langspec \
                    256k 1 "$lr" 100M r64-k128 5 "5.3"
        submit_cell extension eval nl "$model" tur_Latn eng langspec \
                    256k 1 "$lr" 100M r64-k128 5 "5.3"
    done
done

# §5.4 Original-tokenizer × unfrozen-embeds ablation
# (Sanity: does unfreezing alone — no LAS — close the gap?)
for model in "${SMALL_MODELS[@]}"; do
    for lang in "${CODE_LANGS[@]:0:3}"; do   # python, javascript, java
        submit_cell extension ft   code "$model" "$lang" eng original \
                    256k 1 2e-4 100M r64-k128 5 "5.4"
        submit_cell extension eval code "$model" "$lang" eng original \
                    256k 1 2e-4 100M r64-k128 5 "5.4"
    done
done
for model in "${ALL_MODELS[@]}"; do
    for lang in "${NL_LANGS[@]}"; do
        submit_cell extension ft   nl "$model" "$lang" eng original \
                    256k 1 2e-4 100M r64-k128 5 "5.4"
        submit_cell extension eval nl "$model" "$lang" eng original \
                    256k 1 2e-4 100M r64-k128 5 "5.4"
    done
done

echo
echo "=========================================================="
echo "Submission summary"
echo "  Submitted: ${SUBMITTED}"
echo "  Skipped (filter mismatch): ${SKIPPED}"
[ "$DRY_RUN" -eq 1 ] && echo "  (DRY RUN — no jobs were actually queued)"
echo "=========================================================="
