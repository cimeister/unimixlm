#!/usr/bin/env bash
# Launch the full 10-family x 7-language MorphScore sweep (70 cells).
#
# Detached by design: re-exec'd under setsid so it survives the launching shell,
# the SSH session and any restart of the driving process. Progress is a per-cell log
# under logs/intrinsic_sweep_v2/cells/ plus one line per finished cell in
# _status.tsv; completion is the DONE (or FAILED) marker file.
#
#   bash scripts/launch_intrinsic_sweep_v2.sh
#
# Protocol (identical for every cell, and re-recorded inside every cell JSON):
#   base-Unigram conversion on the combined tur/fin/hun/arb/hin corpus, soft EM;
#   per-language overlay seeded with the base tokenizer's own trained log-scores
#   (base_log_scores), K=10 EM iterations, 50k script-filtered FineWeb2-100M
#   lines; scored with the offsets-patched MorphScore harness at
#   stem_eq_lemma=False; default gold for deu/fin/hun/tur/hin/tam and the
#   repaired gold for arb.
#
# Idempotent: a cell whose JSON already exists AND whose recorded harness/gold/
# arm-tokenizer hashes still verify is skipped in seconds. The 10 pre-existing
# F15 overlays and the 3 already-measured cells therefore cost almost nothing.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SWEEP_VENV_DIR:?set SWEEP_VENV_DIR to the tokenizer-sweep virtualenv}/bin/python"
BASE_LANGS=tur,fin,hun,arb,hin
PARALLEL=3          # measured: each spm_train pass takes up to 20 threads; node has 48
LOGDIR=$REPO/logs/intrinsic_sweep_v2
CELLLOGS=$LOGDIR/cells
STATUS=$LOGDIR/_status.tsv
MAIN=$LOGDIR/sweep.log
PIDFILE=$LOGDIR/sweep.pid
DONE=$LOGDIR/DONE
FAILED=$LOGDIR/FAILED

# --- re-exec detached ------------------------------------------------------
if [ "${_ISWEEP_DETACHED:-0}" != "1" ]; then
  mkdir -p "$CELLLOGS"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "A sweep is already running (pid $(cat "$PIDFILE"), pidfile $PIDFILE)." >&2
    echo "Refusing to start a second one; kill it or remove the pidfile first." >&2
    exit 1
  fi
  rm -f "$DONE" "$FAILED"
  _ISWEEP_DETACHED=1 setsid nohup bash "$0" "$@" >>"$MAIN" 2>&1 < /dev/null &
  sleep 1
  echo "launched detached; pid $(cat "$PIDFILE" 2>/dev/null || echo '(starting)')"
  echo "  main log    : $MAIN"
  echo "  per-cell    : $CELLLOGS/<family>__<lang>.log"
  echo "  status      : $STATUS"
  echo "  done marker : $DONE   (or $FAILED)"
  exit 0
fi

# --- detached body ---------------------------------------------------------
echo $$ > "$PIDFILE"
cd "$REPO" || exit 1
mkdir -p "$CELLLOGS"
: > "$STATUS"

FAMILIES=(bloom-3b gemma3-1b granite3.0-8b llama3.2-1b mistral-7b-v0.3
          mistral-nemo-12b phi-4 qwen2_5-0_5b xglm-2.9b yi-1.5-6b)
# phi-3-mini is EXCLUDED: its tokenizer.json has pre_tokenizer: null, so the
# base-Unigram conversion cannot run on it. The driver refuses it by name.
# gemma2-2b is NOT in the sweep: the paper's Gemma row is Gemma 3 (proved by
# reproducing the paper's Devanagari subword count).
LANGS=(deu fin hun tur hin arb tam)

echo "=== intrinsic sweep v2 START $(date -Is) ==="
echo "families=${#FAMILIES[@]} langs=${#LANGS[@]} cells=$((${#FAMILIES[@]} * ${#LANGS[@]})) parallel=$PARALLEL"

run_cell() {
  local fam=$1 lang=$2
  local log="$CELLLOGS/${fam}__${lang}.log"
  local t0 t1
  t0=$(date +%s)
  "$PY" -u "$REPO/scripts/sweep_intrinsic_v2.py" cell \
      --family "$fam" --lang "$lang" --base-langs "$BASE_LANGS" > "$log" 2>&1
  local rc=$?
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\t%s\n' "$fam" "$lang" \
      "$([ $rc -eq 0 ] && echo ok || echo FAIL)" "$((t1 - t0))s" "$(date -Is)" \
      >> "$STATUS"
  return $rc
}
export -f run_cell
export PY REPO BASE_LANGS CELLLOGS STATUS

# Emit every cell, family-major so a family's artifacts warm up together.
for fam in "${FAMILIES[@]}"; do
  for lang in "${LANGS[@]}"; do
    printf '%s %s\n' "$fam" "$lang"
  done
done | xargs -P "$PARALLEL" -n 2 bash -c 'run_cell "$0" "$1"'
XARGS_RC=$?

echo "=== all cells attempted (xargs rc=$XARGS_RC) $(date -Is) ==="
NFAIL=$(grep -cP '\tFAIL\t' "$STATUS" || true)
NOK=$(grep -cP '\tok\t' "$STATUS" || true)
echo "ok=$NOK fail=$NFAIL"
if [ "$NFAIL" -gt 0 ]; then
  echo "--- failed cells ---"
  grep -P '\tFAIL\t' "$STATUS"
fi

echo "=== regenerating manifest $(date -Is) ==="
"$PY" -u "$REPO/scripts/sweep_intrinsic_v2.py" manifest

if [ "$NFAIL" -gt 0 ]; then
  { echo "finished $(date -Is)"; echo "ok=$NOK fail=$NFAIL";
    echo "see $STATUS and $CELLLOGS/"; } > "$FAILED"
  echo "=== FINISHED WITH FAILURES $(date -Is) ==="
else
  { echo "finished $(date -Is)"; echo "ok=$NOK fail=0";
    echo "manifest: $REPO/eval_results/intrinsic_sweep_v2/manifest.json"; } > "$DONE"
  echo "=== DONE $(date -Is) ==="
fi
rm -f "$PIDFILE"
