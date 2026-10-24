#!/usr/bin/env bash
# OLMo generality phase (user decisions 5(b) + 6(c)): 3 families x 4 languages
# = 12 cells, into the SAME eval_results/intrinsic_sweep_v2/cells/ dir as the
# completed 70-cell sweep (path is family+lang+protocol-encoded, so no
# collision). Modeled directly on scripts/launch_intrinsic_sweep_v2.sh.
#
# Families: olmo3-7b (NEW, converted this session), bloom-3b, mistral-7b-v0.3
#   (both already converted + already have deu/fin/hun/tur cells from the
#   70-cell sweep -- those 8 invocations below will idempotently reverify in
#   seconds; only the 4 olmo3-7b cells do real work).
# Languages: deu, fin, hun, tur -- the sweep's Latin languages with existing
#   Hub slices. The OLMo-19 Latin candidates (fil cat afr ind por ita glg nob
#   uzn dan swe azj cym ron lvs als) are NOT included: none has a
#   MultilingualUnigramLM/FineWeb2-<code>-100M Hub slice (re-verified via the
#   full org catalog listing, not just per-name lookups), and none of the
#   other Hub datasets on the org (FineWeb2-10K/5M/10M/100M-olmo3-7b-toks) is
#   both compatible with this driver's fixed per-language-repo loading path
#   AND large enough / language-complete for the 50k-line protocol. See the
#   session report for the per-language gold verdicts (2 of 16 fail n<300:
#   fil n=78, azj n=251; ita's existing gold splits validated separately and
#   pass) and the corpus-availability evidence. No corpus was substituted.
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SWEEP_VENV_DIR:?set SWEEP_VENV_DIR to the tokenizer-sweep virtualenv}/bin/python"
BASE_LANGS=tur,fin,hun,arb,hin
PARALLEL=3
LOGDIR=$REPO/logs/intrinsic_sweep_v2/olmo_generality
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
    echo "A grid run is already running (pid $(cat "$PIDFILE"), pidfile $PIDFILE)." >&2
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

FAMILIES=(olmo3-7b bloom-3b mistral-7b-v0.3)
LANGS=(deu fin hun tur)

echo "=== olmo generality grid START $(date -Is) ==="
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
