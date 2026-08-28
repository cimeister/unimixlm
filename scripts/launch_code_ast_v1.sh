#!/usr/bin/env bash
# Launch the full 8-family x 9-programming-language AST-leaf-alignment grid
# (72 cells) that replaces appendix.tex tab:code-overall-grid /
# tab:code-overall-values.
#
#   bash scripts/launch_code_ast_v1.sh
#
# Detached by design: re-exec'd under setsid so it survives the launching shell,
# the SSH session and any Claude-session restart. Progress is a per-cell log
# under logs/code_ast_v1/cells/ plus one line per finished cell in _status.tsv;
# completion is the DONE (or FAILED) marker file.
#
# Protocol (identical for every cell, and re-recorded inside every cell JSON):
#   * the family's BASE Unigram conversion is REUSED READ-ONLY from the
#     natural-language sweep (base-langs tur,fin,hun,arb,hin). Nothing is refit;
#     a family whose base is missing aborts naming the `sweep_intrinsic_v2.py
#     convert` command that produces it.
#   * per-language LangMAP overlay seeded with that base's own trained
#     log-scores, soft EM, 10 sub-iterations, on the language's EVAL-DISJOINT
#     StarCoder corpus (every row of the repo's code-tokenizer-training sample
#     whose id is not in the AST evaluation parquet, verified again by content
#     sha256).
#   * three arms -- Original(BPE) -> the table's "Base", base_tokenizer.json,
#     LangMAP(langspec) -> the table's "LM" -- scored in one
#     ASTBoundaryMetrics.compute() call over a materialised, sha256-manifested
#     100-file-per-language snippet set.
#
# Phi-3-mini is EXCLUDED (its tokenizer.json has pre_tokenizer: null on the Hub
# and save_pretrained does not synthesise one; Microsoft published no
# non-instruct Phi-3-mini). The grid is 8 x 9 = 72, not 81.
#
# Idempotent: a cell whose JSON already exists AND whose recorded base/overlay
# hashes still verify is skipped in seconds.
set -u

REPO=/users/cmeister747/unimixlm
PY=/users/cmeister747/tokenizer_lm_exps_env/bin/python
BASE_LANGS=tur,fin,hun,arb,hin
# Each overlay fit runs spm_train with up to 20 threads on a ~150-240 MB corpus.
# Node has 48 cores; 3 concurrent cells is the setting the natural-language
# sweep ran at.
PARALLEL=3
LOGDIR=$REPO/logs/code_ast_v1
CELLLOGS=$LOGDIR/cells
STATUS=$LOGDIR/_status.tsv
MAIN=$LOGDIR/sweep.log
PIDFILE=$LOGDIR/sweep.pid
DONE=$LOGDIR/DONE
FAILED=$LOGDIR/FAILED

# --- re-exec detached ------------------------------------------------------
if [ "${_CAST_DETACHED:-0}" != "1" ]; then
  mkdir -p "$CELLLOGS"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "A code-AST sweep is already running (pid $(cat "$PIDFILE"), pidfile $PIDFILE)." >&2
    echo "Refusing to start a second one; kill it or remove the pidfile first." >&2
    exit 1
  fi
  rm -f "$DONE" "$FAILED"
  _CAST_DETACHED=1 setsid nohup bash "$0" "$@" >>"$MAIN" 2>&1 < /dev/null &
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

FAMILIES=(bloom-3b gemma2-2b granite3.0-8b llama3.2-1b mistral-nemo-12b
          phi-4 qwen2_5-0_5b xglm-2.9b)
LANGS=(c cpp csharp go java javascript php python typescript)

echo "=== code AST grid START $(date -Is) ==="
echo "families=${#FAMILIES[@]} langs=${#LANGS[@]} cells=$((${#FAMILIES[@]} * ${#LANGS[@]})) parallel=$PARALLEL"

# Build every language's evalset and eval-disjoint corpus ONCE, serially, before
# any cell runs. Both are per-language artifacts shared by all eight families, so
# building them here removes the only write-write race the grid has and keeps
# the disjointness check off the per-cell critical path.
echo "=== preparing per-language artifacts $(date -Is) ==="
for lang in "${LANGS[@]}"; do
  "$PY" -u "$REPO/scripts/sweep_code_ast_v1.py" evalset --lang "$lang" \
      > "$CELLLOGS/_evalset__${lang}.log" 2>&1 || {
    echo "FATAL: evalset build failed for $lang; see $CELLLOGS/_evalset__${lang}.log" >&2
    echo "evalset build failed for $lang" > "$FAILED"; rm -f "$PIDFILE"; exit 1; }
  "$PY" -u "$REPO/scripts/sweep_code_ast_v1.py" corpus --lang "$lang" \
      > "$CELLLOGS/_corpus__${lang}.log" 2>&1 || {
    echo "FATAL: corpus build failed for $lang; see $CELLLOGS/_corpus__${lang}.log" >&2
    echo "corpus build failed for $lang" > "$FAILED"; rm -f "$PIDFILE"; exit 1; }
  echo "  prepared $lang"
done

run_cell() {
  local fam=$1 lang=$2
  local log="$CELLLOGS/${fam}__${lang}.log"
  local t0 t1
  t0=$(date +%s)
  "$PY" -u "$REPO/scripts/sweep_code_ast_v1.py" cell \
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

# Family-major so a family's artifacts warm up together.
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
"$PY" -u "$REPO/scripts/sweep_code_ast_v1.py" manifest

if [ "$NFAIL" -gt 0 ]; then
  { echo "finished $(date -Is)"; echo "ok=$NOK fail=$NFAIL";
    echo "see $STATUS and $CELLLOGS/"; } > "$FAILED"
  echo "=== FINISHED WITH FAILURES $(date -Is) ==="
else
  { echo "finished $(date -Is)"; echo "ok=$NOK fail=0";
    echo "manifest: $REPO/eval_results/code_ast_v1/manifest.json"; } > "$DONE"
  echo "=== DONE $(date -Is) ==="
fi
rm -f "$PIDFILE"
