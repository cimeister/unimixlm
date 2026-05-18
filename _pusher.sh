#!/usr/bin/env bash
# _pusher.sh — durable, REGULAR HF pushing. Every cycle, upload any
# stable (unmodified >120s) checkpoint-* dir that isn't pushed yet to
# its cell's HF repo, so progress survives a pod loss long before the
# ~3h cell-end upload. Idempotent via a .pushed marker per checkpoint.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
# MUST activate the venv: the `hf` CLI lives in .venv/bin and is not on
# the system PATH. Without this every push fails "hf: command not found".
source .venv/bin/activate
INTERVAL="${PUSH_INTERVAL:-75}"
echo "[pusher] started; scanning $OUT_BASE every ${INTERVAL}s"
while true; do
  while IFS= read -r ck; do
    [ -d "$ck" ] || continue
    [ -f "$ck/.pushed" ] && continue
    # stable? no file modified in the last 120s
    if [ -n "$(find "$ck" -type f -newermt '-45 seconds' 2>/dev/null | head -1)" ]; then
      continue
    fi
    run_dir="$(dirname "$ck")"; model="$(basename "$(dirname "$run_dir")")"
    base="$(basename "$run_dir")"; lang="${base#ft_}"; lang="${lang%%-*}"
    repo="${HF_ORG}/ft-langmap-${model}-fineweb100M-${lang}-original"
    cstep="$(basename "$ck")"
    echo "[pusher][$(date -u +%T)] $model/$lang $cstep -> $repo"
    hf repo create "$repo" --repo-type model -y >/dev/null 2>&1 || true
    if hf upload "$repo" "$ck" "$cstep" --repo-type model \
         >>"$SHM/logs/pusher.log" 2>&1; then
      touch "$ck/.pushed"
      echo "[pusher][$(date -u +%T)] OK $repo/$cstep"
    else
      echo "[pusher][$(date -u +%T)] retry-later $repo/$cstep"
    fi
  done < <(find "$OUT_BASE" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null)
  sleep "$INTERVAL"
done
