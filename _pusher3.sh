#!/usr/bin/env bash
# _pusher3.sh — Phase-3 durable checkpoint pusher. Scans ONLY
# $SHM/ft_models3 and rebuilds the repo (with -<arm>-<loratag> suffix)
# from the path, so it can never mis-push into P1/P2 repos. Path layout:
#   ft_models3/<model>/<arm>/<loratag>/ft_<lang>-eng_<size>/checkpoint-*
# Idempotent via a .pushed marker per checkpoint.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
source .venv/bin/activate
BASE3="$SHM/ft_models3"
INTERVAL="${PUSH_INTERVAL:-75}"
mkdir -p "$BASE3"
echo "[pusher3] started; scanning $BASE3 every ${INTERVAL}s"
while true; do
  while IFS= read -r ck; do
    [ -d "$ck" ] || continue
    [ -f "$ck/.pushed" ] && continue
    if [ -n "$(find "$ck" -type f -newermt '-45 seconds' 2>/dev/null | head -1)" ]; then
      continue
    fi
    run_dir="$(dirname "$ck")"                       # .../ft_<lang>-eng_<size>
    loratag="$(basename "$(dirname "$run_dir")")"
    arm="$(basename "$(dirname "$(dirname "$run_dir")")")"
    model="$(basename "$(dirname "$(dirname "$(dirname "$run_dir")")")")"
    base="$(basename "$run_dir")"; base="${base#ft_}" # <lang>-eng_<size>
    lang="${base%%-*}"
    size="${base##*_}"
    repo="${HF_ORG}/ft-langmap-${model}-fineweb${size}-${lang}-${arm}-${loratag}"
    cstep="$(basename "$ck")"
    echo "[pusher3][$(date -u +%T)] $model/$arm/$loratag/$lang/$size $cstep -> $repo"
    hf repo create "$repo" --repo-type model -y >/dev/null 2>&1 || true
    if hf upload "$repo" "$ck" "$cstep" --repo-type model \
         >>"$SHM/logs/pusher3.log" 2>&1; then
      echo "[pusher3][$(date -u +%T)] OK $repo/$cstep (freeing local)"
      # Dense weights-only ckpts -> delete after a confirmed push so
      # /dev/shm never fills. It's already on HF; end-of-cell RUN_DIR
      # upload covers root model + final-full/ (not checkpoint-* dirs).
      rm -rf "$ck"
    else
      echo "[pusher3][$(date -u +%T)] retry-later $repo/$cstep"
    fi
  done < <(find "$BASE3" -maxdepth 5 -type d -name 'checkpoint-*' 2>/dev/null)
  sleep "$INTERVAL"
done
