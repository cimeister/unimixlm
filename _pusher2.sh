#!/usr/bin/env bash
# _pusher2.sh — Phase-2 durable checkpoint pusher. Independent of the
# live Phase-1 _pusher.sh: it scans ONLY $SHM/ft_models2 and rebuilds the
# repo name (with size + -<loratag> suffix) from the path, so it can
# never mis-push into the Phase-1 100M r16 repos. Idempotent via a
# .pushed marker per checkpoint.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
source .venv/bin/activate
BASE2="$SHM/ft_models2"
INTERVAL="${PUSH_INTERVAL:-75}"
mkdir -p "$BASE2"
echo "[pusher2] started; scanning $BASE2 every ${INTERVAL}s"
while true; do
  while IFS= read -r ck; do
    [ -d "$ck" ] || continue
    [ -f "$ck/.pushed" ] && continue
    # stable? nothing written in the last 45s
    if [ -n "$(find "$ck" -type f -newermt '-45 seconds' 2>/dev/null | head -1)" ]; then
      continue
    fi
    run_dir="$(dirname "$ck")"                 # .../ft_<lang>-eng_<size>
    loratag="$(basename "$(dirname "$run_dir")")"
    model="$(basename "$(dirname "$(dirname "$run_dir")")")"
    base="$(basename "$run_dir")"; base="${base#ft_}"   # <lang>-eng_<size>
    lang="${base%%-*}"
    size="${base##*_}"
    repo="${HF_ORG}/ft-langmap-${model}-fineweb${size}-${lang}-original-${loratag}"
    cstep="$(basename "$ck")"
    echo "[pusher2][$(date -u +%T)] $model/$loratag/$lang/$size $cstep -> $repo"
    hf repo create "$repo" --repo-type model -y >/dev/null 2>&1 || true
    if hf upload "$repo" "$ck" "$cstep" --repo-type model \
         >>"$SHM/logs/pusher2.log" 2>&1; then
      touch "$ck/.pushed"
      echo "[pusher2][$(date -u +%T)] OK $repo/$cstep"
    else
      echo "[pusher2][$(date -u +%T)] retry-later $repo/$cstep"
    fi
  done < <(find "$BASE2" -maxdepth 4 -type d -name 'checkpoint-*' 2>/dev/null)
  sleep "$INTERVAL"
done
