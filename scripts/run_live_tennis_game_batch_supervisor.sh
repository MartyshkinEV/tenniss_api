#!/usr/bin/env bash

set -u

ROOT_DIR="/opt/tennis_ai"
LOG_FILE="$ROOT_DIR/artifacts/live_betting/bankroll900_games_real_diag_supervisor.out"

cd "$ROOT_DIR" || exit 1

while true; do
  printf '[%s] starting game runner\n' "$(date -Iseconds)" >> "$LOG_FILE"
  ./venv/bin/python scripts/run_live_tennis_game_batch.py \
    --interval 5 \
    --run-tag bankroll900_games_real_diag \
    --bankroll 900 \
    --default-stake 30 \
    --workers 6 \
    --real >> "$LOG_FILE" 2>&1
  rc=$?
  printf '[%s] game runner exited rc=%s, restarting in 5s\n' "$(date -Iseconds)" "$rc" >> "$LOG_FILE"
  sleep 5
done
