#!/usr/bin/env bash
set -euo pipefail

EVENT_LIST_URL="${MATCH_STATS_EVENT_LIST_URL:-https://line-lb61-w.bk6bba-resources.com/ma/events/list?lang=en&version=72526853737&scopeMarket=1600}"
SOURCE="${MATCH_STATS_SOURCE:-fonbet_sportradar}"
MAX_EVENTS="${MATCH_STATS_MAX_EVENTS:-200}"
TIMEOUT_SECONDS="${MATCH_STATS_TIMEOUT_SECONDS:-20}"

cd /opt/tennis_ai
exec /opt/tennis_ai/venv/bin/python scripts/ingest_match_player_stats.py \
  --source "${SOURCE}" \
  --event-list-url "${EVENT_LIST_URL}" \
  --only-new \
  --max-events "${MAX_EVENTS}" \
  --timeout "${TIMEOUT_SECONDS}"
