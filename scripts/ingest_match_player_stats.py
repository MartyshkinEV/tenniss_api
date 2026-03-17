try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import gzip
import json
from urllib.request import Request, urlopen

from sqlalchemy import text

from src.db.engine import get_engine
from src.live.fonbet import extract_markets_from_fonbet_catalog
from src.live.yunai_stats import (
    DatabaseMatchStatsRecorder,
    fetch_fonbet_sportradar_stats,
    fetch_match_stats,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch live player stats and persist them into PostgreSQL.")
    parser.add_argument(
        "--source",
        choices=["yunai", "fonbet_sportradar"],
        default="yunai",
        help="Source URL template to use for each event id.",
    )
    parser.add_argument("--event-id", type=int, action="append", dest="event_ids", help="Explicit event id to fetch.")
    parser.add_argument(
        "--event-list-url",
        help="Fonbet events/list URL to auto-discover event ids from a frozen or current tennis feed.",
    )
    parser.add_argument(
        "--from-live-db",
        action="store_true",
        help="Load candidate event ids from live_market_snapshots.",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Skip event ids that already exist in live_match_player_stats for the selected source.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=200,
        help="Maximum number of event ids to fetch from live_market_snapshots or --event-list-url.",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds.")
    return parser


def load_live_event_ids(limit: int) -> list[int]:
    engine = get_engine()
    query = text(
        """
        SELECT DISTINCT event_id
        FROM live_market_snapshots
        WHERE timestamp_utc >= NOW() - INTERVAL '6 hours'
        ORDER BY event_id
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()
    event_ids: list[int] = []
    for row in rows:
        try:
            event_ids.append(int(row[0]))
        except (TypeError, ValueError):
            continue
    return event_ids


def load_event_ids_from_feed_url(url: str, timeout: float, limit: int) -> list[int]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        encoding = ""
        headers = getattr(response, "headers", None)
        if headers is not None:
            encoding = str(headers.get("Content-Encoding", "")).lower()
        if encoding == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        payload = json.loads(raw.decode("utf-8"))
    markets = extract_markets_from_fonbet_catalog(payload)
    event_ids: list[int] = []
    for market in markets:
        try:
            event_id = int(market.event_id)
        except (TypeError, ValueError):
            continue
        event_ids.append(event_id)
    deduped = sorted(dict.fromkeys(event_ids))
    return deduped[:limit]


def load_existing_event_ids(source: str) -> set[int]:
    engine = get_engine()
    query = text(
        """
        SELECT DISTINCT event_id
        FROM live_match_player_stats
        WHERE source = :source
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(query, {"source": source}).fetchall()
    event_ids: set[int] = set()
    for row in rows:
        try:
            event_ids.add(int(row[0]))
        except (TypeError, ValueError):
            continue
    return event_ids


def fetch_one(source: str, event_id: int, timeout: float):
    if source == "fonbet_sportradar":
        return fetch_fonbet_sportradar_stats(event_id, timeout=timeout)
    return fetch_match_stats(event_id, timeout=timeout)


def main() -> None:
    args = build_parser().parse_args()
    event_ids = list(args.event_ids or [])
    if args.from_live_db:
        event_ids.extend(load_live_event_ids(args.max_events))
    if args.event_list_url:
        event_ids.extend(load_event_ids_from_feed_url(args.event_list_url, args.timeout, args.max_events))
    deduped_event_ids = sorted(dict.fromkeys(event_ids))
    if not deduped_event_ids:
        raise SystemExit("No event ids provided. Use --event-id, --from-live-db or --event-list-url.")
    skipped_existing = 0
    if args.only_new:
        existing_event_ids = load_existing_event_ids(args.source)
        before = len(deduped_event_ids)
        deduped_event_ids = [event_id for event_id in deduped_event_ids if event_id not in existing_event_ids]
        skipped_existing = before - len(deduped_event_ids)
    if not deduped_event_ids:
        print(json.dumps({"source": args.source, "written": 0, "skipped_existing": skipped_existing, "results": []}, ensure_ascii=True, indent=2))
        return

    recorder = DatabaseMatchStatsRecorder()
    results = []
    for event_id in deduped_event_ids:
        result = fetch_one(args.source, event_id, args.timeout)
        recorder.write_result(args.source, result)
        results.append(
            {
                "event_id": event_id,
                "status": result.status,
                "message": result.message,
                "players": result.players,
                "candidate_count": result.candidate_count,
                "resolved_url": result.resolved_url,
            }
        )
    print(
        json.dumps(
            {
                "source": args.source,
                "written": len(results),
                "skipped_existing": skipped_existing,
                "results": results,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
