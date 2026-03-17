try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import logging
import time
from datetime import datetime, timezone

from config import settings
from src.live import (
    DatabaseFonbetEventFeedClient,
    DatabaseSnapshotRecorder,
    FileMarketFeedClient,
    FonbetFeedClient,
    SnapshotRecorder,
    SpoyerFeedClient,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one ingest cycle and exit.")
    parser.add_argument("--feed-file", help="Read normalized live markets JSON from file instead of HTTP.")
    parser.add_argument(
        "--provider",
        choices=["spoyer", "events", "fonbet"],
        default="events",
        help="Source provider for market ingestion. Default reads latest raw Fonbet snapshot from DB.",
    )
    parser.add_argument("--interval", type=int, default=settings.live_poll_interval_seconds, help="Polling interval in seconds.")
    return parser


def _feed_client(args: argparse.Namespace):
    if args.feed_file:
        return FileMarketFeedClient(args.feed_file)
    if args.provider == "spoyer":
        return SpoyerFeedClient()
    if args.provider == "events":
        return DatabaseFonbetEventFeedClient()
    return FonbetFeedClient()


def _run_cycle(feed_client, recorder: SnapshotRecorder, db_recorder: DatabaseSnapshotRecorder | None = None) -> int:
    markets = feed_client.fetch_live_markets()
    timestamp = datetime.now(timezone.utc).isoformat()
    normalized = []
    for market in markets:
        raw = dict(market.raw)
        raw["timestamp_utc"] = timestamp
        normalized.append(type(market)(**{**market.__dict__, "raw": raw}))
    written = recorder.write_markets(normalized)
    db_written = db_recorder.write_markets(normalized) if db_recorder is not None else 0
    logging.info("ingested %s live markets into %s and %s rows into live_market_snapshots", written, recorder.path, db_written)
    return written


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feed_client = _feed_client(args)
    recorder = SnapshotRecorder()
    db_recorder = DatabaseSnapshotRecorder()

    if args.once:
        _run_cycle(feed_client, recorder, db_recorder)
        return

    while True:
        _run_cycle(feed_client, recorder, db_recorder)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
