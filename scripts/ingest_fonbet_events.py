try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import logging
import time
from datetime import datetime, timezone

from config import settings
from src.live.fonbet import DatabaseFonbetEventRecorder, extract_fonbet_events, FonbetEventsClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one ingest cycle and exit.")
    parser.add_argument("--interval", type=int, default=settings.live_poll_interval_seconds, help="Polling interval in seconds.")
    parser.add_argument("--url", default=settings.fonbet_feed_url, help="Fonbet events/list URL.")
    return parser


def _run_cycle(client: FonbetEventsClient, recorder: DatabaseFonbetEventRecorder) -> int:
    snapshot_utc = datetime.now(timezone.utc).isoformat()
    requested_url, payload, response_headers = client.fetch_payload()
    events = extract_fonbet_events(payload)
    tennis_events = [event for event in events if event["root_sport_id"] == 4]
    live_tennis_events = [event for event in tennis_events if event["place"] == "live"]
    snapshot_id, written = recorder.write_payload(
        payload=payload,
        requested_url=requested_url,
        response_headers=response_headers,
        snapshot_utc=snapshot_utc,
    )
    logging.info(
        "snapshot_id=%s events=%s tennis=%s live_tennis=%s url=%s",
        snapshot_id,
        written,
        len(tennis_events),
        len(live_tennis_events),
        requested_url,
    )
    return written


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = FonbetEventsClient(feed_url=args.url)
    recorder = DatabaseFonbetEventRecorder()

    if args.once:
        _run_cycle(client, recorder)
        return

    while True:
        _run_cycle(client, recorder)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
