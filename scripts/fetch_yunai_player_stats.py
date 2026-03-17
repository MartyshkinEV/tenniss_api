try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json

from src.live.yunai_stats import (
    DatabaseMatchStatsRecorder,
    fetch_fonbet_sportradar_stats,
    fetch_match_stats,
    fetch_match_stats_from_url,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch latest player stats from supported match-stat sources by match event.")
    parser.add_argument("--event-id", type=int, help="Fonbet event id, e.g. 63295539.")
    parser.add_argument("--url", help="Explicit yunai URL. If omitted, --event-id is used.")
    parser.add_argument(
        "--source",
        choices=["yunai", "fonbet_sportradar"],
        default="yunai",
        help="Source URL template to use with --event-id.",
    )
    parser.add_argument("--write-db", action="store_true", help="Persist fetched stats into PostgreSQL.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.url and not args.event_id:
        raise SystemExit("Provide either --event-id or --url")

    if args.url:
        result = fetch_match_stats_from_url(args.url, timeout=args.timeout, event_id=args.event_id)
    elif args.source == "fonbet_sportradar":
        result = fetch_fonbet_sportradar_stats(args.event_id, timeout=args.timeout)
    else:
        result = fetch_match_stats(args.event_id, timeout=args.timeout)

    payload = {
        "requested_url": result.requested_url,
        "resolved_url": result.resolved_url,
        "event_id": result.event_id,
        "status": result.status,
        "message": result.message,
        "players": result.players,
        "stats": result.stats,
        "normalized_player_stats": result.normalized_player_stats,
        "candidate_count": result.candidate_count,
    }
    if args.write_db:
        payload["db_written"] = DatabaseMatchStatsRecorder().write_result(args.source, result)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
