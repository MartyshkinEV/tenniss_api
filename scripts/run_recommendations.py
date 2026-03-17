try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import logging
import time
from typing import Any

from config import settings
from src.live import (
    DatabaseMarketFeedClient,
    FileMarketFeedClient,
    FonbetBetExecutor,
    FonbetFeedClient,
    LiveBettingRuntime,
    RuntimeConfig,
    SnapshotMarketFeedClient,
    SpoyerFeedClient,
    append_recommendations,
    build_recommendations,
)


class FilteredMarketFeedClient:
    def __init__(self, inner: Any, market_type: str | None = None):
        self.inner = inner
        self.market_type = market_type

    def fetch_live_markets(self):
        markets = self.inner.fetch_live_markets()
        if self.market_type == "total":
            return [market for market in markets if market.market_type == "set_total_over_under"]
        if self.market_type == "match":
            return [market for market in markets if market.market_type == "match_winner"]
        return [market for market in markets if market.market_type in {"match_winner", "set_total_over_under"}]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one recommendation cycle and exit.")
    parser.add_argument("--feed-file", help="Read normalized live markets JSON from file instead of HTTP.")
    parser.add_argument(
        "--provider",
        choices=["spoyer", "fonbet", "snapshots", "table"],
        default=settings.odds_provider,
        help="Odds provider for live tennis markets.",
    )
    parser.add_argument("--interval", type=int, default=settings.live_poll_interval_seconds, help="Polling interval in seconds.")
    parser.add_argument(
        "--market-type",
        choices=["all", "match", "total"],
        default="all",
        help="Restrict recommendations to a single market family.",
    )
    parser.add_argument("--top", type=int, default=20, help="How many top recommendations to print.")
    return parser


def _feed_client(args: argparse.Namespace):
    if args.feed_file:
        return FileMarketFeedClient(args.feed_file)
    if args.provider == "spoyer":
        return SpoyerFeedClient()
    if args.provider == "snapshots":
        return SnapshotMarketFeedClient()
    if args.provider == "table":
        return DatabaseMarketFeedClient()
    return FonbetFeedClient()


def _run_cycle(runtime: LiveBettingRuntime, feed_client: Any, top_n: int) -> list[dict[str, Any]]:
    recommendations = build_recommendations(runtime, feed_client.fetch_live_markets())
    append_recommendations(settings.live_recommendations_path, recommendations)
    logging.info("wrote %s recommendations into %s", len(recommendations), settings.live_recommendations_path)
    return recommendations[:top_n]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = RuntimeConfig.from_settings()
    feed_client = FilteredMarketFeedClient(_feed_client(args), None if args.market_type == "all" else args.market_type)
    runtime = LiveBettingRuntime(
        market_feed_client=feed_client,
        bet_executor=FonbetBetExecutor(dry_run=True),
        config=config,
    )
    if args.once:
        for recommendation in _run_cycle(runtime, feed_client, args.top):
            print(recommendation)
        return
    while True:
        for recommendation in _run_cycle(runtime, feed_client, args.top):
            print(recommendation)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
