try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import logging
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
)


class FilteredMarketFeedClient:
    def __init__(self, inner: Any, market_type: str | None = None):
        self.inner = inner
        self.market_type = market_type

    def fetch_live_markets(self):
        markets = self.inner.fetch_live_markets()
        if self.market_type == "point":
            return [market for market in markets if market.market_type == "point_plus_one_winner"]
        if self.market_type == "total":
            return [market for market in markets if market.market_type == "set_total_over_under"]
        if self.market_type == "game":
            return [market for market in markets if market.market_type == "next_game_winner"]
        if self.market_type == "match":
            return [market for market in markets if market.market_type == "match_winner"]
        return markets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one polling cycle and exit.")
    parser.add_argument("--feed-file", help="Read normalized live markets JSON from file instead of HTTP.")
    parser.add_argument(
        "--provider",
        choices=["spoyer", "fonbet", "snapshots", "table"],
        default=settings.odds_provider,
        help="Odds provider for live tennis markets.",
    )
    parser.add_argument("--interval", type=int, help="Polling interval override in seconds.")
    parser.add_argument(
        "--market-type",
        choices=["all", "match", "game", "point", "total"],
        default="all",
        help="Restrict runtime to a single market family.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = RuntimeConfig.from_settings()
    if args.interval:
        config = RuntimeConfig(
            model_path=config.model_path,
            poll_interval_seconds=args.interval,
            edge_threshold=config.edge_threshold,
            min_model_probability=config.min_model_probability,
            min_odds=config.min_odds,
            max_odds=config.max_odds,
            default_stake=config.default_stake,
            bankroll=config.bankroll,
            kelly_fraction=config.kelly_fraction,
            dry_run=config.dry_run,
            state_path=config.state_path,
            decisions_path=config.decisions_path,
            rl_snapshots_path=config.rl_snapshots_path,
            rl_actions_path=config.rl_actions_path,
            rl_outcomes_path=config.rl_outcomes_path,
            rl_tracker_state_path=config.rl_tracker_state_path,
            rl_market_close_cycles=config.rl_market_close_cycles,
            point_trajectories_path=config.point_trajectories_path,
            point_fast_mode=config.point_fast_mode,
            game_target_offset=config.game_target_offset,
            point_target_offset=config.point_target_offset,
            bet_mode=config.bet_mode,
            express_size=config.express_size,
        )

    if args.feed_file:
        feed_client = FileMarketFeedClient(args.feed_file)
    elif args.provider == "spoyer":
        feed_client = SpoyerFeedClient()
    elif args.provider == "snapshots":
        feed_client = SnapshotMarketFeedClient()
    elif args.provider == "table":
        feed_client = DatabaseMarketFeedClient()
    else:
        feed_client = FonbetFeedClient()
    feed_client = FilteredMarketFeedClient(feed_client, None if args.market_type == "all" else args.market_type)
    bet_executor = FonbetBetExecutor(dry_run=config.dry_run)
    runtime = LiveBettingRuntime(
        market_feed_client=feed_client,
        bet_executor=bet_executor,
        config=config,
    )

    if args.once:
        actions = runtime.run_cycle()
        for action in actions:
            print(action)
        return
    runtime.run_forever()


if __name__ == "__main__":
    main()
