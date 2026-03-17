try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import logging
from pathlib import Path

from config import settings
from src.live import EventMarketFeedClient, FonbetBetExecutor, LiveBettingRuntime, RuntimeConfig


class FilteredMarketFeedClient:
    def __init__(self, inner, market_type: str | None = None):
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
    parser.add_argument("--event-id", type=int, required=True, help="Root live Fonbet event id to monitor continuously.")
    parser.add_argument("--once", action="store_true", help="Run one polling cycle and exit.")
    parser.add_argument("--interval", type=int, default=settings.live_poll_interval_seconds, help="Polling interval in seconds.")
    parser.add_argument(
        "--market-type",
        choices=["all", "match", "game", "point", "total"],
        default="all",
        help="Restrict runtime to a single market family.",
    )
    parser.add_argument(
        "--run-tag",
        default="single_event",
        help="Suffix for event-specific state/log files so multiple watchers do not collide.",
    )
    return parser


def _event_runtime_config(event_id: int, interval_seconds: int, run_tag: str) -> RuntimeConfig:
    base = RuntimeConfig.from_settings()
    suffix = f"event_{event_id}_{run_tag}"
    live_dir = settings.artifacts_dir / "live_betting"
    return RuntimeConfig(
        model_path=base.model_path,
        poll_interval_seconds=interval_seconds,
        edge_threshold=base.edge_threshold,
        min_model_probability=base.min_model_probability,
        min_odds=base.min_odds,
        max_odds=base.max_odds,
        default_stake=base.default_stake,
        bankroll=base.bankroll,
        kelly_fraction=base.kelly_fraction,
        dry_run=base.dry_run,
        state_path=live_dir / f"{suffix}_state.json",
        decisions_path=live_dir / f"{suffix}_decisions.jsonl",
        rl_snapshots_path=live_dir / f"{suffix}_rl_snapshots.jsonl",
        rl_actions_path=live_dir / f"{suffix}_rl_actions.jsonl",
        rl_outcomes_path=live_dir / f"{suffix}_rl_outcomes.jsonl",
        rl_tracker_state_path=live_dir / f"{suffix}_rl_tracker_state.json",
        rl_market_close_cycles=base.rl_market_close_cycles,
        point_trajectories_path=live_dir / f"{suffix}_point_trajectories.jsonl",
        point_fast_mode=base.point_fast_mode,
        game_target_offset=base.game_target_offset,
        point_target_offset=base.point_target_offset,
        point_model_weight=base.point_model_weight,
        point_markov_weight=base.point_markov_weight,
        point_execution_min_probability=base.point_execution_min_probability,
        bet_mode=base.bet_mode,
        express_size=base.express_size,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = _event_runtime_config(args.event_id, args.interval, args.run_tag)
    feed_client = FilteredMarketFeedClient(
        EventMarketFeedClient(args.event_id),
        None if args.market_type == "all" else args.market_type,
    )
    runtime = LiveBettingRuntime(
        market_feed_client=feed_client,
        bet_executor=FonbetBetExecutor(dry_run=config.dry_run),
        config=config,
    )

    if args.once:
        actions = runtime.run_cycle()
        for action in actions:
            print(action)
        return

    logging.info(
        "starting single-event runtime for event_id=%s market_type=%s interval=%ss dry_run=%s decisions=%s",
        args.event_id,
        args.market_type,
        args.interval,
        config.dry_run,
        Path(config.decisions_path),
    )
    runtime.run_forever()


if __name__ == "__main__":
    main()
