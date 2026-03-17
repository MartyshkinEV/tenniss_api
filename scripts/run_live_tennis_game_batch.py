try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sqlalchemy import text

from config import settings
from src.db.engine import get_engine
from src.live import FonbetBetExecutor, FonbetEventDetailsClient, FonbetEventsClient, LiveBettingRuntime, RuntimeConfig
from src.live.fonbet import extract_fonbet_events, extract_markets_from_fonbet_catalog


LOGGER = logging.getLogger(__name__)


class LiveTennisEventBatchFeedClient:
    def __init__(self, max_workers: int = 8, event_source: str = "api"):
        self.engine = get_engine() if event_source == "db" else None
        self.events_client = FonbetEventsClient() if event_source == "api" else None
        self.details_client = FonbetEventDetailsClient()
        self.max_workers = max(1, int(max_workers))
        self.event_source = event_source

    def _current_live_event_ids(self) -> list[int]:
        if self.event_source == "api":
            _url, payload, _headers = self.events_client.fetch_payload()
            rows = extract_fonbet_events(payload)
            return [
                int(row["event_id"])
                for row in rows
                if int(row.get("root_sport_id") or 0) == 4
                and str(row.get("place") or "") == "live"
                and int(row.get("level") or 0) == 1
            ]

        query = text(
            """
            SELECT event_id
            FROM fonbet_tennis_live_events_latest
            WHERE level = 1
            ORDER BY event_id
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).fetchall()
        return [int(row[0]) for row in rows]

    def _fetch_event_markets(self, event_id: int):
        _url, payload, _headers = self.details_client.fetch_payload(event_id, 0)
        return [
            market
            for market in extract_markets_from_fonbet_catalog(payload)
            if market.market_type == "next_game_winner"
        ]

    def fetch_live_markets(self):
        event_ids = self._current_live_event_ids()
        if not event_ids:
            return []

        markets = []
        worker_count = min(self.max_workers, len(event_ids))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(self._fetch_event_markets, event_id): event_id for event_id in event_ids}
            for future in as_completed(future_map):
                event_id = future_map[future]
                try:
                    markets.extend(future.result())
                except Exception as exc:
                    LOGGER.warning("event %s fetch failed: %s", event_id, exc)
        return markets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one batch polling cycle and exit.")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds.")
    parser.add_argument("--run-tag", default="live_tennis_games", help="Suffix for batch state/log files.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel fetch worker count for event details.")
    parser.add_argument(
        "--event-source",
        choices=["api", "db"],
        default="api",
        help="Source of current live tennis root event ids.",
    )
    parser.add_argument("--real", action="store_true", help="Place real bets instead of dry-run.")
    parser.add_argument("--bankroll", type=float, default=None, help="Override runtime bankroll.")
    parser.add_argument("--default-stake", type=float, default=None, help="Override base stake before bandit sizing.")
    return parser


def _batch_runtime_config(run_tag: str, interval_seconds: int, bankroll: float | None, default_stake: float | None, dry_run: bool) -> RuntimeConfig:
    base = RuntimeConfig.from_settings()
    suffix = f"batch_{run_tag}"
    live_dir = settings.artifacts_dir / "live_betting"
    return RuntimeConfig(
        model_path=base.model_path,
        poll_interval_seconds=interval_seconds,
        edge_threshold=base.edge_threshold,
        min_model_probability=base.min_model_probability,
        min_odds=base.min_odds,
        max_odds=base.max_odds,
        default_stake=base.default_stake if default_stake is None else float(default_stake),
        bankroll=base.bankroll if bankroll is None else float(bankroll),
        kelly_fraction=base.kelly_fraction,
        dry_run=dry_run,
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
        game_model_weight=base.game_model_weight,
        game_markov_weight=base.game_markov_weight,
        point_target_offset=base.point_target_offset,
        point_model_weight=base.point_model_weight,
        point_markov_weight=base.point_markov_weight,
        point_execution_min_probability=base.point_execution_min_probability,
        bet_mode="single",
        express_size=base.express_size,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = _batch_runtime_config(
        run_tag=args.run_tag,
        interval_seconds=args.interval,
        bankroll=args.bankroll,
        default_stake=args.default_stake,
        dry_run=not args.real,
    )
    feed_client = LiveTennisEventBatchFeedClient(max_workers=args.workers, event_source=args.event_source)
    runtime = LiveBettingRuntime(
        market_feed_client=feed_client,
        bet_executor=FonbetBetExecutor(dry_run=config.dry_run),
        config=config,
    )

    logging.info(
        "starting live tennis batch runner dry_run=%s bankroll=%s default_stake=%s interval=%ss run_tag=%s event_source=%s decisions=%s",
        config.dry_run,
        config.bankroll,
        config.default_stake,
        config.poll_interval_seconds,
        args.run_tag,
        args.event_source,
        Path(config.decisions_path),
    )

    if args.once:
        for action in runtime.run_cycle():
            print(action)
        return

    while True:
        runtime.run_cycle()
        time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":
    main()
