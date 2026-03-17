try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from config import settings
from src.db.engine import get_engine
from src.live import FonbetBetExecutor, FonbetEventDetailsClient, LiveBettingRuntime, RuntimeConfig
from src.live.fonbet import extract_markets_from_fonbet_catalog


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", type=int, action="append", help="Specific root event id to watch. Can be repeated.")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval per event in seconds.")
    parser.add_argument("--top", type=int, default=1, help="How many best candidates to keep per event per cycle.")
    parser.add_argument("--autobet", action="store_true", help="Place the top candidate when runtime produces one.")
    parser.add_argument("--once", action="store_true", help="Run one cycle per event and exit.")
    parser.add_argument(
        "--output",
        default=str(settings.artifacts_dir / "live_betting" / "event_ml_watch.jsonl"),
        help="JSONL file for per-event ML analysis records.",
    )
    return parser


def _current_live_event_ids() -> list[int]:
    engine = get_engine()
    query = text(
        """
        SELECT event_id
        FROM fonbet_tennis_live_events_latest
        WHERE level = 1
        ORDER BY event_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [int(row[0]) for row in rows]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _candidate_payload(event_id: int, market: Any, candidate: Any, player1_probability: float, player2_probability: float, features: dict[str, Any]) -> dict[str, Any]:
    selected_factor_id = market.raw.get("player1_factor_id") if candidate.side == "player1" else market.raw.get("player2_factor_id")
    selected_param = market.raw.get("player1_param") if candidate.side == "player1" else market.raw.get("player2_param")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        "market_event_id": market.event_id,
        "market_id": market.market_id,
        "market_type": market.market_type,
        "competition": market.competition,
        "round_name": market.round_name,
        "player1_name": market.player1_name,
        "player2_name": market.player2_name,
        "live_score": market.raw.get("score"),
        "live_comment": market.raw.get("comment"),
        "serving_team": market.raw.get("serveT"),
        "selection_id": candidate.selection_id,
        "selection_side": candidate.side,
        "selection_name": candidate.player_name,
        "odds": candidate.odds,
        "edge": candidate.edge,
        "stake": candidate.stake,
        "model_probability": candidate.model_probability,
        "implied_probability": candidate.implied_probability,
        "acceptance_probability": getattr(candidate, "acceptance_probability", 1.0),
        "ranking_score": getattr(candidate, "ranking_score", candidate.edge),
        "factor_id": selected_factor_id,
        "param": selected_param,
        "player1_probability": player1_probability,
        "player2_probability": player2_probability,
        "state_features": features,
    }


def _score_event(runtime: LiveBettingRuntime, payload: dict[str, Any], root_event_id: int, top_n: int) -> list[dict[str, Any]]:
    markets = extract_markets_from_fonbet_catalog(payload)
    candidates: list[dict[str, Any]] = []
    for market in markets:
        try:
            _, features, player1_probability, player2_probability, candidate = runtime._score_market_details(market)
        except Exception as exc:
            candidates.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "event_id": root_event_id,
                    "market_id": market.market_id,
                    "market_type": market.market_type,
                    "status": "score_failed",
                    "reason": str(exc),
                }
            )
            continue
        if candidate is None:
            continue
        candidates.append(_candidate_payload(root_event_id, market, candidate, player1_probability, player2_probability, features))
    candidates.sort(key=lambda item: float(item.get("ranking_score") or 0.0), reverse=True)
    return candidates[:top_n]


async def _watch_event(
    event_id: int,
    runtime: LiveBettingRuntime,
    client: FonbetEventDetailsClient,
    output_path: Path,
    interval_seconds: float,
    top_n: int,
    autobet: bool,
    once: bool,
) -> None:
    while True:
        started = datetime.now(timezone.utc).isoformat()
        try:
            url, payload, _headers = await asyncio.to_thread(client.fetch_payload, event_id, 0)
            candidates = await asyncio.to_thread(_score_event, runtime, payload, event_id, top_n)
            if not candidates:
                _append_jsonl(
                    output_path,
                    {
                        "timestamp_utc": started,
                        "event_id": event_id,
                        "status": "no_candidate",
                        "source_url": url,
                    },
                )
            else:
                for index, candidate in enumerate(candidates):
                    candidate["source_url"] = url
                    candidate["event_rank"] = index + 1
                    candidate["status"] = "candidate"
                    _append_jsonl(output_path, candidate)
                if autobet:
                    # Reuse the existing runtime flow on the top candidate's market/candidate pair once the watcher proves stable.
                    LOGGER.warning("autobet requested for event %s but automatic placement is not wired in this watcher yet", event_id)
        except Exception as exc:
            _append_jsonl(
                output_path,
                {
                    "timestamp_utc": started,
                    "event_id": event_id,
                    "status": "event_failed",
                    "reason": str(exc),
                },
            )
            LOGGER.warning("event %s failed: %s", event_id, exc)
        if once:
            return
        await asyncio.sleep(interval_seconds)


async def _main_async(args: argparse.Namespace) -> None:
    event_ids = args.event_id or _current_live_event_ids()
    if not event_ids:
        raise SystemExit("No live tennis event ids found to watch")

    output_path = Path(args.output).resolve()
    config = RuntimeConfig.from_settings()
    runtime = LiveBettingRuntime(
        market_feed_client=None,
        bet_executor=FonbetBetExecutor(dry_run=not args.autobet),
        config=config,
    )
    client = FonbetEventDetailsClient()
    tasks = [
        asyncio.create_task(
            _watch_event(
                event_id=event_id,
                runtime=runtime,
                client=client,
                output_path=output_path,
                interval_seconds=args.interval,
                top_n=args.top,
                autobet=args.autobet,
                once=args.once,
            )
        )
        for event_id in event_ids
    ]
    await asyncio.gather(*tasks)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
