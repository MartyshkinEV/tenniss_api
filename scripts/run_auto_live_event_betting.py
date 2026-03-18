try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import asyncio
import gzip
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from datetime import timedelta

from config import settings
from src.live.fonbet import FonbetBetExecutor, FonbetEventsClient, extract_fonbet_events, extract_markets_from_fonbet_catalog
from src.live.runtime import LiveBettingRuntime, RuntimeConfig


LOGGER = logging.getLogger(__name__)
AGGRESSIVE_MARKET_TYPES = {
    "next_game_winner",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=10.0, help="How often to refresh live root events.")
    parser.add_argument(
        "--max-events-per-cycle",
        type=int,
        default=0,
        help="How many live events to process in one cycle. Use 0 to process all live events.",
    )
    parser.add_argument(
        "--max-bets-per-cycle",
        type=int,
        default=5,
        help="How many top-ranked bets can be placed in a single cycle.",
    )
    parser.add_argument(
        "--target-active-bets",
        type=int,
        default=10,
        help="Try to keep at least this many active bets by increasing placement budget when possible.",
    )
    parser.add_argument(
        "--min-acceptance-probability",
        type=float,
        default=0.50,
        help="Minimum acceptance probability required for a candidate to be considered valid.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count for scoring live events.",
    )
    parser.add_argument(
        "--journal",
        default="artifacts/live_betting/auto_live_event_betting.jsonl",
        help="Batch orchestrator journal path.",
    )
    parser.add_argument(
        "--state-path",
        default="artifacts/live_betting/auto_live_event_betting_state.json",
        help="Persistent duplicate-protection state.",
    )
    parser.add_argument(
        "--training-log",
        default="artifacts/live_betting/auto_live_training_queue.jsonl",
        help="Append-only log for continual ML/RL dataset growth.",
    )
    parser.add_argument(
        "--max-total-exposure",
        type=float,
        default=2000.0,
        help="Maximum total open stake exposure in RUB across active bets.",
    )
    parser.add_argument(
        "--active-bet-ttl-minutes",
        type=float,
        default=6.0,
        help="How long to keep an open bet in exposure accounting if no settlement signal is available.",
    )
    parser.add_argument(
        "--duplicate-key-ttl-minutes",
        type=float,
        default=8.0,
        help="How long to block exact duplicate match/game/side bets.",
    )
    parser.add_argument(
        "--retrain-every-minutes",
        type=float,
        default=30.0,
        help="Periodic retraining interval for live decision side-models. Use 0 to disable.",
    )
    parser.add_argument(
        "--retrain-min-rows",
        type=int,
        default=200,
        help="Minimum training-log rows required before automatic retraining runs.",
    )
    parser.add_argument("--send-bet", action="store_true", help="Actually place bets instead of dry-run logging.")
    return parser


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"placed_selection_keys": [], "recent_selection_keys": [], "active_bets": [], "event_cursor": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"placed_selection_keys": [], "recent_selection_keys": [], "active_bets": [], "event_cursor": 0}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def _live_root_event_ids() -> list[int]:
    _url, payload, _headers = FonbetEventsClient().fetch_payload()
    rows = extract_fonbet_events(payload)
    return sorted(
        {
            int(row["event_id"])
            for row in rows
            if int(row.get("root_sport_id") or 0) == 4
            and str(row.get("place") or "") == "live"
            and int(row.get("level") or 0) == 1
        }
    )


def _load_event_payload(event_id: int) -> dict:
    url = (
        f"https://line-lb61-w.bk6bba-resources.com/ma/events/event"
        f"?lang={settings.fonbet_lang}&version=0&eventId={event_id}&scopeMarket={settings.fonbet_scope_market_id}"
    )
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=settings.fonbet_timeout_seconds) as response:
        raw = response.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _score_event(event_id: int, runtime: LiveBettingRuntime, min_acceptance_probability: float) -> dict:
    payload = _load_event_payload(event_id)
    markets = [
        market
        for market in extract_markets_from_fonbet_catalog(payload)
        if market.market_type in AGGRESSIVE_MARKET_TYPES
    ]
    accepted_market_candidates = 0
    ranked_candidates: list[dict] = []
    for market in markets:
        candidate = runtime.score_market(market)
        if candidate is None:
            continue
        if float(candidate.acceptance_probability) < min_acceptance_probability:
            continue
        accepted_market_candidates += 1
        player1_probability = candidate.model_probability if candidate.side == "player1" else 1.0 - candidate.model_probability
        player2_probability = 1.0 - player1_probability
        ranked_candidates.append(
            {
                "event_id": event_id,
                "status": "candidate",
                "candidate": candidate,
                "market": market,
                "market_id": market.market_id,
                "market_type": market.market_type,
                "target_game_number": market.raw.get("target_game_number"),
                "selection_side": candidate.side,
                "selection_name": candidate.player_name,
                "stake": candidate.stake,
                "odds_before_refresh": candidate.odds,
                "model_probability": candidate.model_probability,
                "implied_probability": candidate.implied_probability,
                "edge": candidate.edge,
                "acceptance_probability": candidate.acceptance_probability,
                "ranking_score": candidate.ranking_score,
                "player1_probability": player1_probability,
                "player2_probability": player2_probability,
            }
        )
    if not ranked_candidates:
        return {
            "event_id": event_id,
            "status": "no_candidate",
            "reason": "no_valid_market_after_scoring",
            "market_count": len(markets),
            "accepted_market_candidates": accepted_market_candidates,
        }
    return {
        "event_id": event_id,
        "status": "candidates",
        "market_count": len(markets),
        "accepted_market_candidates": accepted_market_candidates,
        "candidates": ranked_candidates,
    }


def _selection_key(candidate) -> str:
    raw = candidate.market.raw
    factor_id = raw.get("player1_factor_id") if candidate.side == "player1" else raw.get("player2_factor_id")
    param = raw.get("player1_param") if candidate.side == "player1" else raw.get("player2_param")
    return f"{candidate.market.market_id}:{candidate.side}:{factor_id}:{param}"


def _prune_active_bets(active_bets: list[dict], current_event_ids: set[int], ttl_minutes: float) -> list[dict]:
    now = datetime.now(timezone.utc)
    ttl = timedelta(minutes=max(ttl_minutes, 1.0))
    kept: list[dict] = []
    for item in active_bets:
        event_id = int(item.get("event_id") or 0)
        placed_at_raw = item.get("placed_at_utc")
        try:
            placed_at = datetime.fromisoformat(str(placed_at_raw))
        except Exception:
            placed_at = now
        if placed_at.tzinfo is None:
            placed_at = placed_at.replace(tzinfo=timezone.utc)
        if event_id not in current_event_ids:
            continue
        if now - placed_at > ttl:
            continue
        kept.append(item)
    return kept


def _current_exposure(active_bets: list[dict]) -> float:
    return round(sum(float(item.get("stake") or 0.0) for item in active_bets), 2)


def _selection_keys_from_active_bets(active_bets: list[dict]) -> set[str]:
    return {
        str(item["selection_key"])
        for item in active_bets
        if item.get("selection_key")
    }


def _prune_recent_selection_keys(items: list[dict], ttl_minutes: float) -> list[dict]:
    now = datetime.now(timezone.utc)
    ttl = timedelta(minutes=max(ttl_minutes, 1.0))
    kept: list[dict] = []
    for item in items:
        placed_at_raw = item.get("placed_at_utc")
        try:
            placed_at = datetime.fromisoformat(str(placed_at_raw))
        except Exception:
            placed_at = now
        if placed_at.tzinfo is None:
            placed_at = placed_at.replace(tzinfo=timezone.utc)
        if now - placed_at > ttl:
            continue
        kept.append(item)
    return kept


def _selection_keys_from_recent(recent_items: list[dict]) -> set[str]:
    return {
        str(item["selection_key"])
        for item in recent_items
        if item.get("selection_key")
    }


def _training_log_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _run_periodic_retrain(
    journal_path: Path,
    timestamp_utc: str,
    training_log_path: Path,
    min_rows: int,
) -> tuple[bool, str]:
    rows = _training_log_rows(training_log_path)
    if rows < min_rows:
        return False, f"not_enough_rows:{rows}"
    _log_cycle_event(
        journal_path,
        timestamp_utc,
        "retrain_started",
        training_rows=rows,
        commands=[
            "scripts/train_leg_acceptance_model.py",
            "scripts/train_rl_policy.py",
        ],
    )
    python_bin = Path(sys.executable)
    commands = [
        [str(python_bin), str(Path("/opt/tennis_ai/scripts/train_leg_acceptance_model.py"))],
        [str(python_bin), str(Path("/opt/tennis_ai/scripts/train_rl_policy.py"))],
    ]
    try:
        for cmd in commands:
            subprocess.run(cmd, check=True, cwd="/opt/tennis_ai", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        _log_cycle_event(
            journal_path,
            timestamp_utc,
            "retrain_failed",
            training_rows=rows,
            reason=str(exc),
        )
        return False, str(exc)
    _log_cycle_event(
        journal_path,
        timestamp_utc,
        "retrain_finished",
        training_rows=rows,
    )
    return True, "ok"


def _log_cycle_event(path: Path, timestamp_utc: str, action: str, **payload: object) -> None:
    _append_jsonl(
        path,
        {
            "timestamp_utc": timestamp_utc,
            "action": action,
            **payload,
        },
    )


def _journal_event_payload(payload: dict) -> dict:
    fields = (
        "event_id",
        "status",
        "reason",
        "traceback",
        "market_id",
        "market_type",
        "target_game_number",
        "selection_key",
        "selection_side",
        "selection_name",
        "stake",
        "odds_before_refresh",
        "model_probability",
        "implied_probability",
        "edge",
        "acceptance_probability",
        "ranking_score",
        "player1_probability",
        "player2_probability",
        "market_count",
        "accepted_market_candidates",
        "candidate_count",
        "current_exposure",
        "max_total_exposure",
        "refreshed_selection",
        "bet_slip_info_response",
        "bet_execution",
    )
    compact = {field: payload.get(field) for field in fields if field in payload}
    if payload.get("status") == "candidates":
        compact["candidates"] = [
            {
                "market_id": item.get("market_id"),
                "market_type": item.get("market_type"),
                "target_game_number": item.get("target_game_number"),
                "selection_side": item.get("selection_side"),
                "selection_name": item.get("selection_name"),
                "stake": item.get("stake"),
                "odds_before_refresh": item.get("odds_before_refresh"),
                "model_probability": item.get("model_probability"),
                "implied_probability": item.get("implied_probability"),
                "edge": item.get("edge"),
                "acceptance_probability": item.get("acceptance_probability"),
                "ranking_score": item.get("ranking_score"),
                "player1_probability": item.get("player1_probability"),
                "player2_probability": item.get("player2_probability"),
            }
            for item in (payload.get("candidates") or [])
        ]
    return compact


def _ranking_tuple(payload: dict) -> tuple[float, float, float]:
    return (
        float(payload.get("ranking_score") or 0.0),
        float(payload.get("acceptance_probability") or 0.0),
        float(payload.get("edge") or 0.0),
    )


def _top_ranked(payloads: list[dict], limit: int | None = None) -> list[dict]:
    decorated = [
        (_ranking_tuple(payload), index, payload)
        for index, payload in enumerate(payloads)
    ]
    decorated.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    ordered = [payload for _ranking, _index, payload in decorated]
    if limit is None or limit <= 0:
        return ordered
    return ordered[:limit]


def _process_event(
    scored: dict,
    executor: FonbetBetExecutor,
    placed_selection_keys: set[str],
    current_exposure: float,
    max_total_exposure: float,
) -> dict:
    candidate = scored["candidate"]
    selection_key = _selection_key(candidate)
    if selection_key in placed_selection_keys:
        return {
            "event_id": scored["event_id"],
            "status": "duplicate_skipped",
            "selection_key": selection_key,
            "market_id": candidate.market.market_id,
            "selection_side": candidate.side,
            "selection_name": candidate.player_name,
            "acceptance_probability": scored.get("acceptance_probability"),
            "ranking_score": scored.get("ranking_score"),
        }
    if current_exposure + float(candidate.stake) > max_total_exposure:
        return {
            "event_id": scored["event_id"],
            "status": "exposure_blocked",
            "selection_key": selection_key,
            "market_id": candidate.market.market_id,
            "selection_side": candidate.side,
            "selection_name": candidate.player_name,
            "stake": candidate.stake,
            "current_exposure": current_exposure,
            "max_total_exposure": max_total_exposure,
            "acceptance_probability": scored.get("acceptance_probability"),
            "ranking_score": scored.get("ranking_score"),
        }
    refreshed_selection, slip_info_response = executor.refresh_candidate(candidate)
    result = executor.place_prepared_bet(candidate, refreshed_selection, slip_info_response)
    return {
        "event_id": scored["event_id"],
        "status": result.get("status"),
        "selection_key": selection_key,
        "market_id": candidate.market.market_id,
        "market_type": candidate.market.market_type,
        "target_game_number": candidate.market.raw.get("target_game_number"),
        "selection_side": candidate.side,
        "selection_name": candidate.player_name,
        "stake": candidate.stake,
        "odds_before_refresh": candidate.odds,
        "model_probability": candidate.model_probability,
        "implied_probability": candidate.implied_probability,
        "edge": candidate.edge,
        "acceptance_probability": candidate.acceptance_probability,
        "ranking_score": candidate.ranking_score,
        "player1_probability": scored["player1_probability"],
        "player2_probability": scored["player2_probability"],
        "refreshed_selection": refreshed_selection,
        "bet_slip_info_response": slip_info_response,
        "bet_execution": result,
    }


async def main_async(args: argparse.Namespace) -> None:
    journal_path = Path(args.journal).resolve()
    state_path = Path(args.state_path).resolve()
    training_log_path = Path(args.training_log).resolve()
    state = _load_state(state_path)
    active_bets = list(state.get("active_bets", []))
    recent_selection_items = list(state.get("recent_selection_keys", []))
    recent_selection_items = _prune_recent_selection_keys(recent_selection_items, float(args.duplicate_key_ttl_minutes))
    placed_selection_keys = _selection_keys_from_recent(recent_selection_items) or set(state.get("placed_selection_keys", []))
    config = RuntimeConfig.from_settings()
    runtime = LiveBettingRuntime(
        market_feed_client=None,
        bet_executor=None,
        config=config,
    )
    executor = FonbetBetExecutor(dry_run=not args.send_bet)
    last_retrain_raw = state.get("last_retrain_utc")
    try:
        last_retrain_at = datetime.fromisoformat(str(last_retrain_raw)) if last_retrain_raw else None
    except Exception:
        last_retrain_at = None

    _log_cycle_event(
        journal_path,
        datetime.now(timezone.utc).isoformat(),
        "runtime_started",
        journal=str(journal_path),
        state_path=str(state_path),
        training_log=str(training_log_path),
        interval_seconds=float(args.interval),
        send_bet=bool(args.send_bet),
        workers=int(args.workers),
        max_bets_per_cycle=int(args.max_bets_per_cycle),
        target_active_bets=int(args.target_active_bets),
    )

    while True:
        now = datetime.now(timezone.utc).isoformat()
        try:
            current_ids = await asyncio.to_thread(_live_root_event_ids)
            active_bets = _prune_active_bets(active_bets, set(current_ids), args.active_bet_ttl_minutes)
            recent_selection_items = _prune_recent_selection_keys(recent_selection_items, float(args.duplicate_key_ttl_minutes))
            placed_selection_keys = _selection_keys_from_recent(recent_selection_items)
            state["placed_selection_keys"] = sorted(placed_selection_keys)
            state["recent_selection_keys"] = recent_selection_items
            state["active_bets"] = active_bets
            cursor = int(state.get("event_cursor") or 0)
            if current_ids:
                if args.max_events_per_cycle <= 0 or args.max_events_per_cycle >= len(current_ids):
                    selected_ids = list(current_ids)
                    state["event_cursor"] = 0
                else:
                    cursor = cursor % len(current_ids)
                    ordered_ids = current_ids[cursor:] + current_ids[:cursor]
                    selected_ids = ordered_ids[: max(args.max_events_per_cycle, 1)]
                    state["event_cursor"] = (cursor + max(args.max_events_per_cycle, 1)) % len(current_ids)
            else:
                selected_ids = []
                state["event_cursor"] = 0
            _save_state(state_path, state)
            exposure_before_cycle = _current_exposure(active_bets)
            cycle_results = []
            if float(args.retrain_every_minutes) > 0:
                now_dt = datetime.now(timezone.utc)
                should_retrain = (
                    last_retrain_at is None
                    or (now_dt - last_retrain_at) >= timedelta(minutes=float(args.retrain_every_minutes))
                )
                if should_retrain:
                    retrained, _reason = _run_periodic_retrain(
                        journal_path=journal_path,
                        timestamp_utc=now,
                        training_log_path=training_log_path,
                        min_rows=int(args.retrain_min_rows),
                    )
                    if retrained:
                        runtime = LiveBettingRuntime(
                            market_feed_client=None,
                            bet_executor=None,
                            config=config,
                        )
                        last_retrain_at = now_dt
                        state["last_retrain_utc"] = now_dt.isoformat()
                        _save_state(state_path, state)
            _log_cycle_event(
                journal_path,
                now,
                "cycle_started",
                live_events_total=len(current_ids),
                selected_events=selected_ids,
                active_bets_total=len(active_bets),
                active_selection_keys=sorted(placed_selection_keys),
                exposure_before_cycle=exposure_before_cycle,
                max_total_exposure=float(args.max_total_exposure),
                min_acceptance_probability=float(args.min_acceptance_probability),
                max_bets_per_cycle=int(args.max_bets_per_cycle),
                workers=int(args.workers),
                duplicate_key_ttl_minutes=float(args.duplicate_key_ttl_minutes),
                retrain_every_minutes=float(args.retrain_every_minutes),
            )
            if selected_ids:
                semaphore = asyncio.Semaphore(max(args.workers, 1))

                async def _score_one_event(event_id: int) -> dict:
                    async with semaphore:
                        try:
                            return await asyncio.to_thread(
                                _score_event,
                                event_id,
                                runtime,
                                float(args.min_acceptance_probability),
                            )
                        except Exception as exc:
                            return {
                                "event_id": event_id,
                                "status": "event_failed",
                                "reason": str(exc),
                                "traceback": traceback.format_exc(),
                            }

                scored_events = await asyncio.gather(*[_score_one_event(event_id) for event_id in selected_ids])
            else:
                scored_events = []

            candidate_pool: list[dict] = []
            for scored in scored_events:
                payload = _journal_event_payload(scored)
                if scored.get("status") == "candidates":
                    payload["candidate_count"] = len(scored.get("candidates") or [])
                _log_cycle_event(
                    journal_path,
                    now,
                    "event_scored",
                    **payload,
                )
                if scored.get("status") == "candidates":
                    candidate_pool.extend(scored.get("candidates") or [])

            ranked_candidates = _top_ranked(candidate_pool)
            placement_budget = max(
                int(args.max_bets_per_cycle),
                max(int(args.target_active_bets) - len(active_bets), 0),
            )
            placement_keys = {
                _selection_key(item["candidate"])
                for item in ranked_candidates[:placement_budget]
                if item.get("candidate") is not None
            }
            _log_cycle_event(
                journal_path,
                now,
                "cycle_ranked",
                candidate_events_total=len(candidate_pool),
                ranked_events=[
                    {
                        "event_id": item["event_id"],
                        "market_id": item.get("market_id"),
                        "market_type": item.get("market_type"),
                        "selection_name": item.get("selection_name"),
                        "selection_side": item.get("selection_side"),
                        "ranking_score": item.get("ranking_score"),
                        "acceptance_probability": item.get("acceptance_probability"),
                        "edge": item.get("edge"),
                        "stake": item.get("stake"),
                    }
                    for item in ranked_candidates[: max(placement_budget, 10)]
                ],
                placement_budget=placement_budget,
                placement_keys=sorted(placement_keys),
                target_active_bets=int(args.target_active_bets),
            )

            for scored in scored_events:
                if scored.get("status") != "candidates":
                    result = scored
                else:
                    result = {
                        "event_id": scored["event_id"],
                        "status": "event_scored_only",
                        "candidate_count": len(scored.get("candidates") or []),
                        "market_count": scored.get("market_count"),
                        "accepted_market_candidates": scored.get("accepted_market_candidates"),
                    }
                    for candidate_payload in scored.get("candidates") or []:
                        selection_key = _selection_key(candidate_payload["candidate"])
                        if placement_budget == 0 or selection_key not in placement_keys:
                            deferred = {
                                **candidate_payload,
                                "status": "candidate_deferred",
                                "reason": "ranked_below_cycle_cutoff",
                            }
                            _log_cycle_event(
                                journal_path,
                                now,
                                "placement_deferred",
                                **_journal_event_payload(deferred),
                            )
                            continue
                        _log_cycle_event(
                            journal_path,
                            now,
                            "placement_attempt",
                            **_journal_event_payload(candidate_payload),
                        )
                        placed_result = _process_event(
                            candidate_payload,
                            executor,
                            placed_selection_keys,
                            _current_exposure(active_bets),
                            float(args.max_total_exposure),
                        )
                        if placed_result.get("status") == "placed" and placed_result.get("selection_key"):
                            placed_selection_keys.add(str(placed_result["selection_key"]))
                            state["placed_selection_keys"] = sorted(placed_selection_keys)
                            active_bets.append(
                                {
                                    "selection_key": placed_result["selection_key"],
                                    "event_id": placed_result.get("event_id"),
                                    "market_id": placed_result.get("market_id"),
                                    "stake": placed_result.get("stake"),
                                    "placed_at_utc": now,
                                }
                            )
                            recent_selection_items.append(
                                {
                                    "selection_key": placed_result["selection_key"],
                                    "event_id": placed_result.get("event_id"),
                                    "market_id": placed_result.get("market_id"),
                                    "placed_at_utc": now,
                                }
                            )
                            state["recent_selection_keys"] = recent_selection_items
                            state["active_bets"] = active_bets
                            _save_state(state_path, state)
                        cycle_results.append(placed_result)
                        _append_jsonl(
                            journal_path,
                            {
                                "timestamp_utc": now,
                                "action": "event_cycle",
                                **placed_result,
                            },
                        )
                        training_payload = {
                            "timestamp_utc": now,
                            "event_id": placed_result.get("event_id"),
                            "status": placed_result.get("status"),
                            "selection_key": placed_result.get("selection_key"),
                            "market_id": placed_result.get("market_id"),
                            "market_type": placed_result.get("market_type"),
                            "target_game_number": placed_result.get("target_game_number"),
                            "selection_side": placed_result.get("selection_side"),
                            "selection_name": placed_result.get("selection_name"),
                            "stake": placed_result.get("stake"),
                            "odds_before_refresh": placed_result.get("odds_before_refresh"),
                            "model_probability": placed_result.get("model_probability"),
                            "implied_probability": placed_result.get("implied_probability"),
                            "edge": placed_result.get("edge"),
                            "acceptance_probability": placed_result.get("acceptance_probability"),
                            "ranking_score": placed_result.get("ranking_score"),
                            "player1_probability": placed_result.get("player1_probability"),
                            "player2_probability": placed_result.get("player2_probability"),
                            "refreshed_selection": placed_result.get("refreshed_selection"),
                            "bet_result_response": (placed_result.get("bet_execution") or {}).get("bet_result_response"),
                        }
                        _append_jsonl(training_log_path, training_payload)
                    cycle_results.append(result)
                    continue
                if result.get("status") == "placed" and result.get("selection_key"):
                    placed_selection_keys.add(str(result["selection_key"]))
                    state["placed_selection_keys"] = sorted(placed_selection_keys)
                    active_bets.append(
                        {
                            "selection_key": result["selection_key"],
                            "event_id": result.get("event_id"),
                            "market_id": result.get("market_id"),
                            "stake": result.get("stake"),
                            "placed_at_utc": now,
                        }
                    )
                    recent_selection_items.append(
                        {
                            "selection_key": result["selection_key"],
                            "event_id": result.get("event_id"),
                            "market_id": result.get("market_id"),
                            "placed_at_utc": now,
                        }
                    )
                    state["recent_selection_keys"] = recent_selection_items
                    state["active_bets"] = active_bets
                    _save_state(state_path, state)
                cycle_results.append(result)
                _append_jsonl(
                    journal_path,
                    {
                        "timestamp_utc": now,
                        "action": "event_cycle",
                        **result,
                    },
                )
                training_payload = {
                    "timestamp_utc": now,
                    "event_id": result.get("event_id"),
                    "status": result.get("status"),
                    "selection_key": result.get("selection_key"),
                    "market_id": result.get("market_id"),
                    "market_type": result.get("market_type"),
                    "target_game_number": result.get("target_game_number"),
                    "selection_side": result.get("selection_side"),
                    "selection_name": result.get("selection_name"),
                    "stake": result.get("stake"),
                    "odds_before_refresh": result.get("odds_before_refresh"),
                    "model_probability": result.get("model_probability"),
                    "implied_probability": result.get("implied_probability"),
                    "edge": result.get("edge"),
                    "acceptance_probability": result.get("acceptance_probability"),
                    "ranking_score": result.get("ranking_score"),
                    "player1_probability": result.get("player1_probability"),
                    "player2_probability": result.get("player2_probability"),
                    "refreshed_selection": result.get("refreshed_selection"),
                    "bet_result_response": (result.get("bet_execution") or {}).get("bet_result_response"),
                }
                _append_jsonl(training_log_path, training_payload)

            _append_jsonl(
                journal_path,
                {
                    "timestamp_utc": now,
                    "action": "heartbeat",
                    "live_events_total": len(current_ids),
                    "selected_events": selected_ids,
                    "placed_events": [row["event_id"] for row in cycle_results if row.get("status") == "placed"],
                    "skipped_events": [row["event_id"] for row in cycle_results if row.get("status") != "placed"],
                    "placed_selection_keys_total": len(placed_selection_keys),
                    "active_bets_total": len(active_bets),
                    "exposure_before_cycle": exposure_before_cycle,
                    "exposure_after_cycle": _current_exposure(active_bets),
                    "max_total_exposure": float(args.max_total_exposure),
                    "candidate_events_total": len(candidate_pool),
                    "ranked_for_placement": sorted(placement_keys),
                },
            )
        except Exception as exc:
            LOGGER.exception("auto live event betting cycle failed")
            _log_cycle_event(
                journal_path,
                now,
                "cycle_failed",
                reason=str(exc),
                traceback=traceback.format_exc(),
            )
        await asyncio.sleep(max(args.interval, 1.0))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
