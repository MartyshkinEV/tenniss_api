try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import gzip
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import joblib

from config import settings
from src.live.fonbet import (
    FonbetBetExecutor,
    FonbetEventsClient,
    extract_fonbet_events,
    extract_markets_from_fonbet_catalog,
)
from src.live.game_model import LayeredGamePredictor
from src.live.markov import MarkovGameModel
from src.live.runtime import RuntimeConfig, build_candidate_options


STAKE_LEVELS = (30.0, 60.0, 90.0, 120.0, 180.0)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


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


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"recent": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"recent": []}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def _prune_recent(items: list[dict], ttl_minutes: float) -> list[dict]:
    now = datetime.now(timezone.utc)
    ttl = timedelta(minutes=max(ttl_minutes, 1.0))
    kept = []
    for item in items:
        try:
            placed_at = datetime.fromisoformat(str(item.get("placed_at_utc")))
        except Exception:
            continue
        if placed_at.tzinfo is None:
            placed_at = placed_at.replace(tzinfo=timezone.utc)
        if now - placed_at <= ttl:
            kept.append(item)
    return kept


def _selection_key(candidate) -> str:
    raw = candidate.market.raw
    factor_id = raw.get("player1_factor_id") if candidate.side == "player1" else raw.get("player2_factor_id")
    param = raw.get("player1_param") if candidate.side == "player1" else raw.get("player2_param")
    return f"{candidate.market.market_id}:{candidate.side}:{factor_id}:{param}"


def _select_stake(edge: float, model_probability: float, odds: float, max_stake: float) -> float:
    allowed_levels = [stake for stake in STAKE_LEVELS if stake <= max_stake]
    if not allowed_levels:
        return STAKE_LEVELS[0]

    score = (
        edge * 2.4
        + max(model_probability - 0.5, 0.0) * 1.6
        - max(odds - 2.4, 0.0) * 0.18
    )
    if score >= 0.82:
        target = 180.0
    elif score >= 0.72:
        target = 120.0
    elif score >= 0.6:
        target = 90.0
    elif score >= 0.48:
        target = 60.0
    else:
        target = 30.0
    eligible = [stake for stake in allowed_levels if stake <= target]
    return eligible[-1] if eligible else allowed_levels[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-events-per-cycle", type=int, default=20)
    parser.add_argument("--max-bets-per-cycle", type=int, default=8)
    parser.add_argument("--max-candidates-per-event", type=int, default=2)
    parser.add_argument("--min-stake", type=float, default=30.0)
    parser.add_argument("--max-stake", type=float, default=180.0)
    parser.add_argument("--duplicate-ttl-minutes", type=float, default=3.0)
    parser.add_argument("--blocked-ttl-minutes", type=float, default=3.0)
    parser.add_argument("--min-model-probability", type=float, default=0.50)
    parser.add_argument("--edge-threshold", type=float, default=0.025)
    parser.add_argument("--min-odds", type=float, default=1.35)
    parser.add_argument("--max-odds", type=float, default=4.2)
    parser.add_argument("--journal", default="artifacts/live_betting/state_only_live_batch.jsonl")
    parser.add_argument("--state-path", default="artifacts/live_betting/state_only_live_batch_state.json")
    parser.add_argument("--send-bet", action="store_true")
    args = parser.parse_args()

    journal_path = Path(args.journal).resolve()
    state_path = Path(args.state_path).resolve()
    state = _load_state(state_path)
    recent_items = _prune_recent(list(state.get("recent", [])), float(args.duplicate_ttl_minutes))
    blocked_items = _prune_recent(list(state.get("blocked", [])), float(args.blocked_ttl_minutes))
    seen_keys = {str(item.get("selection_key")) for item in recent_items if item.get("selection_key")}
    blocked_keys = {str(item.get("selection_key")) for item in blocked_items if item.get("selection_key")}
    state["recent"] = recent_items
    state["blocked"] = blocked_items
    _save_state(state_path, state)

    config = RuntimeConfig.from_settings()
    config = RuntimeConfig(
        **{
            **config.__dict__,
            "edge_threshold": float(args.edge_threshold),
            "min_model_probability": float(args.min_model_probability),
            "min_odds": float(args.min_odds),
            "max_odds": float(args.max_odds),
        }
    )
    markov_model = MarkovGameModel()
    historical_model_path = settings.models_dir / "historical_game_model.joblib"
    predictor = LayeredGamePredictor(
        game_model=joblib.load(historical_model_path) if historical_model_path.exists() else None,
        game_model_weight=config.game_model_weight,
        markov_weight=config.game_markov_weight,
    )
    executor = FonbetBetExecutor(dry_run=not args.send_bet)
    _append_jsonl(
        journal_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": "process_started",
            "send_bet": bool(args.send_bet),
            "interval": float(args.interval),
            "max_events_per_cycle": int(args.max_events_per_cycle),
            "max_bets_per_cycle": int(args.max_bets_per_cycle),
            "max_candidates_per_event": int(args.max_candidates_per_event),
            "duplicate_ttl_minutes": float(args.duplicate_ttl_minutes),
            "blocked_ttl_minutes": float(args.blocked_ttl_minutes),
            "min_model_probability": float(config.min_model_probability),
            "edge_threshold": float(config.edge_threshold),
            "min_odds": float(config.min_odds),
            "max_odds": float(config.max_odds),
            "min_stake": float(args.min_stake),
            "max_stake": float(args.max_stake),
        },
    )

    while True:
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        event_ids = _live_root_event_ids()
        if args.max_events_per_cycle > 0:
            event_ids = event_ids[: args.max_events_per_cycle]

        scored_candidates = []
        for event_id in event_ids:
            try:
                payload = _load_event_payload(event_id)
                markets = [
                    market
                    for market in extract_markets_from_fonbet_catalog(payload)
                    if market.market_type == "next_game_winner"
                ]
            except Exception as exc:
                _append_jsonl(
                    journal_path,
                    {"timestamp_utc": timestamp_utc, "action": "event_failed", "event_id": event_id, "reason": str(exc)},
                )
                continue

            event_candidates = []
            for market in markets:
                prediction = predictor.predict(
                    market=market,
                    markov_probability=markov_model.predict_next_game({}, market.raw),
                )
                player1_probability = prediction.player1_probability
                player2_probability = 1.0 - player1_probability
                candidates = build_candidate_options(
                    ("player1", market.player1_name, player1_probability, market.player1_odds, 0),
                    ("player2", market.player2_name, player2_probability, market.player2_odds, 0),
                    config=config,
                    market=market,
                )
                for candidate in candidates:
                    candidate = type(candidate)(
                        market=candidate.market,
                        side=candidate.side,
                        player_name=candidate.player_name,
                        model_probability=candidate.model_probability,
                        implied_probability=candidate.implied_probability,
                        edge=candidate.edge,
                        odds=candidate.odds,
                        stake=_select_stake(
                            edge=candidate.edge,
                            model_probability=candidate.model_probability,
                            odds=candidate.odds,
                            max_stake=max(float(args.max_stake), float(args.min_stake)),
                        ),
                        player_id=0,
                        acceptance_probability=1.0,
                        ranking_score=candidate.edge,
                    )
                    if candidate.stake < float(args.min_stake):
                        candidate = type(candidate)(
                            market=candidate.market,
                            side=candidate.side,
                            player_name=candidate.player_name,
                            model_probability=candidate.model_probability,
                            implied_probability=candidate.implied_probability,
                            edge=candidate.edge,
                            odds=candidate.odds,
                            stake=float(args.min_stake),
                            player_id=candidate.player_id,
                            acceptance_probability=candidate.acceptance_probability,
                            ranking_score=candidate.ranking_score,
                        )
                    selection_key = _selection_key(candidate)
                    payload = {
                        "timestamp_utc": timestamp_utc,
                        "action": "candidate",
                        "event_id": event_id,
                        "market_id": market.market_id,
                        "target_game_number": market.raw.get("target_game_number"),
                        "selection_key": selection_key,
                        "selection_side": candidate.side,
                        "selection_name": candidate.player_name,
                        "stake": candidate.stake,
                        "odds_before_refresh": candidate.odds,
                        "model_probability": candidate.model_probability,
                        "implied_probability": candidate.implied_probability,
                        "edge": candidate.edge,
                        "historical_probability": prediction.historical_probability,
                        "markov_probability": prediction.markov_probability,
                        "player1_probability": player1_probability,
                        "player2_probability": player2_probability,
                    }
                    event_candidates.append({"candidate": candidate, "payload": payload})
            if not event_candidates:
                _append_jsonl(journal_path, {"timestamp_utc": timestamp_utc, "action": "no_candidate", "event_id": event_id})
                continue
            event_candidates.sort(key=lambda item: item["candidate"].edge, reverse=True)
            scored_candidates.extend(event_candidates[: max(int(args.max_candidates_per_event), 0)])

        scored_candidates.sort(key=lambda item: item["candidate"].edge, reverse=True)
        for item in scored_candidates[: max(int(args.max_bets_per_cycle), 0)]:
            candidate = item["candidate"]
            payload = dict(item["payload"])
            selection_key = payload["selection_key"]
            if selection_key in seen_keys:
                _append_jsonl(
                    journal_path,
                    {
                        "timestamp_utc": timestamp_utc,
                        "action": "duplicate_skipped",
                        "event_id": payload["event_id"],
                        "market_id": payload["market_id"],
                        "selection_key": selection_key,
                    },
                )
                continue
            if selection_key in blocked_keys:
                _append_jsonl(
                    journal_path,
                    {
                        "timestamp_utc": timestamp_utc,
                        "action": "blocked_skipped",
                        "event_id": payload["event_id"],
                        "market_id": payload["market_id"],
                        "selection_key": selection_key,
                    },
                )
                continue
            refreshed_selection, slip_info_response = executor.refresh_candidate(candidate)
            result = executor.place_prepared_bet(candidate, refreshed_selection, slip_info_response)
            record = {
                **payload,
                "action": "bet_attempt",
                "refreshed_selection": refreshed_selection,
                "bet_slip_info_response": slip_info_response,
                "bet_execution": result,
                "status": result.get("status"),
            }
            _append_jsonl(journal_path, record)
            if result.get("status") == "placed":
                seen_keys.add(selection_key)
                recent_items.append(
                    {
                        "selection_key": selection_key,
                        "event_id": payload["event_id"],
                        "market_id": payload["market_id"],
                        "placed_at_utc": timestamp_utc,
                    }
                )
            elif result.get("status") in {"temporarily_suspended", "odds_changed"}:
                blocked_keys.add(selection_key)
                blocked_items.append(
                    {
                        "selection_key": selection_key,
                        "event_id": payload["event_id"],
                        "market_id": payload["market_id"],
                        "placed_at_utc": timestamp_utc,
                    }
                )
        state["recent"] = _prune_recent(recent_items, float(args.duplicate_ttl_minutes))
        state["blocked"] = _prune_recent(blocked_items, float(args.blocked_ttl_minutes))
        _save_state(state_path, state)
        time.sleep(max(float(args.interval), 1.0))


if __name__ == "__main__":
    main()
