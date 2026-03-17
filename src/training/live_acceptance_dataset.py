from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings


ACCEPTANCE_ACTIONS = {"bet", "bet_express"}


@dataclass(frozen=True)
class AcceptanceTrainingFrames:
    leg_acceptance: pd.DataFrame


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.stat().st_size:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _selection_index(result: dict[str, Any], record: dict[str, Any]) -> int | None:
    coupon_bets = result.get("bet_result_response", {}).get("coupon", {}).get("bets", [])
    selection_id = record.get("selection_id")
    market_id = record.get("market_id") or ""
    event_id = record.get("event_id")
    for idx, bet in enumerate(coupon_bets):
        event = bet.get("event")
        if event is not None and str(event) == str(event_id):
            return idx
    if ":" in market_id:
        event_from_market = market_id.split(":", 1)[0]
        for idx, bet in enumerate(coupon_bets):
            event = bet.get("event")
            if event is not None and str(event) == event_from_market:
                return idx
    if selection_id:
        event_from_selection = selection_id.split(":", 1)[0]
        for idx, bet in enumerate(coupon_bets):
            event = bet.get("event")
            if event is not None and str(event) == event_from_selection:
                return idx
    return None


def _slip_bet(result: dict[str, Any], index: int | None) -> dict[str, Any]:
    bets = result.get("bet_slip_info_response", {}).get("bets", [])
    if index is None or index >= len(bets):
        return {}
    return bets[index]


def _coupon_bet(result: dict[str, Any], index: int | None) -> dict[str, Any]:
    bets = result.get("bet_result_response", {}).get("coupon", {}).get("bets", [])
    if index is None or index >= len(bets):
        return {}
    return bets[index]


def _extract_row(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("action") not in ACCEPTANCE_ACTIONS:
        return None
    result = record.get("result")
    if not isinstance(result, dict):
        return None
    coupon = result.get("bet_result_response", {}).get("coupon", {})
    result_code = coupon.get("resultCode")
    if result_code is None:
        return None

    index = _selection_index(result, record)
    slip_bet = _slip_bet(result, index)
    slip_factor = slip_bet.get("factor", {}) if isinstance(slip_bet, dict) else {}
    slip_event = slip_bet.get("event", {}) if isinstance(slip_bet, dict) else {}
    return {
        "event_id": record.get("event_id"),
        "market_id": record.get("market_id"),
        "action": record.get("action"),
        "market_type": record.get("market_type"),
        "side": record.get("side"),
        "player_name": record.get("player_name"),
        "odds": record.get("odds"),
        "stake": record.get("stake"),
        "model_probability": record.get("model_probability"),
        "implied_probability": record.get("implied_probability"),
        "edge": record.get("edge"),
        "coupon_legs": len(coupon.get("bets", [])),
        "slip_factor_id": slip_factor.get("id"),
        "slip_factor_value": slip_factor.get("v"),
        "slip_factor_param": slip_factor.get("p"),
        "slip_score": slip_event.get("score"),
        "label": int(result_code == 0),
    }


def build_leg_acceptance_frame(
    actions_path: Path | None = None,
) -> AcceptanceTrainingFrames:
    records = _load_jsonl(actions_path or settings.live_rl_actions_path)
    rows = []
    for record in records:
        row = _extract_row(record)
        if row is not None:
            rows.append(row)
    return AcceptanceTrainingFrames(leg_acceptance=pd.DataFrame(rows))
