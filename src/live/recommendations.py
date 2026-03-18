from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _match_key(market: Any) -> tuple[str, str, str]:
    return (
        str(getattr(market, "competition", "") or ""),
        str(getattr(market, "player1_name", "") or ""),
        str(getattr(market, "player2_name", "") or ""),
    )


def _recommendation_action(market_type: str) -> str:
    if market_type == "set_total_over_under":
        return "bet_total"
    if market_type == "next_game_winner":
        return "bet_games"
    return "bet_winner"


def build_recommendations(runtime: Any, markets: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for market in markets:
        if getattr(market, "market_type", "") not in {"match_winner", "set_total_over_under", "next_game_winner"}:
            continue
        grouped[_match_key(market)].append(market)

    recommendations: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for match_key, match_markets in grouped.items():
        best_payload: dict[str, Any] | None = None
        reasons: list[str] = []
        for market in match_markets:
            try:
                _, features, player1_probability, player2_probability, candidate = runtime._score_market_details(market)
            except Exception as exc:
                reasons.append(str(exc))
                continue
            if candidate is None:
                continue
            ranking_score = float(getattr(candidate, "ranking_score", candidate.edge))
            payload = {
                "timestamp_utc": timestamp,
                "competition": market.competition,
                "event_id": market.event_id,
                "market_id": market.market_id,
                "market_type": market.market_type,
                "player1_name": market.player1_name,
                "player2_name": market.player2_name,
                "action": _recommendation_action(market.market_type),
                "selection": candidate.player_name,
                "selection_side": candidate.side,
                "odds": candidate.odds,
                "stake": candidate.stake,
                "edge": candidate.edge,
                "model_probability": candidate.model_probability,
                "implied_probability": candidate.implied_probability,
                "acceptance_probability": getattr(candidate, "acceptance_probability", 1.0),
                "ranking_score": ranking_score,
                "live_score": market.raw.get("score"),
                "factor_id": market.raw.get("player1_factor_id" if candidate.side == "player1" else "player2_factor_id"),
                "param": market.raw.get("player1_param" if candidate.side == "player1" else "player2_param"),
                "player1_probability": player1_probability,
                "player2_probability": player2_probability,
                "state_features": features,
            }
            if best_payload is None or payload["ranking_score"] > best_payload["ranking_score"]:
                best_payload = payload

        if best_payload is None:
            recommendations.append(
                {
                    "timestamp_utc": timestamp,
                    "competition": match_key[0],
                    "event_id": None,
                    "market_id": None,
                    "market_type": None,
                    "player1_name": match_key[1],
                    "player2_name": match_key[2],
                    "action": "no_bet",
                    "reason": reasons[0] if reasons else "no_edge",
                }
            )
        else:
            recommendations.append(best_payload)

    recommendations.sort(key=lambda item: (item["action"] == "no_bet", -float(item.get("ranking_score") or 0.0)))
    return recommendations


def append_recommendations(path: Path, recommendations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in recommendations:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
