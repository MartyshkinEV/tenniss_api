from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _parse_score(score: str | None) -> tuple[int, int]:
    if not score or ":" not in score:
        return 0, 0
    left, right = score.split(":", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return 0, 0


@dataclass(frozen=True)
class TotalPrediction:
    over_probability: float
    heuristic_probability: float
    historical_probability: float | None


class SetTotalModel:
    def __init__(self, historical_model: Any | None = None, historical_weight: float = 0.55):
        self.historical_model = historical_model
        self.historical_weight = historical_weight

    def _heuristic_probability(self, market: Any, state_features: dict[str, Any]) -> float:
        games_p1, games_p2 = _parse_score(market.raw.get("score"))
        current_total = games_p1 + games_p2
        line = float(market.raw.get("total_line") or 0.0)
        hold_strength = (float(state_features.get("p1_hold_rate", 0.65)) + float(state_features.get("p2_hold_rate", 0.65))) / 2.0
        break_strength = (float(state_features.get("p1_break_rate", 0.25)) + float(state_features.get("p2_break_rate", 0.25))) / 2.0
        pace = 0.5 + (hold_strength - break_strength) * 0.35
        remaining_margin = line - current_total
        base = 0.55 + (pace - 0.5) * 0.8 - remaining_margin * 0.09
        return _clamp(base, 0.05, 0.95)

    def _historical_probability(self, market: Any, state_features: dict[str, Any]) -> float | None:
        if self.historical_model is None:
            return None
        games_p1, games_p2 = _parse_score(market.raw.get("score"))
        frame = pd.DataFrame(
            [
                {
                    "surface": getattr(market, "surface", "Hard"),
                    "tourney_level": getattr(market, "tourney_level", "tour"),
                    "best_of": getattr(market, "best_of", 3),
                    "line": float(market.raw.get("total_line") or 0.0),
                    "games_p1": games_p1,
                    "games_p2": games_p2,
                    "current_total_games": games_p1 + games_p2,
                    "hold_rate_diff": float(state_features.get("p1_hold_rate", 0.65)) - float(state_features.get("p2_hold_rate", 0.65)),
                    "break_rate_diff": float(state_features.get("p1_break_rate", 0.25)) - float(state_features.get("p2_break_rate", 0.25)),
                    "rank_diff": float(state_features.get("rank_diff", 0.0)),
                    "elo_diff": float(state_features.get("elo_diff", 0.0)),
                }
            ]
        )
        proba = self.historical_model.predict_proba(frame)
        if len(proba[0]) < 2:
            return None
        return float(proba[0][1])

    def predict_over(self, market: Any, state_features: dict[str, Any]) -> TotalPrediction:
        heuristic_probability = self._heuristic_probability(market, state_features)
        historical_probability = self._historical_probability(market, state_features)
        if historical_probability is None:
            over_probability = heuristic_probability
        else:
            over_probability = (
                historical_probability * self.historical_weight
                + heuristic_probability * (1.0 - self.historical_weight)
            )
        return TotalPrediction(
            over_probability=_clamp(over_probability, 0.05, 0.95),
            heuristic_probability=heuristic_probability,
            historical_probability=historical_probability,
        )
