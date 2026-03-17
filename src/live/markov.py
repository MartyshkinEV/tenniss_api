from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def infer_server_side(raw: dict[str, Any]) -> str | None:
    serve_t = raw.get("serveT")
    if serve_t in (1, "1", "player1", "p1", "home"):
        return "player1"
    if serve_t in (2, "2", "player2", "p2", "away"):
        return "player2"
    return None


def estimate_service_game_win_probability(
    hold_rate: float,
    opponent_break_rate: float,
    server_advantage: float = 0.02,
) -> float:
    # Blend player hold ability with opponent return pressure into a stable game-win prior.
    raw_probability = (hold_rate + (1.0 - opponent_break_rate)) / 2.0 + server_advantage
    return _clamp(raw_probability, 0.05, 0.95)


def estimate_point_win_probability(
    hold_rate: float,
    opponent_break_rate: float,
    server_advantage: float = 0.02,
) -> float:
    baseline = (hold_rate + (1.0 - opponent_break_rate)) / 2.0
    return _clamp(0.5 + (baseline - 0.5) * 0.55 + server_advantage, 0.1, 0.9)


@dataclass(frozen=True)
class MarkovGameModel:
    server_advantage: float = 0.02

    def predict_next_game(self, features: dict[str, Any], raw: dict[str, Any]) -> float:
        p1_hold = float(features.get("p1_hold_rate", 0.65))
        p2_hold = float(features.get("p2_hold_rate", 0.65))
        p1_break = float(features.get("p1_break_rate", 0.25))
        p2_break = float(features.get("p2_break_rate", 0.25))

        p1_on_serve = estimate_service_game_win_probability(
            hold_rate=p1_hold,
            opponent_break_rate=p2_break,
            server_advantage=self.server_advantage,
        )
        p2_on_serve = estimate_service_game_win_probability(
            hold_rate=p2_hold,
            opponent_break_rate=p1_break,
            server_advantage=self.server_advantage,
        )
        server_side = infer_server_side(raw)
        if server_side == "player1":
            return p1_on_serve
        if server_side == "player2":
            return 1.0 - p2_on_serve

        neutral_probability = (p1_on_serve + (1.0 - p2_on_serve)) / 2.0
        return _clamp(neutral_probability, 0.05, 0.95)

    def predict_point_plus_one(self, features: dict[str, Any], raw: dict[str, Any]) -> float:
        p1_hold = float(features.get("p1_hold_rate", 0.65))
        p2_hold = float(features.get("p2_hold_rate", 0.65))
        p1_break = float(features.get("p1_break_rate", 0.25))
        p2_break = float(features.get("p2_break_rate", 0.25))
        p1_on_serve = estimate_point_win_probability(
            hold_rate=p1_hold,
            opponent_break_rate=p2_break,
            server_advantage=self.server_advantage,
        )
        p2_on_serve = estimate_point_win_probability(
            hold_rate=p2_hold,
            opponent_break_rate=p1_break,
            server_advantage=self.server_advantage,
        )
        server_side = infer_server_side(raw)
        if server_side == "player1":
            return p1_on_serve
        if server_side == "player2":
            return 1.0 - p2_on_serve
        return _clamp((p1_on_serve + (1.0 - p2_on_serve)) / 2.0, 0.1, 0.9)
