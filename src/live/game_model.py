from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


POINT_SCORE_MAP = {"0": 0, "00": 0, "15": 1, "30": 2, "40": 3, "A": 4}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _parse_point_comment(comment: str | None) -> tuple[int, int]:
    if not comment:
        return 0, 0
    body = comment.strip().strip("()")
    if " " in body:
        body = body.split(" ", 1)[0]
    body = body.replace("*", "")
    if "-" not in body:
        return 0, 0
    left, right = body.split("-", 1)
    return POINT_SCORE_MAP.get(left.strip(), 0), POINT_SCORE_MAP.get(right.strip(), 0)


def _parse_game_score(score: str | None) -> tuple[int, int]:
    if not score or ":" not in score:
        return 0, 0
    left, right = score.split(":", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return 0, 0


def _parse_set_no(round_name: str | None) -> int:
    if not round_name:
        return 1
    match = re.search(r"(\d+)", round_name)
    if match:
        return int(match.group(1))
    return 1


def build_live_game_feature_frame(market: Any) -> pd.DataFrame:
    games_p1, games_p2 = _parse_game_score(market.raw.get("score"))
    points_p1, points_p2 = _parse_point_comment(market.raw.get("comment"))
    server_side = int(market.raw.get("serveT") or 0)
    is_break_point_p1 = int(server_side == 2 and points_p1 >= 3 and points_p1 > points_p2)
    is_break_point_p2 = int(server_side == 1 and points_p2 >= 3 and points_p2 > points_p1)
    frame = pd.DataFrame(
        [
            {
                "tour": str(getattr(market, "tourney_level", "tour") or "tour"),
                "draw": "live",
                "adf_flag": 0,
                "set_no": _parse_set_no(getattr(market, "round_name", "")),
                "game_no": max(int(market.raw.get("target_game_number") or (games_p1 + games_p2 + 1)), 1),
                "point_no": max(points_p1 + points_p2 + 1, 1),
                "is_tiebreak": 1 if "tiebreak" in str(getattr(market, "round_name", "")).lower() else 0,
                "server_side": server_side,
                "points_p1": points_p1,
                "points_p2": points_p2,
                "point_score_diff": points_p1 - points_p2,
                "is_break_point_p1": is_break_point_p1,
                "is_break_point_p2": is_break_point_p2,
            }
        ]
    )
    return frame


@dataclass(frozen=True)
class LayeredGamePrediction:
    player1_probability: float
    markov_probability: float
    historical_probability: float | None


class LayeredGamePredictor:
    def __init__(
        self,
        game_model: Any | None,
        game_model_weight: float,
        markov_weight: float,
    ):
        self.game_model = game_model
        self.game_model_weight = game_model_weight
        self.markov_weight = markov_weight

    def _positive_probability(self, model: Any, frame: pd.DataFrame, fallback: float | None = None) -> float | None:
        proba = model.predict_proba(frame)
        if len(proba[0]) >= 2:
            return float(proba[0][1])
        return fallback

    def predict(self, market: Any, markov_probability: float) -> LayeredGamePrediction:
        historical_probability: float | None = None
        if self.game_model is not None:
            game_frame = build_live_game_feature_frame(market)
            historical_probability = self._positive_probability(self.game_model, game_frame, fallback=None)

        if historical_probability is None:
            blended_probability = markov_probability
        else:
            total_weight = self.game_model_weight + self.markov_weight
            blended_probability = (
                historical_probability * self.game_model_weight
                + markov_probability * self.markov_weight
            ) / total_weight

        return LayeredGamePrediction(
            player1_probability=_clamp(blended_probability, 0.05, 0.95),
            markov_probability=markov_probability,
            historical_probability=historical_probability,
        )
