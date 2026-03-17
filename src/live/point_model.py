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


def build_live_point_feature_frame(
    market: Any,
    target_offset: int,
) -> pd.DataFrame:
    games_p1, games_p2 = _parse_game_score(market.raw.get("score"))
    points_p1, points_p2 = _parse_point_comment(market.raw.get("comment"))
    current_point_number = max(int(market.raw.get("target_point_number", target_offset)) - target_offset, 1)
    server_side = int(market.raw.get("serveT") or 0)
    frame = pd.DataFrame(
        [
            {
                "tour": str(getattr(market, "tourney_level", "tour") or "tour"),
                "draw": "live",
                "adf_flag": 0,
                "set_no": _parse_set_no(getattr(market, "round_name", "")),
                "game_no": games_p1 + games_p2 + 1,
                "point_no": current_point_number,
                "is_tiebreak": 1 if "tiebreak" in str(getattr(market, "round_name", "")).lower() else 0,
                "server_side": server_side,
                "points_p1": points_p1,
                "points_p2": points_p2,
                "point_score_diff": points_p1 - points_p2,
                "target_offset": target_offset,
            }
        ]
    )
    return frame


def build_execution_feature_frame(
    market: Any,
    state_features: dict[str, Any],
    player1_probability: float,
    player2_probability: float,
) -> pd.DataFrame:
    games_p1, games_p2 = _parse_game_score(market.raw.get("score"))
    points_p1, points_p2 = _parse_point_comment(market.raw.get("comment"))
    row = {
        "competition": getattr(market, "competition", None),
        "surface": getattr(market, "surface", None),
        "round_name": getattr(market, "round_name", None),
        "market_type": getattr(market, "market_type", None),
        "serving_team": market.raw.get("serveT"),
        "target_point_number": market.raw.get("target_point_number"),
        "player1_odds": getattr(market, "player1_odds", None),
        "player2_odds": getattr(market, "player2_odds", None),
        "player1_probability": player1_probability,
        "player2_probability": player2_probability,
        "selected_side": None,
        "selected_odds": max(getattr(market, "player1_odds", 0.0), getattr(market, "player2_odds", 0.0)),
        "selected_edge": None,
        "selected_stake": None,
        "games_p1": games_p1,
        "games_p2": games_p2,
        "points_p1": points_p1,
        "points_p2": points_p2,
    }
    for key, value in state_features.items():
        row[f"sf_{key}"] = value
    return pd.DataFrame([row])


@dataclass(frozen=True)
class LayeredPointPrediction:
    player1_probability: float
    execution_probability: float | None
    markov_probability: float
    historical_probability: float | None


class LayeredPointPredictor:
    def __init__(
        self,
        point_model: Any | None,
        execution_model: Any | None,
        target_offset: int,
        point_model_weight: float,
        markov_weight: float,
    ):
        self.point_model = point_model
        self.execution_model = execution_model
        self.target_offset = target_offset
        self.point_model_weight = point_model_weight
        self.markov_weight = markov_weight

    def _positive_probability(self, model: Any, frame: pd.DataFrame, fallback: float | None = None) -> float | None:
        proba = model.predict_proba(frame)
        if len(proba[0]) >= 2:
            return float(proba[0][1])
        return fallback

    def predict(self, market: Any, state_features: dict[str, Any], markov_probability: float) -> LayeredPointPrediction:
        historical_probability: float | None = None
        if self.point_model is not None:
            point_frame = build_live_point_feature_frame(market, target_offset=self.target_offset)
            historical_probability = self._positive_probability(self.point_model, point_frame, fallback=None)

        if historical_probability is None:
            blended_probability = markov_probability
        else:
            total_weight = self.point_model_weight + self.markov_weight
            blended_probability = (
                historical_probability * self.point_model_weight
                + markov_probability * self.markov_weight
            ) / total_weight

        execution_probability: float | None = None
        if self.execution_model is not None:
            execution_frame = build_execution_feature_frame(
                market=market,
                state_features=state_features,
                player1_probability=blended_probability,
                player2_probability=1.0 - blended_probability,
            )
            execution_probability = self._positive_probability(self.execution_model, execution_frame, fallback=None)

        return LayeredPointPrediction(
            player1_probability=_clamp(blended_probability, 0.05, 0.95),
            execution_probability=execution_probability,
            markov_probability=markov_probability,
            historical_probability=historical_probability,
        )
