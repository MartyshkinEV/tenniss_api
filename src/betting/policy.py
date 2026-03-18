from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


def _clamp_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True)
class MarketCandidate:
    match_id: str
    market: str
    pick: str
    odds: float
    model_prob: float
    stake: float
    tournament: str | None = None
    tournament_level: str | None = None
    surface: str | None = None
    explanation: dict[str, Any] = field(default_factory=dict)
    data_quality_score: float = 1.0
    recent_form: float | None = None
    odds_movement: float | None = None


@dataclass(frozen=True)
class BettingPolicyConfig:
    min_value: float = 0.03
    min_model_probability: float = 0.55
    min_data_quality_score: float = 0.7
    allowed_surfaces: tuple[str, ...] = ("Hard", "Clay", "Grass", "Carpet")
    allowed_tournament_levels: tuple[str, ...] = ("G", "M", "A", "B", "D", "F", "P", "I", "C", "tour", "challenger", "itf")
    min_recent_form: float = 0.35
    require_value: bool = True


@dataclass(frozen=True)
class BettingDecision:
    should_bet: bool
    reason: str
    bookmaker_prob: float
    value: float
    confidence: float
    passed_filters: bool
    explanation: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class BettingPolicy:
    def __init__(self, config: BettingPolicyConfig | None = None):
        self.config = config or BettingPolicyConfig()

    def evaluate(self, candidate: MarketCandidate) -> BettingDecision:
        bookmaker_prob = 1.0 / float(candidate.odds) if candidate.odds > 0 else 0.0
        model_prob = _clamp_probability(candidate.model_prob)
        value = model_prob - bookmaker_prob
        reasons: list[str] = []
        passed_filters = True

        if candidate.data_quality_score < self.config.min_data_quality_score:
            reasons.append(f"data_quality_below_min:{round(candidate.data_quality_score, 4)}")
            passed_filters = False
        if candidate.surface and candidate.surface not in self.config.allowed_surfaces:
            reasons.append(f"surface_blocked:{candidate.surface}")
            passed_filters = False
        if candidate.tournament_level and candidate.tournament_level not in self.config.allowed_tournament_levels:
            reasons.append(f"tournament_level_blocked:{candidate.tournament_level}")
            passed_filters = False
        if candidate.recent_form is not None and float(candidate.recent_form) < self.config.min_recent_form:
            reasons.append(f"recent_form_below_min:{round(float(candidate.recent_form), 4)}")
            passed_filters = False
        if model_prob < self.config.min_model_probability:
            reasons.append(f"model_probability_below_min:{round(model_prob, 4)}")
            passed_filters = False
        if self.config.require_value and value <= 0:
            reasons.append(f"missing_value:{round(value, 4)}")
            passed_filters = False
        if value < self.config.min_value:
            reasons.append(f"value_below_threshold:{round(value, 4)}")
            passed_filters = False

        explanation = {
            "match_id": candidate.match_id,
            "market": candidate.market,
            "pick": candidate.pick,
            "odds": float(candidate.odds),
            "stake": float(candidate.stake),
            "surface": candidate.surface,
            "tournament": candidate.tournament,
            "tournament_level": candidate.tournament_level,
            "data_quality_score": float(candidate.data_quality_score),
            "model_prob": model_prob,
            "bookmaker_prob": bookmaker_prob,
            "value": value,
            "recent_form": candidate.recent_form,
            "odds_movement": candidate.odds_movement,
            "filters": {
                "allowed_surfaces": list(self.config.allowed_surfaces),
                "allowed_tournament_levels": list(self.config.allowed_tournament_levels),
                "min_value": self.config.min_value,
                "min_model_probability": self.config.min_model_probability,
                "min_data_quality_score": self.config.min_data_quality_score,
                "min_recent_form": self.config.min_recent_form,
            },
        }
        if candidate.explanation:
            explanation["feature_context"] = dict(candidate.explanation)
        if passed_filters:
            reasons.append("eligible")
        confidence = max(model_prob - abs(value), 0.0)
        return BettingDecision(
            should_bet=passed_filters,
            reason=";".join(reasons),
            bookmaker_prob=bookmaker_prob,
            value=value,
            confidence=confidence,
            passed_filters=passed_filters,
            explanation=explanation,
        )


def candidate_from_frame_row(
    row: pd.Series,
    *,
    match_id_column: str = "match_id",
    market_column: str = "market",
    pick_column: str = "pick",
    odds_column: str = "odds",
    probability_column: str = "model_prob",
    stake_column: str = "stake",
) -> MarketCandidate:
    return MarketCandidate(
        match_id=str(row.get(match_id_column) or ""),
        market=str(row.get(market_column) or ""),
        pick=str(row.get(pick_column) or ""),
        odds=float(row.get(odds_column) or 0.0),
        model_prob=float(row.get(probability_column) or 0.0),
        stake=float(row.get(stake_column) or 0.0),
        tournament=row.get("tournament"),
        tournament_level=row.get("tournament_level"),
        surface=row.get("surface"),
        data_quality_score=float(row.get("data_quality_score") or 0.0),
        recent_form=row.get("recent_form"),
        odds_movement=row.get("odds_movement"),
        explanation={
            key: row[key]
            for key in (
                "elo_diff",
                "surface_elo_diff",
                "avg_total_games",
                "hold_rate_diff",
                "break_rate_diff",
                "three_set_rate_diff",
                "line_movement",
            )
            if key in row.index
        },
    )
