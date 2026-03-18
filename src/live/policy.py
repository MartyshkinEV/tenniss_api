from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.live.runtime import ScoredSelection


def _bucket_odds(odds: float) -> str:
    if odds < 1.5:
        return "<1.5"
    if odds < 2.0:
        return "1.5-2.0"
    if odds < 3.0:
        return "2.0-3.0"
    return "3.0+"


class BankrollBanditPolicy:
    STAKE_LEVELS = (30.0, 60.0, 90.0)

    def __init__(
        self,
        outcomes_path: Path,
        bankroll: float,
        min_samples: int = 10,
        starting_bankroll: float | None = None,
        recent_window: int = 200,
    ):
        self.outcomes_path = outcomes_path
        self.bankroll = bankroll
        self.min_samples = min_samples
        self.starting_bankroll = float(starting_bankroll if starting_bankroll is not None else bankroll)
        self.recent_window = max(int(recent_window), int(min_samples))

    def _edge_bucket(self, edge: float) -> str:
        if edge < 0.08:
            return "0.05-0.08"
        if edge < 0.15:
            return "0.08-0.15"
        if edge < 0.25:
            return "0.15-0.25"
        return "0.25+"

    def _arm_key(self, market_type: str, side: str, odds: float, stake: float, edge: float) -> str:
        return f"{market_type}:{side}:{_bucket_odds(odds)}:{int(round(stake))}:{self._edge_bucket(edge)}"

    def _load_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "reward_sum": 0.0, "win_count": 0.0, "loss_count": 0.0}
        )
        if not self.outcomes_path.exists():
            return {}
        for raw_line in self.outcomes_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            reward = row.get("reward")
            if reward is None:
                continue
            key = self._arm_key(
                str(row.get("market_type") or "match_winner"),
                str(row.get("side") or "player1"),
                float(row.get("odds_taken") or 0.0),
                float(row.get("stake") or 0.0),
                float(row.get("edge") or 0.0),
            )
            stats[key]["count"] += 1.0
            stats[key]["reward_sum"] += float(reward)
            if float(reward) > 0:
                stats[key]["win_count"] += 1.0
            elif float(reward) < 0:
                stats[key]["loss_count"] += 1.0
        return dict(stats)

    def _recent_outcomes(self) -> list[dict[str, float]]:
        if not self.outcomes_path.exists():
            return []
        rows: list[dict[str, float]] = []
        for raw_line in self.outcomes_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            reward = row.get("reward")
            probability = row.get("model_probability")
            if reward is None or probability is None:
                continue
            rows.append(
                {
                    "reward": float(reward),
                    "model_probability": float(probability),
                    "won": 1.0 if float(reward) > 0 else 0.0,
                }
            )
        if len(rows) <= self.recent_window:
            return rows
        return rows[-self.recent_window:]

    def performance_profile(self) -> dict[str, float]:
        rows = self._recent_outcomes()
        if len(rows) < self.min_samples:
            return {
                "settled_count": float(len(rows)),
                "avg_model_probability": 0.0,
                "win_rate": 0.0,
                "avg_reward": 0.0,
                "probability_scale": 1.0,
                "risk_multiplier": 1.0,
                "bankroll_ratio": (
                    float(self.bankroll / self.starting_bankroll)
                    if self.starting_bankroll > 0 and self.bankroll > 0
                    else 1.0
                ),
            }

        count = float(len(rows))
        avg_model_probability = sum(item["model_probability"] for item in rows) / count
        win_rate = sum(item["won"] for item in rows) / count
        avg_reward = sum(item["reward"] for item in rows) / count

        probability_scale = min(max(win_rate / max(avg_model_probability, 1e-6), 0.55), 1.0)
        bankroll_ratio = (
            float(self.bankroll / self.starting_bankroll)
            if self.starting_bankroll > 0 and self.bankroll > 0
            else 1.0
        )
        risk_multiplier = probability_scale
        if avg_reward < 0:
            risk_multiplier *= max(0.5, 1.0 + (avg_reward * 12.0))
        if bankroll_ratio < 0.85:
            risk_multiplier *= 0.8
        if bankroll_ratio < 0.7:
            risk_multiplier *= 0.7
        risk_multiplier = min(max(risk_multiplier, 0.35), 1.0)

        return {
            "settled_count": count,
            "avg_model_probability": avg_model_probability,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "probability_scale": probability_scale,
            "risk_multiplier": risk_multiplier,
            "bankroll_ratio": bankroll_ratio,
        }

    def calibrate_probability(self, probability: float) -> float:
        clamped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        profile = self.performance_profile()
        scale = float(profile["probability_scale"])
        return min(max(0.5 + ((clamped - 0.5) * scale), 1e-6), 1.0 - 1e-6)

    def risk_multiplier(self) -> float:
        return float(self.performance_profile()["risk_multiplier"])

    def _allowed_stakes(self, bankroll: float) -> list[float]:
        allowed = [stake for stake in self.STAKE_LEVELS if bankroll <= 0 or stake <= bankroll]
        return allowed or [self.STAKE_LEVELS[0]]

    def _cap_candidate_stake(self, candidate: "ScoredSelection") -> float:
        capped_stake = max(self.STAKE_LEVELS[0], min(candidate.stake, self.STAKE_LEVELS[-1]))
        risk_multiplier = self.risk_multiplier()
        if self.bankroll > 0:
            capped_stake = min(capped_stake, round(self.bankroll * 0.02 * risk_multiplier, 2))
        if capped_stake < self.STAKE_LEVELS[0]:
            capped_stake = self.STAKE_LEVELS[0]
        nearest = min(self._allowed_stakes(self.bankroll), key=lambda stake: abs(stake - capped_stake))
        return round(nearest, 2)

    def recommend(self, candidates: list["ScoredSelection"], market_type: str) -> "ScoredSelection | None":
        if not candidates:
            return None

        stats = self._load_stats()
        total_samples = sum(item["count"] for item in stats.values())
        best_choice: ScoredSelection | None = None
        best_score: float | None = None

        for candidate in candidates:
            preferred_stake = self._cap_candidate_stake(candidate)
            calibrated_probability = self.calibrate_probability(candidate.model_probability)
            risk_multiplier = self.risk_multiplier()
            for stake in self._allowed_stakes(self.bankroll):
                stake_ratio = stake / self.STAKE_LEVELS[-1]
                base_score = (
                    float(candidate.edge) * 1.4 * risk_multiplier
                    + float(calibrated_probability) * 0.6
                    + (0.04 if stake == preferred_stake else 0.0)
                    - (0.12 * abs(stake_ratio - (preferred_stake / self.STAKE_LEVELS[-1])))
                    - (0.08 * stake_ratio)
                )
                arm = stats.get(self._arm_key(market_type, candidate.side, candidate.odds, stake, candidate.edge))
                if arm and arm["count"] > 0:
                    average_reward = arm["reward_sum"] / arm["count"]
                    success_rate = arm["win_count"] / arm["count"]
                    exploration_bonus = math.sqrt(math.log(total_samples + 2.0) / arm["count"])
                    score = base_score + average_reward + (0.15 * success_rate) + (0.05 * exploration_bonus)
                else:
                    score = base_score + 0.12
                choice = replace(candidate, stake=round(stake, 2))
                if best_score is None or score > best_score:
                    best_choice = choice
                    best_score = score
        return best_choice

    def recommend_stake(self, candidate: "ScoredSelection", market_type: str) -> float:
        recommended = self.recommend([candidate], market_type)
        if recommended is None:
            return 0.0
        return recommended.stake
