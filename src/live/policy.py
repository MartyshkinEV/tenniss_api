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

    def __init__(self, outcomes_path: Path, bankroll: float, min_samples: int = 10):
        self.outcomes_path = outcomes_path
        self.bankroll = bankroll
        self.min_samples = min_samples

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

    def _allowed_stakes(self, bankroll: float) -> list[float]:
        allowed = [stake for stake in self.STAKE_LEVELS if bankroll <= 0 or stake <= bankroll]
        return allowed or [self.STAKE_LEVELS[0]]

    def _cap_candidate_stake(self, candidate: "ScoredSelection") -> float:
        capped_stake = max(self.STAKE_LEVELS[0], min(candidate.stake, self.STAKE_LEVELS[-1]))
        if self.bankroll > 0:
            capped_stake = min(capped_stake, round(self.bankroll * 0.03, 2))
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
            for stake in self._allowed_stakes(self.bankroll):
                stake_ratio = stake / self.STAKE_LEVELS[-1]
                base_score = (
                    float(candidate.edge) * 1.7
                    + float(candidate.model_probability) * 0.8
                    + (0.04 if stake == preferred_stake else 0.0)
                    - (0.08 * abs(stake_ratio - (preferred_stake / self.STAKE_LEVELS[-1])))
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
