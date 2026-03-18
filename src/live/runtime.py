from __future__ import annotations

import json
import logging
import time
import hashlib
from functools import lru_cache
from dataclasses import replace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import (
    discover_player_csv_files,
    player_id_namespace_offset,
    resolve_default_model_artifact_path,
    settings,
)
from src.betting import BettingAuditLogger, BettingPolicy, BettingPolicyConfig, DatabaseBetLogRecorder, MarketCandidate
from src.db.engine import get_engine
from src.data import ELO_FEATURES, align_frame_to_model, load_match_features_elo, load_player_match_stats
from src.features.feature_builder import rank_diff_bucket
from src.live.game_model import LayeredGamePredictor
from src.live.markov import MarkovGameModel
from src.live.point_model import LayeredPointPredictor
from src.live.policy import BankrollBanditPolicy
from src.live.total_model import SetTotalModel

LOGGER = logging.getLogger(__name__)


SUCCESSFUL_PLACEMENT_STATUSES = {"placed", "dry_run"}


@dataclass(frozen=True)
class LiveMarket:
    market_id: str
    event_id: str
    competition: str
    surface: str
    round_name: str
    best_of: int
    tourney_level: str
    player1_name: str
    player2_name: str
    player1_odds: float
    player2_odds: float
    market_type: str = "match_winner"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredSelection:
    market: LiveMarket
    side: str
    player_name: str
    model_probability: float
    implied_probability: float
    edge: float
    odds: float
    stake: float
    player_id: int
    acceptance_probability: float = 1.0
    ranking_score: float = 0.0

    @property
    def selection_id(self) -> str:
        return f"{self.market.market_id}:{self.side}"


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: Path
    poll_interval_seconds: int
    edge_threshold: float
    min_model_probability: float
    min_odds: float
    max_odds: float
    default_stake: float
    bankroll: float
    kelly_fraction: float
    dry_run: bool
    state_path: Path
    decisions_path: Path
    rl_snapshots_path: Path
    rl_actions_path: Path
    rl_outcomes_path: Path
    rl_tracker_state_path: Path
    rl_market_close_cycles: int
    point_trajectories_path: Path = Path("point_trajectories.jsonl")
    point_fast_mode: bool = True
    game_target_offset: int = 2
    game_model_weight: float = 0.60
    game_markov_weight: float = 0.40
    point_target_offset: int = 2
    point_model_weight: float = 0.65
    point_markov_weight: float = 0.35
    point_execution_min_probability: float = 0.40
    bet_mode: str = "single"
    express_size: int = 2

    @classmethod
    def from_settings(cls) -> "RuntimeConfig":
        return cls(
            model_path=resolve_default_model_artifact_path(),
            poll_interval_seconds=settings.live_poll_interval_seconds,
            edge_threshold=settings.live_edge_threshold,
            min_model_probability=settings.live_min_model_probability,
            min_odds=settings.live_min_odds,
            max_odds=settings.live_max_odds,
            default_stake=settings.live_default_stake,
            bankroll=settings.live_bankroll,
            kelly_fraction=settings.live_kelly_fraction,
            dry_run=settings.live_dry_run,
            state_path=settings.live_state_path,
            decisions_path=settings.live_decisions_path,
            rl_snapshots_path=settings.live_rl_snapshots_path,
            rl_actions_path=settings.live_rl_actions_path,
            rl_outcomes_path=settings.live_rl_outcomes_path,
            rl_tracker_state_path=settings.live_rl_tracker_state_path,
            rl_market_close_cycles=settings.live_rl_market_close_cycles,
            point_trajectories_path=settings.live_point_trajectories_path,
            point_fast_mode=True,
            game_target_offset=settings.live_game_target_offset,
            game_model_weight=settings.live_game_model_weight,
            game_markov_weight=settings.live_game_markov_weight,
            point_target_offset=settings.live_point_target_offset,
            point_model_weight=settings.live_point_model_weight,
            point_markov_weight=settings.live_point_markov_weight,
            point_execution_min_probability=settings.live_point_execution_min_probability,
            bet_mode=settings.live_bet_mode,
            express_size=settings.live_express_size,
        )


def normalize_name(value: str) -> str:
    letters = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return " ".join(letters.split())


def build_name_aliases(normalized: str) -> set[str]:
    aliases = {normalized}
    parts = normalized.split()
    if len(parts) >= 2:
        aliases.add(" ".join(reversed(parts)))
        first = parts[0]
        last = parts[-1]
        if first:
            aliases.add(f"{last} {first[0]}")
            aliases.add(f"{first[0]} {last}")
        if len(parts) >= 3:
            aliases.add(f"{last} {parts[1][0]}")
    return {alias for alias in aliases if alias}


class HistoricalLookup:
    def __init__(self, player_match_stats: pd.DataFrame | None = None, match_features_elo: pd.DataFrame | None = None):
        self.player_match_stats = player_match_stats if player_match_stats is not None else load_player_match_stats()
        self.match_features_elo = match_features_elo if match_features_elo is not None else load_match_features_elo()
        self._name_to_player_id = self._build_name_index()

    def _iter_player_csv_names(self) -> list[tuple[int, str]]:
        rows: list[tuple[int, str]] = []
        for path in discover_player_csv_files(must_exist=False):
            try:
                df = pd.read_csv(path, low_memory=False)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue
            columns = {column.lower(): column for column in df.columns}
            player_id_col = columns.get("player_id")
            first_name_col = columns.get("first_name") or columns.get("name_first")
            last_name_col = columns.get("last_name") or columns.get("name_last")
            if not player_id_col or not first_name_col or not last_name_col:
                continue
            namespace_offset = player_id_namespace_offset(path)
            ids = pd.to_numeric(df[player_id_col], errors="coerce")
            if namespace_offset:
                ids = ids + namespace_offset
            names = (
                df[first_name_col].fillna("").astype(str).str.strip()
                + " "
                + df[last_name_col].fillna("").astype(str).str.strip()
            ).str.strip()
            local = pd.DataFrame({"player_id": ids, "player_name": names})
            local = local.dropna(subset=["player_id"])
            local = local[local["player_name"] != ""]
            for row in local.itertuples(index=False):
                rows.append((int(row.player_id), str(row.player_name)))
        return rows

    def _build_name_index(self) -> dict[str, int]:
        latest_names = (
            self.player_match_stats.sort_values(["tourney_date", "match_id"])
            .dropna(subset=["player_id", "player_name"])
            .drop_duplicates(subset=["player_id"], keep="last")
        )
        mapping: dict[str, int] = {}
        for row in latest_names.itertuples(index=False):
            pid = int(row.player_id)
            normalized = normalize_name(str(row.player_name))
            if normalized:
                for alias in build_name_aliases(normalized):
                    mapping.setdefault(alias, pid)
        for pid, player_name in self._iter_player_csv_names():
            normalized = normalize_name(player_name)
            if normalized:
                for alias in build_name_aliases(normalized):
                    mapping.setdefault(alias, pid)
        return mapping

    @lru_cache(maxsize=32768)
    def resolve_player_id(self, player_name: str) -> int:
        normalized = normalize_name(player_name)
        if normalized in self._name_to_player_id:
            return self._name_to_player_id[normalized]
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]
        return -int(digest, 16)

    @lru_cache(maxsize=32768)
    def player_stats(self, pid: int, surface: str) -> dict[str, float]:
        df = self.player_match_stats[self.player_match_stats["player_id"] == pid].copy()
        df = df.sort_values(["tourney_date", "match_id"], ascending=[False, False]).head(50)
        if df.empty:
            return {
                "rank": 9999.0,
                "rank_points": 0.0,
                "winrate_last10": 0.5,
                "recent_form_last5": 0.5,
                "recent_form_last10": 0.5,
                "winrate_surface": 0.5,
                "surface_recent_form": 0.5,
                "hold_rate": 0.62,
                "break_rate": 0.24,
                "matches_last7days": 0,
                "matches_last14days": 0,
                "minutes_last14days": 0.0,
            }

        df["tourney_date"] = pd.to_datetime(df["tourney_date"])
        winrate_last10 = df.head(10)["is_win"].mean()
        recent_form_last5 = df.head(5)["is_win"].mean()
        recent_form_last10 = df.head(10)["is_win"].mean()

        surface_df = df[df["surface"] == surface]
        winrate_surface = surface_df.head(20)["is_win"].mean()
        if pd.isna(winrate_surface):
            winrate_surface = winrate_last10
        surface_recent_form = surface_df.head(10)["is_win"].mean()
        if pd.isna(surface_recent_form):
            surface_recent_form = recent_form_last10

        hold_num = (
            df["service_games"].fillna(0)
            - (df["bp_faced"].fillna(0) - df["bp_saved"].fillna(0))
        ).clip(lower=0)
        hold_den = df["service_games"].fillna(0)
        hold_rate = hold_num.sum() / hold_den.sum() if hold_den.sum() > 0 else 0.0

        break_num = (
            df["opp_bp_faced"].fillna(0) - df["opp_bp_saved"].fillna(0)
        ).clip(lower=0)
        break_den = df["opp_service_games"].fillna(0)
        break_rate = break_num.sum() / break_den.sum() if break_den.sum() > 0 else 0.0

        rank = df.iloc[0]["player_rank"]
        rank_points = df.iloc[0]["player_rank_points"]
        latest_date = df.iloc[0]["tourney_date"]
        matches_last7 = len(df[df["tourney_date"] >= latest_date - pd.Timedelta(days=7)])
        matches_last14 = len(df[df["tourney_date"] >= latest_date - pd.Timedelta(days=14)])
        minutes_last14 = df[df["tourney_date"] >= latest_date - pd.Timedelta(days=14)]["minutes"].fillna(0).sum()

        return {
            "rank": float(rank) if pd.notna(rank) else 9999.0,
            "rank_points": float(rank_points) if pd.notna(rank_points) else 0.0,
            "winrate_last10": float(winrate_last10) if pd.notna(winrate_last10) else 0.5,
            "recent_form_last5": float(recent_form_last5) if pd.notna(recent_form_last5) else 0.5,
            "recent_form_last10": float(recent_form_last10) if pd.notna(recent_form_last10) else 0.5,
            "winrate_surface": float(winrate_surface) if pd.notna(winrate_surface) else 0.5,
            "surface_recent_form": float(surface_recent_form) if pd.notna(surface_recent_form) else 0.5,
            "hold_rate": float(hold_rate),
            "break_rate": float(break_rate),
            "matches_last7days": int(matches_last7),
            "matches_last14days": int(matches_last14),
            "minutes_last14days": float(minutes_last14),
        }

    @lru_cache(maxsize=32768)
    def current_elo(self, pid: int, surface: str) -> tuple[float, float]:
        df = self.match_features_elo[
            (self.match_features_elo["p1_id"] == pid) | (self.match_features_elo["p2_id"] == pid)
        ].copy()
        df = df.sort_values(["tourney_date", "match_id"], ascending=[False, False]).head(1)
        if df.empty:
            return 1500.0, 1500.0

        row = df.iloc[0]
        if int(row["p1_id"]) == pid:
            elo = row["p1_elo"]
            surface_elo = row["p1_surface_elo"]
        else:
            elo = row["p2_elo"]
            surface_elo = row["p2_surface_elo"]
        return (
            float(elo) if pd.notna(elo) else 1500.0,
            float(surface_elo) if pd.notna(surface_elo) else 1500.0,
        )

    @lru_cache(maxsize=32768)
    def h2h_stats(self, player_1: int, player_2: int, surface: str) -> tuple[int, int, int, int]:
        h2h = self.player_match_stats[
            (
                (self.player_match_stats["player_id"] == player_1)
                & (self.player_match_stats["opponent_id"] == player_2)
            )
            | (
                (self.player_match_stats["player_id"] == player_2)
                & (self.player_match_stats["opponent_id"] == player_1)
            )
        ].copy()

        p1_h2h_wins = int(
            (
                (h2h["player_id"] == player_1)
                & (h2h["opponent_id"] == player_2)
                & (h2h["is_win"] == 1)
            ).sum()
        )
        p2_h2h_wins = int(
            (
                (h2h["player_id"] == player_2)
                & (h2h["opponent_id"] == player_1)
                & (h2h["is_win"] == 1)
            ).sum()
        )

        surface_h2h = h2h[h2h["surface"] == surface]
        p1_h2h_surface_wins = int(
            (
                (surface_h2h["player_id"] == player_1)
                & (surface_h2h["opponent_id"] == player_2)
                & (surface_h2h["is_win"] == 1)
            ).sum()
        )
        p2_h2h_surface_wins = int(
            (
                (surface_h2h["player_id"] == player_2)
                & (surface_h2h["opponent_id"] == player_1)
                & (surface_h2h["is_win"] == 1)
            ).sum()
        )
        return p1_h2h_wins, p2_h2h_wins, p1_h2h_surface_wins, p2_h2h_surface_wins

    def build_prediction_frame(self, market: LiveMarket) -> tuple[pd.DataFrame, dict[str, float], int, int]:
        player_1 = self.resolve_player_id(market.player1_name)
        player_2 = self.resolve_player_id(market.player2_name)

        p1 = self.player_stats(player_1, market.surface)
        p2 = self.player_stats(player_2, market.surface)
        p1_elo, p1_surface_elo = self.current_elo(player_1, market.surface)
        p2_elo, p2_surface_elo = self.current_elo(player_2, market.surface)
        p1_h2h_wins, p2_h2h_wins, p1_h2h_surface_wins, p2_h2h_surface_wins = self.h2h_stats(
            player_1,
            player_2,
            market.surface,
        )

        row = {
            "surface": market.surface,
            "tourney_level": market.tourney_level,
            "round": market.round_name,
            "best_of": market.best_of,
            "rank_diff_bucket": "even",
            "p1_rank": p1["rank"],
            "p2_rank": p2["rank"],
            "p1_rank_points": p1["rank_points"],
            "p2_rank_points": p2["rank_points"],
            "p1_winrate_last10": p1["winrate_last10"],
            "p2_winrate_last10": p2["winrate_last10"],
            "p1_recent_form_last5": p1["recent_form_last5"],
            "p2_recent_form_last5": p2["recent_form_last5"],
            "p1_recent_form_last10": p1["recent_form_last10"],
            "p2_recent_form_last10": p2["recent_form_last10"],
            "p1_winrate_surface": p1["winrate_surface"],
            "p2_winrate_surface": p2["winrate_surface"],
            "p1_surface_recent_form": p1["surface_recent_form"],
            "p2_surface_recent_form": p2["surface_recent_form"],
            "p1_hold_rate": p1["hold_rate"],
            "p2_hold_rate": p2["hold_rate"],
            "p1_break_rate": p1["break_rate"],
            "p2_break_rate": p2["break_rate"],
            "p1_matches_last7days": p1["matches_last7days"],
            "p2_matches_last7days": p2["matches_last7days"],
            "p1_matches_last14days": p1["matches_last14days"],
            "p2_matches_last14days": p2["matches_last14days"],
            "p1_minutes_last14days": p1["minutes_last14days"],
            "p2_minutes_last14days": p2["minutes_last14days"],
            "p1_h2h_wins": p1_h2h_wins,
            "p2_h2h_wins": p2_h2h_wins,
            "p1_h2h_surface_wins": p1_h2h_surface_wins,
            "p2_h2h_surface_wins": p2_h2h_surface_wins,
            "elo_diff": p1_elo - p2_elo,
            "surface_elo_diff": p1_surface_elo - p2_surface_elo,
        }

        row["rank_diff"] = row["p1_rank"] - row["p2_rank"]
        row["rank_diff_bucket"] = rank_diff_bucket(row["rank_diff"])
        row["rank_points_diff"] = row["p1_rank_points"] - row["p2_rank_points"]
        row["winrate_last10_diff"] = row["p1_winrate_last10"] - row["p2_winrate_last10"]
        row["recent_form_last5_diff"] = row["p1_recent_form_last5"] - row["p2_recent_form_last5"]
        row["recent_form_last10_diff"] = row["p1_recent_form_last10"] - row["p2_recent_form_last10"]
        row["winrate_surface_diff"] = row["p1_winrate_surface"] - row["p2_winrate_surface"]
        row["surface_recent_form_diff"] = row["p1_surface_recent_form"] - row["p2_surface_recent_form"]
        row["hold_rate_diff"] = row["p1_hold_rate"] - row["p2_hold_rate"]
        row["break_rate_diff"] = row["p1_break_rate"] - row["p2_break_rate"]
        row["matches_last7days_diff"] = row["p1_matches_last7days"] - row["p2_matches_last7days"]
        row["matches_last14days_diff"] = row["p1_matches_last14days"] - row["p2_matches_last14days"]
        row["minutes_last14days_diff"] = row["p1_minutes_last14days"] - row["p2_minutes_last14days"]
        return pd.DataFrame([row]), row, player_1, player_2


def compute_stake(
    model_probability: float,
    odds: float,
    default_stake: float,
    bankroll: float,
    kelly_fraction: float,
) -> float:
    if bankroll <= 0:
        return round(default_stake, 2)

    b = max(odds - 1.0, 0.0)
    q = 1.0 - model_probability
    if b <= 0:
        return round(default_stake, 2)
    kelly = max(((b * model_probability) - q) / b, 0.0)
    if kelly <= 0:
        return 0.0
    stake = bankroll * kelly * max(min(kelly_fraction, 1.0), 0.0)
    return round(max(stake, default_stake), 2)


def select_candidate(
    market: LiveMarket,
    player1_probability: float,
    player2_probability: float,
    player1_id: int,
    player2_id: int,
    config: RuntimeConfig,
) -> ScoredSelection | None:
    options = build_candidate_options(
        ("player1", market.player1_name, player1_probability, market.player1_odds, player1_id),
        ("player2", market.player2_name, player2_probability, market.player2_odds, player2_id),
        config=config,
        market=market,
    )
    best_candidate: ScoredSelection | None = None
    for candidate in options:
        if best_candidate is None or candidate.edge > best_candidate.edge:
            best_candidate = candidate
    return best_candidate


def build_candidate_options(
    *options: tuple[str, str, float, float, int],
    config: RuntimeConfig,
    market: LiveMarket,
) -> list[ScoredSelection]:
    candidates: list[ScoredSelection] = []
    for side, player_name, model_probability, odds, player_id in options:
        if odds < config.min_odds or odds > config.max_odds:
            continue
        if model_probability < config.min_model_probability:
            continue

        implied_probability = 1.0 / odds
        edge = model_probability - implied_probability
        if edge < config.edge_threshold:
            continue

        stake = compute_stake(
            model_probability=model_probability,
            odds=odds,
            default_stake=config.default_stake,
            bankroll=config.bankroll,
            kelly_fraction=config.kelly_fraction,
        )
        if stake <= 0:
            continue

        candidates.append(
            ScoredSelection(
                market=market,
                side=side,
                player_name=player_name,
                model_probability=model_probability,
                implied_probability=implied_probability,
                edge=edge,
                odds=odds,
                stake=stake,
                player_id=player_id,
            )
        )
    return candidates


def select_total_candidate(
    market: LiveMarket,
    over_probability: float,
    config: RuntimeConfig,
) -> ScoredSelection | None:
    options = [
        ("player1", "over", over_probability, market.player1_odds),
        ("player2", "under", 1.0 - over_probability, market.player2_odds),
    ]
    best_candidate: ScoredSelection | None = None
    for side, label, model_probability, odds in options:
        if odds < config.min_odds or odds > config.max_odds:
            continue
        if model_probability < config.min_model_probability:
            continue
        implied_probability = 1.0 / odds
        edge = model_probability - implied_probability
        if edge < config.edge_threshold:
            continue
        stake = compute_stake(
            model_probability=model_probability,
            odds=odds,
            default_stake=config.default_stake,
            bankroll=config.bankroll,
            kelly_fraction=config.kelly_fraction,
        )
        if stake <= 0:
            continue
        candidate = ScoredSelection(
            market=market,
            side=side,
            player_name=label,
            model_probability=model_probability,
            implied_probability=implied_probability,
            edge=edge,
            odds=odds,
            stake=stake,
            player_id=0,
        )
        if best_candidate is None or candidate.edge > best_candidate.edge:
            best_candidate = candidate
    return best_candidate


class RuntimeState:
    def __init__(self, path: Path, initial_bankroll: float = 0.0):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.placed_selection_ids: set[str] = set()
        self.current_bankroll = float(initial_bankroll)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.placed_selection_ids = set(payload.get("placed_selection_ids", []))
            stored_bankroll = payload.get("current_bankroll")
            if stored_bankroll is not None:
                self.current_bankroll = float(stored_bankroll)
            elif initial_bankroll > 0:
                self.current_bankroll = float(initial_bankroll)

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "placed_selection_ids": sorted(self.placed_selection_ids),
                    "current_bankroll": self.current_bankroll,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    def has_seen(self, selection_id: str) -> bool:
        return selection_id in self.placed_selection_ids

    def mark_placed(self, selection_id: str) -> None:
        self.placed_selection_ids.add(selection_id)
        self._save()

    def apply_profit(self, profit: float) -> float:
        self.current_bankroll = round(self.current_bankroll + profit, 2)
        self._save()
        return self.current_bankroll


class RLDatasetLogger:
    def __init__(self, snapshots_path: Path, actions_path: Path, outcomes_path: Path, point_trajectories_path: Path):
        self.snapshots_path = snapshots_path
        self.actions_path = actions_path
        self.outcomes_path = outcomes_path
        self.point_trajectories_path = point_trajectories_path
        self.snapshots_path.parent.mkdir(parents=True, exist_ok=True)
        self.actions_path.parent.mkdir(parents=True, exist_ok=True)
        self.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        self.point_trajectories_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_path.touch(exist_ok=True)
        self.actions_path.touch(exist_ok=True)
        self.outcomes_path.touch(exist_ok=True)
        self.point_trajectories_path.touch(exist_ok=True)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def log_snapshot(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.snapshots_path, payload)

    def log_action(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.actions_path, payload)

    def log_outcome(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.outcomes_path, payload)

    def log_point_trajectory(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.point_trajectories_path, payload)


class RLOutcomeTracker:
    def __init__(self, state_path: Path, market_close_cycles: int):
        self.state_path = state_path
        self.market_close_cycles = market_close_cycles
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_cycle = 0
        self.pending_by_event: dict[str, list[dict[str, Any]]] = {}
        self.market_last_seen_cycle: dict[str, int] = {}
        self.market_last_snapshot: dict[str, dict[str, Any]] = {}
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.current_cycle = int(payload.get("current_cycle", 0))
            self.pending_by_event = {
                str(key): list(value)
                for key, value in payload.get("pending_by_event", {}).items()
            }
            self.market_last_seen_cycle = {
                str(key): int(value)
                for key, value in payload.get("market_last_seen_cycle", {}).items()
            }
            self.market_last_snapshot = {
                str(key): dict(value)
                for key, value in payload.get("market_last_snapshot", {}).items()
            }

    def save(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "current_cycle": self.current_cycle,
                    "pending_by_event": self.pending_by_event,
                    "market_last_seen_cycle": self.market_last_seen_cycle,
                    "market_last_snapshot": self.market_last_snapshot,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    def start_cycle(self) -> None:
        self.current_cycle += 1
        self.save()

    def observe_market(self, event_id: str, snapshot: dict[str, Any]) -> None:
        self.market_last_seen_cycle[event_id] = self.current_cycle
        self.market_last_snapshot[event_id] = snapshot
        self.save()

    def record_pending_bet(
        self,
        event_id: str,
        market_id: str,
        candidate: ScoredSelection,
        result: dict[str, Any],
        snapshot: dict[str, Any],
        dry_run: bool,
    ) -> dict[str, Any]:
        bet_result = result.get("bet_result_response", {})
        coupon = bet_result.get("coupon", {})
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id,
            "market_id": market_id,
            "market_type": candidate.market.market_type,
            "target_game_number": candidate.market.raw.get("target_game_number"),
            "selection_id": candidate.selection_id,
            "side": candidate.side,
            "player_name": candidate.player_name,
            "odds_taken": candidate.odds,
            "stake": candidate.stake,
            "edge": candidate.edge,
            "model_probability": candidate.model_probability,
            "status": "pending_dry_run" if dry_run else "pending",
            "done": False,
            "reward": None,
            "reg_id": coupon.get("regId"),
            "check_code": coupon.get("checkCode"),
            "request_result": result.get("status"),
            "opening_snapshot": snapshot,
            "final_snapshot": snapshot,
        }
        self.pending_by_event.setdefault(event_id, []).append(record)
        self.save()
        return record

    def close_missing_markets(self, seen_event_ids: set[str]) -> list[dict[str, Any]]:
        closed: list[dict[str, Any]] = []
        for event_id in list(self.pending_by_event):
            if event_id in seen_event_ids:
                continue
            last_seen = self.market_last_seen_cycle.get(event_id, self.current_cycle)
            if self.current_cycle - last_seen < self.market_close_cycles:
                continue
            final_snapshot = self.market_last_snapshot.get(event_id)
            for record in self.pending_by_event.pop(event_id):
                closed.append(
                    {
                        **record,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "market_closed_unsettled",
                        "done": True,
                        "reward": None,
                        "final_snapshot": final_snapshot,
                    }
                )
        if closed:
            self.save()
        return closed

    def pending_exposure(self, include_dry_run: bool = False) -> float:
        total = 0.0
        for records in self.pending_by_event.values():
            for record in records:
                if record.get("done"):
                    continue
                if not include_dry_run and record.get("status") == "pending_dry_run":
                    continue
                total += float(record.get("stake") or 0.0)
        return round(total, 2)


def _winner_from_match_snapshot(snapshot: dict[str, Any] | None) -> str | None:
    if not snapshot:
        return None
    live_score = snapshot.get("live_score")
    if not isinstance(live_score, str) or ":" not in live_score:
        return None
    try:
        left, right = live_score.split(":", 1)
        sets_1 = int(left)
        sets_2 = int(right)
    except ValueError:
        return None
    best_of = int(snapshot.get("best_of") or 3)
    sets_needed = (best_of // 2) + 1
    if sets_1 >= sets_needed and sets_1 > sets_2:
        return "player1"
    if sets_2 >= sets_needed and sets_2 > sets_1:
        return "player2"
    return None


def _games_from_snapshot(snapshot: dict[str, Any] | None) -> tuple[int, int] | None:
    if not snapshot:
        return None
    live_score = snapshot.get("live_score")
    if not isinstance(live_score, str) or ":" not in live_score:
        return None
    try:
        left, right = live_score.split(":", 1)
        return int(left), int(right)
    except ValueError:
        return None


def _exact_points_from_comment(comment: str | None) -> tuple[int, int] | None:
    if not comment:
        return None
    text = str(comment)
    if "points:" in text:
        suffix = text.split("points:", 1)[1].strip()
        if "-" in suffix:
            left, right = suffix.split("-", 1)
            try:
                return int(left.strip()), int(right.strip().split()[0])
            except ValueError:
                return None
    body = text.strip().strip("()")
    if " " in body:
        body = body.split(" ", 1)[0]
    body = body.replace("*", "")
    if "-" not in body:
        return None
    left, right = body.split("-", 1)
    score_map = {"00": 0, "0": 0, "15": 1, "30": 2, "40": 3}
    left_score = score_map.get(left.strip())
    right_score = score_map.get(right.strip())
    if left_score is None or right_score is None:
        return None
    if left_score >= 3 and right_score >= 3:
        return None
    return left_score, right_score


def _game_winner_from_game_delta(
    opening_snapshot: dict[str, Any] | None,
    final_snapshot: dict[str, Any] | None,
) -> str | None:
    opening_games = _games_from_snapshot(opening_snapshot)
    final_games = _games_from_snapshot(final_snapshot)
    if opening_games is None or final_games is None:
        return None
    p1_delta = final_games[0] - opening_games[0]
    p2_delta = final_games[1] - opening_games[1]
    if p1_delta == 1 and p2_delta == 0:
        return "player1"
    if p2_delta == 1 and p1_delta == 0:
        return "player2"
    return None


def _is_game_complete(points: tuple[int, int]) -> bool:
    return max(points) >= 4 and abs(points[0] - points[1]) >= 2


def _winner_from_point_snapshots(
    opening_snapshot: dict[str, Any] | None,
    final_snapshot: dict[str, Any] | None,
    target_point_number: int | None,
) -> str | None:
    if target_point_number is None:
        return None
    if not opening_snapshot or not final_snapshot:
        return None
    opening_points = _exact_points_from_comment(opening_snapshot.get("live_comment"))
    if opening_points is None:
        return None
    opening_total = opening_points[0] + opening_points[1]
    if target_point_number <= opening_total:
        return None

    final_points = None
    final_mode: tuple[str, Any] | None = None
    opening_games = _games_from_snapshot(opening_snapshot)
    final_games = _games_from_snapshot(final_snapshot)
    if opening_games is not None and final_games is not None and opening_games == final_games:
        final_points = _exact_points_from_comment(final_snapshot.get("live_comment"))
        if final_points is None:
            return None
        if final_points[0] + final_points[1] < target_point_number:
            return None
        final_mode = ("points", final_points)
        max_steps = (final_points[0] + final_points[1]) - opening_total
    else:
        game_winner = _game_winner_from_game_delta(opening_snapshot, final_snapshot)
        if game_winner is None:
            return None
        final_mode = ("game_winner", game_winner)
        max_steps = 12

    winners: set[str] = set()

    def search(points: tuple[int, int], points_played: int, target_winner: str | None, steps_left: int) -> None:
        if steps_left < 0:
            return
        if final_mode is None:
            return
        mode, payload = final_mode
        if mode == "points" and points == payload:
            if target_winner is not None:
                winners.add(target_winner)
            return
        if mode == "game_winner" and _is_game_complete(points):
            winner = "player1" if points[0] > points[1] else "player2"
            if winner == payload and target_winner is not None:
                winners.add(target_winner)
            return
        if _is_game_complete(points):
            return

        for side, winner_label in ((0, "player1"), (1, "player2")):
            updated = [points[0], points[1]]
            updated[side] += 1
            next_points = (updated[0], updated[1])
            next_total = points_played + 1
            next_target_winner = target_winner
            if next_total == target_point_number:
                next_target_winner = winner_label
            search(next_points, next_total, next_target_winner, steps_left - 1)

    search(opening_points, opening_total, None, max_steps)
    if len(winners) == 1:
        return next(iter(winners))
    return None


def _winner_from_next_game_snapshots(
    opening_snapshot: dict[str, Any] | None,
    final_snapshot: dict[str, Any] | None,
    target_game_number: int | None,
) -> str | None:
    opening_games = _games_from_snapshot(opening_snapshot)
    final_games = _games_from_snapshot(final_snapshot)
    if opening_games is None or final_games is None:
        return None

    opening_total = opening_games[0] + opening_games[1]
    final_total = final_games[0] + final_games[1]
    if target_game_number is not None and opening_total > max(target_game_number - 1, 0):
        return None
    if final_total <= opening_total:
        return None

    p1_delta = final_games[0] - opening_games[0]
    p2_delta = final_games[1] - opening_games[1]
    if p1_delta <= 0 and p2_delta <= 0:
        return None
    if p1_delta > p2_delta:
        return "player1"
    if p2_delta > p1_delta:
        return "player2"
    return None


def _winner_from_total_snapshot(
    final_snapshot: dict[str, Any] | None,
    total_line: float | None,
) -> str | None:
    if total_line is None:
        return None
    final_games = _games_from_snapshot(final_snapshot)
    if final_games is None:
        return None
    final_total = final_games[0] + final_games[1]
    if final_total > total_line:
        return "player1"
    if final_total < total_line:
        return "player2"
    return None


def settle_outcome_record(
    record: dict[str, Any],
    final_snapshot: dict[str, Any] | None,
    bankroll_before: float,
) -> dict[str, Any]:
    settled = {
        **record,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_snapshot": final_snapshot,
        "bankroll_before": round(bankroll_before, 2),
        "bankroll_after": round(bankroll_before, 2),
    }
    market_type = str(record.get("market_type") or "")
    if market_type == "match_winner":
        winner_side = _winner_from_match_snapshot(final_snapshot)
    elif market_type == "next_game_winner":
        target_game_number = record.get("target_game_number")
        try:
            parsed_target_game_number = int(target_game_number) if target_game_number is not None else None
        except (TypeError, ValueError):
            parsed_target_game_number = None
        winner_side = _winner_from_next_game_snapshots(
            opening_snapshot=record.get("opening_snapshot"),
            final_snapshot=final_snapshot,
            target_game_number=parsed_target_game_number,
        )
    elif market_type == "point_plus_one_winner":
        target_point_number = record.get("opening_snapshot", {}).get("state_features", {}).get("target_point_number")
        if target_point_number is None:
            target_point_number = record.get("opening_snapshot", {}).get("target_point_number")
        try:
            parsed_target_point_number = int(target_point_number) if target_point_number is not None else None
        except (TypeError, ValueError):
            parsed_target_point_number = None
        winner_side = _winner_from_point_snapshots(
            opening_snapshot=record.get("opening_snapshot"),
            final_snapshot=final_snapshot,
            target_point_number=parsed_target_point_number,
        )
    elif market_type == "set_total_over_under":
        opening_snapshot = record.get("opening_snapshot") or {}
        total_line = opening_snapshot.get("state_features", {}).get("total_line")
        if total_line is None:
            total_line = opening_snapshot.get("player1_param")
            try:
                total_line = float(total_line) / 100.0 if total_line is not None else None
            except (TypeError, ValueError):
                total_line = None
        winner_side = _winner_from_total_snapshot(final_snapshot, total_line)
    else:
        settled["status"] = "market_closed_unsettled"
        settled["done"] = True
        settled["reward"] = None
        settled["profit"] = None
        return settled

    if winner_side is None:
        settled["status"] = "market_closed_unsettled"
        settled["done"] = True
        settled["reward"] = None
        settled["profit"] = None
        return settled

    stake = float(record.get("stake") or 0.0)
    odds = float(record.get("odds_taken") or 0.0)
    won = winner_side == record.get("side")
    profit = round(stake * max(odds - 1.0, 0.0), 2) if won else round(-stake, 2)
    bankroll_after = round(bankroll_before + profit, 2)
    reward = round(profit / bankroll_before, 4) if bankroll_before > 0 else round(profit / max(stake, 1.0), 4)
    settled.update(
        {
            "status": "won" if won else "lost",
            "done": True,
            "winner_side": winner_side,
            "profit": profit,
            "reward": reward,
            "bankroll_after": bankroll_after,
        }
    )
    return settled


class LiveBettingRuntime:
    def __init__(
        self,
        market_feed_client: Any,
        bet_executor: Any,
        config: RuntimeConfig | None = None,
        lookup: HistoricalLookup | None = None,
    ):
        self.market_feed_client = market_feed_client
        self.bet_executor = bet_executor
        self.config = config or RuntimeConfig.from_settings()
        self.lookup = lookup or HistoricalLookup()
        self.audit_logger = BettingAuditLogger(settings.market_bet_log_path)
        try:
            self.db_bet_log_recorder = DatabaseBetLogRecorder(get_engine())
        except Exception:
            self.db_bet_log_recorder = DatabaseBetLogRecorder(None)
        self.betting_policy = BettingPolicy(
            BettingPolicyConfig(
                min_value=max(self.config.edge_threshold, 0.07),
                min_model_probability=max(self.config.min_model_probability, 0.58),
                min_data_quality_score=0.7,
                min_recent_form=0.4,
            )
        )
        self.model = joblib.load(self.config.model_path)
        point_model_path = settings.models_dir / "historical_point_model.joblib"
        execution_model_path = settings.models_dir / "execution_survival_model.joblib"
        self.point_predictor = LayeredPointPredictor(
            point_model=joblib.load(point_model_path) if point_model_path.exists() else None,
            execution_model=joblib.load(execution_model_path) if execution_model_path.exists() else None,
            target_offset=self.config.point_target_offset,
            point_model_weight=self.config.point_model_weight,
            markov_weight=self.config.point_markov_weight,
        )
        total_model_path = settings.models_dir / "historical_total_model.joblib"
        self.total_model = SetTotalModel(
            historical_model=joblib.load(total_model_path) if total_model_path.exists() else None
        )
        self.state = RuntimeState(self.config.state_path, initial_bankroll=self.config.bankroll)
        self.game_model = MarkovGameModel()
        game_model_path = settings.models_dir / "historical_game_model.joblib"
        self.game_predictor = LayeredGamePredictor(
            game_model=joblib.load(game_model_path) if game_model_path.exists() else None,
            game_model_weight=self.config.game_model_weight,
            markov_weight=self.config.game_markov_weight,
        )
        self.rl_logger = RLDatasetLogger(
            self.config.rl_snapshots_path,
            self.config.rl_actions_path,
            self.config.rl_outcomes_path,
            self.config.point_trajectories_path,
        )
        self.rl_outcome_tracker = RLOutcomeTracker(
            self.config.rl_tracker_state_path,
            self.config.rl_market_close_cycles,
        )
        self.bankroll_policy = BankrollBanditPolicy(
            outcomes_path=self.config.rl_outcomes_path,
            bankroll=self.state.current_bankroll,
            starting_bankroll=self.config.bankroll if self.config.bankroll > 0 else self.state.current_bankroll,
        )
        acceptance_model_path = settings.models_dir / "leg_acceptance_model.joblib"
        metadata_path = settings.models_dir / "leg_acceptance_model.json"
        self.acceptance_model = joblib.load(acceptance_model_path) if acceptance_model_path.exists() else None
        self.acceptance_metadata = {}
        if metadata_path.exists():
            self.acceptance_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.rl_policy_path = settings.models_dir / "rl_bankroll_policy.json"

    def _available_bankroll(self) -> float:
        if self.config.bankroll <= 0:
            return 0.0
        pending_exposure = self.rl_outcome_tracker.pending_exposure(include_dry_run=False)
        return round(max(self.state.current_bankroll - pending_exposure, 0.0), 2)

    def score_market(self, market: LiveMarket) -> ScoredSelection | None:
        _, _, _, _, candidate = self._score_market_details(market)
        return candidate

    def _export_rl_policy(self) -> None:
        stats = self.bankroll_policy._load_stats()
        self.rl_policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.rl_policy_path.write_text(
            json.dumps(
                {
                    "model_type": "bankroll_bandit_policy",
                    "bankroll": self.state.current_bankroll,
                    "stake_levels": list(BankrollBanditPolicy.STAKE_LEVELS),
                    "performance": self.bankroll_policy.performance_profile(),
                    "arms": stats,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _acceptance_feature_frame(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
    ) -> pd.DataFrame:
        row = {
            "action": "bet_express" if self.config.bet_mode == "express" else "bet",
            "market_type": market.market_type,
            "side": candidate.side,
            "player_name": candidate.player_name,
            "odds": candidate.odds,
            "stake": candidate.stake,
            "model_probability": candidate.model_probability,
            "implied_probability": candidate.implied_probability,
            "edge": candidate.edge,
            "coupon_legs": self.config.express_size if self.config.bet_mode == "express" else 1,
            "slip_factor_id": market.raw.get("player1_factor_id") if candidate.side == "player1" else market.raw.get("player2_factor_id"),
            "slip_factor_value": candidate.odds,
            "slip_factor_param": market.raw.get("player1_param") if candidate.side == "player1" else market.raw.get("player2_param"),
            "slip_score": market.raw.get("score"),
        }
        return pd.DataFrame([row])

    def _acceptance_probability(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
    ) -> float:
        metadata_rows = int(self.acceptance_metadata.get("rows") or 0)
        metadata_model_type = str(self.acceptance_metadata.get("model_type") or "")
        if metadata_model_type == "dummy_classifier" or metadata_rows < 50:
            return 0.5
        if self.acceptance_model is None:
            return 1.0
        try:
            frame = self._acceptance_feature_frame(market, candidate)
            if hasattr(self.acceptance_model, "predict_proba"):
                probability = float(self.acceptance_model.predict_proba(frame)[0][1])
                return min(max(probability, 0.0), 1.0)
        except Exception as exc:
            LOGGER.debug("Acceptance model failed for %s: %s", candidate.selection_id, exc)
        positive_rate = self.acceptance_metadata.get("positive_rate")
        if isinstance(positive_rate, (float, int)):
            return min(max(float(positive_rate), 0.35), 0.65)
        return 1.0

    def _candidate_ranking_score(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
    ) -> tuple[float, float]:
        acceptance_probability = self._acceptance_probability(market, candidate)
        ranking_score = float(candidate.edge) * acceptance_probability * self.bankroll_policy.risk_multiplier()
        return acceptance_probability, ranking_score

    def _calibrate_probability(self, probability: float) -> float:
        self.bankroll_policy.bankroll = self._available_bankroll()
        return self.bankroll_policy.calibrate_probability(probability)

    def _score_market_details(
        self,
        market: LiveMarket,
    ) -> tuple[pd.DataFrame, dict[str, Any], float, float, ScoredSelection | None]:
        frame, row, player1_id, player2_id = self.lookup.build_prediction_frame(market)
        if market.market_type == "next_game_winner":
            game_prediction = self.game_predictor.predict(
                market=market,
                markov_probability=self.game_model.predict_next_game(frame.iloc[0].to_dict(), market.raw),
            )
            player1_probability = game_prediction.player1_probability
            row["game_markov_probability"] = game_prediction.markov_probability
            row["game_historical_probability"] = game_prediction.historical_probability
        elif market.market_type == "point_plus_one_winner":
            layered_prediction = self.point_predictor.predict(
                market=market,
                state_features=frame.iloc[0].to_dict(),
                markov_probability=self.game_model.predict_point_plus_one(frame.iloc[0].to_dict(), market.raw),
            )
            player1_probability = layered_prediction.player1_probability
            row["point_markov_probability"] = layered_prediction.markov_probability
            row["point_historical_probability"] = layered_prediction.historical_probability
            row["point_execution_probability"] = layered_prediction.execution_probability
        elif market.market_type == "set_total_over_under":
            total_prediction = self.total_model.predict_over(market, frame.iloc[0].to_dict())
            player1_probability = total_prediction.over_probability
            row["total_heuristic_probability"] = total_prediction.heuristic_probability
            row["total_historical_probability"] = total_prediction.historical_probability
        else:
            features = align_frame_to_model(self.model, frame, ELO_FEATURES)
            player1_probability = float(self.model.predict_proba(features)[0][1])
        player1_probability = self._calibrate_probability(player1_probability)
        player2_probability = float(1.0 - player1_probability)
        if market.market_type == "point_plus_one_winner":
            execution_probability = row.get("point_execution_probability")
            if execution_probability is not None and execution_probability < self.config.point_execution_min_probability:
                return frame, row, player1_probability, player2_probability, None
        if market.market_type == "set_total_over_under":
            candidate = select_total_candidate(
                market=market,
                over_probability=player1_probability,
                config=self.config,
            )
        else:
            candidate_options = build_candidate_options(
                ("player1", market.player1_name, player1_probability, market.player1_odds, player1_id),
                ("player2", market.player2_name, player2_probability, market.player2_odds, player2_id),
                config=self.config,
                market=market,
            )
            self.bankroll_policy.bankroll = self._available_bankroll()
            candidate = self.bankroll_policy.recommend(candidate_options, market.market_type)
        if candidate is None:
            return frame, row, player1_probability, player2_probability, None
        acceptance_probability, ranking_score = self._candidate_ranking_score(market, candidate)
        return (
            frame,
            row,
            player1_probability,
            player2_probability,
            replace(
                candidate,
                acceptance_probability=acceptance_probability,
                ranking_score=ranking_score,
            ),
        )

    def _market_with_refresh(self, market: LiveMarket, refreshed_selection: dict[str, Any]) -> LiveMarket:
        raw = dict(market.raw)
        factor_id = refreshed_selection.get("factor_id")
        if refreshed_selection.get("score") not in (None, ""):
            raw["score"] = refreshed_selection["score"]
        if refreshed_selection.get("value") not in (None, ""):
            if factor_id == raw.get("player1_factor_id"):
                raw["player1_value"] = refreshed_selection["value"]
            if factor_id == raw.get("player2_factor_id"):
                raw["player2_value"] = refreshed_selection["value"]
        if refreshed_selection.get("param") not in (None, ""):
            if factor_id == raw.get("player1_factor_id"):
                raw["player1_param"] = refreshed_selection["param"]
            if factor_id == raw.get("player2_factor_id"):
                raw["player2_param"] = refreshed_selection["param"]
        player1_odds = float(raw.get("player1_value", market.player1_odds))
        player2_odds = float(raw.get("player2_value", market.player2_odds))
        return replace(market, player1_odds=player1_odds, player2_odds=player2_odds, raw=raw)

    def _candidate_rejection_reason(
        self,
        market: LiveMarket,
        player1_probability: float,
        player2_probability: float,
    ) -> str:
        reasons: list[str] = []
        options = (
            ("player1", market.player1_name, player1_probability, market.player1_odds),
            ("player2", market.player2_name, player2_probability, market.player2_odds),
        )
        for side, _player_name, model_probability, odds in options:
            if odds < self.config.min_odds or odds > self.config.max_odds:
                reasons.append(f"{side}:odds_out_of_range:{odds}")
                continue
            if model_probability < self.config.min_model_probability:
                reasons.append(f"{side}:model_probability_below_min:{round(model_probability, 6)}")
                continue
            implied_probability = 1.0 / odds
            edge = model_probability - implied_probability
            if edge < self.config.edge_threshold:
                reasons.append(f"{side}:edge_below_threshold:{round(edge, 6)}")
                continue
            reasons.append(f"{side}:eligible")
        return ";".join(reasons)

    def _refresh_rejection_reason(
        self,
        candidate: ScoredSelection,
        refreshed_market: LiveMarket,
        player1_probability: float,
        player2_probability: float,
    ) -> str:
        selected_odds = refreshed_market.player1_odds if candidate.side == "player1" else refreshed_market.player2_odds
        if selected_odds <= 0:
            return (
                "selected_side_closed_after_refresh:"
                f"{candidate.side}:odds={selected_odds};"
                f"{self._candidate_rejection_reason(refreshed_market, player1_probability, player2_probability)}"
            )
        return (
            "candidate_invalid_after_refresh:"
            f"{self._candidate_rejection_reason(refreshed_market, player1_probability, player2_probability)}"
        )

    def _refresh_and_rescore(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
    ) -> tuple[LiveMarket, dict[str, Any], dict[str, Any], dict[str, Any], float, float, ScoredSelection | None]:
        refreshed_selection, slip_info_response = self.bet_executor.refresh_candidate(candidate)
        refreshed_market = self._market_with_refresh(market, refreshed_selection)
        _, refreshed_row, player1_probability, player2_probability, refreshed_candidate = self._score_market_details(
            refreshed_market
        )
        return (
            refreshed_market,
            refreshed_selection,
            slip_info_response,
            refreshed_row,
            player1_probability,
            player2_probability,
            refreshed_candidate,
        )

    def _data_quality_score(self, features: dict[str, Any]) -> float:
        if len(features) <= 2:
            return 1.0

        groups = (
            ("surface", "tourney_level"),
            ("rank_diff", "elo_diff", "surface_elo_diff"),
            ("recent_form_last5_diff", "p1_recent_form_last5", "p2_recent_form_last5"),
            ("hold_rate_diff", "p1_hold_rate", "p2_hold_rate"),
            ("break_rate_diff", "p1_break_rate", "p2_break_rate"),
        )
        present = 0
        for group in groups:
            for key in group:
                value = features.get(key)
                if value is None:
                    continue
                if isinstance(value, float) and pd.isna(value):
                    continue
                present += 1
                break
        return round(present / len(groups), 4)

    def _recent_form_value(self, features: dict[str, Any], candidate: ScoredSelection) -> float | None:
        if candidate.market.market_type == "set_total_over_under":
            p1 = features.get("p1_recent_form_last5")
            p2 = features.get("p2_recent_form_last5")
            if p1 is None or p2 is None:
                return None
            return float((float(p1) + float(p2)) / 2.0)
        key = "p1_recent_form_last5" if candidate.side == "player1" else "p2_recent_form_last5"
        value = features.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)

    def _betting_policy_decision(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
        features: dict[str, Any],
    ):
        line_movement = market.raw.get("line_movement")
        if line_movement is None:
            line_movement = features.get("odds_movement")
        try:
            parsed_line_movement = None if line_movement is None else float(line_movement)
        except (TypeError, ValueError):
            parsed_line_movement = None
        market_label = {
            "match_winner": "match_winner",
            "set_total_over_under": "games_total",
            "next_game_winner": "games_handicap",
            "point_plus_one_winner": "point_plus_one",
        }.get(market.market_type, market.market_type)
        return self.betting_policy.evaluate(
            MarketCandidate(
                match_id=str(market.event_id),
                market=market_label,
                pick=candidate.player_name,
                odds=float(candidate.odds),
                model_prob=float(candidate.model_probability),
                stake=float(candidate.stake),
                tournament=market.competition,
                tournament_level=market.tourney_level,
                surface=market.surface,
                explanation={
                    "elo_diff": features.get("elo_diff"),
                    "surface_elo_diff": features.get("surface_elo_diff"),
                    "avg_total_games": features.get("avg_total_games"),
                    "hold_rate_diff": features.get("hold_rate_diff"),
                    "break_rate_diff": features.get("break_rate_diff"),
                    "three_set_rate_diff": features.get("three_set_rate_diff"),
                    "line_movement": parsed_line_movement,
                },
                data_quality_score=self._data_quality_score(features),
                recent_form=self._recent_form_value(features, candidate),
                odds_movement=parsed_line_movement,
            )
        )

    def _record_bet_log(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
        decision: Any,
        *,
        status: str,
        features: dict[str, Any],
        profit: float | None = None,
        settled_at: str | None = None,
    ) -> None:
        payload = {
            "settled_at": settled_at,
            "match_id": str(market.event_id),
            "event_id": str(market.event_id),
            "market": {
                "match_winner": "match_winner",
                "set_total_over_under": "games_total",
                "next_game_winner": "games_handicap",
                "point_plus_one_winner": "point_plus_one",
            }.get(market.market_type, market.market_type),
            "market_type": market.market_type,
            "pick": candidate.player_name,
            "odds": float(candidate.odds),
            "stake": float(candidate.stake),
            "result": status,
            "profit": profit,
            "model_prob": float(candidate.model_probability),
            "bookmaker_prob": float(decision.bookmaker_prob),
            "value": float(decision.value),
            "confidence": float(decision.confidence),
            "threshold_value": float(self.betting_policy.config.min_value),
            "min_probability": float(self.betting_policy.config.min_model_probability),
            "data_quality_score": float(self._data_quality_score(features)),
            "filter_surface": market.surface,
            "filter_tourney_level": market.tourney_level,
            "filter_form_window": 5,
            "filter_passed": bool(decision.passed_filters),
            "decision_reason": str(decision.reason),
            "explanation_json": decision.explanation,
            "source_json": {
                "market_id": market.market_id,
                "competition": market.competition,
                "player1_name": market.player1_name,
                "player2_name": market.player2_name,
                "live_score": market.raw.get("score"),
                "state_features": features,
            },
        }
        self.audit_logger.append(payload)
        try:
            self.db_bet_log_recorder.write(payload)
        except Exception:
            LOGGER.debug("Failed to persist bet_log record for %s", market.market_id)

    def _append_decision_log(self, record: dict[str, Any]) -> None:
        self.config.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _place_single_candidate(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
        decision: Any,
        features: dict[str, Any],
        player1_probability: float,
        player2_probability: float,
        refreshed_selection: dict[str, Any] | None,
        slip_info_response: dict[str, Any] | None,
        actions: list[dict[str, Any]],
        mode: str | None = None,
    ) -> None:
        available_bankroll = self._available_bankroll()
        if available_bankroll > 0 and candidate.stake > available_bankroll:
            self.rl_logger.log_action(
                self._action_record(
                    market=market,
                    candidate=candidate,
                    action="bankroll_blocked",
                    reason=f"stake_exceeds_available_bankroll:{available_bankroll}",
                )
            )
            self._append_decision_log(
                {
                    "market_id": market.market_id,
                    "event_id": market.event_id,
                    "selection_id": candidate.selection_id,
                    "status": "bankroll_blocked",
                    "stake": candidate.stake,
                    "available_bankroll": available_bankroll,
                }
            )
            return
        try:
            result = self.bet_executor.place_prepared_bet(
                candidate,
                refreshed_selection or {},
                slip_info_response or {},
            )
        except Exception as exc:
            LOGGER.warning("Bet placement failed for %s: %s", candidate.selection_id, exc)
            self.rl_logger.log_action(
                self._action_record(
                    market=market,
                    candidate=candidate,
                    action="execution_error",
                    reason=str(exc),
                )
            )
            self._append_decision_log(
                {
                    "market_id": market.market_id,
                    "event_id": market.event_id,
                    "selection_id": candidate.selection_id,
                    "status": "execution_error",
                    "reason": str(exc),
                }
            )
            return
        result_status = str(result.get("status") or "unknown")
        if result_status not in SUCCESSFUL_PLACEMENT_STATUSES:
            self.rl_logger.log_snapshot(
                self._snapshot_record(
                    market=market,
                    features=features,
                    candidate=candidate,
                    status=result_status,
                    reason=str(result.get("error") or decision.reason),
                    player1_probability=player1_probability,
                    player2_probability=player2_probability,
                )
            )
            self.rl_logger.log_action(
                self._action_record(
                    market=market,
                    candidate=candidate,
                    action=result_status,
                    result=result,
                    reason=str(result.get("error") or decision.reason),
                )
            )
            self._append_decision_log(
                {
                    "market_id": market.market_id,
                    "event_id": market.event_id,
                    "selection_id": candidate.selection_id,
                    "status": result_status,
                    "reason": str(result.get("error") or decision.reason),
                }
            )
            return
        self.state.mark_placed(candidate.selection_id)
        placed_snapshot = self._snapshot_record(
            market=market,
            features=features,
            candidate=candidate,
            status=result_status,
            player1_probability=player1_probability,
            player2_probability=player2_probability,
        )
        self.rl_logger.log_snapshot(placed_snapshot)
        self.rl_outcome_tracker.observe_market(market.event_id, placed_snapshot)
        stage = "bet_placed" if result_status != "dry_run" else "bet_dry_run"
        if mode == "point_fast_mode":
            stage = "bet_placed_fast_mode" if result_status != "dry_run" else "bet_dry_run_fast_mode"
        self._log_point_trajectory(
            market=market,
            stage=stage,
            features=features,
            candidate=candidate,
            result=result,
            player1_probability=player1_probability,
            player2_probability=player2_probability,
        )
        action_name = "bet" if result_status != "dry_run" else "bet_dry_run"
        if mode == "point_fast_mode":
            action_name = "bet_fast_mode" if result_status != "dry_run" else "bet_dry_run_fast_mode"
        self.rl_logger.log_action(
            self._action_record(
                market=market,
                candidate=candidate,
                action=action_name,
                result=result,
                reason="point_fast_mode_fallback" if mode == "point_fast_mode" else None,
            )
        )
        self.rl_logger.log_outcome(
            self.rl_outcome_tracker.record_pending_bet(
                event_id=market.event_id,
                market_id=market.market_id,
                candidate=candidate,
                result=result,
                snapshot=placed_snapshot,
                dry_run=result_status == "dry_run",
            )
        )
        record = {
            "market_id": market.market_id,
            "event_id": market.event_id,
            "selection_id": candidate.selection_id,
            "player_name": candidate.player_name,
            "odds": candidate.odds,
            "stake": candidate.stake,
            "edge": candidate.edge,
            "status": result_status,
            "reason": decision.reason,
        }
        if mode is not None:
            record["mode"] = mode
        self._append_decision_log(record)
        self._record_bet_log(
            market=market,
            candidate=candidate,
            decision=decision,
            status=result_status,
            features=features,
        )
        actions.append(record)

    def _select_express_candidates(self, queued_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        for payload in sorted(
            queued_candidates,
            key=lambda item: (item["candidate"].ranking_score, item["candidate"].edge),
            reverse=True,
        ):
            event_id = payload["market"].event_id
            if event_id in seen_event_ids:
                continue
            selected.append(payload)
            seen_event_ids.add(event_id)
            if len(selected) >= self.config.express_size:
                break
        return selected

    def _place_express_candidates(
        self,
        queued_candidates: list[dict[str, Any]],
        actions: list[dict[str, Any]],
    ) -> None:
        express_candidates = self._select_express_candidates(queued_candidates)
        if len(express_candidates) < self.config.express_size:
            self._append_decision_log(
                {
                    "status": "express_skipped",
                    "reason": "not_enough_distinct_events",
                    "available_candidates": len(express_candidates),
                    "required_candidates": self.config.express_size,
                }
            )
            return

        express_stake = float(express_candidates[0]["candidate"].stake)
        available_bankroll = self._available_bankroll()
        if available_bankroll > 0 and express_stake > available_bankroll:
            self._append_decision_log(
                {
                    "status": "express_skipped",
                    "reason": "bankroll_blocked",
                    "stake": express_stake,
                    "available_bankroll": available_bankroll,
                }
            )
            return

        try:
            result = self.bet_executor.place_express_bet(
                [item["candidate"] for item in express_candidates],
                stake=express_stake,
            )
        except Exception as exc:
            LOGGER.warning("Express placement failed: %s", exc)
            self._append_decision_log(
                {
                    "status": "express_execution_error",
                    "reason": str(exc),
                    "express_size": len(express_candidates),
                }
            )
            return
        result_status = str(result.get("status") or "unknown")
        if result_status not in SUCCESSFUL_PLACEMENT_STATUSES:
            self._append_decision_log(
                {
                    "status": "express_not_placed",
                    "reason": str(result.get("error") or result_status),
                    "express_size": len(express_candidates),
                }
            )
            return
        for payload in express_candidates:
            market = payload["market"]
            candidate = payload["candidate"]
            decision = payload["decision"]
            features = payload["features"]
            player1_probability = payload["player1_probability"]
            player2_probability = payload["player2_probability"]
            mode = payload.get("mode")
            self.state.mark_placed(candidate.selection_id)
            placed_snapshot = self._snapshot_record(
                market=market,
                features=features,
                candidate=candidate,
                status=result_status,
                player1_probability=player1_probability,
                player2_probability=player2_probability,
            )
            self.rl_logger.log_snapshot(placed_snapshot)
            self.rl_outcome_tracker.observe_market(market.event_id, placed_snapshot)
            self._log_point_trajectory(
                market=market,
                stage="bet_placed_express" if result_status != "dry_run" else "bet_dry_run_express",
                features=features,
                candidate=candidate,
                result=result,
                player1_probability=player1_probability,
                player2_probability=player2_probability,
            )
            self.rl_logger.log_action(
                self._action_record(
                    market=market,
                    candidate=candidate,
                    action="bet_express" if result_status != "dry_run" else "bet_dry_run_express",
                    result=result,
                    reason=mode,
                )
            )
            self.rl_logger.log_outcome(
                self.rl_outcome_tracker.record_pending_bet(
                    event_id=market.event_id,
                    market_id=market.market_id,
                    candidate=candidate,
                    result=result,
                    snapshot=placed_snapshot,
                    dry_run=result_status == "dry_run",
                )
            )
            record = {
                "market_id": market.market_id,
                "event_id": market.event_id,
                "selection_id": candidate.selection_id,
                "player_name": candidate.player_name,
                "odds": candidate.odds,
                "stake": candidate.stake,
                "edge": candidate.edge,
                "status": result_status,
                "mode": "express",
                "express_size": len(express_candidates),
                "reason": decision.reason,
            }
            self._append_decision_log(record)
            self._record_bet_log(
                market=market,
                candidate=candidate,
                decision=decision,
                status=result_status,
                features=features,
            )
            actions.append(record)

    def _snapshot_record(
        self,
        market: LiveMarket,
        features: dict[str, Any] | None,
        candidate: ScoredSelection | None,
        status: str,
        reason: str | None = None,
        player1_probability: float | None = None,
        player2_probability: float | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        raw = market.raw
        return {
            "timestamp_utc": timestamp,
            "event_id": market.event_id,
            "market_id": market.market_id,
            "competition": market.competition,
            "surface": market.surface,
            "round_name": market.round_name,
            "best_of": market.best_of,
            "tourney_level": market.tourney_level,
            "player1_name": market.player1_name,
            "player2_name": market.player2_name,
            "player1_odds": market.player1_odds,
            "player2_odds": market.player2_odds,
            "market_type": market.market_type,
            "live_score": raw.get("score"),
            "live_comment": raw.get("comment"),
            "live_delay": raw.get("liveDelay"),
            "serving_team": raw.get("serveT"),
            "player1_factor_id": raw.get("player1_factor_id") or raw.get("factor"),
            "player2_factor_id": raw.get("player2_factor_id"),
            "player1_param": raw.get("player1_param") or raw.get("param"),
            "player2_param": raw.get("player2_param"),
            "scope_market_id": raw.get("scopeMarketId"),
            "status": status,
            "reason": reason,
            "player1_probability": player1_probability,
            "player2_probability": player2_probability,
            "selected_side": candidate.side if candidate else None,
            "selected_player_name": candidate.player_name if candidate else None,
            "selected_odds": candidate.odds if candidate else None,
            "selected_edge": candidate.edge if candidate else None,
            "selected_stake": candidate.stake if candidate else None,
            "selected_implied_probability": candidate.implied_probability if candidate else None,
            "state_features": features,
        }

    def _action_record(
        self,
        market: LiveMarket,
        candidate: ScoredSelection | None,
        action: str,
        result: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp_utc": timestamp,
            "event_id": market.event_id,
            "market_id": market.market_id,
            "market_type": market.market_type,
            "action": action,
            "reason": reason,
        }
        if candidate is not None:
            record.update(
                {
                    "selection_id": candidate.selection_id,
                    "side": candidate.side,
                    "player_name": candidate.player_name,
                    "odds": candidate.odds,
                    "stake": candidate.stake,
                    "model_probability": candidate.model_probability,
                    "implied_probability": candidate.implied_probability,
                    "edge": candidate.edge,
                }
            )
        if result is not None:
            record["result"] = result
        return record

    def _point_trajectory_record(
        self,
        market: LiveMarket,
        stage: str,
        features: dict[str, Any] | None = None,
        candidate: ScoredSelection | None = None,
        result: dict[str, Any] | None = None,
        reason: str | None = None,
        player1_probability: float | None = None,
        player2_probability: float | None = None,
    ) -> dict[str, Any] | None:
        if market.market_type != "point_plus_one_winner":
            return None
        raw = market.raw
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "event_id": market.event_id,
            "market_id": market.market_id,
            "market_type": market.market_type,
            "competition": market.competition,
            "round_name": market.round_name,
            "player1_name": market.player1_name,
            "player2_name": market.player2_name,
            "player1_odds": market.player1_odds,
            "player2_odds": market.player2_odds,
            "live_score": raw.get("score"),
            "live_comment": raw.get("comment"),
            "serving_team": raw.get("serveT"),
            "target_point_number": raw.get("target_point_number"),
            "player1_factor_id": raw.get("player1_factor_id"),
            "player2_factor_id": raw.get("player2_factor_id"),
            "player1_param": raw.get("player1_param"),
            "player2_param": raw.get("player2_param"),
            "scope_market_id": raw.get("scopeMarketId"),
            "reason": reason,
            "player1_probability": player1_probability,
            "player2_probability": player2_probability,
            "state_features": features,
        }
        if candidate is not None:
            record.update(
                {
                    "selection_id": candidate.selection_id,
                    "selected_side": candidate.side,
                    "selected_player_name": candidate.player_name,
                    "selected_odds": candidate.odds,
                    "selected_stake": candidate.stake,
                    "selected_edge": candidate.edge,
                    "selected_implied_probability": candidate.implied_probability,
                    "selected_factor_id": raw.get("player1_factor_id") if candidate.side == "player1" else raw.get("player2_factor_id"),
                    "selected_param": raw.get("player1_param") if candidate.side == "player1" else raw.get("player2_param"),
                }
            )
        if result is not None:
            record["result"] = result
        return record

    def _log_point_trajectory(self, **kwargs: Any) -> None:
        record = self._point_trajectory_record(**kwargs)
        if record is not None:
            self.rl_logger.log_point_trajectory(record)

    def _should_use_point_fast_mode(
        self,
        market: LiveMarket,
        candidate: ScoredSelection,
        refreshed_selection: dict[str, Any] | None,
    ) -> bool:
        if not self.config.point_fast_mode:
            return False
        if market.market_type != "point_plus_one_winner":
            return False
        if candidate.edge < self.config.edge_threshold:
            return False
        if not refreshed_selection:
            return True
        if refreshed_selection.get("value") in (None, "", 0, 0.0):
            return True
        return False

    def run_cycle(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        queued_express_candidates: list[dict[str, Any]] = []
        self.rl_outcome_tracker.start_cycle()
        for market in self.market_feed_client.fetch_live_markets():
            seen_event_ids.add(market.event_id)
            try:
                _, row, player1_probability, player2_probability, candidate = self._score_market_details(market)
            except Exception as exc:
                LOGGER.warning("Skipping market %s: %s", market.market_id, exc)
                self._log_point_trajectory(
                    market=market,
                    stage="score_failed",
                    reason=str(exc),
                )
                self.rl_logger.log_snapshot(
                    self._snapshot_record(
                        market=market,
                        features=None,
                        candidate=None,
                        status="skipped",
                        reason=str(exc),
                    )
                )
                self.rl_outcome_tracker.observe_market(
                    market.event_id,
                    self._snapshot_record(
                        market=market,
                        features=None,
                        candidate=None,
                        status="skipped",
                        reason=str(exc),
                    ),
                )
                self.rl_logger.log_action(
                    self._action_record(market=market, candidate=None, action="skip", reason=str(exc))
                )
                self._append_decision_log(
                    {"market_id": market.market_id, "status": "skipped", "reason": str(exc)}
                )
                continue

            current_snapshot = self._snapshot_record(
                market=market,
                features=row,
                candidate=candidate,
                status="candidate" if candidate is not None else "no_edge",
                player1_probability=player1_probability,
                player2_probability=player2_probability,
            )
            self._log_point_trajectory(
                market=market,
                stage="scored" if candidate is not None else "observed_no_edge",
                features=row,
                candidate=candidate,
                player1_probability=player1_probability,
                player2_probability=player2_probability,
                reason=None if candidate is not None else "no_edge",
            )
            self.rl_outcome_tracker.observe_market(market.event_id, current_snapshot)

            if candidate is None:
                no_bet_reason = self._candidate_rejection_reason(market, player1_probability, player2_probability)
                self.rl_logger.log_snapshot(current_snapshot)
                self.rl_logger.log_action(
                    self._action_record(market=market, candidate=None, action="no_bet", reason=no_bet_reason)
                )
                self._append_decision_log({"market_id": market.market_id, "status": "no_edge", "reason": no_bet_reason})
                continue
            policy_decision = self._betting_policy_decision(market, candidate, row)
            if not policy_decision.should_bet:
                self.rl_logger.log_snapshot(
                    self._snapshot_record(
                        market=market,
                        features=row,
                        candidate=candidate,
                        status="policy_blocked",
                        reason=policy_decision.reason,
                        player1_probability=player1_probability,
                        player2_probability=player2_probability,
                    )
                )
                self.rl_logger.log_action(
                    self._action_record(
                        market=market,
                        candidate=candidate,
                        action="policy_blocked",
                        reason=policy_decision.reason,
                    )
                )
                self._append_decision_log(
                    {
                        "market_id": market.market_id,
                        "event_id": market.event_id,
                        "selection_id": candidate.selection_id,
                        "status": "policy_blocked",
                        "reason": policy_decision.reason,
                    }
                )
                continue
            if self.state.has_seen(candidate.selection_id):
                self._log_point_trajectory(
                    market=market,
                    stage="duplicate",
                    features=row,
                    candidate=candidate,
                    player1_probability=player1_probability,
                    player2_probability=player2_probability,
                    reason="selection_already_seen",
                )
                self.rl_logger.log_snapshot(
                    self._snapshot_record(
                        market=market,
                        features=row,
                        candidate=candidate,
                        status="duplicate",
                        reason="selection_already_seen",
                        player1_probability=player1_probability,
                        player2_probability=player2_probability,
                    )
                )
                self.rl_logger.log_action(
                    self._action_record(market=market, candidate=candidate, action="duplicate", reason="selection_already_seen")
                )
                self._append_decision_log(
                    {"market_id": market.market_id, "status": "duplicate", "selection_id": candidate.selection_id}
                )
                continue

            refreshed_market = market
            refreshed_candidate = candidate
            refreshed_selection: dict[str, Any] | None = None
            slip_info_response: dict[str, Any] | None = None
            try:
                (
                    refreshed_market,
                    refreshed_selection,
                    slip_info_response,
                    refreshed_row,
                    refreshed_player1_probability,
                    refreshed_player2_probability,
                    refreshed_candidate,
                ) = self._refresh_and_rescore(market, candidate)
            except Exception as exc:
                LOGGER.warning("Pre-bet refresh failed for market %s: %s", market.market_id, exc)
                self._log_point_trajectory(
                    market=market,
                    stage="refresh_failed",
                    features=row,
                    candidate=candidate,
                    player1_probability=player1_probability,
                    player2_probability=player2_probability,
                    reason=str(exc),
                )
                self.rl_logger.log_action(
                    self._action_record(market=market, candidate=candidate, action="refresh_failed", reason=str(exc))
                )
                self._append_decision_log(
                    {"market_id": market.market_id, "status": "refresh_failed", "reason": str(exc)}
                )
                continue

            if refreshed_candidate is None:
                if self._should_use_point_fast_mode(refreshed_market, candidate, refreshed_selection):
                    self._log_point_trajectory(
                        market=market,
                        stage="point_fast_mode_fallback",
                        features=row,
                        candidate=candidate,
                        player1_probability=player1_probability,
                        player2_probability=player2_probability,
                        reason="refresh_candidate_invalid_using_snapshot_value",
                    )
                    if self.config.bet_mode == "express":
                        queued_express_candidates.append(
                            {
                                "market": market,
                                "candidate": candidate,
                                "decision": policy_decision,
                                "features": row,
                                "player1_probability": player1_probability,
                                "player2_probability": player2_probability,
                                "mode": "point_fast_mode",
                            }
                        )
                    else:
                        self._place_single_candidate(
                            market=market,
                            candidate=candidate,
                            decision=policy_decision,
                            features=row,
                            player1_probability=player1_probability,
                            player2_probability=player2_probability,
                            refreshed_selection={},
                            slip_info_response=slip_info_response,
                            actions=actions,
                            mode="point_fast_mode",
                        )
                    continue
                self._log_point_trajectory(
                    market=refreshed_market,
                    stage="refresh_no_edge",
                    features=refreshed_row,
                    candidate=candidate,
                    player1_probability=refreshed_player1_probability,
                    player2_probability=refreshed_player2_probability,
                    reason="candidate_invalid_after_refresh",
                )
                self.rl_logger.log_snapshot(
                    self._snapshot_record(
                        market=refreshed_market,
                        features=refreshed_row,
                        candidate=None,
                        status="refresh_no_edge",
                        reason="candidate_invalid_after_refresh",
                        player1_probability=refreshed_player1_probability,
                        player2_probability=refreshed_player2_probability,
                    )
                )
                self.rl_logger.log_action(
                    self._action_record(
                        market=refreshed_market,
                        candidate=candidate,
                        action="refresh_no_bet",
                        reason=self._refresh_rejection_reason(
                            candidate,
                            refreshed_market,
                            refreshed_player1_probability,
                            refreshed_player2_probability,
                        ),
                    )
                )
                self._append_decision_log(
                    {
                        "market_id": market.market_id,
                        "status": "refresh_no_edge",
                        "selection_id": candidate.selection_id,
                        "reason": self._refresh_rejection_reason(
                            candidate,
                            refreshed_market,
                            refreshed_player1_probability,
                            refreshed_player2_probability,
                        ),
                    }
                )
                continue

            self._log_point_trajectory(
                market=refreshed_market,
                stage="refreshed_candidate",
                features=refreshed_row,
                candidate=refreshed_candidate,
                player1_probability=refreshed_player1_probability,
                player2_probability=refreshed_player2_probability,
            )
            if self.config.bet_mode == "express":
                queued_express_candidates.append(
                    {
                        "market": refreshed_market,
                        "candidate": refreshed_candidate,
                        "decision": self._betting_policy_decision(refreshed_market, refreshed_candidate, refreshed_row),
                        "features": refreshed_row,
                        "player1_probability": refreshed_player1_probability,
                        "player2_probability": refreshed_player2_probability,
                    }
                )
            else:
                refreshed_policy_decision = self._betting_policy_decision(
                    refreshed_market,
                    refreshed_candidate,
                    refreshed_row,
                )
                if not refreshed_policy_decision.should_bet:
                    self.rl_logger.log_action(
                        self._action_record(
                            market=refreshed_market,
                            candidate=refreshed_candidate,
                            action="policy_blocked_refresh",
                            reason=refreshed_policy_decision.reason,
                        )
                    )
                    self._append_decision_log(
                        {
                            "market_id": refreshed_market.market_id,
                            "event_id": refreshed_market.event_id,
                            "selection_id": refreshed_candidate.selection_id,
                            "status": "policy_blocked_refresh",
                            "reason": refreshed_policy_decision.reason,
                        }
                    )
                    continue
                self._place_single_candidate(
                    market=refreshed_market,
                    candidate=refreshed_candidate,
                    decision=refreshed_policy_decision,
                    features=refreshed_row,
                    player1_probability=refreshed_player1_probability,
                    player2_probability=refreshed_player2_probability,
                    refreshed_selection=refreshed_selection,
                    slip_info_response=slip_info_response,
                    actions=actions,
                )
        if self.config.bet_mode == "express" and queued_express_candidates:
            self._place_express_candidates(queued_express_candidates, actions)
        settled_any = False
        for closed_record in self.rl_outcome_tracker.close_missing_markets(seen_event_ids):
            bankroll_before = self.state.current_bankroll
            settled_record = settle_outcome_record(
                closed_record,
                final_snapshot=closed_record.get("final_snapshot"),
                bankroll_before=bankroll_before,
            )
            profit = settled_record.get("profit")
            if isinstance(profit, (int, float)):
                self.state.apply_profit(float(profit))
                settled_record["bankroll_after"] = self.state.current_bankroll
            self.rl_logger.log_outcome(settled_record)
            self.audit_logger.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "match_id": closed_record.get("event_id"),
                    "event_id": closed_record.get("event_id"),
                    "market": closed_record.get("market_type"),
                    "pick": closed_record.get("player_name"),
                    "odds": closed_record.get("odds_taken"),
                    "stake": closed_record.get("stake"),
                    "result": settled_record.get("status"),
                    "profit": settled_record.get("profit"),
                    "model_prob": closed_record.get("model_probability"),
                    "bookmaker_prob": (1.0 / float(closed_record.get("odds_taken"))) if float(closed_record.get("odds_taken") or 0.0) > 0 else 0.0,
                    "value": closed_record.get("edge"),
                    "decision_reason": "settled_outcome",
                    "source_json": settled_record,
                }
            )
            settled_any = True
            if settled_record.get("market_type") == "point_plus_one_winner":
                self.rl_logger.log_point_trajectory(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "stage": "outcome_closed",
                        **settled_record,
                    }
                )
        if settled_any:
            self.bankroll_policy.bankroll = self._available_bankroll()
            self._export_rl_policy()
        return actions

    def run_forever(self) -> None:
        while True:
            self.run_cycle()
            time.sleep(self.config.poll_interval_seconds)
