from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings


POINT_SCORE_MAP = {"0": 0, "00": 0, "15": 1, "30": 2, "40": 3, "A": 4}
EXECUTION_POSITIVE_STAGES = {"bet_placed", "bet_dry_run", "bet_placed_fast_mode", "bet_dry_run_fast_mode"}
EXECUTION_NEGATIVE_STAGES = {"refresh_no_edge", "refresh_failed", "score_failed"}
BASE_POINT_STAGES = {"scored", "refreshed_candidate", "observed_no_edge"}


@dataclass(frozen=True)
class PointTrainingFrames:
    point_outcome: pd.DataFrame
    execution_survival: pd.DataFrame


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.stat().st_size:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_point_comment(comment: str | None) -> tuple[int | None, int | None]:
    if not comment:
        return None, None
    body = comment.strip().strip("()")
    if " " in body:
        body = body.split(" ", 1)[0]
    body = body.replace("*", "")
    if "-" not in body:
        return None, None
    left, right = body.split("-", 1)
    return POINT_SCORE_MAP.get(left.strip()), POINT_SCORE_MAP.get(right.strip())


def _parse_game_score(score: str | None) -> tuple[int | None, int | None]:
    if not score or ":" not in score:
        return None, None
    left, right = score.split(":", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None, None


def _flatten_state_features(features: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(features, dict):
        return {}
    return {f"sf_{key}": value for key, value in features.items()}


def _base_features(record: dict[str, Any]) -> dict[str, Any]:
    games_1, games_2 = _parse_game_score(record.get("live_score"))
    points_1, points_2 = _parse_point_comment(record.get("live_comment"))
    selected_side = record.get("selected_side")
    return {
        "competition": record.get("competition"),
        "surface": record.get("surface"),
        "round_name": record.get("round_name"),
        "market_type": record.get("market_type"),
        "serving_team": record.get("serving_team"),
        "target_point_number": record.get("target_point_number"),
        "player1_odds": record.get("player1_odds"),
        "player2_odds": record.get("player2_odds"),
        "player1_probability": record.get("player1_probability"),
        "player2_probability": record.get("player2_probability"),
        "selected_side": selected_side,
        "selected_odds": record.get("selected_odds"),
        "selected_edge": record.get("selected_edge"),
        "selected_stake": record.get("selected_stake"),
        "games_p1": games_1,
        "games_p2": games_2,
        "points_p1": points_1,
        "points_p2": points_2,
        **_flatten_state_features(record.get("state_features")),
    }


def _possible_next_states(points_1: int, points_2: int) -> dict[str, tuple[int | None, int | None, bool]]:
    result: dict[str, tuple[int | None, int | None, bool]] = {}
    if points_1 <= 2 and points_2 <= 2:
        result["player1"] = (points_1 + 1, points_2, False)
        result["player2"] = (points_1, points_2 + 1, False)
        return result
    if points_1 == 3 and points_2 <= 2:
        result["player1"] = (None, None, True)
        result["player2"] = (points_1, points_2 + 1, False)
        return result
    if points_2 == 3 and points_1 <= 2:
        result["player1"] = (points_1 + 1, points_2, False)
        result["player2"] = (None, None, True)
        return result
    if points_1 == 3 and points_2 == 3:
        result["player1"] = (4, 3, False)
        result["player2"] = (3, 4, False)
        return result
    if points_1 == 4 and points_2 == 3:
        result["player1"] = (None, None, True)
        result["player2"] = (3, 3, False)
        return result
    if points_1 == 3 and points_2 == 4:
        result["player1"] = (3, 3, False)
        result["player2"] = (None, None, True)
        return result
    return result


def _infer_next_point_winner(current: dict[str, Any], next_record: dict[str, Any]) -> int | None:
    cg1, cg2 = _parse_game_score(current.get("live_score"))
    ng1, ng2 = _parse_game_score(next_record.get("live_score"))
    cp1, cp2 = _parse_point_comment(current.get("live_comment"))
    np1, np2 = _parse_point_comment(next_record.get("live_comment"))
    if None in {cg1, cg2, cp1, cp2, ng1, ng2, np1, np2}:
        return None
    transitions = _possible_next_states(cp1, cp2)
    for side, (expected_p1, expected_p2, is_game_win) in transitions.items():
        if not is_game_win and (np1, np2) == (expected_p1, expected_p2) and (ng1, ng2) == (cg1, cg2):
            return 1 if side == "player1" else 0
        if is_game_win:
            if side == "player1" and ng1 == cg1 + 1 and ng2 == cg2 and np1 == 0 and np2 == 0:
                return 1
            if side == "player2" and ng2 == cg2 + 1 and ng1 == cg1 and np1 == 0 and np2 == 0:
                return 0
    return None


def build_point_training_frames(
    point_trajectories_path: Path | None = None,
) -> PointTrainingFrames:
    records = _load_jsonl(point_trajectories_path or settings.live_point_trajectories_path)
    point_records = [
        record for record in records
        if record.get("market_type") == "point_plus_one_winner"
    ]
    if not point_records:
        empty = pd.DataFrame()
        return PointTrainingFrames(point_outcome=empty, execution_survival=empty)

    frame = pd.DataFrame(point_records)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.sort_values(["event_id", "timestamp_utc", "market_id"]).reset_index(drop=True)

    execution_rows: list[dict[str, Any]] = []
    grouped_by_key = frame.groupby(["event_id", "market_id", "selected_side"], dropna=False)
    for _, group in grouped_by_key:
        group_records = group.to_dict("records")
        base_record = next((record for record in group_records if record.get("stage") in BASE_POINT_STAGES and record.get("selected_side")), None)
        if base_record is None:
            continue
        label = None
        for record in group_records:
            stage = record.get("stage")
            if stage in EXECUTION_POSITIVE_STAGES:
                label = 1
                break
            if stage in EXECUTION_NEGATIVE_STAGES:
                label = 0
                break
        if label is None:
            continue
        execution_rows.append(
            {
                **_base_features(base_record),
                "event_id": base_record.get("event_id"),
                "market_id": base_record.get("market_id"),
                "label": label,
            }
        )

    point_outcome_rows: list[dict[str, Any]] = []
    grouped_by_event = frame.groupby(["event_id", "round_name"], dropna=False)
    for _, group in grouped_by_event:
        group_records = group.sort_values("timestamp_utc").to_dict("records")
        for index, current in enumerate(group_records[:-1]):
            if current.get("stage") not in BASE_POINT_STAGES:
                continue
            next_record = group_records[index + 1]
            label = _infer_next_point_winner(current, next_record)
            if label is None:
                continue
            point_outcome_rows.append(
                {
                    **_base_features(current),
                    "event_id": current.get("event_id"),
                    "market_id": current.get("market_id"),
                    "label": label,
                }
            )

    return PointTrainingFrames(
        point_outcome=pd.DataFrame(point_outcome_rows),
        execution_survival=pd.DataFrame(execution_rows),
    )
