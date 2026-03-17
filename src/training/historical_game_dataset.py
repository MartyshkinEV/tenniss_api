from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import discover_pbp_csv_files
from src.data.point_parser import ParsedPoint, parse_pbp_points


PBP_SOURCE_COLUMNS = [
    "pbp_id",
    "date",
    "tny_name",
    "tour",
    "draw",
    "server1",
    "server2",
    "winner",
    "pbp",
    "score",
    "adf_flag",
    "wh_minutes",
]

POINT_SCORE_MAP = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4}


@dataclass(frozen=True)
class HistoricalGameDatasetResult:
    frame: pd.DataFrame
    metadata: dict[str, Any]


def _score_to_num(value: str) -> int:
    return POINT_SCORE_MAP.get(value, 0)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in PBP_SOURCE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, PBP_SOURCE_COLUMNS]


def _build_row(match_row: dict[str, Any], current: ParsedPoint, game_winner_side: int) -> dict[str, Any]:
    score_before_p1 = _score_to_num(current.score_before_p1)
    score_before_p2 = _score_to_num(current.score_before_p2)
    server_side = 1 if current.server_name == str(match_row.get("server1") or "") else 2
    is_break_point_p1 = int(server_side == 2 and score_before_p1 >= 3 and score_before_p1 > score_before_p2)
    is_break_point_p2 = int(server_side == 1 and score_before_p2 >= 3 and score_before_p2 > score_before_p1)
    return {
        "tour": str(match_row.get("tour") or "unknown"),
        "draw": str(match_row.get("draw") or "unknown"),
        "adf_flag": int(match_row.get("adf_flag") or 0),
        "set_no": current.set_no,
        "game_no": current.game_no,
        "point_no": current.point_no,
        "is_tiebreak": int(current.is_tiebreak),
        "server_side": server_side,
        "points_p1": score_before_p1,
        "points_p2": score_before_p2,
        "point_score_diff": score_before_p1 - score_before_p2,
        "is_break_point_p1": is_break_point_p1,
        "is_break_point_p2": is_break_point_p2,
        "label": 1 if game_winner_side == 1 else 0,
    }


def build_historical_game_training_frame(
    max_rows: int | None = 300_000,
    current_only: bool = True,
) -> HistoricalGameDatasetResult:
    files = discover_pbp_csv_files()
    if current_only:
        files = [path for path in files if "current" in path.name.lower()]
    files = sorted(files, key=lambda path: ("current" not in path.name.lower(), path.name.lower()))

    rows: list[dict[str, Any]] = []
    matches_seen = 0
    files_used: list[str] = []

    for path in files:
        df = _ensure_columns(pd.read_csv(path, low_memory=False))
        files_used.append(path.name)
        for match_row in df.to_dict(orient="records"):
            parsed_points = parse_pbp_points(
                str(match_row.get("pbp") or ""),
                str(match_row.get("server1") or ""),
                str(match_row.get("server2") or ""),
            )
            if not parsed_points:
                continue
            matches_seen += 1
            points_by_game: dict[tuple[int, int], list[ParsedPoint]] = {}
            for point in parsed_points:
                points_by_game.setdefault((point.set_no, point.game_no), []).append(point)
            for game_points in points_by_game.values():
                if not game_points:
                    continue
                game_winner_side = game_points[-1].point_winner_side
                for point in game_points:
                    rows.append(_build_row(match_row=match_row, current=point, game_winner_side=game_winner_side))
                    if max_rows is not None and len(rows) >= max_rows:
                        frame = pd.DataFrame(rows)
                        return HistoricalGameDatasetResult(
                            frame=frame,
                            metadata={
                                "rows": int(len(frame)),
                                "matches_seen": int(matches_seen),
                                "files_used": files_used,
                                "current_only": current_only,
                                "truncated": True,
                            },
                        )

    frame = pd.DataFrame(rows)
    return HistoricalGameDatasetResult(
        frame=frame,
        metadata={
            "rows": int(len(frame)),
            "matches_seen": int(matches_seen),
            "files_used": files_used,
            "current_only": current_only,
            "truncated": False,
        },
    )
