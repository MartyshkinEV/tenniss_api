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
class HistoricalPointDatasetResult:
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


def _build_row(
    match_row: dict[str, Any],
    current: ParsedPoint,
    future: ParsedPoint,
    target_offset: int,
) -> dict[str, Any]:
    score_before_p1 = _score_to_num(current.score_before_p1)
    score_before_p2 = _score_to_num(current.score_before_p2)
    server_side = 1 if current.server_name == str(match_row.get("server1") or "") else 2
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
        "target_offset": target_offset,
        "label": 1 if future.point_winner_side == 1 else 0,
    }


def build_historical_point_training_frame(
    target_offset: int = 2,
    max_rows: int | None = 200_000,
    current_only: bool = True,
) -> HistoricalPointDatasetResult:
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
            if len(parsed_points) <= target_offset:
                continue
            matches_seen += 1
            for index in range(len(parsed_points) - target_offset):
                rows.append(
                    _build_row(
                        match_row=match_row,
                        current=parsed_points[index],
                        future=parsed_points[index + target_offset],
                        target_offset=target_offset,
                    )
                )
                if max_rows is not None and len(rows) >= max_rows:
                    frame = pd.DataFrame(rows)
                    return HistoricalPointDatasetResult(
                        frame=frame,
                        metadata={
                            "rows": int(len(frame)),
                            "matches_seen": int(matches_seen),
                            "files_used": files_used,
                            "target_offset": target_offset,
                            "current_only": current_only,
                            "truncated": True,
                        },
                    )

    frame = pd.DataFrame(rows)
    return HistoricalPointDatasetResult(
        frame=frame,
        metadata={
            "rows": int(len(frame)),
            "matches_seen": int(matches_seen),
            "files_used": files_used,
            "target_offset": target_offset,
            "current_only": current_only,
            "truncated": False,
        },
    )
