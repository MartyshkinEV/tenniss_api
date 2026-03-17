from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import discover_match_csv_files


@dataclass(frozen=True)
class HistoricalTotalDataset:
    frame: pd.DataFrame
    metadata: dict[str, Any]


LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)


def _first_set_total(score: str | None) -> int | None:
    if not score:
        return None
    first_set = str(score).strip().split()[0]
    if "-" not in first_set:
        return None
    left, right = first_set.split("-", 1)
    try:
        return int(left) + int(right)
    except ValueError:
        return None


def build_historical_total_training_frame(max_rows: int | None = 250_000) -> HistoricalTotalDataset:
    rows: list[dict[str, Any]] = []
    files_used: list[str] = []
    matches_seen = 0
    for path in discover_match_csv_files():
        files_used.append(path.name)
        df = pd.read_csv(
            path,
            usecols=["surface", "tourney_level", "best_of", "score", "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"],
            low_memory=False,
        )
        for row in df.to_dict(orient="records"):
            first_set_total = _first_set_total(row.get("score"))
            if first_set_total is None:
                continue
            matches_seen += 1
            rank_diff = float(pd.to_numeric(row.get("winner_rank"), errors="coerce") or 9999) - float(
                pd.to_numeric(row.get("loser_rank"), errors="coerce") or 9999
            )
            rank_points_diff = float(pd.to_numeric(row.get("winner_rank_points"), errors="coerce") or 0.0) - float(
                pd.to_numeric(row.get("loser_rank_points"), errors="coerce") or 0.0
            )
            for line in LINES:
                rows.append(
                    {
                        "surface": row.get("surface") or "Hard",
                        "tourney_level": row.get("tourney_level") or "tour",
                        "best_of": int(pd.to_numeric(row.get("best_of"), errors="coerce") or 3),
                        "line": line,
                        "games_p1": 0,
                        "games_p2": 0,
                        "current_total_games": 0,
                        "hold_rate_diff": 0.0,
                        "break_rate_diff": 0.0,
                        "rank_diff": rank_diff,
                        "elo_diff": rank_points_diff / 10.0,
                        "label": 1 if first_set_total > line else 0,
                    }
                )
                if max_rows is not None and len(rows) >= max_rows:
                    frame = pd.DataFrame(rows)
                    return HistoricalTotalDataset(
                        frame=frame,
                        metadata={"rows": len(frame), "matches_seen": matches_seen, "files_used": files_used, "truncated": True},
                    )
    frame = pd.DataFrame(rows)
    return HistoricalTotalDataset(
        frame=frame,
        metadata={"rows": len(frame), "matches_seen": matches_seen, "files_used": files_used, "truncated": False},
    )
