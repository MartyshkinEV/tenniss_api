try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from psycopg2.extras import execute_values
from sqlalchemy import text

from config import settings
from scripts.load_pointbypoint import ensure_table
from src.db.engine import get_engine

_SCORE_MAP = {"AD": "A"}
_SLAM_LABELS = {
    "ausopen": "Australian Open",
    "frenchopen": "French Open",
    "wimbledon": "Wimbledon",
    "usopen": "US Open",
}
_POINT_FILE_RE = re.compile(r"^(?P<year>\d{4})-(?P<slam>[a-z]+)-points\.csv$")


def _normalize_score(value: str | None) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return "0"
    return _SCORE_MAP.get(text, text)


def _score_to_num(value: str) -> int:
    if value == "0":
        return 0
    if value == "15":
        return 1
    if value == "30":
        return 2
    if value == "40":
        return 3
    if value == "A":
        return 4
    try:
        return int(value)
    except ValueError:
        return 0


def _num_to_regular_score(p1: int, p2: int) -> tuple[str, str]:
    display = {0: "0", 1: "15", 2: "30", 3: "40", 4: "A"}
    if p1 >= 3 and p2 >= 3:
        if p1 == p2:
            return "40", "40"
        if p1 == p2 + 1:
            return "A", "40"
        if p2 == p1 + 1:
            return "40", "A"
    return display.get(p1, str(p1)), display.get(p2, str(p2))


def _derive_after_score(score_before_p1: str, score_before_p2: str, winner_side: int, is_tiebreak: bool) -> tuple[str, str]:
    p1 = _score_to_num(score_before_p1)
    p2 = _score_to_num(score_before_p2)
    if winner_side == 1:
        p1 += 1
    else:
        p2 += 1
    if is_tiebreak:
        return str(p1), str(p2)
    return _num_to_regular_score(p1, p2)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _winner_side(meta: dict[str, str]) -> int | None:
    raw = str(meta.get("winner") or "").strip()
    if raw == "1":
        return 1
    if raw == "2":
        return 2
    if raw and raw == str(meta.get("player1") or "").strip():
        return 1
    if raw and raw == str(meta.get("player2") or "").strip():
        return 2
    return None


def _is_tiebreak(row: dict[str, str]) -> bool:
    games_p1 = _parse_int(row.get("P1GamesWon"))
    games_p2 = _parse_int(row.get("P2GamesWon"))
    score_1 = _normalize_score(row.get("P1Score"))
    score_2 = _normalize_score(row.get("P2Score"))
    if games_p1 == games_p2 and games_p1 >= 6:
        return True
    regular_scores = {"0", "15", "30", "40", "A"}
    return score_1 not in regular_scores or score_2 not in regular_scores


def _point_code(row: dict[str, str], winner_side: int, server_side: int) -> str:
    if winner_side == 1 and _parse_int(row.get("P1Ace")):
        return "A"
    if winner_side == 2 and _parse_int(row.get("P2Ace")):
        return "A"
    if server_side == 1 and winner_side == 2 and _parse_int(row.get("P1DoubleFault")):
        return "D"
    if server_side == 2 and winner_side == 1 and _parse_int(row.get("P2DoubleFault")):
        return "D"
    return "S" if winner_side == server_side else "R"


def _load_matches(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("match_id") or "").strip(): row for row in reader if row.get("match_id")}


def _iter_point_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            winner_side = _parse_int(row.get("PointWinner"))
            server_side = _parse_int(row.get("PointServer"))
            point_no = _parse_int(row.get("PointNumber"), default=-1)
            if winner_side not in {1, 2} or server_side not in {1, 2} or point_no < 0:
                continue
            rows.append(row)
    return rows


def _build_insert_rows(points_path: Path) -> list[dict[str, Any]]:
    match_path = points_path.with_name(points_path.name.replace("-points.csv", "-matches.csv"))
    match_meta = _load_matches(match_path)
    rows = _iter_point_rows(points_path)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        match_id = str(row.get("match_id") or "").strip()
        meta = match_meta.get(match_id, {})
        player1 = str(meta.get("player1") or "").strip()
        player2 = str(meta.get("player2") or "").strip()
        if not match_id or not player1 or not player2:
            continue
        set_no = _parse_int(row.get("SetNo"))
        game_no = _parse_int(row.get("GameNo"))
        point_no = _parse_int(row.get("PointNumber"))
        winner_side = _parse_int(row.get("PointWinner"))
        server_side = _parse_int(row.get("PointServer"))
        if set_no <= 0 or game_no <= 0 or point_no < 0:
            continue
        is_tiebreak = _is_tiebreak(row)
        score_before_p1 = _normalize_score(row.get("P1Score"))
        score_before_p2 = _normalize_score(row.get("P2Score"))
        next_row = rows[index + 1] if index + 1 < len(rows) else None
        if (
            next_row
            and str(next_row.get("match_id") or "").strip() == match_id
            and _parse_int(next_row.get("SetNo")) == set_no
            and _parse_int(next_row.get("GameNo")) == game_no
        ):
            score_after_p1 = _normalize_score(next_row.get("P1Score"))
            score_after_p2 = _normalize_score(next_row.get("P2Score"))
        else:
            score_after_p1, score_after_p2 = _derive_after_score(score_before_p1, score_before_p2, winner_side, is_tiebreak)
        server_name = player1 if server_side == 1 else player2
        returner_name = player2 if server_side == 1 else player1
        out.append(
            {
                "pbp_id": f"slam:{match_id}",
                "source_file": points_path.name,
                "match_date": None,
                "tny_name": _SLAM_LABELS.get(str(meta.get("slam") or "").strip(), str(meta.get("slam") or "").strip()),
                "tour": "slam",
                "draw": str(meta.get("round") or "main").strip() or "main",
                "server1": player1,
                "server2": player2,
                "winner": _winner_side(meta),
                "match_score": None,
                "adf_flag": 0,
                "wh_minutes": None,
                "set_no": set_no,
                "game_no": game_no,
                "point_no": point_no,
                "is_tiebreak": is_tiebreak,
                "server_name": server_name,
                "returner_name": returner_name,
                "point_winner_name": player1 if winner_side == 1 else player2,
                "point_winner_side": winner_side,
                "score_before_p1": score_before_p1,
                "score_before_p2": score_before_p2,
                "score_after_p1": score_after_p1,
                "score_after_p2": score_after_p2,
                "point_code": _point_code(row, winner_side=winner_side, server_side=server_side),
            }
        )
    return out


def discover_slam_point_files(root: Path, year_from: int, year_to: int | None) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.glob("*-points.csv")):
        match = _POINT_FILE_RE.match(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        if year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        paths.append(path)
    return paths


def flush_rows(engine, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    columns = [
        "pbp_id",
        "source_file",
        "match_date",
        "tny_name",
        "tour",
        "draw",
        "server1",
        "server2",
        "winner",
        "match_score",
        "adf_flag",
        "wh_minutes",
        "set_no",
        "game_no",
        "point_no",
        "is_tiebreak",
        "server_name",
        "returner_name",
        "point_winner_name",
        "point_winner_side",
        "score_before_p1",
        "score_before_p2",
        "score_after_p1",
        "score_after_p2",
        "point_code",
    ]
    tuples = [tuple(row.get(column) for column in columns) for row in rows]
    sql = f"""
        INSERT INTO pointbypoint ({", ".join(columns)})
        VALUES %s
        ON CONFLICT (pbp_id, set_no, game_no, point_no) DO NOTHING
    """
    raw_connection = engine.raw_connection()
    try:
        with raw_connection.cursor() as cursor:
            execute_values(cursor, sql, tuples, page_size=5000)
            inserted = cursor.rowcount if cursor.rowcount is not None else 0
        raw_connection.commit()
    finally:
        raw_connection.close()
    return max(inserted, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Grand Slam point-by-point files into pointbypoint.")
    parser.add_argument(
        "--source-root",
        default=str(settings.project_root / "tennis_slam_pointbypoint"),
        help="Directory containing *-points.csv and *-matches.csv files.",
    )
    parser.add_argument("--year-from", type=int, default=2018)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    files = discover_slam_point_files(source_root, year_from=args.year_from, year_to=args.year_to)
    if not files:
        raise SystemExit(f"No slam point files found in {source_root}")

    engine = get_engine()
    ensure_table(engine)
    if args.truncate:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE pointbypoint"))

    total_inserted = 0
    for path in files:
        insert_rows = _build_insert_rows(path)
        inserted = flush_rows(engine, insert_rows)
        total_inserted += inserted
        print(f"{path.name}: prepared={len(insert_rows)} inserted={inserted}")
    print(f"total_inserted={total_inserted}")


if __name__ == "__main__":
    main()
