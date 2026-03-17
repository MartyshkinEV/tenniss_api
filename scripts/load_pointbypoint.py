try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import discover_pbp_csv_files, settings
from src.data.point_parser import parse_pbp_points, point_row_from_match
from src.db.engine import get_engine


POINTBYPPOINT_DDL = """
CREATE TABLE IF NOT EXISTS pointbypoint (
    pbp_id TEXT,
    source_file TEXT,
    match_date DATE,
    tny_name TEXT,
    tour TEXT,
    draw TEXT,
    server1 TEXT,
    server2 TEXT,
    winner INTEGER,
    match_score TEXT,
    adf_flag INTEGER,
    wh_minutes INTEGER,
    set_no INTEGER NOT NULL,
    game_no INTEGER NOT NULL,
    point_no INTEGER NOT NULL,
    is_tiebreak BOOLEAN NOT NULL,
    server_name TEXT,
    returner_name TEXT,
    point_winner_name TEXT,
    point_winner_side INTEGER,
    score_before_p1 TEXT,
    score_before_p2 TEXT,
    score_after_p1 TEXT,
    score_after_p2 TEXT,
    point_code TEXT,
    PRIMARY KEY (pbp_id, set_no, game_no, point_no)
)
"""


@dataclass
class ImportProgress:
    status: str
    total_files: int
    processed_files: int
    total_matches: int
    processed_matches: int
    total_points_inserted: int
    percent_complete: float
    elapsed_seconds: float
    eta_seconds: float | None
    current_file: str | None
    started_at_utc: str
    updated_at_utc: str
    message: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_matches(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def ensure_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(POINTBYPPOINT_DDL))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pointbypoint_match_date ON pointbypoint(match_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pointbypoint_pbp_id ON pointbypoint(pbp_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pointbypoint_tour ON pointbypoint(tour)"))


def truncate_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE pointbypoint"))


def write_progress(path: Path, progress: ImportProgress) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(progress), ensure_ascii=True, indent=2), encoding="utf-8")


def build_progress(
    *,
    status: str,
    total_files: int,
    processed_files: int,
    total_matches: int,
    processed_matches: int,
    total_points_inserted: int,
    started_at: float,
    current_file: str | None,
    message: str | None = None,
) -> ImportProgress:
    elapsed_seconds = max(time.time() - started_at, 0.0)
    percent_complete = 0.0 if total_matches <= 0 else min(processed_matches / total_matches * 100.0, 100.0)
    eta_seconds = None
    if processed_matches > 0 and total_matches > processed_matches:
        rate = elapsed_seconds / processed_matches
        eta_seconds = max((total_matches - processed_matches) * rate, 0.0)
    return ImportProgress(
        status=status,
        total_files=total_files,
        processed_files=processed_files,
        total_matches=total_matches,
        processed_matches=processed_matches,
        total_points_inserted=total_points_inserted,
        percent_complete=round(percent_complete, 2),
        elapsed_seconds=round(elapsed_seconds, 2),
        eta_seconds=None if eta_seconds is None else round(eta_seconds, 2),
        current_file=current_file,
        started_at_utc=datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        updated_at_utc=utc_now(),
        message=message,
    )


def normalize_point_rows(rows: list[dict[str, object]], source_file: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "date": "match_date",
            "score": "match_score",
        }
    )
    frame["source_file"] = source_file
    frame["match_date"] = pd.to_datetime(frame["match_date"], format="%d %b %y", errors="coerce").dt.date
    ordered_columns = [
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
    return frame.loc[:, ordered_columns]


def flush_batch(engine, rows: list[dict[str, object]], source_file: str) -> int:
    if not rows:
        return 0
    frame = normalize_point_rows(rows, source_file=source_file)
    tuples = [tuple(row) for row in frame.itertuples(index=False, name=None)]
    columns = list(frame.columns)
    insert_sql = f"""
        INSERT INTO pointbypoint ({", ".join(columns)})
        VALUES %s
        ON CONFLICT (pbp_id, set_no, game_no, point_no) DO NOTHING
    """
    inserted = 0
    raw_connection = engine.raw_connection()
    try:
        with raw_connection.cursor() as cursor:
            execute_values(cursor, insert_sql, tuples, page_size=5000)
            inserted = cursor.rowcount if cursor.rowcount is not None else 0
        raw_connection.commit()
    finally:
        raw_connection.close()
    return max(inserted, 0)


def run_import(progress_path: Path, truncate: bool, retry_wait_seconds: int) -> None:
    files = discover_pbp_csv_files()
    total_matches = sum(count_matches(path) for path in files)
    total_files = len(files)
    started_at = time.time()
    processed_files = 0
    processed_matches = 0
    total_points_inserted = 0
    engine = get_engine()

    while True:
        try:
            ensure_table(engine)
            if truncate:
                truncate_table(engine)
            break
        except SQLAlchemyError as exc:
            progress = build_progress(
                status="waiting_for_db",
                total_files=total_files,
                processed_files=processed_files,
                total_matches=total_matches,
                processed_matches=processed_matches,
                total_points_inserted=total_points_inserted,
                started_at=started_at,
                current_file=None,
                message=str(exc),
            )
            write_progress(progress_path, progress)
            time.sleep(retry_wait_seconds)

    for path in files:
        current_file = path.name
        batch_rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, match_row in enumerate(reader):
                if "pbp_id" not in match_row or not match_row.get("pbp_id"):
                    match_row["pbp_id"] = f"{path.stem}:{index}"
                parsed_points = parse_pbp_points(
                    str(match_row.get("pbp") or ""),
                    str(match_row.get("server1") or ""),
                    str(match_row.get("server2") or ""),
                )
                for parsed_point in parsed_points:
                    batch_rows.append(point_row_from_match(match_row, parsed_point))
                processed_matches += 1
                if len(batch_rows) >= 5000:
                    total_points_inserted += flush_batch(engine, batch_rows, source_file=path.name)
                    batch_rows = []
                    write_progress(
                        progress_path,
                        build_progress(
                            status="running",
                            total_files=total_files,
                            processed_files=processed_files,
                            total_matches=total_matches,
                            processed_matches=processed_matches,
                            total_points_inserted=total_points_inserted,
                            started_at=started_at,
                            current_file=current_file,
                        ),
                    )
        total_points_inserted += flush_batch(engine, batch_rows, source_file=path.name)
        processed_files += 1
        write_progress(
            progress_path,
            build_progress(
                status="running",
                total_files=total_files,
                processed_files=processed_files,
                total_matches=total_matches,
                processed_matches=processed_matches,
                total_points_inserted=total_points_inserted,
                started_at=started_at,
                current_file=current_file,
            ),
        )

    write_progress(
        progress_path,
        build_progress(
            status="completed",
            total_files=total_files,
            processed_files=processed_files,
            total_matches=total_matches,
            processed_matches=processed_matches,
            total_points_inserted=total_points_inserted,
            started_at=started_at,
            current_file=None,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--progress-path",
        default=str(settings.artifacts_dir / "imports" / "pointbypoint_progress.json"),
    )
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--retry-wait-seconds", type=int, default=30)
    args = parser.parse_args()

    progress_path = Path(args.progress_path).resolve()
    try:
        run_import(
            progress_path=progress_path,
            truncate=args.truncate,
            retry_wait_seconds=args.retry_wait_seconds,
        )
    except Exception as exc:  # pragma: no cover
        write_progress(
            progress_path,
            ImportProgress(
                status="failed",
                total_files=0,
                processed_files=0,
                total_matches=0,
                processed_matches=0,
                total_points_inserted=0,
                percent_complete=0.0,
                elapsed_seconds=0.0,
                eta_seconds=None,
                current_file=None,
                started_at_utc=utc_now(),
                updated_at_utc=utc_now(),
                message=str(exc),
            ),
        )
        raise


if __name__ == "__main__":
    main()
