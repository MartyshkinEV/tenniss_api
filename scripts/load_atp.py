try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from config import (
    discover_match_csv_files,
    discover_player_csv_files,
    discover_ranking_csv_files,
    player_id_namespace_offset,
)
from src.db.engine import get_engine

engine = get_engine()

INT_COLS_MATCHES = [
    "draw_size", "match_num", "best_of", "minutes",
    "winner_id", "winner_ht", "loser_id", "loser_ht",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
    "w_ace", "w_df", "w_svpt", "w_1stin", "w_1stwon", "w_2ndwon", "w_svgms", "w_bpsaved", "w_bpfaced",
    "l_ace", "l_df", "l_svpt", "l_1stin", "l_1stwon", "l_2ndwon", "l_svgms", "l_bpsaved", "l_bpfaced",
]
FLOAT_COLS_MATCHES = ["winner_age", "loser_age"]


def load_players():
    frames = []
    for path in discover_player_csv_files():
        df = pd.read_csv(path, low_memory=False)
        cols = {c.lower(): c for c in df.columns}

        player_id_col = cols.get("player_id")
        first_name_col = cols.get("first_name") or cols.get("name_first")
        last_name_col = cols.get("last_name") or cols.get("name_last")
        hand_col = cols.get("hand")
        country_col = cols.get("country_code") or cols.get("ioc")
        height_col = cols.get("height") or cols.get("ht")
        birth_col = cols.get("birth_date") or cols.get("dob") or cols.get("birthdate")

        if not player_id_col:
            print("skip players:", path.name, "missing player_id")
            continue

        out = pd.DataFrame()
        out["player_id"] = pd.to_numeric(df[player_id_col], errors="coerce")
        namespace_offset = player_id_namespace_offset(path)
        if namespace_offset:
            out["player_id"] = out["player_id"] + namespace_offset
        out["first_name"] = df[first_name_col] if first_name_col else None
        out["last_name"] = df[last_name_col] if last_name_col else None
        out["hand"] = df[hand_col] if hand_col else None
        out["country_code"] = df[country_col] if country_col else None
        out["height_cm"] = pd.to_numeric(df[height_col], errors="coerce") if height_col else None
        out["birth_date"] = pd.to_datetime(df[birth_col], format="%Y%m%d", errors="coerce").dt.date if birth_col else None

        out = out.dropna(subset=["player_id"])
        out["player_id"] = out["player_id"].astype("int64")
        out["height_cm"] = pd.to_numeric(out["height_cm"], errors="coerce").astype("Int64")
        out = out.drop_duplicates(subset=["player_id"])
        frames.append(out)
        print("players stage:", path.name, len(out))

    if not frames:
        raise FileNotFoundError("No compatible player CSV files found")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["player_id"])

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE players"))
    out.to_sql("players", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    print("players loaded:", len(out))


def load_rankings():
    files = discover_ranking_csv_files()

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS rankings_stage"))
        conn.execute(text("CREATE TABLE rankings_stage (ranking_date DATE, ranking INTEGER, player_id BIGINT, ranking_points INTEGER)"))
        conn.execute(text("TRUNCATE TABLE rankings"))

    total_stage = 0
    for path in files:
        df = pd.read_csv(path, header=None, low_memory=False)
        if df.shape[1] == 4:
            df.columns = ["ranking_date", "ranking", "player_id", "ranking_points"]
        elif df.shape[1] == 3:
            df.columns = ["ranking_date", "ranking", "player_id"]
            df["ranking_points"] = None
        else:
            print("skip rankings:", path.name, "unexpected cols:", df.shape[1])
            continue

        df["ranking_date"] = pd.to_datetime(df["ranking_date"], format="%Y%m%d", errors="coerce").dt.date
        df["ranking"] = pd.to_numeric(df["ranking"], errors="coerce")
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        namespace_offset = player_id_namespace_offset(path)
        if namespace_offset:
            df["player_id"] = df["player_id"] + namespace_offset
        df["ranking_points"] = pd.to_numeric(df["ranking_points"], errors="coerce")

        df = df.dropna(subset=["ranking_date", "ranking", "player_id"])
        df["ranking"] = df["ranking"].astype("int64")
        df["player_id"] = df["player_id"].astype("int64")
        df["ranking_points"] = df["ranking_points"].astype("Int64")

        df.to_sql("rankings_stage", engine, if_exists="append", index=False, method="multi", chunksize=10000)
        total_stage += len(df)
        print("rankings stage:", path.name, len(df))

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO rankings (ranking_date, ranking, player_id, ranking_points)
            SELECT DISTINCT ranking_date, ranking, player_id, ranking_points
            FROM rankings_stage
            ON CONFLICT (ranking_date, player_id) DO NOTHING
        """))
        conn.execute(text("DROP TABLE IF EXISTS rankings_stage"))

    print("rankings staged total:", total_stage)
    with engine.begin() as conn:
        print("rankings final:", conn.execute(text("SELECT COUNT(*) FROM rankings")).scalar())


def normalize_matches(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = path.name
    if "qual_chall" in path.name:
        df["level_group"] = "qual_chall"
    elif "futures" in path.name:
        df["level_group"] = "futures"
    else:
        df["level_group"] = "tour"

    df["tourney_date"] = pd.to_datetime(df.get("tourney_date"), format="%Y%m%d", errors="coerce").dt.date

    wanted = [
        "source_file", "level_group", "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
        "match_num", "best_of", "round", "minutes", "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc",
        "winner_age", "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age", "winner_rank",
        "winner_rank_points", "loser_rank", "loser_rank_points", "score", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
        "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
    ]
    for col in wanted:
        if col not in df.columns:
            df[col] = None

    out = df[wanted].rename(columns={
        "w_1stIn": "w_1stin", "w_1stWon": "w_1stwon", "w_2ndWon": "w_2ndwon", "w_SvGms": "w_svgms",
        "w_bpSaved": "w_bpsaved", "w_bpFaced": "w_bpfaced", "l_1stIn": "l_1stin", "l_1stWon": "l_1stwon",
        "l_2ndWon": "l_2ndwon", "l_SvGms": "l_svgms", "l_bpSaved": "l_bpsaved", "l_bpFaced": "l_bpfaced",
    })

    for col in INT_COLS_MATCHES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLS_MATCHES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    text_cols = [c for c in out.columns if c not in INT_COLS_MATCHES + FLOAT_COLS_MATCHES + ["tourney_date"]]
    for col in text_cols:
        out[col] = out[col].where(out[col].notna(), None)
    return out


def load_matches():
    files = discover_match_csv_files()
    total = 0
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE matches"))

    for path in files:
        print("loading matches file:", path.name)
        df = normalize_matches(pd.read_csv(path, low_memory=False), path)
        namespace_offset = player_id_namespace_offset(path)
        if namespace_offset:
            for column in ("winner_id", "loser_id"):
                df[column] = pd.to_numeric(df[column], errors="coerce")
                df[column] = df[column] + namespace_offset
        df.to_sql("matches", engine, if_exists="append", index=False, method="multi", chunksize=2000)
        total += len(df)
        print("matches:", path.name, len(df))

    print("matches total:", total)


if __name__ == "__main__":
    print("load players")
    load_players()
    print("load rankings")
    load_rankings()
    print("load matches")
    load_matches()
    print("done")
