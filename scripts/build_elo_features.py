import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.engine import get_engine
from config import settings

engine = get_engine()


BASE_ELO = 1500.0
K_FACTOR = 32.0


def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra, rb, score_a, k=K_FACTOR):
    ea = expected_score(ra, rb)
    eb = expected_score(rb, ra)
    new_ra = ra + k * (score_a - ea)
    new_rb = rb + k * ((1.0 - score_a) - eb)
    return new_ra, new_rb


def main():
    query = """
    SELECT
        match_id,
        tourney_date,
        surface,
        p1_id,
        p1_name,
        p2_id,
        p2_name,
        label
    FROM match_features
    WHERE match_id IS NOT NULL
      AND p1_id IS NOT NULL
      AND p2_id IS NOT NULL
      AND tourney_date IS NOT NULL
      AND label IS NOT NULL
    ORDER BY tourney_date, match_id
    """
    df = pd.read_sql(query, engine)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    elo_all = {}
    elo_surface = {}

    p1_elo_list = []
    p2_elo_list = []
    elo_diff_list = []

    p1_surface_elo_list = []
    p2_surface_elo_list = []
    surface_elo_diff_list = []

    for _, row in df.iterrows():
        p1 = int(row["p1_id"])
        p2 = int(row["p2_id"])
        surf = row["surface"]
        label = int(row["label"])

        # Общий ELO до матча
        r1 = elo_all.get(p1, BASE_ELO)
        r2 = elo_all.get(p2, BASE_ELO)

        p1_elo_list.append(r1)
        p2_elo_list.append(r2)
        elo_diff_list.append(r1 - r2)

        # Surface ELO до матча
        key1 = (p1, surf)
        key2 = (p2, surf)

        sr1 = elo_surface.get(key1, BASE_ELO)
        sr2 = elo_surface.get(key2, BASE_ELO)

        p1_surface_elo_list.append(sr1)
        p2_surface_elo_list.append(sr2)
        surface_elo_diff_list.append(sr1 - sr2)

        # Обновление после матча
        new_r1, new_r2 = update_elo(r1, r2, label, k=K_FACTOR)
        elo_all[p1] = new_r1
        elo_all[p2] = new_r2

        new_sr1, new_sr2 = update_elo(sr1, sr2, label, k=K_FACTOR)
        elo_surface[key1] = new_sr1
        elo_surface[key2] = new_sr2

    elo_df = pd.DataFrame({
        "match_id": df["match_id"],
        "p1_elo": p1_elo_list,
        "p2_elo": p2_elo_list,
        "elo_diff": elo_diff_list,
        "p1_surface_elo": p1_surface_elo_list,
        "p2_surface_elo": p2_surface_elo_list,
        "surface_elo_diff": surface_elo_diff_list,
    })

    full_query = """
    SELECT *
    FROM match_features
    ORDER BY tourney_date, match_id
    """
    mf = pd.read_sql(full_query, engine)
    out = mf.merge(elo_df, on="match_id", how="left")

    out.to_sql(
        "match_features_elo",
        engine,
        if_exists="replace",
        index=False,
        chunksize=10000,
        method="multi"
    )

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_match_features_elo_match_id
            ON match_features_elo (match_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_match_features_elo_date
            ON match_features_elo (tourney_date)
        """))
        conn.execute(text("ANALYZE match_features_elo"))

    print("Saved table: match_features_elo")
    print("Rows:", len(out))


if __name__ == "__main__":
    main()
