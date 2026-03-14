import pandas as pd
from sqlalchemy import text

from config import settings
from src.db.engine import get_engine
from src.features.feature_builder import build_match_features

engine = get_engine()


def main():
    print("Loading player_match_stats...")
    query = """
    SELECT
        match_id,
        player_id,
        player_name,
        opponent_id,
        opponent_name,
        is_win,
        surface,
        tourney_date,
        tourney_level,
        round,
        best_of,
        minutes,
        player_rank,
        player_rank_points,
        opponent_rank,
        opponent_rank_points,
        ace,
        df,
        svpt,
        first_in,
        first_won,
        second_won,
        service_games,
        bp_saved,
        bp_faced,
        opp_ace,
        opp_df,
        opp_svpt,
        opp_first_in,
        opp_first_won,
        opp_second_won,
        opp_service_games,
        opp_bp_saved,
        opp_bp_faced
    FROM player_match_stats
    WHERE player_id IS NOT NULL
      AND opponent_id IS NOT NULL
      AND tourney_date IS NOT NULL
    ORDER BY tourney_date, match_id, player_id
    """
    df = pd.read_sql(query, engine)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    print(f"Loaded rows: {len(df):,}")
    match_df = build_match_features(df)

    print(f"Final match_features rows: {len(match_df):,}")
    print("Writing to PostgreSQL...")
    match_df.to_sql("match_features", engine, if_exists="replace", index=False, method="multi", chunksize=10000)

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_date ON match_features (tourney_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_match_id ON match_features (match_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_surface ON match_features (surface)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_level ON match_features (tourney_level)"))
        conn.execute(text("ANALYZE match_features"))

    print(f"Done. SQL dir configured: {settings.sql_dir}")


if __name__ == "__main__":
    main()
