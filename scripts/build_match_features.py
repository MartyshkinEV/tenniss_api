try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import pandas as pd
from sqlalchemy import text

from config import settings
from src.data.pipeline import _db_table_available, load_match_features


def main():
    print("Building match_features...")
    match_df = load_match_features(refresh=True)
    print(f"Final match_features rows: {len(match_df):,}")
    if _db_table_available("player_match_stats"):
        from src.db.engine import get_engine

        engine = get_engine()
        print("Writing to PostgreSQL...")
        match_df.to_sql("match_features", engine, if_exists="replace", index=False, method="multi", chunksize=10000)

        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_date ON match_features (tourney_date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_match_id ON match_features (match_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_surface ON match_features (surface)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_level ON match_features (tourney_level)"))
            conn.execute(text("ANALYZE match_features"))
        print(f"Done. SQL dir configured: {settings.sql_dir}")
        return

    print("PostgreSQL unavailable; wrote local cache under artifacts/cache/")


if __name__ == "__main__":
    main()
