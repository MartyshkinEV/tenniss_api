try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import numpy as np
from sqlalchemy import text

from config import settings
from src.data.pipeline import _db_table_available, load_match_features_elo


def main():
    out = load_match_features_elo(refresh=True)

    if _db_table_available("match_features"):
        from src.db.engine import get_engine

        engine = get_engine()
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
        return

    print("PostgreSQL unavailable; wrote local cache under artifacts/cache/")
    print("Rows:", len(out))


if __name__ == "__main__":
    main()
