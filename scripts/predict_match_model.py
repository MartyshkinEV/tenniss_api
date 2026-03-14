import os

import joblib
import pandas as pd
from sqlalchemy import text

from config import settings
from src.db.engine import get_engine

engine = get_engine()
MODEL_PATH = settings.model_path("logreg_baseline.joblib")
LEGACY_MODEL_PATH = settings.project_root / "models" / "match_winner" / "logreg_baseline.joblib"


def resolve_model_path() -> str:
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    if LEGACY_MODEL_PATH.exists():
        return str(LEGACY_MODEL_PATH)
    raise FileNotFoundError(f"Model not found: {MODEL_PATH} or legacy fallback {LEGACY_MODEL_PATH}")


def main():
    model = joblib.load(resolve_model_path())

    query = """
    SELECT *
    FROM match_features
    WHERE label IS NOT NULL
      AND tourney_date IS NOT NULL
    ORDER BY tourney_date
    """
    df = pd.read_sql(query, engine)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    candidate_features = [
        "surface", "tourney_level", "round", "best_of", "minutes", "p1_rank", "p2_rank",
        "p1_rank_points", "p2_rank_points", "p1_winrate_last10", "p2_winrate_last10",
        "p1_winrate_surface", "p2_winrate_surface", "p1_hold_rate", "p2_hold_rate",
        "p1_break_rate", "p2_break_rate", "p1_recent_form", "p2_recent_form",
        "p1_matches_last7days", "p2_matches_last7days", "rank_diff", "rank_points_diff",
        "winrate_last10_diff", "winrate_surface_diff", "hold_rate_diff", "break_rate_diff",
        "recent_form_diff", "matches_last7days_diff", "p1_h2h_wins", "p2_h2h_wins",
        "p1_h2h_surface", "p2_h2h_surface",
    ]

    features = [c for c in candidate_features if c in df.columns]
    X = df[features]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = pd.DataFrame({
        "match_id": df["match_id"], "tourney_date": df["tourney_date"], "surface": df["surface"],
        "tourney_level": df["tourney_level"], "round": df["round"], "p1_id": df["p1_id"],
        "p1_name": df["p1_name"], "p2_id": df["p2_id"], "p2_name": df["p2_name"],
        "actual_label": df["label"], "pred_proba": proba, "pred_label": pred, "model_name": "logreg_baseline",
    })

    out.to_sql("match_predictions", engine, if_exists="replace", index=False, chunksize=10000, method="multi")

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_predictions_match_id ON match_predictions (match_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_predictions_date ON match_predictions (tourney_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_predictions_model ON match_predictions (model_name)"))
        conn.execute(text("ANALYZE match_predictions"))

    print("Saved predictions to table: match_predictions")


if __name__ == "__main__":
    main()
