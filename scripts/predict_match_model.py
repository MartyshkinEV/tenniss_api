try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import os

import joblib
import pandas as pd

from config import (
    get_model_spec,
    resolve_default_model_artifact_path,
    resolve_default_model_name,
    resolve_model_artifact_path,
    settings,
)
from src.data import BASELINE_FEATURES, ELO_FEATURES, align_frame_to_model
from src.data.pipeline import _db_table_available, load_match_features
from src.data.pipeline import load_match_features_elo


def resolve_model_path(model_name: str | None = None) -> str:
    if model_name is None:
        return str(resolve_default_model_artifact_path())
    return str(resolve_model_artifact_path(model_name))


def resolve_prediction_inputs(model_name: str):
    spec = get_model_spec(model_name)
    if spec["feature_set"] == "elo":
        return load_match_features_elo(), ELO_FEATURES
    return load_match_features(), BASELINE_FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=resolve_default_model_name(),
        help="Model artifact name under artifacts/models/",
    )
    args = parser.parse_args()

    model_name = os.path.basename(args.model)
    model = joblib.load(resolve_model_path(model_name))
    df, feature_list = resolve_prediction_inputs(model_name)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    X = align_frame_to_model(model, df, feature_list)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = pd.DataFrame({
        "match_id": df["match_id"], "tourney_date": df["tourney_date"], "surface": df["surface"],
        "tourney_level": df["tourney_level"], "round": df["round"], "p1_id": df["p1_id"],
        "p1_name": df["p1_name"], "p2_id": df["p2_id"], "p2_name": df["p2_name"],
        "actual_label": df["label"], "pred_proba": proba, "pred_label": pred, "model_name": model_name.removesuffix(".joblib"),
    })

    if _db_table_available("match_features"):
        from sqlalchemy import text

        from src.db.engine import get_engine

        engine = get_engine()
        out.to_sql("match_predictions", engine, if_exists="replace", index=False, chunksize=10000, method="multi")

        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_predictions_match_id ON match_predictions (match_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_predictions_date ON match_predictions (tourney_date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_predictions_model ON match_predictions (model_name)"))
            conn.execute(text("ANALYZE match_predictions"))

        print("Saved predictions to table: match_predictions")
        return

    output_dir = settings.artifacts_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name.removesuffix('.joblib')}_predictions.csv"
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to file: {output_path}")


if __name__ == "__main__":
    main()
