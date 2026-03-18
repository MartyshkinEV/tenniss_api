try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json

import joblib
import pandas as pd

from config import settings
from src.db.engine import get_engine
from src.training.market_training import build_market_training_frames, train_market_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["catboost", "logreg"],
        choices=["catboost", "logreg"],
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["match_winner", "games_total", "games_handicap", "three_sets"],
        choices=["match_winner", "games_total", "games_handicap", "three_sets"],
    )
    args = parser.parse_args()

    engine = get_engine()
    stats_df = pd.read_sql("SELECT * FROM player_match_stats", engine)
    frames = build_market_training_frames(stats_df)

    for market in args.markets:
        frame = frames.get(market)
        if frame is None or frame.empty:
            print(f"Skipping {market}: empty training frame")
            continue
        for model_type in args.model_types:
            artifact = train_market_model(frame, market=market, model_type=model_type)
            model_path = settings.models_dir / f"{market}_{model_type}.joblib"
            metadata_path = settings.models_dir / f"{market}_{model_type}.json"
            joblib.dump(artifact.model, model_path)
            metadata_path.write_text(
                json.dumps(artifact.metadata, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            print(model_path)
            print(metadata_path)


if __name__ == "__main__":
    main()
