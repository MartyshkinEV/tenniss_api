try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import json

import joblib

from config import settings
from src.live.markov import MarkovGameModel


def main():
    output_path = settings.models_dir / "game_markov_model.joblib"
    metadata_path = settings.models_dir / "game_markov_model.json"
    model = MarkovGameModel()
    joblib.dump(model, output_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_type": "markov_next_game",
                "server_advantage": model.server_advantage,
                "source": "historical_hold_break_rates",
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
