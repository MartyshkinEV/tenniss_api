try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json

import joblib

from config import settings
from src.training.historical_total_dataset import build_historical_total_training_frame
from src.training.live_point_models import train_binary_classifier


def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=250000)
    args = parser.parse_args()

    dataset = build_historical_total_training_frame(max_rows=args.max_rows)
    result = train_binary_classifier(dataset.frame, target_name="historical_set_total_over")
    metadata = dict(result.metadata)
    metadata.update(dataset.metadata)

    output_path = settings.models_dir / "historical_total_model.joblib"
    metadata_path = settings.models_dir / "historical_total_model.json"
    joblib.dump(result.model, output_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    print(output_path)
    print(metadata_path)
    return str(output_path)


if __name__ == "__main__":
    main()
