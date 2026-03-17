try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json

import joblib

from config import settings
from src.training.historical_point_dataset import build_historical_point_training_frame
from src.training.live_point_models import train_binary_classifier


def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-offset", type=int, default=settings.live_point_target_offset)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--all-files", action="store_true")
    args = parser.parse_args()

    dataset = build_historical_point_training_frame(
        target_offset=args.target_offset,
        max_rows=args.max_rows,
        current_only=not args.all_files,
    )
    result = train_binary_classifier(dataset.frame, target_name=f"historical_point_t+{args.target_offset}")
    metadata = dict(result.metadata)
    metadata.update(dataset.metadata)
    metadata["target_offset"] = args.target_offset

    output_path = settings.models_dir / "historical_point_model.joblib"
    metadata_path = settings.models_dir / "historical_point_model.json"
    joblib.dump(result.model, output_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    print(output_path)
    print(metadata_path)
    return str(output_path)


if __name__ == "__main__":
    main()
