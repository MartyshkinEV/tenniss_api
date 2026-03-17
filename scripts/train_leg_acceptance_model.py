try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import json

import joblib

from config import settings
from src.training.live_acceptance_dataset import build_leg_acceptance_frame
from src.training.live_point_models import train_binary_classifier


def main():
    frames = build_leg_acceptance_frame()
    result = train_binary_classifier(frames.leg_acceptance, target_name="leg_acceptance")
    output_path = settings.models_dir / "leg_acceptance_model.joblib"
    metadata_path = settings.models_dir / "leg_acceptance_model.json"
    joblib.dump(result.model, output_path)
    metadata_path.write_text(json.dumps(result.metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    print(output_path)
    print(metadata_path)
    return output_path


if __name__ == "__main__":
    main()
