import subprocess
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

from config import resolve_default_model_artifact_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_match import build_prediction_frame
from scripts.predict_match_model import resolve_model_path, resolve_prediction_inputs
from src.data import ELO_FEATURES, align_frame_to_model


class PredictMatchTest(unittest.TestCase):
    def test_default_model_path_uses_production_lightgbm_elo(self):
        resolved = Path(resolve_model_path())
        self.assertEqual(resolved, resolve_default_model_artifact_path())
        self.assertEqual(resolved.name, "lightgbm_elo.joblib")

    def test_build_prediction_frame_matches_model_schema(self):
        model = joblib.load(resolve_default_model_artifact_path())
        frame, row = build_prediction_frame(106168, 210000)

        self.assertFalse(frame.empty)
        self.assertIn("elo_diff", row)

        aligned = align_frame_to_model(model, frame, ELO_FEATURES)
        expected = list(getattr(model, "feature_names_in_", ELO_FEATURES))

        self.assertListEqual(list(aligned.columns), expected)
        self.assertEqual(aligned.shape[0], 1)
        self.assertTrue(pd.notna(aligned["elo_diff"].iloc[0]))

    def test_batch_prediction_uses_matching_feature_set(self):
        model = joblib.load(resolve_default_model_artifact_path())
        frame, features = resolve_prediction_inputs("lightgbm_elo.joblib")
        aligned = align_frame_to_model(model, frame.head(5), features)
        expected = list(getattr(model, "feature_names_in_", ELO_FEATURES))

        self.assertListEqual(list(aligned.columns), expected)
        self.assertEqual(aligned.shape[0], 5)
        self.assertIn("surface_elo_diff", aligned.columns)

    def test_point_prediction_command_runs(self):
        cmd = [
            str(PROJECT_ROOT / "venv" / "bin" / "python"),
            str(PROJECT_ROOT / "scripts" / "predict_match.py"),
            "106168",
            "210000",
        ]
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("P(player1 wins) =", completed.stdout)
        self.assertIn("elo_diff =", completed.stdout)


if __name__ == "__main__":
    unittest.main()
