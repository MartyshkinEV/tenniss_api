import tempfile
import unittest
from pathlib import Path

from src.training.live_acceptance_dataset import build_leg_acceptance_frame
from src.training.live_point_models import train_binary_classifier


class LiveAcceptanceTrainingTest(unittest.TestCase):
    def test_build_leg_acceptance_frame_extracts_single_and_express_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rl_actions.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"timestamp_utc":"2026-03-15T11:00:00+00:00","event_id":"e1","market_id":"m1","market_type":"set_total_over_under","action":"bet","selection_id":"m1:player1","side":"player1","player_name":"over","odds":2.1,"stake":30.0,"model_probability":0.7,"implied_probability":0.476,"edge":0.224,"result":{"status":"placed","bet_response":{"betDelay":1200},"bet_slip_info_response":{"bets":[{"oldIndex":0,"event":{"id":"e1","score":"3:2"},"factor":{"id":1848,"v":2.0,"p":950}}]},"bet_result_response":{"coupon":{"resultCode":0,"originalK":2.0,"bets":[{"event":"e1","factor":1848,"value":2.0,"param":950,"score":"3:2"}]}}}}',
                        '{"timestamp_utc":"2026-03-15T11:00:01+00:00","event_id":"e2","market_id":"m2","market_type":"match_winner","action":"bet_express","selection_id":"m2:player2","side":"player2","player_name":"B","odds":1.8,"stake":30.0,"model_probability":0.6,"implied_probability":0.555,"edge":0.045,"result":{"status":"placed","bet_response":{"betDelay":900},"bet_slip_info_response":{"bets":[{"oldIndex":0,"event":{"id":"e2","score":"1:1"},"factor":{"id":923,"v":0}},{"oldIndex":1,"event":{"id":"e3","score":"0:0"},"factor":{"id":921,"v":1.9}}]},"bet_result_response":{"coupon":{"resultCode":100,"originalK":3.1,"bets":[{"event":"e2","factor":923,"value":1.8,"score":"1:1"},{"event":"e3","factor":921,"value":1.9,"score":"0:0"}]}}}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            frames = build_leg_acceptance_frame(path)

            self.assertEqual(len(frames.leg_acceptance), 2)
            self.assertEqual(sorted(frames.leg_acceptance["label"].tolist()), [0, 1])
            self.assertIn("coupon_legs", frames.leg_acceptance.columns)
            self.assertIn("slip_factor_value", frames.leg_acceptance.columns)

    def test_train_binary_classifier_acceptance_falls_back_for_empty_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rl_actions.jsonl"
            path.write_text("", encoding="utf-8")
            frames = build_leg_acceptance_frame(path)
            result = train_binary_classifier(frames.leg_acceptance, "leg_acceptance")
            self.assertEqual(result.metadata["model_type"], "dummy_classifier")


if __name__ == "__main__":
    unittest.main()
