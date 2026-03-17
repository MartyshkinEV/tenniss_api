import tempfile
import unittest
from pathlib import Path

from src.training.live_point_dataset import build_point_training_frames
from src.training.live_point_models import train_binary_classifier


class LivePointTrainingTest(unittest.TestCase):
    def test_build_point_training_frames_extracts_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "point_trajectories.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"timestamp_utc":"2026-03-15T10:00:00+00:00","stage":"scored","event_id":"e1","market_id":"m1","market_type":"point_plus_one_winner","competition":"ITF","round_name":"2nd set","player1_name":"A","player2_name":"B","player1_odds":2.0,"player2_odds":1.8,"live_score":"2:4","live_comment":"(40-15*)","serving_team":2,"target_point_number":7,"player1_factor_id":2995,"player2_factor_id":2996,"player1_param":458759,"player2_param":458759,"scope_market_id":"1600","player1_probability":0.35,"player2_probability":0.65,"selected_side":"player2","selected_player_name":"B","selected_odds":1.8,"selected_edge":0.09,"selected_stake":30.0,"selected_implied_probability":0.56,"state_features":{"p1_hold_rate":0.78,"p2_hold_rate":0.72}}',
                        '{"timestamp_utc":"2026-03-15T10:00:01+00:00","stage":"bet_placed_fast_mode","event_id":"e1","market_id":"m1","market_type":"point_plus_one_winner","competition":"ITF","round_name":"2nd set","player1_name":"A","player2_name":"B","player1_odds":2.0,"player2_odds":1.8,"live_score":"2:4","live_comment":"(40-15*)","serving_team":2,"target_point_number":7,"player1_factor_id":2995,"player2_factor_id":2996,"player1_param":458759,"player2_param":458759,"scope_market_id":"1600","player1_probability":0.35,"player2_probability":0.65,"selected_side":"player2","selected_player_name":"B","selected_odds":1.8,"selected_edge":0.09,"selected_stake":30.0,"selected_implied_probability":0.56,"state_features":{"p1_hold_rate":0.78,"p2_hold_rate":0.72}}',
                        '{"timestamp_utc":"2026-03-15T10:00:02+00:00","stage":"scored","event_id":"e1","market_id":"m2","market_type":"point_plus_one_winner","competition":"ITF","round_name":"2nd set","player1_name":"A","player2_name":"B","player1_odds":1.9,"player2_odds":1.9,"live_score":"2:4","live_comment":"(40-15*)","serving_team":2,"target_point_number":8,"player1_factor_id":2998,"player2_factor_id":2999,"player1_param":458760,"player2_param":458760,"scope_market_id":"1600","player1_probability":0.35,"player2_probability":0.65,"selected_side":"player2","selected_player_name":"B","selected_odds":1.9,"selected_edge":0.12,"selected_stake":30.0,"selected_implied_probability":0.53,"state_features":{"p1_hold_rate":0.78,"p2_hold_rate":0.72}}',
                        '{"timestamp_utc":"2026-03-15T10:00:03+00:00","stage":"observed_no_edge","event_id":"e1","market_id":"m3","market_type":"point_plus_one_winner","competition":"ITF","round_name":"2nd set","player1_name":"A","player2_name":"B","player1_odds":1.8,"player2_odds":2.0,"live_score":"2:4","live_comment":"(40-30*)","serving_team":2,"target_point_number":9,"player1_factor_id":3010,"player2_factor_id":3011,"player1_param":720899,"player2_param":720899,"scope_market_id":"1600","player1_probability":0.4,"player2_probability":0.6,"state_features":{"p1_hold_rate":0.78,"p2_hold_rate":0.72}}',
                        '{"timestamp_utc":"2026-03-15T10:00:04+00:00","stage":"scored","event_id":"e1","market_id":"m4","market_type":"point_plus_one_winner","competition":"ITF","round_name":"2nd set","player1_name":"A","player2_name":"B","player1_odds":1.7,"player2_odds":2.1,"live_score":"2:4","live_comment":"(40-30*)","serving_team":2,"target_point_number":9,"player1_factor_id":3001,"player2_factor_id":3002,"player1_param":458761,"player2_param":458761,"scope_market_id":"1600","player1_probability":0.42,"player2_probability":0.58,"selected_side":"player2","selected_player_name":"B","selected_odds":2.1,"selected_edge":0.10,"selected_stake":30.0,"selected_implied_probability":0.47,"state_features":{"p1_hold_rate":0.78,"p2_hold_rate":0.72}}',
                        '{"timestamp_utc":"2026-03-15T10:00:05+00:00","stage":"refresh_no_edge","event_id":"e1","market_id":"m4","market_type":"point_plus_one_winner","competition":"ITF","round_name":"2nd set","player1_name":"A","player2_name":"B","player1_odds":1.7,"player2_odds":0.0,"live_score":"2:4","live_comment":"(40-30*)","serving_team":2,"target_point_number":9,"player1_factor_id":3001,"player2_factor_id":3002,"player1_param":458761,"player2_param":458761,"scope_market_id":"1600","player1_probability":0.42,"player2_probability":0.58,"selected_side":"player2","selected_player_name":"B","selected_odds":2.1,"selected_edge":0.10,"selected_stake":30.0,"selected_implied_probability":0.47,"state_features":{"p1_hold_rate":0.78,"p2_hold_rate":0.72}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            frames = build_point_training_frames(path)
            self.assertEqual(len(frames.execution_survival), 2)
            self.assertEqual(sorted(frames.execution_survival["label"].tolist()), [0, 1])
            self.assertEqual(len(frames.point_outcome), 1)
            self.assertIn("sf_p1_hold_rate", frames.point_outcome.columns)

    def test_train_binary_classifier_falls_back_for_tiny_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "point_trajectories.jsonl"
            path.write_text("", encoding="utf-8")
            frames = build_point_training_frames(path)
            result = train_binary_classifier(frames.execution_survival, "execution_survival")
            self.assertEqual(result.metadata["model_type"], "dummy_classifier")


if __name__ == "__main__":
    unittest.main()
