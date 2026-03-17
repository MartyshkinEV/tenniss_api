import tempfile
import unittest
from pathlib import Path

from src.live.game_model import LayeredGamePredictor, build_live_game_feature_frame
from src.live.point_model import LayeredPointPredictor, build_live_point_feature_frame
from src.live.runtime import LiveMarket
from src.training.historical_point_dataset import build_historical_point_training_frame


class StubProbaModel:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, frame):
        return [[1.0 - self.probability, self.probability] for _ in range(len(frame))]


class HistoricalPointTrainingTest(unittest.TestCase):
    def test_build_historical_point_training_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pbp_matches_test_current.csv"
            path.write_text(
                "\n".join(
                    [
                        "pbp_id,date,tny_name,tour,draw,server1,server2,winner,pbp,score,adf_flag,wh_minutes",
                        "1,01 Jan 17,ATP Test,ATP,Main,Player A,Player B,1,SSSS;RRRR,6-0 6-0,1,60",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            from unittest.mock import patch

            with patch("src.training.historical_point_dataset.discover_pbp_csv_files", return_value=[path]):
                dataset = build_historical_point_training_frame(target_offset=2, max_rows=32, current_only=True)

        self.assertGreater(len(dataset.frame), 0)
        self.assertIn("server_side", dataset.frame.columns)
        self.assertIn("label", dataset.frame.columns)
        self.assertEqual(dataset.metadata["target_offset"], 2)

    def test_layered_point_predictor_blends_models(self):
        market = LiveMarket(
            market_id="m1",
            event_id="e1",
            competition="ITF",
            surface="Hard",
            round_name="2nd set",
            best_of=3,
            tourney_level="itf",
            player1_name="Player A",
            player2_name="Player B",
            player1_odds=1.8,
            player2_odds=2.0,
            market_type="point_plus_one_winner",
            raw={"score": "2:4", "comment": "(40-15*)", "serveT": 1, "target_point_number": 7},
        )
        predictor = LayeredPointPredictor(
            point_model=StubProbaModel(0.70),
            execution_model=StubProbaModel(0.80),
            target_offset=2,
            point_model_weight=0.6,
            markov_weight=0.4,
        )
        prediction = predictor.predict(
            market=market,
            state_features={"p1_hold_rate": 0.7, "p2_hold_rate": 0.6},
            markov_probability=0.55,
        )
        self.assertAlmostEqual(prediction.player1_probability, 0.64, places=2)
        self.assertAlmostEqual(prediction.execution_probability, 0.80, places=2)
        frame = build_live_point_feature_frame(market, target_offset=2)
        self.assertEqual(frame.iloc[0]["set_no"], 2)
        self.assertEqual(frame.iloc[0]["points_p1"], 3)

    def test_layered_game_predictor_blends_models(self):
        market = LiveMarket(
            market_id="m2",
            event_id="e2",
            competition="ITF",
            surface="Hard",
            round_name="3rd set",
            best_of=3,
            tourney_level="itf",
            player1_name="Player A",
            player2_name="Player B",
            player1_odds=1.8,
            player2_odds=2.0,
            market_type="next_game_winner",
            raw={"score": "1:2", "comment": "(40*-30)", "serveT": 1, "target_game_number": 4},
        )
        predictor = LayeredGamePredictor(
            game_model=StubProbaModel(0.70),
            game_model_weight=0.6,
            markov_weight=0.4,
        )
        prediction = predictor.predict(
            market=market,
            markov_probability=0.55,
        )
        self.assertAlmostEqual(prediction.player1_probability, 0.64, places=2)
        frame = build_live_game_feature_frame(market)
        self.assertEqual(frame.iloc[0]["set_no"], 3)
        self.assertEqual(frame.iloc[0]["game_no"], 4)
        self.assertEqual(frame.iloc[0]["points_p1"], 3)


if __name__ == "__main__":
    unittest.main()
