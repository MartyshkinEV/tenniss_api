import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.live.fonbet import extract_markets_from_fonbet_catalog
from src.live.total_model import SetTotalModel
from src.training.historical_total_dataset import build_historical_total_training_frame


class StubTotalModel:
    def predict_proba(self, frame):
        return [[0.35, 0.65] for _ in range(len(frame))]


class TotalModelTest(unittest.TestCase):
    def test_extract_total_market_from_fonbet_catalog(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 18148, "parentId": 4, "kind": "segment", "name": "ITF. Women. France. Qualification"},
            ],
            "events": [
                {"id": 63295837, "level": 1, "sportId": 18148, "place": "live", "team1": "Gram O", "team2": "Ahti Venla"},
                {"id": 63312370, "parentId": 63295837, "name": "1st set", "level": 2, "sportId": 18148, "place": "live"},
            ],
            "eventMiscs": [{"id": 63312370, "score1": 1, "score2": 1, "liveDelay": 0, "comment": "(15-15*)", "servingTeam": 2}],
            "customFactors": [{"e": 63312370, "factors": [{"f": 1848, "v": 2.15, "p": 950}, {"f": 1849, "v": 1.62, "p": 950}]}],
        }
        markets = extract_markets_from_fonbet_catalog(payload)
        total_markets = [market for market in markets if market.market_type == "set_total_over_under"]
        self.assertEqual(len(total_markets), 1)
        self.assertEqual(total_markets[0].raw["total_line"], 9.5)
        self.assertEqual(total_markets[0].player1_odds, 2.15)

    def test_set_total_model_blends_historical_and_heuristic(self):
        class Market:
            surface = "Hard"
            tourney_level = "itf"
            best_of = 3
            raw = {"score": "1:1", "total_line": 9.5}

        model = SetTotalModel(historical_model=StubTotalModel(), historical_weight=0.5)
        prediction = model.predict_over(Market(), {"p1_hold_rate": 0.7, "p2_hold_rate": 0.68, "p1_break_rate": 0.2, "p2_break_rate": 0.21})
        self.assertGreater(prediction.over_probability, 0.0)
        self.assertLess(prediction.over_probability, 1.0)
        self.assertEqual(prediction.historical_probability, 0.65)

    def test_build_historical_total_training_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "atp_matches_2024.csv"
            path.write_text(
                "\n".join(
                    [
                        "surface,tourney_level,best_of,score,winner_rank,loser_rank,winner_rank_points,loser_rank_points",
                        "Hard,tour,3,7-5 6-3,50,100,1000,600",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("src.training.historical_total_dataset.discover_match_csv_files", return_value=[path]):
                dataset = build_historical_total_training_frame(max_rows=20)
        self.assertGreater(len(dataset.frame), 0)
        self.assertIn("line", dataset.frame.columns)
        self.assertIn("label", dataset.frame.columns)


if __name__ == "__main__":
    unittest.main()
