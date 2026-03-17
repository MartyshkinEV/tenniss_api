import tempfile
import unittest
from pathlib import Path

from src.training.historical_game_dataset import build_historical_game_training_frame


class HistoricalGameTrainingTest(unittest.TestCase):
    def test_build_historical_game_training_frame(self):
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

            with patch("src.training.historical_game_dataset.discover_pbp_csv_files", return_value=[path]):
                dataset = build_historical_game_training_frame(max_rows=32, current_only=True)

        self.assertGreater(len(dataset.frame), 0)
        self.assertIn("server_side", dataset.frame.columns)
        self.assertIn("is_break_point_p1", dataset.frame.columns)
        self.assertIn("label", dataset.frame.columns)
        self.assertTrue(set(dataset.frame["label"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
