import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.point_parser import parse_pbp_points, point_row_from_match
from src.data.pipeline import _load_historical_points_from_files


class PointParserTest(unittest.TestCase):
    def test_parse_regular_game(self):
        points = parse_pbp_points("SSSS", "Player A", "Player B")
        self.assertEqual(len(points), 4)
        self.assertTrue(all(point.server_name == "Player A" for point in points))
        self.assertEqual(points[-1].score_before_p1, "40")
        self.assertEqual(points[-1].score_before_p2, "0")
        self.assertEqual(points[-1].score_after_p2, "0")
        self.assertEqual(points[-1].point_winner_name, "Player A")

    def test_parse_multiple_games_alternates_server(self):
        points = parse_pbp_points("SSSS;RRRR", "Player A", "Player B")
        self.assertEqual(len(points), 8)
        self.assertEqual(points[0].server_name, "Player A")
        self.assertEqual(points[4].server_name, "Player B")
        self.assertEqual(points[4].point_winner_name, "Player A")

    def test_parse_tiebreak_changes_server_sequence(self):
        points = parse_pbp_points("SR/SR", "Player A", "Player B")
        self.assertEqual(len(points), 4)
        self.assertEqual([point.server_name for point in points], ["Player A", "Player B", "Player B", "Player A"])
        self.assertTrue(all(point.is_tiebreak for point in points))

    def test_point_row_from_match_keeps_metadata(self):
        parsed_point = parse_pbp_points("SSSS", "Player A", "Player B")[0]
        row = point_row_from_match(
            {
                "pbp_id": 1,
                "date": "01 Jan 17",
                "tny_name": "ATP Test",
                "tour": "ATP",
                "draw": "Main",
                "server1": "Player A",
                "server2": "Player B",
                "winner": 1,
                "score": "6-0 6-0",
                "adf_flag": 1,
                "wh_minutes": 60,
            },
            parsed_point,
        )
        self.assertEqual(row["pbp_id"], 1)
        self.assertEqual(row["tny_name"], "ATP Test")
        self.assertEqual(row["point_code"], "S")

    def test_load_historical_points_from_files_builds_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "pbp_matches_test.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "pbp_id,date,tny_name,tour,draw,server1,server2,winner,pbp,score,adf_flag,wh_minutes",
                        "1,01 Jan 17,ATP Test,ATP,Main,Player A,Player B,1,SSSS;RRRR,6-0 6-0,1,60",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("src.data.pipeline._local_pbp_files", return_value=[csv_path]):
                frame = _load_historical_points_from_files()

        self.assertEqual(len(frame), 8)
        self.assertEqual(frame.iloc[0]["server_name"], "Player A")
        self.assertEqual(frame.iloc[4]["server_name"], "Player B")
        self.assertEqual(str(frame.iloc[0]["date"].date()), "2017-01-01")


if __name__ == "__main__":
    unittest.main()
