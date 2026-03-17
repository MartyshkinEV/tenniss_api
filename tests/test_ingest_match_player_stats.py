import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ingest_match_player_stats import load_event_ids_from_feed_url


class IngestMatchPlayerStatsTest(unittest.TestCase):
    def test_load_event_ids_from_feed_url_extracts_unique_event_ids(self):
        payload = {
            "sports": [
                {"id": 4, "kind": "sport", "name": "Tennis", "alias": "tennis"},
                {"id": 71736, "parentId": 4, "kind": "segment", "name": "Open tournament Liga Pro. Men. Balashikha"},
            ],
            "events": [
                {"id": 1001, "level": 1, "sportId": 71736, "place": "live", "team1": "A", "team2": "B"},
                {"id": 1002, "level": 1, "sportId": 71736, "place": "live", "team1": "C", "team2": "D"},
                {"id": 1001, "level": 1, "sportId": 71736, "place": "live", "team1": "A", "team2": "B"},
            ],
            "customFactors": [
                {"e": 1001, "factors": [{"f": 921, "v": 1.8}, {"f": 923, "v": 2.0}]},
                {"e": 1002, "factors": [{"f": 921, "v": 1.9}, {"f": 923, "v": 1.9}]},
            ],
            "eventMiscs": [
                {"id": 1001, "score1": 0, "score2": 0, "liveDelay": 0, "comment": "(0-0)"},
                {"id": 1002, "score1": 0, "score2": 0, "liveDelay": 0, "comment": "(0-0)"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "feed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            def _open(request, timeout=15.0):
                class _Resp:
                    headers = {}

                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                    def read(self_inner):
                        return path.read_bytes()

                return _Resp()

            with patch("scripts.ingest_match_player_stats.urlopen", _open):
                event_ids = load_event_ids_from_feed_url("https://example.test/feed", timeout=5.0, limit=10)

        self.assertEqual(event_ids, [1001, 1002])


if __name__ == "__main__":
    unittest.main()
