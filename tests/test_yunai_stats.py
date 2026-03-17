import unittest

from src.live.yunai_stats import (
    build_fonbet_sportradar_url,
    build_match_url,
    extract_match_stats_from_html,
    normalize_player_stat_lines,
    _is_antibot_page,
)


class YunaiStatsTest(unittest.TestCase):
    def test_build_match_url_uses_event_id(self):
        self.assertEqual(
            build_match_url(63295539),
            "https://s5.sir.yunai.tech/fp/en/match/63295539",
        )

    def test_build_fonbet_sportradar_url_uses_event_id(self):
        self.assertEqual(
            build_fonbet_sportradar_url(63296383),
            "https://fon.bet/specificApplication/sportRadarDesktop?id=63296383",
        )

    def test_extract_match_stats_from_embedded_json(self):
        html = """
        <html>
          <head></head>
          <body>
            <script>
              window.__INITIAL_STATE__ = {
                "match": {
                  "id": 63295539,
                  "players": [
                    {"name": "Otakar Krutykh"},
                    {"name": "Andrew Paulson"}
                  ],
                  "statistics": {
                    "aces": {"home": 5, "away": 2},
                    "doubleFaults": {"home": 1, "away": 3},
                    "breakPointsWon": {"home": "2/5", "away": "1/4"}
                  },
                  "form": {
                    "home": ["W", "W", "L"],
                    "away": ["L", "W", "W"]
                  }
                }
              };
            </script>
          </body>
        </html>
        """

        players, stats, candidate_count = extract_match_stats_from_html(html)

        self.assertEqual(players, ["Otakar Krutykh", "Andrew Paulson"])
        self.assertGreaterEqual(candidate_count, 1)
        self.assertIn("statistics", stats)
        self.assertEqual(stats["statistics"]["aces"]["home"], 5)
        self.assertIn("form", stats)

    def test_normalize_player_stat_lines_maps_home_and_away_to_players(self):
        normalized = normalize_player_stat_lines(
            ["Otakar Krutykh", "Andrew Paulson"],
            {
                "statistics": {
                    "home": {"aces": 5, "doubleFaults": 1},
                    "away": {"aces": 2, "doubleFaults": 3},
                },
                "form": {
                    "home": ["W", "W", "L"],
                    "away": ["L", "W", "W"],
                },
            },
        )

        self.assertEqual(normalized["Otakar Krutykh"]["statistics"]["aces"], 5)
        self.assertEqual(normalized["Andrew Paulson"]["statistics"]["doubleFaults"], 3)
        self.assertEqual(normalized["Otakar Krutykh"]["form"], ["W", "W", "L"])

    def test_detects_antibot_page(self):
        html = """
        <html>
          <body>
            <div id="id_captcha_frame_div"></div>
            <script src="//servicepipe.ru/static/checkjs/abc/script.js"></script>
          </body>
        </html>
        """
        self.assertTrue(_is_antibot_page(html))


if __name__ == "__main__":
    unittest.main()
