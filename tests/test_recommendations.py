import unittest

from src.live.recommendations import build_recommendations
from src.live.runtime import LiveMarket, ScoredSelection


class StubRuntime:
    def __init__(self, payloads):
        self.payloads = payloads

    def _score_market_details(self, market):
        return self.payloads[market.market_id]


class RecommendationTest(unittest.TestCase):
    def test_build_recommendations_prefers_higher_ranking_score(self):
        match_market = LiveMarket(
            market_id="m1",
            event_id="e1",
            competition="ITF",
            surface="Hard",
            round_name="R32",
            best_of=3,
            tourney_level="tour",
            player1_name="A",
            player2_name="B",
            player1_odds=2.1,
            player2_odds=1.8,
            market_type="match_winner",
            raw={"score": "1:1", "player1_factor_id": 921},
        )
        total_market = LiveMarket(
            market_id="m2",
            event_id="e2",
            competition="ITF",
            surface="Hard",
            round_name="1st set",
            best_of=3,
            tourney_level="tour",
            player1_name="A",
            player2_name="B",
            player1_odds=2.4,
            player2_odds=1.5,
            market_type="set_total_over_under",
            raw={"score": "1:1", "player1_factor_id": 1848, "player1_param": 950},
        )
        empty_market = LiveMarket(
            market_id="m3",
            event_id="e3",
            competition="ATP",
            surface="Clay",
            round_name="R16",
            best_of=3,
            tourney_level="tour",
            player1_name="C",
            player2_name="D",
            player1_odds=1.8,
            player2_odds=2.0,
            market_type="match_winner",
            raw={"score": "0:0"},
        )
        runtime = StubRuntime(
            {
                "m1": (
                    None,
                    {"feature": 1},
                    0.6,
                    0.4,
                    ScoredSelection(
                        market=match_market,
                        side="player1",
                        player_name="A",
                        model_probability=0.6,
                        implied_probability=1 / 2.1,
                        edge=0.12,
                        odds=2.1,
                        stake=30.0,
                        player_id=1,
                        acceptance_probability=0.9,
                        ranking_score=0.108,
                    ),
                ),
                "m2": (
                    None,
                    {"feature": 2},
                    0.7,
                    0.3,
                    ScoredSelection(
                        market=total_market,
                        side="player1",
                        player_name="over",
                        model_probability=0.7,
                        implied_probability=1 / 2.4,
                        edge=0.16,
                        odds=2.4,
                        stake=30.0,
                        player_id=0,
                        acceptance_probability=0.95,
                        ranking_score=0.152,
                    ),
                ),
                "m3": (None, {"feature": 3}, 0.51, 0.49, None),
            }
        )

        recommendations = build_recommendations(runtime, [match_market, total_market, empty_market])
        self.assertEqual(recommendations[0]["action"], "bet_total")
        self.assertEqual(recommendations[0]["selection"], "over")
        self.assertEqual(recommendations[1]["action"], "no_bet")
        self.assertEqual(recommendations[1]["player1_name"], "C")


if __name__ == "__main__":
    unittest.main()
