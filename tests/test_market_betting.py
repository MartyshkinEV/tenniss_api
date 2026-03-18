import unittest

import pandas as pd

from src.betting.analytics import backtest_market_predictions, build_profitability_report
from src.betting.policy import BettingPolicy, BettingPolicyConfig, MarketCandidate
from src.training.market_training import build_market_training_frames, train_market_model


class MarketBettingTest(unittest.TestCase):
    def test_betting_policy_requires_value_and_filters(self):
        policy = BettingPolicy(
            BettingPolicyConfig(
                min_value=0.04,
                min_model_probability=0.58,
                min_data_quality_score=0.8,
                allowed_surfaces=("Hard", "Clay"),
            )
        )
        accepted = policy.evaluate(
            MarketCandidate(
                match_id="m1",
                market="games_total",
                pick="over",
                odds=2.1,
                model_prob=0.61,
                stake=100,
                surface="Hard",
                tournament_level="tour",
                data_quality_score=0.9,
                recent_form=0.6,
            )
        )
        rejected = policy.evaluate(
            MarketCandidate(
                match_id="m2",
                market="games_total",
                pick="under",
                odds=1.8,
                model_prob=0.53,
                stake=100,
                surface="Grass",
                tournament_level="tour",
                data_quality_score=0.5,
                recent_form=0.2,
            )
        )
        self.assertTrue(accepted.should_bet)
        self.assertIn("eligible", accepted.reason)
        self.assertFalse(rejected.should_bet)
        self.assertIn("surface_blocked:Grass", rejected.reason)
        self.assertIn("data_quality_below_min", rejected.reason)

    def test_backtest_and_report(self):
        frame = pd.DataFrame(
            [
                {
                    "market": "games_total",
                    "tournament": "ATP Finals",
                    "odds": 2.0,
                    "stake": 100,
                    "model_prob": 0.62,
                    "actual": 1,
                },
                {
                    "market": "games_handicap",
                    "tournament": "Rome",
                    "odds": 1.9,
                    "stake": 100,
                    "model_prob": 0.59,
                    "actual": 0,
                },
            ]
        )
        backtest = backtest_market_predictions(frame)
        report = build_profitability_report(backtest)
        self.assertIn("profit", backtest.columns)
        self.assertEqual(report["overall"]["bets"], 2)
        self.assertEqual(len(report["roi_by_market"]), 2)
        self.assertGreaterEqual(len(report["calibration"]), 1)

    def test_build_market_training_frames_and_logreg_training(self):
        stats = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "player_id": 10,
                    "player_name": "A",
                    "opponent_id": 20,
                    "opponent_name": "B",
                    "is_win": 1,
                    "surface": "Hard",
                    "tourney_date": "2024-01-01",
                    "tourney_level": "tour",
                    "round": "R32",
                    "best_of": 3,
                    "score": "7-5 4-6 6-4",
                    "minutes": 120,
                    "player_rank": 15,
                    "player_rank_points": 2500,
                    "opponent_rank": 22,
                    "opponent_rank_points": 2100,
                    "elo": 1680,
                    "surface_elo": 1660,
                    "odds_movement": -0.05,
                    "ace": 8,
                    "df": 3,
                    "svpt": 80,
                    "first_in": 50,
                    "first_won": 35,
                    "second_won": 15,
                    "service_games": 15,
                    "bp_saved": 4,
                    "bp_faced": 6,
                    "opp_ace": 5,
                    "opp_df": 4,
                    "opp_svpt": 75,
                    "opp_first_in": 45,
                    "opp_first_won": 30,
                    "opp_second_won": 14,
                    "opp_service_games": 15,
                    "opp_bp_saved": 3,
                    "opp_bp_faced": 7,
                },
                {
                    "match_id": 1,
                    "player_id": 20,
                    "player_name": "B",
                    "opponent_id": 10,
                    "opponent_name": "A",
                    "is_win": 0,
                    "surface": "Hard",
                    "tourney_date": "2024-01-01",
                    "tourney_level": "tour",
                    "round": "R32",
                    "best_of": 3,
                    "score": "7-5 4-6 6-4",
                    "minutes": 120,
                    "player_rank": 22,
                    "player_rank_points": 2100,
                    "opponent_rank": 15,
                    "opponent_rank_points": 2500,
                    "elo": 1615,
                    "surface_elo": 1600,
                    "odds_movement": 0.04,
                    "ace": 5,
                    "df": 4,
                    "svpt": 75,
                    "first_in": 45,
                    "first_won": 30,
                    "second_won": 14,
                    "service_games": 15,
                    "bp_saved": 3,
                    "bp_faced": 7,
                    "opp_ace": 8,
                    "opp_df": 3,
                    "opp_svpt": 80,
                    "opp_first_in": 50,
                    "opp_first_won": 35,
                    "opp_second_won": 15,
                    "opp_service_games": 15,
                    "opp_bp_saved": 4,
                    "opp_bp_faced": 6,
                },
                {
                    "match_id": 2,
                    "player_id": 10,
                    "player_name": "A",
                    "opponent_id": 30,
                    "opponent_name": "C",
                    "is_win": 0,
                    "surface": "Clay",
                    "tourney_date": "2024-01-10",
                    "tourney_level": "tour",
                    "round": "R16",
                    "best_of": 3,
                    "score": "6-2 6-2",
                    "minutes": 70,
                    "player_rank": 15,
                    "player_rank_points": 2500,
                    "opponent_rank": 12,
                    "opponent_rank_points": 2800,
                    "elo": 1685,
                    "surface_elo": 1630,
                    "odds_movement": 0.03,
                    "ace": 3,
                    "df": 2,
                    "svpt": 55,
                    "first_in": 34,
                    "first_won": 22,
                    "second_won": 8,
                    "service_games": 8,
                    "bp_saved": 1,
                    "bp_faced": 5,
                    "opp_ace": 6,
                    "opp_df": 1,
                    "opp_svpt": 54,
                    "opp_first_in": 35,
                    "opp_first_won": 24,
                    "opp_second_won": 9,
                    "opp_service_games": 8,
                    "opp_bp_saved": 4,
                    "opp_bp_faced": 5,
                },
                {
                    "match_id": 2,
                    "player_id": 30,
                    "player_name": "C",
                    "opponent_id": 10,
                    "opponent_name": "A",
                    "is_win": 1,
                    "surface": "Clay",
                    "tourney_date": "2024-01-10",
                    "tourney_level": "tour",
                    "round": "R16",
                    "best_of": 3,
                    "score": "6-2 6-2",
                    "minutes": 70,
                    "player_rank": 12,
                    "player_rank_points": 2800,
                    "opponent_rank": 15,
                    "opponent_rank_points": 2500,
                    "elo": 1710,
                    "surface_elo": 1705,
                    "odds_movement": -0.02,
                    "ace": 6,
                    "df": 1,
                    "svpt": 54,
                    "first_in": 35,
                    "first_won": 24,
                    "second_won": 9,
                    "service_games": 8,
                    "bp_saved": 4,
                    "bp_faced": 5,
                    "opp_ace": 3,
                    "opp_df": 2,
                    "opp_svpt": 55,
                    "opp_first_in": 34,
                    "opp_first_won": 22,
                    "opp_second_won": 8,
                    "opp_service_games": 8,
                    "opp_bp_saved": 1,
                    "opp_bp_faced": 5,
                },
            ]
        )
        stats["tourney_date"] = pd.to_datetime(stats["tourney_date"])
        frames = build_market_training_frames(stats)
        self.assertIn("games_total", frames)
        self.assertIn("games_handicap", frames)
        self.assertIn("three_sets", frames)
        self.assertGreater(len(frames["games_total"]), 0)
        artifact = train_market_model(frames["games_total"], market="games_total", model_type="logreg")
        self.assertEqual(artifact.market, "games_total")
        self.assertEqual(artifact.model_type, "logreg")
        self.assertIn("feature_columns", artifact.metadata)


if __name__ == "__main__":
    unittest.main()
