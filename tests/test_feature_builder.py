import pytest
pd = pytest.importorskip("pandas")

from src.features.feature_builder import build_match_features


def test_build_match_features_smoke():
    df = pd.DataFrame(
        [
            {
                "match_id": 1,
                "player_id": 10,
                "player_name": "A",
                "opponent_id": 20,
                "opponent_name": "B",
                "is_win": 1,
                "surface": "Clay",
                "tourney_date": "2024-01-01",
                "tourney_level": "challenger",
                "round": "R32",
                "best_of": 3,
                "minutes": 90,
                "player_rank": 100,
                "player_rank_points": 500,
                "opponent_rank": 110,
                "opponent_rank_points": 450,
                "ace": 5,
                "df": 2,
                "svpt": 60,
                "first_in": 35,
                "first_won": 25,
                "second_won": 10,
                "service_games": 10,
                "bp_saved": 3,
                "bp_faced": 5,
                "opp_ace": 2,
                "opp_df": 3,
                "opp_svpt": 55,
                "opp_first_in": 30,
                "opp_first_won": 20,
                "opp_second_won": 12,
                "opp_service_games": 10,
                "opp_bp_saved": 2,
                "opp_bp_faced": 4,
            },
            {
                "match_id": 1,
                "player_id": 20,
                "player_name": "B",
                "opponent_id": 10,
                "opponent_name": "A",
                "is_win": 0,
                "surface": "Clay",
                "tourney_date": "2024-01-01",
                "tourney_level": "challenger",
                "round": "R32",
                "best_of": 3,
                "minutes": 90,
                "player_rank": 110,
                "player_rank_points": 450,
                "opponent_rank": 100,
                "opponent_rank_points": 500,
                "ace": 2,
                "df": 3,
                "svpt": 55,
                "first_in": 30,
                "first_won": 20,
                "second_won": 12,
                "service_games": 10,
                "bp_saved": 2,
                "bp_faced": 4,
                "opp_ace": 5,
                "opp_df": 2,
                "opp_svpt": 60,
                "opp_first_in": 35,
                "opp_first_won": 25,
                "opp_second_won": 10,
                "opp_service_games": 10,
                "opp_bp_saved": 3,
                "opp_bp_faced": 5,
            },
        ]
    )
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    out = build_match_features(df)
    assert len(out) == 1
    assert "rank_diff" in out.columns
    assert int(out.iloc[0]["label"]) in (0, 1)
