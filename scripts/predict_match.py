try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse

import joblib
import pandas as pd

from config import resolve_default_model_artifact_path
from src.data import ELO_FEATURES, align_frame_to_model, load_match_features_elo, load_player_match_stats
from src.features.feature_builder import rank_diff_bucket

SURFACE = "Clay"
TOURNEY_LEVEL = "challenger"
ROUND = "R32"
BEST_OF = 3


def player_stats(player_match_stats: pd.DataFrame, pid: int) -> dict[str, float]:
    df = player_match_stats[player_match_stats["player_id"] == pid].copy()
    df = df.sort_values(["tourney_date", "match_id"], ascending=[False, False]).head(50)

    if df.empty:
        raise ValueError(f"No matches for player {pid}")

    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    winrate_last10 = df.head(10)["is_win"].mean()
    recent_form_last5 = df.head(5)["is_win"].mean()
    recent_form_last10 = df.head(10)["is_win"].mean()

    surface_df = df[df["surface"] == SURFACE]
    winrate_surface = surface_df.head(20)["is_win"].mean()
    if pd.isna(winrate_surface):
        winrate_surface = winrate_last10
    surface_recent_form = surface_df.head(10)["is_win"].mean()
    if pd.isna(surface_recent_form):
        surface_recent_form = recent_form_last10

    hold_num = (
        df["service_games"].fillna(0)
        - (df["bp_faced"].fillna(0) - df["bp_saved"].fillna(0))
    ).clip(lower=0)
    hold_den = df["service_games"].fillna(0)
    hold_rate = hold_num.sum() / hold_den.sum() if hold_den.sum() > 0 else 0.0

    break_num = (
        df["opp_bp_faced"].fillna(0) - df["opp_bp_saved"].fillna(0)
    ).clip(lower=0)
    break_den = df["opp_service_games"].fillna(0)
    break_rate = break_num.sum() / break_den.sum() if break_den.sum() > 0 else 0.0

    rank = df.iloc[0]["player_rank"]
    rank_points = df.iloc[0]["player_rank_points"]

    latest_date = df.iloc[0]["tourney_date"]
    matches_last7 = len(df[df["tourney_date"] >= latest_date - pd.Timedelta(days=7)])
    matches_last14 = len(df[df["tourney_date"] >= latest_date - pd.Timedelta(days=14)])
    minutes_last14 = df[df["tourney_date"] >= latest_date - pd.Timedelta(days=14)]["minutes"].fillna(0).sum()

    return {
        "rank": float(rank) if pd.notna(rank) else 9999.0,
        "rank_points": float(rank_points) if pd.notna(rank_points) else 0.0,
        "winrate_last10": float(winrate_last10) if pd.notna(winrate_last10) else 0.5,
        "recent_form_last5": float(recent_form_last5) if pd.notna(recent_form_last5) else 0.5,
        "recent_form_last10": float(recent_form_last10) if pd.notna(recent_form_last10) else 0.5,
        "winrate_surface": float(winrate_surface) if pd.notna(winrate_surface) else 0.5,
        "surface_recent_form": float(surface_recent_form) if pd.notna(surface_recent_form) else 0.5,
        "hold_rate": float(hold_rate),
        "break_rate": float(break_rate),
        "matches_last7days": int(matches_last7),
        "matches_last14days": int(matches_last14),
        "minutes_last14days": float(minutes_last14),
    }


def get_current_elo(match_features_elo: pd.DataFrame, pid: int) -> tuple[float, float]:
    df = match_features_elo[
        (match_features_elo["p1_id"] == pid) | (match_features_elo["p2_id"] == pid)
    ].copy()
    df = df.sort_values(["tourney_date", "match_id"], ascending=[False, False]).head(1)

    if df.empty:
        return 1500.0, 1500.0

    row = df.iloc[0]

    if int(row["p1_id"]) == pid:
        elo = row["p1_elo"]
        surface_elo = row["p1_surface_elo"]
    else:
        elo = row["p2_elo"]
        surface_elo = row["p2_surface_elo"]

    elo = float(elo) if pd.notna(elo) else 1500.0
    surface_elo = float(surface_elo) if pd.notna(surface_elo) else 1500.0
    return elo, surface_elo


def get_h2h_stats(player_match_stats: pd.DataFrame, player_1: int, player_2: int) -> tuple[int, int, int, int]:
    h2h = player_match_stats[
        (
            (player_match_stats["player_id"] == player_1)
            & (player_match_stats["opponent_id"] == player_2)
        )
        | (
            (player_match_stats["player_id"] == player_2)
            & (player_match_stats["opponent_id"] == player_1)
        )
    ].copy()
    h2h = h2h.sort_values(["tourney_date", "match_id"])

    p1_h2h_wins = int(
        (
            (h2h["player_id"] == player_1)
            & (h2h["opponent_id"] == player_2)
            & (h2h["is_win"] == 1)
        ).sum()
    )
    p2_h2h_wins = int(
        (
            (h2h["player_id"] == player_2)
            & (h2h["opponent_id"] == player_1)
            & (h2h["is_win"] == 1)
        ).sum()
    )

    surface_h2h = h2h[h2h["surface"] == SURFACE]
    p1_h2h_surface_wins = int(
        (
            (surface_h2h["player_id"] == player_1)
            & (surface_h2h["opponent_id"] == player_2)
            & (surface_h2h["is_win"] == 1)
        ).sum()
    )
    p2_h2h_surface_wins = int(
        (
            (surface_h2h["player_id"] == player_2)
            & (surface_h2h["opponent_id"] == player_1)
            & (surface_h2h["is_win"] == 1)
        ).sum()
    )
    return p1_h2h_wins, p2_h2h_wins, p1_h2h_surface_wins, p2_h2h_surface_wins


def build_prediction_frame(player_1: int, player_2: int) -> tuple[pd.DataFrame, dict[str, float]]:
    player_match_stats = load_player_match_stats()
    match_features_elo = load_match_features_elo()

    p1 = player_stats(player_match_stats, player_1)
    p2 = player_stats(player_match_stats, player_2)
    p1_elo, p1_surface_elo = get_current_elo(match_features_elo, player_1)
    p2_elo, p2_surface_elo = get_current_elo(match_features_elo, player_2)
    p1_h2h_wins, p2_h2h_wins, p1_h2h_surface_wins, p2_h2h_surface_wins = get_h2h_stats(
        player_match_stats,
        player_1,
        player_2,
    )

    row = {
        "surface": SURFACE,
        "tourney_level": TOURNEY_LEVEL,
        "round": ROUND,
        "best_of": BEST_OF,
        "rank_diff_bucket": "even",
        "p1_rank": p1["rank"],
        "p2_rank": p2["rank"],
        "p1_rank_points": p1["rank_points"],
        "p2_rank_points": p2["rank_points"],
        "p1_winrate_last10": p1["winrate_last10"],
        "p2_winrate_last10": p2["winrate_last10"],
        "p1_recent_form_last5": p1["recent_form_last5"],
        "p2_recent_form_last5": p2["recent_form_last5"],
        "p1_recent_form_last10": p1["recent_form_last10"],
        "p2_recent_form_last10": p2["recent_form_last10"],
        "p1_winrate_surface": p1["winrate_surface"],
        "p2_winrate_surface": p2["winrate_surface"],
        "p1_surface_recent_form": p1["surface_recent_form"],
        "p2_surface_recent_form": p2["surface_recent_form"],
        "p1_hold_rate": p1["hold_rate"],
        "p2_hold_rate": p2["hold_rate"],
        "p1_break_rate": p1["break_rate"],
        "p2_break_rate": p2["break_rate"],
        "p1_matches_last7days": p1["matches_last7days"],
        "p2_matches_last7days": p2["matches_last7days"],
        "p1_matches_last14days": p1["matches_last14days"],
        "p2_matches_last14days": p2["matches_last14days"],
        "p1_minutes_last14days": p1["minutes_last14days"],
        "p2_minutes_last14days": p2["minutes_last14days"],
        "p1_h2h_wins": p1_h2h_wins,
        "p2_h2h_wins": p2_h2h_wins,
        "p1_h2h_surface_wins": p1_h2h_surface_wins,
        "p2_h2h_surface_wins": p2_h2h_surface_wins,
        "elo_diff": p1_elo - p2_elo,
        "surface_elo_diff": p1_surface_elo - p2_surface_elo,
    }

    row["rank_diff"] = row["p1_rank"] - row["p2_rank"]
    row["rank_diff_bucket"] = rank_diff_bucket(row["rank_diff"])
    row["rank_points_diff"] = row["p1_rank_points"] - row["p2_rank_points"]
    row["winrate_last10_diff"] = row["p1_winrate_last10"] - row["p2_winrate_last10"]
    row["recent_form_last5_diff"] = row["p1_recent_form_last5"] - row["p2_recent_form_last5"]
    row["recent_form_last10_diff"] = row["p1_recent_form_last10"] - row["p2_recent_form_last10"]
    row["winrate_surface_diff"] = row["p1_winrate_surface"] - row["p2_winrate_surface"]
    row["surface_recent_form_diff"] = row["p1_surface_recent_form"] - row["p2_surface_recent_form"]
    row["hold_rate_diff"] = row["p1_hold_rate"] - row["p2_hold_rate"]
    row["break_rate_diff"] = row["p1_break_rate"] - row["p2_break_rate"]
    row["matches_last7days_diff"] = row["p1_matches_last7days"] - row["p2_matches_last7days"]
    row["matches_last14days_diff"] = row["p1_matches_last14days"] - row["p2_matches_last14days"]
    row["minutes_last14days_diff"] = row["p1_minutes_last14days"] - row["p2_minutes_last14days"]
    return pd.DataFrame([row]), row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("player_1", type=int)
    parser.add_argument("player_2", type=int)
    args = parser.parse_args()

    model = joblib.load(resolve_default_model_artifact_path())
    frame, row = build_prediction_frame(args.player_1, args.player_2)
    X = align_frame_to_model(model, frame, ELO_FEATURES)
    proba = model.predict_proba(X)[0][1]

    print()
    print("MODEL PREDICTION")
    print("--------------------------")
    print("player1 id:", args.player_1)
    print("player2 id:", args.player_2)
    print()
    print("P(player1 wins) =", round(float(proba), 3))
    print("P(player2 wins) =", round(float(1 - proba), 3))
    print()
    print("elo_diff =", round(float(row["elo_diff"]), 3))
    print("surface_elo_diff =", round(float(row["surface_elo_diff"]), 3))


if __name__ == "__main__":
    main()
