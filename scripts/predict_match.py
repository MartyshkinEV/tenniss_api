import sys
import joblib
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://tennis_user:tennis_pass@localhost:5432/tennis")

MODEL_PATH = "/opt/tennis_ai/models/match_winner/lightgbm_elo.joblib"
model = joblib.load(MODEL_PATH)

P1 = int(sys.argv[1])
P2 = int(sys.argv[2])

SURFACE = "Clay"
TOURNEY_LEVEL = "challenger"
ROUND = "R32"
BEST_OF = 3


def player_stats(pid: int):
    q = f"""
    SELECT *
    FROM player_match_stats
    WHERE player_id = {pid}
      AND tourney_date IS NOT NULL
    ORDER BY tourney_date DESC, match_id DESC
    LIMIT 50
    """
    df = pd.read_sql(q, engine)

    if len(df) == 0:
        raise Exception(f"No matches for player {pid}")

    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    winrate_last10 = df.head(10)["is_win"].mean()

    surface_df = df[df["surface"] == SURFACE]
    winrate_surface = surface_df.head(20)["is_win"].mean()
    if pd.isna(winrate_surface):
        winrate_surface = winrate_last10

    # match hold rate
    hold_num = (
        df["service_games"].fillna(0)
        - (df["bp_faced"].fillna(0) - df["bp_saved"].fillna(0))
    ).clip(lower=0)
    hold_den = df["service_games"].fillna(0)
    hold_rate = hold_num.sum() / hold_den.sum() if hold_den.sum() > 0 else 0.0

    # match break rate
    break_num = (
        df["opp_bp_faced"].fillna(0) - df["opp_bp_saved"].fillna(0)
    ).clip(lower=0)
    break_den = df["opp_service_games"].fillna(0)
    break_rate = break_num.sum() / break_den.sum() if break_den.sum() > 0 else 0.0

    rank = df.iloc[0]["player_rank"]
    rank_points = df.iloc[0]["player_rank_points"]

    latest_date = df.iloc[0]["tourney_date"]
    matches_last7 = len(df[df["tourney_date"] >= latest_date - pd.Timedelta(days=7)])

    return {
        "rank": float(rank) if pd.notna(rank) else 9999.0,
        "rank_points": float(rank_points) if pd.notna(rank_points) else 0.0,
        "winrate_last10": float(winrate_last10) if pd.notna(winrate_last10) else 0.5,
        "winrate_surface": float(winrate_surface) if pd.notna(winrate_surface) else 0.5,
        "hold_rate": float(hold_rate),
        "break_rate": float(break_rate),
        "matches_last7days": int(matches_last7),
    }


def get_current_elo(pid: int, surface: str):
    q = f"""
    SELECT
        p1_id, p2_id,
        p1_elo, p2_elo,
        p1_surface_elo, p2_surface_elo,
        tourney_date
    FROM match_features_elo
    WHERE p1_id = {pid} OR p2_id = {pid}
    ORDER BY tourney_date DESC, match_id DESC
    LIMIT 1
    """
    df = pd.read_sql(q, engine)

    if len(df) == 0:
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


p1 = player_stats(P1)
p2 = player_stats(P2)

p1_elo, p1_surface_elo = get_current_elo(P1, SURFACE)
p2_elo, p2_surface_elo = get_current_elo(P2, SURFACE)

row = {
    "surface": SURFACE,
    "tourney_level": TOURNEY_LEVEL,
    "round": ROUND,
    "best_of": BEST_OF,

    "p1_rank": p1["rank"],
    "p2_rank": p2["rank"],

    "p1_rank_points": p1["rank_points"],
    "p2_rank_points": p2["rank_points"],

    "p1_winrate_last10": p1["winrate_last10"],
    "p2_winrate_last10": p2["winrate_last10"],

    "p1_winrate_surface": p1["winrate_surface"],
    "p2_winrate_surface": p2["winrate_surface"],

    "p1_hold_rate": p1["hold_rate"],
    "p2_hold_rate": p2["hold_rate"],

    "p1_break_rate": p1["break_rate"],
    "p2_break_rate": p2["break_rate"],

    "p1_matches_last7days": p1["matches_last7days"],
    "p2_matches_last7days": p2["matches_last7days"],

    "p1_h2h_wins": 0,
    "p2_h2h_wins": 0,

    "elo_diff": p1_elo - p2_elo,
    "surface_elo_diff": p1_surface_elo - p2_surface_elo,
}

row["rank_diff"] = row["p1_rank"] - row["p2_rank"]
row["rank_points_diff"] = row["p1_rank_points"] - row["p2_rank_points"]
row["winrate_last10_diff"] = row["p1_winrate_last10"] - row["p2_winrate_last10"]
row["winrate_surface_diff"] = row["p1_winrate_surface"] - row["p2_winrate_surface"]
row["hold_rate_diff"] = row["p1_hold_rate"] - row["p2_hold_rate"]
row["break_rate_diff"] = row["p1_break_rate"] - row["p2_break_rate"]
row["matches_last7days_diff"] = row["p1_matches_last7days"] - row["p2_matches_last7days"]

df = pd.DataFrame([row])

proba = model.predict_proba(df)[0][1]

print()
print("MODEL PREDICTION")
print("--------------------------")
print("player1 id:", P1)
print("player2 id:", P2)
print()
print("P(player1 wins) =", round(proba, 3))
print("P(player2 wins) =", round(1 - proba, 3))
print()
print("elo_diff =", round(row["elo_diff"], 3))
print("surface_elo_diff =", round(row["surface_elo_diff"], 3))
