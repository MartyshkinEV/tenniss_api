import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://tennis_user:tennis_pass@localhost:5432/tennis"

engine = create_engine(DB_URL)


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b.notna()) & (b != 0), a / b, np.nan)


def compute_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["player_id", "tourney_date", "match_id"]).reset_index(drop=True)

    # базовые матчевые метрики
    df["hold_rate_match"] = safe_div(df["service_games"] - (df["bp_faced"] - df["bp_saved"]), df["service_games"])
    df["break_rate_match"] = safe_div(df["opp_bp_faced"] - df["opp_bp_saved"], df["opp_service_games"])

    # last10 winrate
    df["p_win_shift"] = df.groupby("player_id")["is_win"].shift(1)
    df["winrate_last10"] = (
        df.groupby("player_id")["p_win_shift"]
        .transform(lambda s: s.rolling(10, min_periods=1).mean())
    )

    # общие rolling метрики подачи/брейка
    df["hold_rate_shift"] = df.groupby("player_id")["hold_rate_match"].shift(1)
    df["break_rate_shift"] = df.groupby("player_id")["break_rate_match"].shift(1)

    df["hold_rate"] = (
        df.groupby("player_id")["hold_rate_shift"]
        .transform(lambda s: s.rolling(20, min_periods=3).mean())
    )

    df["break_rate"] = (
        df.groupby("player_id")["break_rate_shift"]
        .transform(lambda s: s.rolling(20, min_periods=3).mean())
    )

    # surface-specific winrate
    df["surface_win_shift"] = df.groupby(["player_id", "surface"])["is_win"].shift(1)
    df["winrate_surface"] = (
        df.groupby(["player_id", "surface"])["surface_win_shift"]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )

    # матчи за последние 7 дней
    df["matches_last7days"] = 0

    for player_id, idx in df.groupby("player_id").groups.items():
        idx = list(idx)
        dates = df.loc[idx, "tourney_date"].values.astype("datetime64[D]")
        counts = np.zeros(len(idx), dtype=int)

        left = 0
        for right in range(len(idx)):
            while dates[right] - dates[left] > np.timedelta64(7, "D"):
                left += 1
            counts[right] = right - left  # до текущего матча, без него самого
        df.loc[idx, "matches_last7days"] = counts

    return df


def build_h2h_features(match_df: pd.DataFrame) -> pd.DataFrame:
    match_df = match_df.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)

    h2h_counts = {}
    p1_h2h_wins = []
    p2_h2h_wins = []

    for _, row in match_df.iterrows():
        p1 = int(row["p1_id"])
        p2 = int(row["p2_id"])

        key = tuple(sorted((p1, p2)))
        rec = h2h_counts.get(key, {p1: 0, p2: 0})

        p1_h2h_wins.append(rec.get(p1, 0))
        p2_h2h_wins.append(rec.get(p2, 0))

        if row["label"] == 1:
            rec[p1] = rec.get(p1, 0) + 1
        else:
            rec[p2] = rec.get(p2, 0) + 1

        h2h_counts[key] = rec

    match_df["p1_h2h_wins"] = p1_h2h_wins
    match_df["p2_h2h_wins"] = p2_h2h_wins
    return match_df


def main():
    print("Loading player_match_stats...")
    query = """
    SELECT
        match_id,
        player_id,
        player_name,
        opponent_id,
        opponent_name,
        is_win,
        surface,
        tourney_date,
        tourney_level,
        round,
        best_of,
        minutes,
        player_rank,
        player_rank_points,
        opponent_rank,
        opponent_rank_points,
        ace,
        df,
        svpt,
        first_in,
        first_won,
        second_won,
        service_games,
        bp_saved,
        bp_faced,
        opp_ace,
        opp_df,
        opp_svpt,
        opp_first_in,
        opp_first_won,
        opp_second_won,
        opp_service_games,
        opp_bp_saved,
        opp_bp_faced
    FROM player_match_stats
    WHERE player_id IS NOT NULL
      AND opponent_id IS NOT NULL
      AND tourney_date IS NOT NULL
    ORDER BY tourney_date, match_id, player_id
    """
    df = pd.read_sql(query, engine)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    print(f"Loaded rows: {len(df):,}")

    # rolling features per player
    print("Computing rolling player features...")
    df = compute_player_rolling_features(df)

    # соберем матч из двух строк player_match_stats
    print("Building one-row-per-match dataset...")
    counts = df.groupby("match_id").size()
    valid_match_ids = counts[counts == 2].index

    df2 = df[df["match_id"].isin(valid_match_ids)].copy()
    df2 = df2.sort_values(["match_id", "player_id"]).reset_index(drop=True)

    p1 = df2.groupby("match_id").nth(0).reset_index()
    p2 = df2.groupby("match_id").nth(1).reset_index()

    match_df = pd.DataFrame({
        "match_id": p1["match_id"],
        "tourney_date": p1["tourney_date"],
        "surface": p1["surface"],
        "tourney_level": p1["tourney_level"],
        "round": p1["round"],
        "best_of": p1["best_of"],

        "p1_id": p1["player_id"],
        "p1_name": p1["player_name"],
        "p2_id": p2["player_id"],
        "p2_name": p2["player_name"],

        "p1_rank": p1["player_rank"],
        "p2_rank": p2["player_rank"],
        "p1_rank_points": p1["player_rank_points"],
        "p2_rank_points": p2["player_rank_points"],

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

        # если первая строка player-centric была победителем => p1_win=1
        "label": p1["is_win"].astype(int),
    })

    # разницы
    match_df["rank_diff"] = match_df["p1_rank"] - match_df["p2_rank"]
    match_df["rank_points_diff"] = match_df["p1_rank_points"] - match_df["p2_rank_points"]
    match_df["winrate_last10_diff"] = match_df["p1_winrate_last10"] - match_df["p2_winrate_last10"]
    match_df["winrate_surface_diff"] = match_df["p1_winrate_surface"] - match_df["p2_winrate_surface"]
    match_df["hold_rate_diff"] = match_df["p1_hold_rate"] - match_df["p2_hold_rate"]
    match_df["break_rate_diff"] = match_df["p1_break_rate"] - match_df["p2_break_rate"]
    match_df["matches_last7days_diff"] = match_df["p1_matches_last7days"] - match_df["p2_matches_last7days"]

    # H2H
    print("Computing H2H...")
    match_df = build_h2h_features(match_df)

    # очистка явных дублей
    match_df = match_df.drop_duplicates(subset=["match_id"]).copy()

    print(f"Final match_features rows: {len(match_df):,}")

    # сохраняем как таблицу
    print("Writing to PostgreSQL...")
    match_df.to_sql("match_features", engine, if_exists="replace", index=False, method="multi", chunksize=10000)

    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_date ON match_features (tourney_date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_match_id ON match_features (match_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_surface ON match_features (surface)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_match_features_level ON match_features (tourney_level)"))
        conn.execute(text("ANALYZE match_features"))

    print("Done.")


if __name__ == "__main__":
    main()
