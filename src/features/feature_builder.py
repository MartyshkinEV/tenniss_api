from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(a: pd.Series, b: pd.Series):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b.notna()) & (b != 0), a / b, np.nan)


def rank_diff_bucket(rank_diff: float) -> str:
    if pd.isna(rank_diff):
        return "unknown"
    if rank_diff <= -50:
        return "p1_clear_edge"
    if rank_diff <= -10:
        return "p1_edge"
    if rank_diff < 10:
        return "even"
    if rank_diff < 50:
        return "p2_edge"
    return "p2_clear_edge"


def compute_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["player_id", "tourney_date", "match_id"]).reset_index(drop=True)

    df["hold_rate_match"] = safe_div(
        df["service_games"] - (df["bp_faced"] - df["bp_saved"]),
        df["service_games"],
    )
    df["break_rate_match"] = safe_div(
        df["opp_bp_faced"] - df["opp_bp_saved"],
        df["opp_service_games"],
    )

    df["p_win_shift"] = df.groupby("player_id")["is_win"].shift(1)
    df["recent_form_last5"] = (
        df.groupby("player_id")["p_win_shift"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    df["recent_form_last10"] = (
        df.groupby("player_id")["p_win_shift"]
        .transform(lambda s: s.rolling(10, min_periods=1).mean())
    )
    df["winrate_last10"] = (
        df.groupby("player_id")["p_win_shift"]
        .transform(lambda s: s.rolling(10, min_periods=1).mean())
    )

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

    df["surface_win_shift"] = df.groupby(["player_id", "surface"])["is_win"].shift(1)
    df["winrate_surface"] = (
        df.groupby(["player_id", "surface"])["surface_win_shift"]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    df["surface_recent_form"] = (
        df.groupby(["player_id", "surface"])["surface_win_shift"]
        .transform(lambda s: s.rolling(10, min_periods=1).mean())
    )

    df["matches_last7days"] = 0
    df["matches_last14days"] = 0
    df["minutes_last14days"] = 0.0
    for _, idx in df.groupby("player_id").groups.items():
        idx = list(idx)
        dates = df.loc[idx, "tourney_date"].values.astype("datetime64[D]")
        minutes = pd.to_numeric(df.loc[idx, "minutes"], errors="coerce").fillna(0).to_numpy()
        counts_7 = np.zeros(len(idx), dtype=int)
        counts_14 = np.zeros(len(idx), dtype=int)
        minutes_14 = np.zeros(len(idx), dtype=float)
        left_7 = 0
        left_14 = 0
        for right in range(len(idx)):
            while dates[right] - dates[left_7] > np.timedelta64(7, "D"):
                left_7 += 1
            while dates[right] - dates[left_14] > np.timedelta64(14, "D"):
                left_14 += 1
            counts_7[right] = right - left_7
            counts_14[right] = right - left_14
            minutes_14[right] = minutes[left_14:right].sum() if right > left_14 else 0.0
        df.loc[idx, "matches_last7days"] = counts_7
        df.loc[idx, "matches_last14days"] = counts_14
        df.loc[idx, "minutes_last14days"] = minutes_14

    return df


def build_h2h_features(match_df: pd.DataFrame) -> pd.DataFrame:
    match_df = match_df.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)
    h2h_counts = {}
    surface_h2h_counts = {}
    p1_h2h_wins = []
    p2_h2h_wins = []
    p1_h2h_surface_wins = []
    p2_h2h_surface_wins = []

    for _, row in match_df.iterrows():
        p1 = int(row["p1_id"])
        p2 = int(row["p2_id"])
        surface = row["surface"]

        key = tuple(sorted((p1, p2)))
        surface_key = (key, surface)
        rec = h2h_counts.get(key, {p1: 0, p2: 0})
        surface_rec = surface_h2h_counts.get(surface_key, {p1: 0, p2: 0})

        p1_h2h_wins.append(rec.get(p1, 0))
        p2_h2h_wins.append(rec.get(p2, 0))
        p1_h2h_surface_wins.append(surface_rec.get(p1, 0))
        p2_h2h_surface_wins.append(surface_rec.get(p2, 0))

        if row["label"] == 1:
            rec[p1] = rec.get(p1, 0) + 1
            surface_rec[p1] = surface_rec.get(p1, 0) + 1
        else:
            rec[p2] = rec.get(p2, 0) + 1
            surface_rec[p2] = surface_rec.get(p2, 0) + 1

        h2h_counts[key] = rec
        surface_h2h_counts[surface_key] = surface_rec

    match_df["p1_h2h_wins"] = p1_h2h_wins
    match_df["p2_h2h_wins"] = p2_h2h_wins
    match_df["p1_h2h_surface_wins"] = p1_h2h_surface_wins
    match_df["p2_h2h_surface_wins"] = p2_h2h_surface_wins
    return match_df


def build_match_features(stats_df: pd.DataFrame) -> pd.DataFrame:
    df = compute_player_rolling_features(stats_df)

    counts = df.groupby("match_id").size()
    valid_match_ids = counts[counts == 2].index

    df2 = df[df["match_id"].isin(valid_match_ids)].copy()
    df2 = df2.sort_values(["match_id", "player_id"]).reset_index(drop=True)

    p1 = df2.groupby("match_id").nth(0).reset_index()
    p2 = df2.groupby("match_id").nth(1).reset_index()

    match_df = pd.DataFrame(
        {
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
            "label": p1["is_win"].astype(int),
        }
    )

    match_df["rank_diff"] = match_df["p1_rank"] - match_df["p2_rank"]
    match_df["rank_diff_bucket"] = match_df["rank_diff"].apply(rank_diff_bucket)
    match_df["rank_points_diff"] = match_df["p1_rank_points"] - match_df["p2_rank_points"]
    match_df["winrate_last10_diff"] = match_df["p1_winrate_last10"] - match_df["p2_winrate_last10"]
    match_df["recent_form_last5_diff"] = (
        match_df["p1_recent_form_last5"] - match_df["p2_recent_form_last5"]
    )
    match_df["recent_form_last10_diff"] = (
        match_df["p1_recent_form_last10"] - match_df["p2_recent_form_last10"]
    )
    match_df["winrate_surface_diff"] = match_df["p1_winrate_surface"] - match_df["p2_winrate_surface"]
    match_df["surface_recent_form_diff"] = (
        match_df["p1_surface_recent_form"] - match_df["p2_surface_recent_form"]
    )
    match_df["hold_rate_diff"] = match_df["p1_hold_rate"] - match_df["p2_hold_rate"]
    match_df["break_rate_diff"] = match_df["p1_break_rate"] - match_df["p2_break_rate"]
    match_df["matches_last7days_diff"] = (
        match_df["p1_matches_last7days"] - match_df["p2_matches_last7days"]
    )
    match_df["matches_last14days_diff"] = (
        match_df["p1_matches_last14days"] - match_df["p2_matches_last14days"]
    )
    match_df["minutes_last14days_diff"] = (
        match_df["p1_minutes_last14days"] - match_df["p2_minutes_last14days"]
    )

    match_df = build_h2h_features(match_df)
    return match_df.drop_duplicates(subset=["match_id"]).copy()
