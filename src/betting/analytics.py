from __future__ import annotations

from typing import Any

import pandas as pd


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def summarize_calibration(frame: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(["probability_bucket", "avg_model_prob", "actual_rate", "bets"])
    scored = frame.dropna(subset=["model_prob", "actual"]).copy()
    if scored.empty:
        return _empty_frame(["probability_bucket", "avg_model_prob", "actual_rate", "bets"])
    bucket_count = min(max(int(bins), 1), max(len(scored), 1))
    scored["probability_bucket"] = pd.qcut(
        scored["model_prob"],
        q=bucket_count,
        duplicates="drop",
    )
    calibration = (
        scored.groupby("probability_bucket", observed=False)
        .agg(
            avg_model_prob=("model_prob", "mean"),
            actual_rate=("actual", "mean"),
            bets=("actual", "size"),
        )
        .reset_index()
    )
    calibration["probability_bucket"] = calibration["probability_bucket"].astype(str)
    return calibration


def build_profitability_report(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "overall": {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": 0.0},
            "roi_by_market": [],
            "roi_by_tournament": [],
            "roi_by_odds_bucket": [],
            "calibration": [],
        }

    report = frame.copy()
    report["stake"] = pd.to_numeric(report["stake"], errors="coerce").fillna(0.0)
    report["profit"] = pd.to_numeric(report["profit"], errors="coerce").fillna(0.0)
    report["odds"] = pd.to_numeric(report["odds"], errors="coerce")
    report["actual"] = pd.to_numeric(report.get("actual"), errors="coerce")

    total_stake = float(report["stake"].sum())
    total_profit = float(report["profit"].sum())

    def _group_roi(group_by: str) -> list[dict[str, Any]]:
        if group_by not in report.columns:
            return []
        grouped = (
            report.groupby(group_by, dropna=False)
            .agg(bets=("profit", "size"), stake=("stake", "sum"), profit=("profit", "sum"))
            .reset_index()
        )
        grouped["roi"] = grouped.apply(
            lambda row: float(row["profit"]) / float(row["stake"]) if float(row["stake"]) else 0.0,
            axis=1,
        )
        return grouped.to_dict(orient="records")

    odds_frame = report.dropna(subset=["odds"]).copy()
    if odds_frame.empty:
        roi_by_odds_bucket: list[dict[str, Any]] = []
    else:
        odds_frame["odds_bucket"] = pd.cut(
            odds_frame["odds"],
            bins=[0.0, 1.5, 1.8, 2.2, 3.0, float("inf")],
            labels=["<=1.5", "1.51-1.80", "1.81-2.20", "2.21-3.00", ">3.00"],
            include_lowest=True,
        )
        roi_by_odds_bucket = _group_roi("odds_bucket")

    return {
        "overall": {
            "bets": int(len(report)),
            "stake": round(total_stake, 2),
            "profit": round(total_profit, 2),
            "roi": round(total_profit / total_stake, 4) if total_stake else 0.0,
        },
        "roi_by_market": _group_roi("market"),
        "roi_by_tournament": _group_roi("tournament"),
        "roi_by_odds_bucket": roi_by_odds_bucket,
        "calibration": summarize_calibration(report).to_dict(orient="records"),
    }


def backtest_market_predictions(
    frame: pd.DataFrame,
    *,
    result_column: str = "actual",
    model_prob_column: str = "model_prob",
    odds_column: str = "odds",
    stake_column: str = "stake",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out[result_column] = pd.to_numeric(out[result_column], errors="coerce").fillna(0.0)
    out[model_prob_column] = pd.to_numeric(out[model_prob_column], errors="coerce").fillna(0.0)
    out[odds_column] = pd.to_numeric(out[odds_column], errors="coerce").fillna(0.0)
    out[stake_column] = pd.to_numeric(out[stake_column], errors="coerce").fillna(0.0)
    out["bookmaker_prob"] = out.apply(
        lambda row: 1.0 / float(row[odds_column]) if float(row[odds_column]) > 0 else 0.0,
        axis=1,
    )
    out["value"] = out[model_prob_column] - out["bookmaker_prob"]
    out["profit"] = out.apply(
        lambda row: round(float(row[stake_column]) * (float(row[odds_column]) - 1.0), 2)
        if float(row[result_column]) >= 1.0
        else round(-float(row[stake_column]), 2),
        axis=1,
    )
    return out
