from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_builder import build_match_features, parse_score_sets, score_total_games
from src.models.catboost_model import build_model as build_catboost_model
from src.models.logistic_model import build_model as build_logistic_model

TOTAL_LINES = (20.5, 21.5, 22.5, 23.5, 24.5)
HANDICAP_LINES = (-4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5)
MARKET_FEATURES = [
    "surface",
    "tourney_level",
    "round",
    "best_of",
    "rank_diff",
    "rank_points_diff",
    "elo_diff",
    "surface_elo_diff",
    "winrate_last10_diff",
    "recent_form_last5_diff",
    "recent_form_last10_diff",
    "hold_rate_diff",
    "break_rate_diff",
    "avg_total_games",
    "avg_games_diff",
    "tiebreak_rate_diff",
    "three_set_rate_diff",
    "odds_movement",
]


@dataclass(frozen=True)
class MarketTrainingArtifacts:
    market: str
    model_type: str
    model: Any
    metadata: dict[str, Any]


def _winner_loser_games(score: str | None) -> tuple[int | None, int | None]:
    sets = parse_score_sets(score)
    if not sets:
        return None, None
    winner_games = sum(left for left, _ in sets)
    loser_games = sum(right for _, right in sets)
    return winner_games, loser_games


def _match_margin(match_row: pd.Series) -> float | None:
    winner_games, loser_games = _winner_loser_games(match_row.get("score"))
    if winner_games is None or loser_games is None:
        return None
    label = int(match_row.get("label") or 0)
    if label == 1:
        return float(winner_games - loser_games)
    return float(loser_games - winner_games)


def _went_three_sets(match_row: pd.Series) -> float | None:
    sets = parse_score_sets(match_row.get("score"))
    if not sets:
        return None
    if int(match_row.get("best_of") or 3) != 3:
        return None
    return float(len(sets) == 3)


def build_market_training_frames(
    stats_df: pd.DataFrame,
    *,
    total_lines: tuple[float, ...] = TOTAL_LINES,
    handicap_lines: tuple[float, ...] = HANDICAP_LINES,
) -> dict[str, pd.DataFrame]:
    match_df = build_match_features(stats_df).copy()
    if match_df.empty:
        return {
            "match_winner": match_df,
            "games_total": pd.DataFrame(),
            "games_handicap": pd.DataFrame(),
            "three_sets": pd.DataFrame(),
        }

    match_df["market"] = "match_winner"
    match_df["pick"] = match_df["label"].map({1: "player1", 0: "player2"})
    match_df["total_games"] = match_df["score"].apply(score_total_games) if "score" in match_df.columns else pd.NA
    match_df["margin_games"] = match_df.apply(_match_margin, axis=1)
    match_df["three_sets_label"] = match_df.apply(_went_three_sets, axis=1)

    total_rows: list[dict[str, Any]] = []
    handicap_rows: list[dict[str, Any]] = []
    for row in match_df.to_dict(orient="records"):
        total_games = row.get("total_games")
        margin_games = row.get("margin_games")
        for line in total_lines:
            if pd.isna(total_games):
                continue
            total_rows.append(
                {
                    **row,
                    "market": "games_total",
                    "pick": "over",
                    "line": float(line),
                    "label": int(float(total_games) > float(line)),
                }
            )
        for line in handicap_lines:
            if pd.isna(margin_games):
                continue
            handicap_rows.append(
                {
                    **row,
                    "market": "games_handicap",
                    "pick": "player1",
                    "line": float(line),
                    "label": int(float(margin_games) + float(line) > 0.0),
                }
            )

    three_sets = match_df.dropna(subset=["three_sets_label"]).copy()
    three_sets["market"] = "three_sets"
    three_sets["pick"] = "yes"
    three_sets["label"] = three_sets["three_sets_label"].astype(int)

    return {
        "match_winner": match_df.copy(),
        "games_total": pd.DataFrame(total_rows),
        "games_handicap": pd.DataFrame(handicap_rows),
        "three_sets": three_sets,
    }


def _feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    available = [column for column in MARKET_FEATURES if column in frame.columns]
    X = frame[available].copy()
    categorical = [column for column in available if not is_numeric_dtype(X[column])]
    numeric = [column for column in available if column not in categorical]
    return X, numeric, categorical


def _build_logistic_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_features,
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_features,
                        ),
                    ],
                ),
            ),
            ("classifier", build_logistic_model(class_weight="balanced")),
        ]
    )


def train_market_model(frame: pd.DataFrame, *, market: str, model_type: str) -> MarketTrainingArtifacts:
    train_frame = frame.dropna(subset=["label"]).copy()
    if train_frame.empty:
        raise ValueError(f"No training rows available for market {market}")

    X, numeric_features, categorical_features = _feature_matrix(train_frame)
    y = train_frame["label"].astype(int)
    if model_type == "catboost":
        model = build_catboost_model(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
        )
        model.fit(X, y, cat_features=categorical_features)
    elif model_type == "logreg":
        model = _build_logistic_pipeline(numeric_features, categorical_features)
        model.fit(X, y)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return MarketTrainingArtifacts(
        market=market,
        model_type=model_type,
        model=model,
        metadata={
            "market": market,
            "model_type": model_type,
            "rows": int(len(train_frame)),
            "positive_rate": float(y.mean()),
            "feature_columns": list(X.columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
    )


def save_market_model(artifact: MarketTrainingArtifacts, output_path: str) -> None:
    joblib.dump(artifact.model, output_path)
