from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    metadata: dict[str, Any]


def _feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {"label", "event_id", "market_id"}
    categorical = [
        column for column in frame.columns
        if column not in excluded and not is_numeric_dtype(frame[column])
    ]
    numeric = [
        column for column in frame.columns
        if column not in excluded and column not in categorical
    ]
    return numeric, categorical


def train_binary_classifier(frame: pd.DataFrame, target_name: str) -> TrainingResult:
    if frame.empty or "label" not in frame.columns:
        model = DummyClassifier(strategy="prior")
        model.fit([[0], [1]], [0, 1])
        return TrainingResult(
            model=model,
            metadata={
                "target": target_name,
                "model_type": "dummy_classifier",
                "rows": int(len(frame)),
                "positive_rate": None,
                "feature_count": 0,
            },
        )

    numeric_features, categorical_features = _feature_columns(frame)
    X = frame.drop(columns=["label", "event_id", "market_id"], errors="ignore")
    y = frame["label"].astype(int)

    if len(frame) < 10 or y.nunique() < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        return TrainingResult(
            model=model,
            metadata={
                "target": target_name,
                "model_type": "dummy_classifier",
                "rows": int(len(frame)),
                "positive_rate": float(y.mean()) if len(y) else None,
                "feature_count": int(X.shape[1]),
            },
        )

    preprocessor = ColumnTransformer(
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
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return TrainingResult(
        model=model,
        metadata={
            "target": target_name,
            "model_type": "logreg_pipeline",
            "rows": int(len(frame)),
            "positive_rate": float(y.mean()),
            "feature_count": int(X.shape[1]),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
    )
