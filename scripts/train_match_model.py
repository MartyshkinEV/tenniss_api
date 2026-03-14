import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

DB_URL = "postgresql://tennis_user:tennis_pass@localhost:5432/tennis"
engine = create_engine(DB_URL)

MODEL_DIR = "/opt/tennis_ai/models/match_winner"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    query = """
    SELECT *
    FROM match_features
    WHERE label IS NOT NULL
      AND tourney_date IS NOT NULL
    ORDER BY tourney_date
    """
    df = pd.read_sql(query, engine)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    train_df = df[df["tourney_date"] < "2022-01-01"].copy()
    valid_df = df[(df["tourney_date"] >= "2022-01-01") & (df["tourney_date"] < "2024-01-01")].copy()
    features = [
        "surface",
        "tourney_level",
        "round",
        "best_of",

        "p1_rank",
        "p2_rank",

        "p1_rank_points",
        "p2_rank_points",

        "p1_winrate_last10",
        "p2_winrate_last10",

        "p1_winrate_surface",
        "p2_winrate_surface",

        "p1_hold_rate",
        "p2_hold_rate",

        "p1_break_rate",
        "p2_break_rate",

        "p1_matches_last7days",
        "p2_matches_last7days",

        "rank_diff",
        "rank_points_diff",

        "winrate_last10_diff",
        "winrate_surface_diff",

        "hold_rate_diff",
        "break_rate_diff",

        "matches_last7days_diff",

        "p1_h2h_wins",
        "p2_h2h_wins",
    ]
    target = "label"
    test_df = df[df["tourney_date"] >= "2024-01-01"].copy()
    X_train = train_df[features]
    y_train = train_df[target]

    X_valid = valid_df[features]
    y_valid = valid_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    categorical = ["surface", "tourney_level", "round"]
    numeric = [c for c in features if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
        ]
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    model.fit(X_train, y_train)

    for name, X, y in [
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)

        print(f"\n{name.upper()} METRICS")
        print("accuracy =", round(accuracy_score(y, pred), 6))
        print("roc_auc  =", round(roc_auc_score(y, proba), 6))
        print("log_loss =", round(log_loss(y, proba), 6))
        print("brier    =", round(brier_score_loss(y, proba), 6))

    model_path = os.path.join(MODEL_DIR, "logreg_baseline.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
