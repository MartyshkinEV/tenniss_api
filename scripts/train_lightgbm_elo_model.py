try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

from config import settings
from src.data import CATEGORICAL_FEATURES, ELO_FEATURES
from src.data.pipeline import load_match_features_elo, split_dataset


def main():
    print("Loading dataset...")
    df = load_match_features_elo()
    splits = split_dataset(df)

    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    target = "label"
    X_train = train[ELO_FEATURES]
    y_train = train[target]
    X_valid = valid[ELO_FEATURES]
    y_valid = valid[target]
    X_test = test[ELO_FEATURES]
    y_test = test[target]

    categorical = [c for c in CATEGORICAL_FEATURES if c in ELO_FEATURES]
    numeric = [c for c in ELO_FEATURES if c not in categorical]

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    model = Pipeline(
        [
            ("prep", pre),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )

    print("Training LightGBM + ELO...")
    model.fit(X_train, y_train)

    for name, X, y in [("VALID", X_valid, y_valid), ("TEST", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"\n{name}")
        print("accuracy", accuracy_score(y, pred))
        print("roc_auc", roc_auc_score(y, proba))
        print("log_loss", log_loss(y, proba))
        print("brier", brier_score_loss(y, proba))

    path = settings.model_path("lightgbm_elo.joblib")
    joblib.dump(model, path)
    print("\nModel saved:", path)


if __name__ == "__main__":
    main()
