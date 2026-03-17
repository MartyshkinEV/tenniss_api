try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import joblib
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import settings
from src.data import BASELINE_FEATURES, CATEGORICAL_FEATURES
from src.data.pipeline import load_match_features, split_dataset


def main():
    df = load_match_features()
    splits = split_dataset(df)
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]

    features = BASELINE_FEATURES
    target = "label"

    X_train, y_train = train_df[features], train_df[target]
    X_valid, y_valid = valid_df[features], valid_df[target]
    X_test, y_test = test_df[features], test_df[target]

    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    numeric = [c for c in features if c not in categorical]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
    ])

    print("Training LightGBM...")
    model.fit(X_train, y_train)

    for name, X, y in [("valid", X_valid, y_valid), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"\n{name.upper()} METRICS")
        print("accuracy =", round(accuracy_score(y, pred), 6))
        print("roc_auc  =", round(roc_auc_score(y, proba), 6))
        print("log_loss =", round(log_loss(y, proba), 6))
        print("brier    =", round(brier_score_loss(y, proba), 6))

    model_path = settings.model_path("lightgbm_baseline.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
