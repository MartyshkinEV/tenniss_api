from lightgbm import LGBMClassifier


def build_model(**kwargs):
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    params.update(kwargs)
    return LGBMClassifier(**params)
