from __future__ import annotations


def build_model(**kwargs):
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:
        raise ImportError(
            "catboost is required for CatBoost models. Add it to the environment before training."
        ) from exc

    params = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
    }
    params.update(kwargs)
    return CatBoostClassifier(**params)
