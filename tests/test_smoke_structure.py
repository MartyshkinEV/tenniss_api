import pytest
from importlib import import_module
from pathlib import Path


def test_sql_files_exist():
    assert Path("sql/init.sql").exists()
    assert Path("sql/player_match_stats.sql").exists()


def test_smoke_import_train_predict_modules():
    pytest.importorskip("numpy")

    import_module("src.training.train")
    import_module("src.inference.predict")
    import_module("src.features.feature_builder")
    import_module("src.models.logistic_model")
    import_module("src.models.lightgbm_model")
