from pathlib import Path

from config import load_settings


def test_load_settings_has_required_paths():
    s = load_settings()
    assert s.project_root.exists()
    assert s.artifacts_dir.exists()
    assert s.models_dir.exists()
    assert isinstance(s.db_port, int)


def test_model_path_builder():
    s = load_settings()
    model_path = s.model_path("demo.joblib")
    assert isinstance(model_path, Path)
    assert model_path.name == "demo.joblib"
    assert model_path.parent == s.models_dir
