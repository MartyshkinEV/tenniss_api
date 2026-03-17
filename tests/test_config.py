import unittest
from pathlib import Path

from config import (
    DEFAULT_PRODUCTION_MODEL,
    load_settings,
    resolve_default_model_artifact_path,
    resolve_default_model_name,
)


class ConfigTest(unittest.TestCase):
    def test_load_settings_has_required_paths(self):
        s = load_settings()
        self.assertTrue(s.project_root.exists())
        self.assertTrue(s.artifacts_dir.exists())
        self.assertTrue(s.models_dir.exists())
        self.assertIsInstance(s.db_port, int)
        self.assertIsInstance(s.fonbet_sys_id, int)
        self.assertIsInstance(s.fonbet_client_id, int)
        self.assertIsInstance(s.fonbet_scope_market_id, int)

    def test_model_path_builder(self):
        s = load_settings()
        model_path = s.model_path("demo.joblib")
        self.assertIsInstance(model_path, Path)
        self.assertEqual(model_path.name, "demo.joblib")
        self.assertEqual(model_path.parent, s.models_dir)

    def test_default_model_resolution(self):
        self.assertEqual(resolve_default_model_name(), DEFAULT_PRODUCTION_MODEL)
        resolved = resolve_default_model_artifact_path()
        self.assertEqual(resolved.name, DEFAULT_PRODUCTION_MODEL)
        self.assertTrue(resolved.exists())


if __name__ == "__main__":
    unittest.main()
