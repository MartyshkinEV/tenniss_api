from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def _load_dotenv_file(dotenv_path: Path) -> Dict[str, str]:
    if not dotenv_path.exists():
        return {}

    loaded: Dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key.strip()] = value.strip().strip('"').strip("'")
    return loaded


def _get_env(name: str, default: str, dotenv_values: Dict[str, str]) -> str:
    if name in os.environ:
        return os.environ[name]
    if name in dotenv_values:
        return dotenv_values[name]
    return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    sql_dir: Path
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    log_level: str

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def model_path(self, model_name: str) -> Path:
        return self.models_dir / model_name


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parent
    dotenv_values = _load_dotenv_file(project_root / ".env")

    resolved_root = Path(_get_env("PROJECT_ROOT", str(project_root), dotenv_values)).resolve()
    artifacts_dir = Path(_get_env("ARTIFACTS_DIR", str(resolved_root / "artifacts"), dotenv_values)).resolve()
    models_dir = Path(_get_env("MODELS_DIR", str(artifacts_dir / "models"), dotenv_values)).resolve()
    data_dir = Path(_get_env("DATA_DIR", str(resolved_root / "data"), dotenv_values)).resolve()
    sql_dir = Path(_get_env("SQL_DIR", str(resolved_root / "sql"), dotenv_values)).resolve()

    settings = Settings(
        project_root=resolved_root,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        sql_dir=sql_dir,
        db_host=_get_env("DB_HOST", "localhost", dotenv_values),
        db_port=int(_get_env("DB_PORT", "5432", dotenv_values)),
        db_name=_get_env("DB_NAME", "tennis", dotenv_values),
        db_user=_get_env("DB_USER", "tennis_user", dotenv_values),
        db_password=_get_env("DB_PASSWORD", "tennis_pass", dotenv_values),
        log_level=_get_env("LOG_LEVEL", "INFO", dotenv_values),
    )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = load_settings()
