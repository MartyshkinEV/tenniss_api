from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

DEFAULT_PRODUCTION_MODEL = "lightgbm_elo.joblib"

MODEL_SPECS = {
    "logreg_baseline.joblib": {
        "label": "logreg_baseline",
        "feature_set": "baseline",
        "valid_auc": 0.706988,
        "test_auc": 0.700249,
        "test_log_loss": 0.627697,
    },
    "lightgbm_baseline.joblib": {
        "label": "lightgbm_baseline",
        "feature_set": "baseline",
        "valid_auc": 0.733865,
        "test_auc": 0.728249,
        "test_log_loss": 0.603539,
    },
    "logreg_elo.joblib": {
        "label": "logreg_elo",
        "feature_set": "elo",
        "valid_auc": 0.737311,
        "test_auc": 0.733303,
        "test_log_loss": 0.602794,
    },
    "lightgbm_elo.joblib": {
        "label": "lightgbm_elo",
        "feature_set": "elo",
        "valid_auc": 0.747888,
        "test_auc": 0.743791,
        "test_log_loss": 0.591725,
    },
}


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


def _get_env_bool(name: str, default: bool, dotenv_values: Dict[str, str]) -> bool:
    raw = _get_env(name, "true" if default else "false", dotenv_values)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_float(name: str, default: float, dotenv_values: Dict[str, str]) -> float:
    return float(_get_env(name, str(default), dotenv_values))


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
    live_poll_interval_seconds: int
    live_edge_threshold: float
    live_min_model_probability: float
    live_min_odds: float
    live_max_odds: float
    live_default_stake: float
    live_bankroll: float
    live_kelly_fraction: float
    live_bet_mode: str
    live_express_size: int
    live_dry_run: bool
    live_state_path: Path
    live_decisions_path: Path
    live_market_snapshots_path: Path
    live_recommendations_path: Path
    market_bet_log_path: Path
    market_backtest_path: Path
    market_analytics_path: Path
    live_rl_snapshots_path: Path
    live_rl_actions_path: Path
    live_rl_outcomes_path: Path
    live_rl_tracker_state_path: Path
    live_point_trajectories_path: Path
    live_rl_market_close_cycles: int
    live_game_target_offset: int
    live_game_model_weight: float
    live_game_markov_weight: float
    live_point_target_offset: int
    live_point_model_weight: float
    live_point_markov_weight: float
    live_point_execution_min_probability: float
    odds_provider: str
    spoyer_base_url: str
    spoyer_token: str
    spoyer_login: str
    spoyer_task: str
    spoyer_sport: str
    spoyer_bookmaker: str
    fonbet_session_info_url: str
    fonbet_feed_url: str
    fonbet_coupon_info_url: str
    fonbet_bet_request_id_url: str
    fonbet_bet_url: str
    fonbet_bet_result_url: str
    fonbet_origin: str
    fonbet_referer: str
    fonbet_user_agent: str
    fonbet_cookie: str
    fonbet_timeout_seconds: float
    fonbet_auth_token: str
    fonbet_tennis_sport_id: str
    fonbet_lang: str
    fonbet_fsid: str
    fonbet_sys_id: int
    fonbet_client_id: int
    fonbet_cdi: int
    fonbet_device_id: str
    fonbet_scope_market_id: int
    fonbet_mirror: str
    fonbet_flex_bet: str
    fonbet_flex_param: bool

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
        live_poll_interval_seconds=int(_get_env("LIVE_POLL_INTERVAL_SECONDS", "20", dotenv_values)),
        live_edge_threshold=_get_env_float("LIVE_EDGE_THRESHOLD", 0.07, dotenv_values),
        live_min_model_probability=_get_env_float("LIVE_MIN_MODEL_PROBABILITY", 0.58, dotenv_values),
        live_min_odds=_get_env_float("LIVE_MIN_ODDS", 1.6, dotenv_values),
        live_max_odds=_get_env_float("LIVE_MAX_ODDS", 3.5, dotenv_values),
        live_default_stake=_get_env_float("LIVE_DEFAULT_STAKE", 30.0, dotenv_values),
        live_bankroll=_get_env_float("LIVE_BANKROLL", 0.0, dotenv_values),
        live_kelly_fraction=_get_env_float("LIVE_KELLY_FRACTION", 0.1, dotenv_values),
        live_bet_mode=_get_env("LIVE_BET_MODE", "single", dotenv_values).strip().lower(),
        live_express_size=max(2, int(_get_env("LIVE_EXPRESS_SIZE", "2", dotenv_values))),
        live_dry_run=_get_env_bool("LIVE_DRY_RUN", True, dotenv_values),
        live_state_path=Path(
            _get_env(
                "LIVE_STATE_PATH",
                str(artifacts_dir / "live_betting" / "state.json"),
                dotenv_values,
            )
        ).resolve(),
        live_decisions_path=Path(
            _get_env(
                "LIVE_DECISIONS_PATH",
                str(artifacts_dir / "live_betting" / "decisions.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        live_market_snapshots_path=Path(
            _get_env(
                "LIVE_MARKET_SNAPSHOTS_PATH",
                str(artifacts_dir / "live_betting" / "market_snapshots.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        live_recommendations_path=Path(
            _get_env(
                "LIVE_RECOMMENDATIONS_PATH",
                str(artifacts_dir / "live_betting" / "rl_recommendations.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        market_bet_log_path=Path(
            _get_env(
                "MARKET_BET_LOG_PATH",
                str(artifacts_dir / "betting" / "bet_log.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        market_backtest_path=Path(
            _get_env(
                "MARKET_BACKTEST_PATH",
                str(artifacts_dir / "betting" / "backtest.json"),
                dotenv_values,
            )
        ).resolve(),
        market_analytics_path=Path(
            _get_env(
                "MARKET_ANALYTICS_PATH",
                str(artifacts_dir / "betting" / "analytics.json"),
                dotenv_values,
            )
        ).resolve(),
        live_rl_snapshots_path=Path(
            _get_env(
                "LIVE_RL_SNAPSHOTS_PATH",
                str(artifacts_dir / "live_betting" / "rl_snapshots.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        live_rl_actions_path=Path(
            _get_env(
                "LIVE_RL_ACTIONS_PATH",
                str(artifacts_dir / "live_betting" / "rl_actions.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        live_rl_outcomes_path=Path(
            _get_env(
                "LIVE_RL_OUTCOMES_PATH",
                str(artifacts_dir / "live_betting" / "rl_outcomes.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        live_rl_tracker_state_path=Path(
            _get_env(
                "LIVE_RL_TRACKER_STATE_PATH",
                str(artifacts_dir / "live_betting" / "rl_tracker_state.json"),
                dotenv_values,
            )
        ).resolve(),
        live_point_trajectories_path=Path(
            _get_env(
                "LIVE_POINT_TRAJECTORIES_PATH",
                str(artifacts_dir / "live_betting" / "point_trajectories.jsonl"),
                dotenv_values,
            )
        ).resolve(),
        live_rl_market_close_cycles=int(_get_env("LIVE_RL_MARKET_CLOSE_CYCLES", "3", dotenv_values)),
        live_game_target_offset=int(_get_env("LIVE_GAME_TARGET_OFFSET", "2", dotenv_values)),
        live_game_model_weight=_get_env_float("LIVE_GAME_MODEL_WEIGHT", 0.60, dotenv_values),
        live_game_markov_weight=_get_env_float("LIVE_GAME_MARKOV_WEIGHT", 0.40, dotenv_values),
        live_point_target_offset=int(_get_env("LIVE_POINT_TARGET_OFFSET", "2", dotenv_values)),
        live_point_model_weight=_get_env_float("LIVE_POINT_MODEL_WEIGHT", 0.65, dotenv_values),
        live_point_markov_weight=_get_env_float("LIVE_POINT_MARKOV_WEIGHT", 0.35, dotenv_values),
        live_point_execution_min_probability=_get_env_float(
            "LIVE_POINT_EXECUTION_MIN_PROBABILITY", 0.40, dotenv_values
        ),
        odds_provider=_get_env("ODDS_PROVIDER", "spoyer", dotenv_values).strip().lower(),
        spoyer_base_url=_get_env("SPOYER_BASE_URL", "https://spoyer.com/api/en/get.php", dotenv_values),
        spoyer_token=_get_env("SPOYER_TOKEN", "", dotenv_values),
        spoyer_login=_get_env("SPOYER_LOGIN", "", dotenv_values),
        spoyer_task=_get_env("SPOYER_TASK", "fonlive", dotenv_values),
        spoyer_sport=_get_env("SPOYER_SPORT", "tennis", dotenv_values),
        spoyer_bookmaker=_get_env("SPOYER_BOOKMAKER", "fonbet", dotenv_values),
        fonbet_session_info_url=_get_env("FONBET_SESSION_INFO_URL", "", dotenv_values),
        fonbet_feed_url=_get_env("FONBET_FEED_URL", "", dotenv_values),
        fonbet_coupon_info_url=_get_env("FONBET_COUPON_INFO_URL", "", dotenv_values),
        fonbet_bet_request_id_url=_get_env("FONBET_BET_REQUEST_ID_URL", "", dotenv_values),
        fonbet_bet_url=_get_env("FONBET_BET_URL", "", dotenv_values),
        fonbet_bet_result_url=_get_env("FONBET_BET_RESULT_URL", "", dotenv_values),
        fonbet_origin=_get_env("FONBET_ORIGIN", "https://fon.bet", dotenv_values),
        fonbet_referer=_get_env("FONBET_REFERER", "https://fon.bet/", dotenv_values),
        fonbet_user_agent=_get_env(
            "FONBET_USER_AGENT",
            (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
            ),
            dotenv_values,
        ),
        fonbet_cookie=_get_env("FONBET_COOKIE", "", dotenv_values),
        fonbet_timeout_seconds=_get_env_float("FONBET_TIMEOUT_SECONDS", 10.0, dotenv_values),
        fonbet_auth_token=_get_env("FONBET_AUTH_TOKEN", "", dotenv_values),
        fonbet_tennis_sport_id=_get_env("FONBET_TENNIS_SPORT_ID", "", dotenv_values),
        fonbet_lang=_get_env("FONBET_LANG", "en", dotenv_values),
        fonbet_fsid=_get_env("FONBET_FSID", "", dotenv_values),
        fonbet_sys_id=int(_get_env("FONBET_SYS_ID", "21", dotenv_values)),
        fonbet_client_id=int(_get_env("FONBET_CLIENT_ID", "0", dotenv_values)),
        fonbet_cdi=int(_get_env("FONBET_CDI", "0", dotenv_values)),
        fonbet_device_id=_get_env("FONBET_DEVICE_ID", "", dotenv_values),
        fonbet_scope_market_id=int(_get_env("FONBET_SCOPE_MARKET_ID", "1600", dotenv_values)),
        fonbet_mirror=_get_env("FONBET_MIRROR", "https://fon.bet", dotenv_values),
        fonbet_flex_bet=_get_env("FONBET_FLEX_BET", "any", dotenv_values),
        fonbet_flex_param=_get_env_bool("FONBET_FLEX_PARAM", True, dotenv_values),
    )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.live_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.live_decisions_path.parent.mkdir(parents=True, exist_ok=True)
    settings.live_recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    settings.market_bet_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.market_backtest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.market_analytics_path.parent.mkdir(parents=True, exist_ok=True)
    settings.live_rl_snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    settings.live_rl_actions_path.parent.mkdir(parents=True, exist_ok=True)
    settings.live_rl_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    settings.live_rl_tracker_state_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


settings = load_settings()

_MATCH_FILE_RE = re.compile(r"^(atp|wta)_matches.*\.csv$", re.IGNORECASE)
_PLAYER_FILE_RE = re.compile(r"^[a-z0-9]+_players\.csv$", re.IGNORECASE)
_RANKING_FILE_RE = re.compile(r"^[a-z0-9]+_rankings.*\.csv$", re.IGNORECASE)
_PBP_FILE_RE = re.compile(r"^pbp_matches_.*\.csv$", re.IGNORECASE)
_IGNORED_MATCH_TOKENS = ("doubles", "mixed", "amateur", "failed")


def resolve_data_dir(must_exist: bool = True) -> Path:
    candidates = resolve_data_roots(must_exist=False)

    for candidate in candidates:
        if discover_match_csv_files([candidate], must_exist=False):
            return candidate

    if must_exist:
        raise FileNotFoundError("Tennis data directory not found. Checked: " + ", ".join(str(path) for path in candidates))
    return settings.data_dir


def resolve_data_roots(must_exist: bool = True) -> list[Path]:
    candidates = [
        settings.data_dir,
        settings.project_root / "tennis_atp",
    ]
    candidates.extend(
        path
        for path in sorted(settings.project_root.glob("tennis_*"))
        if path.is_dir()
    )

    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            ordered.append(resolved)

    if must_exist and not ordered:
        raise FileNotFoundError("No tennis data roots found")
    return ordered


def discover_match_csv_files(roots: list[Path] | None = None, must_exist: bool = True) -> list[Path]:
    roots = roots or resolve_data_roots(must_exist=must_exist)
    files = []
    for root in roots:
        for path in sorted(root.glob("*.csv")):
            name = path.name.lower()
            if not _MATCH_FILE_RE.match(path.name):
                continue
            if any(token in name for token in _IGNORED_MATCH_TOKENS):
                continue
            files.append(path.resolve())
    deduped = sorted(dict.fromkeys(files))
    if must_exist and not deduped:
        raise FileNotFoundError("No compatible tennis match CSV files found")
    return deduped


def discover_player_csv_files(roots: list[Path] | None = None, must_exist: bool = True) -> list[Path]:
    roots = roots or resolve_data_roots(must_exist=must_exist)
    files = [
        path.resolve()
        for root in roots
        for path in sorted(root.glob("*.csv"))
        if _PLAYER_FILE_RE.match(path.name)
    ]
    deduped = sorted(dict.fromkeys(files))
    if must_exist and not deduped:
        raise FileNotFoundError("No compatible tennis player CSV files found")
    return deduped


def discover_ranking_csv_files(roots: list[Path] | None = None, must_exist: bool = True) -> list[Path]:
    roots = roots or resolve_data_roots(must_exist=must_exist)
    files = [
        path.resolve()
        for root in roots
        for path in sorted(root.glob("*.csv"))
        if _RANKING_FILE_RE.match(path.name)
    ]
    deduped = sorted(dict.fromkeys(files))
    if must_exist and not deduped:
        raise FileNotFoundError("No compatible tennis ranking CSV files found")
    return deduped


def discover_pbp_csv_files(roots: list[Path] | None = None, must_exist: bool = True) -> list[Path]:
    roots = roots or resolve_data_roots(must_exist=False)
    candidate_roots = list(roots)
    candidate_roots.append((settings.project_root / "tennis_pointbypoint").resolve())
    files = [
        path.resolve()
        for root in candidate_roots
        if root.exists()
        for path in sorted(root.glob("*.csv"))
        if _PBP_FILE_RE.match(path.name)
    ]
    deduped = sorted(dict.fromkeys(files))
    if must_exist and not deduped:
        raise FileNotFoundError("No compatible tennis point-by-point CSV files found")
    return deduped


def player_id_namespace_offset(path: Path) -> int:
    prefix = path.name.split("_", 1)[0].lower()
    offsets = {
        "atp": 0,
        "wta": 1_000_000_000,
    }
    return offsets.get(prefix, 2_000_000_000)


def resolve_model_artifact_path(model_name: str, must_exist: bool = True) -> Path:
    primary = settings.model_path(model_name)
    legacy = settings.project_root / "models" / "match_winner" / model_name

    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    if must_exist:
        raise FileNotFoundError(f"Model not found: {primary} or legacy fallback {legacy}")
    return primary


def resolve_default_model_name() -> str:
    return DEFAULT_PRODUCTION_MODEL


def resolve_default_model_artifact_path(must_exist: bool = True) -> Path:
    model_name = resolve_default_model_name()
    legacy = settings.project_root / "models" / "match_winner" / model_name
    if legacy.exists():
        return legacy
    return resolve_model_artifact_path(model_name, must_exist=must_exist)


def get_model_spec(model_name: str) -> Dict[str, str | float]:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model spec for {model_name}")
    return MODEL_SPECS[model_name]
