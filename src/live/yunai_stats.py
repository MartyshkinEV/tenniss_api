from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from json import JSONDecoder
from typing import Any
from urllib.request import Request, urlopen

from sqlalchemy import text

from src.db.engine import get_engine


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
)

YUNAI_MATCH_URL_TEMPLATE = "https://s5.sir.yunai.tech/fp/en/match/{event_id}"
FONBET_SPORTRADAR_URL_TEMPLATE = "https://fon.bet/specificApplication/sportRadarDesktop?id={event_id}"
_STAT_HINTS = (
    "stat",
    "point",
    "ace",
    "fault",
    "break",
    "serve",
    "return",
    "winner",
    "error",
    "form",
    "streak",
)
_PLAYER_NAME_KEYS = ("name", "playerName", "shortName")


class _ScriptExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_script = False
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "script":
            self._inside_script = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self._inside_script = False

    def handle_data(self, data: str) -> None:
        if self._inside_script and data.strip():
            self.scripts.append(data)


@dataclass(frozen=True)
class YunaiFetchResult:
    requested_url: str
    resolved_url: str
    event_id: int | None
    players: list[str]
    stats: dict[str, Any]
    normalized_player_stats: dict[str, dict[str, Any]]
    candidate_count: int
    status: str = "ok"
    message: str | None = None


def build_match_url(event_id: int) -> str:
    return YUNAI_MATCH_URL_TEMPLATE.format(event_id=event_id)


def build_fonbet_sportradar_url(event_id: int) -> str:
    return FONBET_SPORTRADAR_URL_TEMPLATE.format(event_id=event_id)


def fetch_match_stats(event_id: int, timeout: float = 15.0, user_agent: str = DEFAULT_USER_AGENT) -> YunaiFetchResult:
    return fetch_match_stats_from_url(build_match_url(event_id), timeout=timeout, user_agent=user_agent, event_id=event_id)


def fetch_fonbet_sportradar_stats(
    event_id: int,
    timeout: float = 15.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> YunaiFetchResult:
    return fetch_match_stats_from_url(
        build_fonbet_sportradar_url(event_id),
        timeout=timeout,
        user_agent=user_agent,
        event_id=event_id,
    )


def fetch_match_stats_from_url(
    url: str,
    timeout: float = 15.0,
    user_agent: str = DEFAULT_USER_AGENT,
    event_id: int | None = None,
) -> YunaiFetchResult:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        resolved_url = response.geturl()
        html = response.read().decode("utf-8", errors="replace")
    if _is_antibot_page(html):
        return YunaiFetchResult(
            requested_url=url,
            resolved_url=resolved_url,
            event_id=event_id,
            players=[],
            stats={},
            normalized_player_stats={},
            candidate_count=0,
            status="blocked_by_antibot",
            message="Source returned anti-bot protection page instead of match statistics.",
        )
    players, stats, candidate_count = extract_match_stats_from_html(html)
    return YunaiFetchResult(
        requested_url=url,
        resolved_url=resolved_url,
        event_id=event_id,
        players=players,
        stats=stats,
        normalized_player_stats=normalize_player_stat_lines(players, stats),
        candidate_count=candidate_count,
    )


def extract_match_stats_from_html(html: str) -> tuple[list[str], dict[str, Any], int]:
    parser = _ScriptExtractor()
    parser.feed(html)
    candidates: list[dict[str, Any]] = []
    for script in parser.scripts:
        for payload in _extract_json_objects(script):
            if isinstance(payload, dict):
                candidates.extend(_collect_stat_candidates(payload))
    if not candidates:
        return [], {}, 0
    best = max(candidates, key=_candidate_score)
    players = _extract_players(best)
    stats = _extract_stats(best)
    return players, stats, len(candidates)


def _extract_json_objects(text: str) -> list[Any]:
    decoder = JSONDecoder()
    compact = unescape(text)
    objects: list[Any] = []
    for match in re.finditer(r"[\{\[]", compact):
        start = match.start()
        try:
            payload, end = decoder.raw_decode(compact[start:])
        except json.JSONDecodeError:
            continue
        if end > 0:
            objects.append(payload)
    return objects


def _collect_stat_candidates(payload: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if _looks_like_stats_container(payload):
            candidates.append(payload)
        for value in payload.values():
            candidates.extend(_collect_stat_candidates(value))
    elif isinstance(payload, list):
        for value in payload:
            candidates.extend(_collect_stat_candidates(value))
    return candidates


def _looks_like_stats_container(payload: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    has_player_shape = any("player" in key for key in keys) or "players" in keys or "home" in keys or "away" in keys
    has_stat_shape = any(any(hint in key for hint in _STAT_HINTS) for key in keys)
    nested_stats = any(isinstance(value, dict) and _looks_like_stat_map(value) for value in payload.values())
    return has_player_shape and (has_stat_shape or nested_stats)


def _looks_like_stat_map(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    keys = {str(key).lower() for key in payload}
    if any(any(hint in key for hint in _STAT_HINTS) for key in keys):
        return True
    scalar_values = sum(1 for value in payload.values() if isinstance(value, (str, int, float, bool)) or value is None)
    return scalar_values >= max(2, len(payload) // 2)


def _candidate_score(payload: dict[str, Any]) -> int:
    score = len(_extract_players(payload)) * 5
    score += len(_extract_stats(payload))
    return score


def _extract_players(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for value in payload.values():
        if isinstance(value, dict):
            for key in _PLAYER_NAME_KEYS:
                name = value.get(key)
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for key in _PLAYER_NAME_KEYS:
                        name = item.get(key)
                        if isinstance(name, str) and name.strip():
                            names.append(name.strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _extract_stats(payload: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = str(key)
        if isinstance(value, dict) and _looks_like_stat_map(value):
            stats[normalized_key] = value
        elif any(hint in normalized_key.lower() for hint in _STAT_HINTS):
            stats[normalized_key] = value
    return stats


def _is_antibot_page(html: str) -> bool:
    lowered = html.lower()
    return "servicepipe.ru/static/checkjs" in lowered or "id_captcha_frame_div" in lowered or "get_cookie_spsn" in lowered


def normalize_player_stat_lines(players: list[str], stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not players:
        return {}
    lines = {player: {} for player in players[:2]}
    aliases = _player_aliases(players)
    for key, value in stats.items():
        if not isinstance(value, dict):
            continue
        normalized_key = str(key)
        mapped = _map_stat_sides(value, aliases)
        for player_name, player_value in mapped.items():
            if player_name in lines:
                lines[player_name][normalized_key] = player_value
    return {player: values for player, values in lines.items() if values}


def _player_aliases(players: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if players:
        aliases["home"] = players[0]
        aliases["player1"] = players[0]
        aliases["p1"] = players[0]
    if len(players) > 1:
        aliases["away"] = players[1]
        aliases["player2"] = players[1]
        aliases["p2"] = players[1]
    for player in players[:2]:
        aliases[player.lower()] = player
    return aliases


def _map_stat_sides(stat_payload: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in stat_payload.items():
        canonical_key = aliases.get(str(key).lower())
        if canonical_key is not None:
            mapped[canonical_key] = value
    return mapped


class DatabaseMatchStatsRecorder:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def ensure_table(self) -> None:
        statement = text(
            """
            CREATE TABLE IF NOT EXISTS live_match_player_stats (
                source TEXT NOT NULL,
                event_id TEXT NOT NULL,
                snapshot_utc TIMESTAMPTZ NOT NULL,
                requested_url TEXT,
                resolved_url TEXT,
                player1_name TEXT,
                player2_name TEXT,
                candidate_count INTEGER,
                status TEXT NOT NULL,
                message TEXT,
                stats_json JSONB NOT NULL,
                normalized_player_stats_json JSONB NOT NULL,
                PRIMARY KEY (source, event_id, snapshot_utc)
            );

            CREATE INDEX IF NOT EXISTS idx_live_match_player_stats_event_id
            ON live_match_player_stats(event_id);

            CREATE INDEX IF NOT EXISTS idx_live_match_player_stats_snapshot_utc
            ON live_match_player_stats(snapshot_utc DESC);

            ALTER TABLE live_match_player_stats
            ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ok';

            ALTER TABLE live_match_player_stats
            ADD COLUMN IF NOT EXISTS message TEXT;
            """
        )
        with self.engine.begin() as conn:
            conn.execute(statement)

    def write_result(self, source: str, result: YunaiFetchResult) -> int:
        self.ensure_table()
        statement = text(
            """
            INSERT INTO live_match_player_stats (
                source,
                event_id,
                snapshot_utc,
                requested_url,
                resolved_url,
                player1_name,
                player2_name,
                candidate_count,
                status,
                message,
                stats_json,
                normalized_player_stats_json
            ) VALUES (
                :source,
                :event_id,
                :snapshot_utc,
                :requested_url,
                :resolved_url,
                :player1_name,
                :player2_name,
                :candidate_count,
                :status,
                :message,
                CAST(:stats_json AS JSONB),
                CAST(:normalized_player_stats_json AS JSONB)
            )
            """
        )
        payload = {
            "source": source,
            "event_id": str(result.event_id or ""),
            "snapshot_utc": datetime.now(timezone.utc).isoformat(),
            "requested_url": result.requested_url,
            "resolved_url": result.resolved_url,
            "player1_name": result.players[0] if len(result.players) > 0 else None,
            "player2_name": result.players[1] if len(result.players) > 1 else None,
            "candidate_count": result.candidate_count,
            "status": result.status,
            "message": result.message,
            "stats_json": json.dumps(result.stats, ensure_ascii=True),
            "normalized_player_stats_json": json.dumps(result.normalized_player_stats, ensure_ascii=True),
        }
        with self.engine.begin() as conn:
            conn.execute(statement, payload)
        return 1
