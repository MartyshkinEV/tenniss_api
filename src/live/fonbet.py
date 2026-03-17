from __future__ import annotations

import copy
import gzip
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from config import settings
from src.db.engine import get_engine
from src.live.runtime import LiveMarket, ScoredSelection


def _coerce_float(value: Any) -> float | None:
    if value in (None, "", 0):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_empty(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_name(value: Any) -> str | None:
    if isinstance(value, dict):
        return str(
            _first_non_empty(
                value,
                "name",
                "title",
                "team_name",
                "competitor_name",
            )
            or ""
        ).strip() or None
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _coerce_numeric_param(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return value


def _load_json_response(response: Any) -> Any:
    raw = response.read()
    encoding = str(response.headers.get("Content-Encoding", "")).lower()
    if encoding == "gzip" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _factor_value(factor: dict[str, Any]) -> float | None:
    return _coerce_float(_first_non_empty(factor, "v", "value", "price"))


def _factor_param(factor: dict[str, Any]) -> Any:
    return _first_non_empty(factor, "p", "param")


def _parse_score(score: str | None) -> tuple[int, int] | None:
    if not score or ":" not in score:
        return None
    try:
        left, right = score.split(":", 1)
        return int(left), int(right)
    except ValueError:
        return None


def _target_game_number(factor: dict[str, Any]) -> int | None:
    param = _factor_param(factor)
    if param not in (None, ""):
        try:
            return int(param) // 100
        except (TypeError, ValueError):
            return None
    pt = _first_non_empty(factor, "pt", "paramText")
    if pt not in (None, ""):
        try:
            return int(pt)
        except (TypeError, ValueError):
            return None
    return None


def _target_point_number(factor: dict[str, Any]) -> int | None:
    lo = _first_non_empty(factor, "lo")
    try:
        return int(lo)
    except (TypeError, ValueError):
        return None


def _parse_point_comment(comment: str | None) -> tuple[int | None, int | None]:
    if not comment:
        return None, None
    body = comment.strip().strip("()").replace("*", "")
    if "-" not in body:
        return None, None
    left, right = body.split("-", 1)
    score_map = {"00": 0, "0": 0, "15": 1, "30": 2, "40": 3, "A": 4}
    left = left.strip()
    right = right.strip().split()[0]
    return score_map.get(left), score_map.get(right)


def _extract_odds(item: dict[str, Any], side: str) -> float | None:
    odds_block = item.get("odds")
    if isinstance(odds_block, dict):
        value = _first_non_empty(
            odds_block,
            side,
            f"{side}_odds",
            "p1" if side == "player1" else "p2",
        )
        parsed = _coerce_float(value)
        if parsed:
            return parsed

    keys = (
        ("player1_odds", "p1_odds", "odds1", "factor1")
        if side == "player1"
        else ("player2_odds", "p2_odds", "odds2", "factor2")
    )
    for key in keys:
        parsed = _coerce_float(item.get(key))
        if parsed:
            return parsed
    if side == "player1":
        home = item.get("home_od")
        if isinstance(home, dict):
            parsed = _coerce_float(_first_non_empty(home, "value", "odds", "price"))
            if parsed:
                return parsed
    if side == "player2":
        away = item.get("away_od")
        if isinstance(away, dict):
            parsed = _coerce_float(_first_non_empty(away, "value", "odds", "price"))
            if parsed:
                return parsed
    return None


def _infer_surface(competition: str) -> str:
    label = competition.lower()
    if "clay" in label:
        return "Clay"
    if "grass" in label:
        return "Grass"
    if "carpet" in label:
        return "Carpet"
    return "Hard"


def _infer_tourney_level(competition: str) -> str:
    label = competition.lower()
    if "itf" in label:
        return "itf"
    if "challenger" in label:
        return "challenger"
    if "liga pro" in label:
        return "tour"
    if "atp" in label or "wta" in label:
        return "tour"
    return "tour"


def extract_markets_from_fonbet_catalog(payload: Any) -> list[LiveMarket]:
    if not isinstance(payload, dict):
        return []

    sports = payload.get("sports")
    events = payload.get("events")
    custom_factors = payload.get("customFactors")
    event_miscs = payload.get("eventMiscs")
    if not isinstance(sports, list) or not isinstance(events, list) or not isinstance(custom_factors, list):
        return []

    tennis_segments = {
        int(item["id"]): item
        for item in sports
        if isinstance(item, dict) and item.get("parentId") == 4
    }
    factors_by_event = {
        int(item["e"]): {
            int(factor["f"]): factor
            for factor in item.get("factors", [])
            if isinstance(factor, dict) and factor.get("f") not in (None, "")
        }
        for item in custom_factors
        if isinstance(item, dict) and item.get("e") not in (None, "")
    }
    misc_by_event = {
        int(item["id"]): item
        for item in event_miscs or []
        if isinstance(item, dict) and item.get("id") not in (None, "")
    }
    events_by_id = {
        int(item["id"]): item
        for item in events
        if isinstance(item, dict) and item.get("id") not in (None, "")
    }

    markets: list[LiveMarket] = []
    game_factor_pairs = ((1747, 1748), (1750, 1751), (1753, 1754))
    point_factor_pairs = ((2995, 2996), (2998, 2999), (3001, 3002), (3004, 3005), (3007, 3008), (3010, 3011))
    total_factor_pairs = ((1848, 1849),)
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("place") != "live":
            continue
        segment = tennis_segments.get(int(event.get("sportId", 0)))
        if not segment:
            continue

        event_id = int(event["id"])
        factors = factors_by_event.get(event_id, {})
        competition = str(segment.get("name") or "")
        parent_event = None
        if event.get("level") == 2 and event.get("parentId") not in (None, ""):
            parent_event = events_by_id.get(int(event["parentId"]))
        misc = misc_by_event.get(event_id, {})
        score = None
        if misc.get("score1") is not None and misc.get("score2") is not None:
            score = f"{misc['score1']}:{misc['score2']}"
        score_pair = _parse_score(score)
        current_games = sum(score_pair) if score_pair is not None else None

        common_raw = {
            "score": score,
            "scopeMarketId": str(settings.fonbet_scope_market_id),
            "liveDelay": misc.get("liveDelay"),
            "comment": misc.get("comment"),
            "serveT": _first_non_empty(misc, "serveT", "servingTeam"),
        }
        point_score_pair = _parse_point_comment(misc.get("comment"))
        current_point_index = None
        if point_score_pair[0] is not None and point_score_pair[1] is not None:
            current_point_index = point_score_pair[0] + point_score_pair[1] + 1

        player1_factor = factors.get(921)
        player2_factor = factors.get(923)
        if event.get("level") == 1 and player1_factor and player2_factor:
            markets.append(
                LiveMarket(
                    market_id=str(event_id),
                    event_id=str(event_id),
                    competition=competition,
                    surface=_infer_surface(competition),
                    round_name="R32",
                    best_of=3,
                    tourney_level=_infer_tourney_level(competition),
                    player1_name=str(event.get("team1") or ""),
                    player2_name=str(event.get("team2") or ""),
                    player1_odds=float(player1_factor["v"]),
                    player2_odds=float(player2_factor["v"]),
                    market_type="match_winner",
                    raw={
                        **common_raw,
                        "factor": 921,
                        "player1_factor_id": 921,
                        "player2_factor_id": 923,
                        "player1_value": float(player1_factor["v"]),
                        "player2_value": float(player2_factor["v"]),
                        "zone": "lv",
                    },
                )
            )

        if event.get("level") == 2 and parent_event:
            for player1_factor_id, player2_factor_id in game_factor_pairs:
                next_game_player1_factor = factors.get(player1_factor_id)
                next_game_player2_factor = factors.get(player2_factor_id)
                if not next_game_player1_factor or not next_game_player2_factor:
                    continue
                target_game_number = _target_game_number(next_game_player1_factor)
                if current_games is not None and target_game_number is not None:
                    # Keep the currently opening game plus later still-listed game markets,
                    # but avoid speculative future games before the first game has started.
                    minimum_open_game = 1 if current_games <= 0 else current_games + 1
                    if target_game_number < minimum_open_game:
                        continue
                    if current_games <= 0 and target_game_number > minimum_open_game:
                        continue
                markets.append(
                    LiveMarket(
                        market_id=f"{event_id}:{target_game_number or player1_factor_id}",
                        event_id=str(event_id),
                        competition=competition,
                        surface=_infer_surface(competition),
                        round_name=str(event.get("name") or "R32"),
                        best_of=3,
                        tourney_level=_infer_tourney_level(competition),
                        player1_name=str(parent_event.get("team1") or ""),
                        player2_name=str(parent_event.get("team2") or ""),
                        player1_odds=float(next_game_player1_factor["v"]),
                        player2_odds=float(next_game_player2_factor["v"]),
                        market_type="next_game_winner",
                        raw={
                            **common_raw,
                            "player1_factor_id": player1_factor_id,
                            "player2_factor_id": player2_factor_id,
                            "player1_value": float(next_game_player1_factor["v"]),
                            "player2_value": float(next_game_player2_factor["v"]),
                            "player1_param": _factor_param(next_game_player1_factor),
                            "player2_param": _factor_param(next_game_player2_factor),
                            "param": _factor_param(next_game_player1_factor),
                            "target_game_number": target_game_number,
                            "zone": "es",
                        },
                    )
                )
            for player1_factor_id, player2_factor_id in point_factor_pairs:
                point_player1_factor = factors.get(player1_factor_id)
                point_player2_factor = factors.get(player2_factor_id)
                if not point_player1_factor or not point_player2_factor:
                    continue
                target_point_number = _target_point_number(point_player1_factor)
                if current_point_index is not None and target_point_number is not None:
                    if target_point_number != current_point_index + settings.live_point_target_offset:
                        continue
                markets.append(
                    LiveMarket(
                        market_id=f"{event_id}:point:{target_point_number or player1_factor_id}",
                        event_id=str(event_id),
                        competition=competition,
                        surface=_infer_surface(competition),
                        round_name=str(event.get("name") or "R32"),
                        best_of=3,
                        tourney_level=_infer_tourney_level(competition),
                        player1_name=str(parent_event.get("team1") or ""),
                        player2_name=str(parent_event.get("team2") or ""),
                        player1_odds=float(point_player1_factor["v"]),
                        player2_odds=float(point_player2_factor["v"]),
                        market_type="point_plus_one_winner",
                        raw={
                            **common_raw,
                            "player1_factor_id": player1_factor_id,
                            "player2_factor_id": player2_factor_id,
                            "player1_value": float(point_player1_factor["v"]),
                            "player2_value": float(point_player2_factor["v"]),
                            "player1_param": _factor_param(point_player1_factor),
                            "player2_param": _factor_param(point_player2_factor),
                            "param": _factor_param(point_player1_factor),
                            "target_point_number": target_point_number,
                            "zone": "es",
                        },
                    )
                )
            for over_factor_id, under_factor_id in total_factor_pairs:
                over_factor = factors.get(over_factor_id)
                under_factor = factors.get(under_factor_id)
                if not over_factor or not under_factor:
                    continue
                total_line = _factor_param(over_factor)
                total_line_value = None
                if total_line not in (None, ""):
                    try:
                        total_line_value = float(total_line) / 100.0
                    except (TypeError, ValueError):
                        total_line_value = None
                markets.append(
                    LiveMarket(
                        market_id=f"{event_id}:total:{total_line or over_factor_id}",
                        event_id=str(event_id),
                        competition=competition,
                        surface=_infer_surface(competition),
                        round_name=str(event.get("name") or "R32"),
                        best_of=3,
                        tourney_level=_infer_tourney_level(competition),
                        player1_name=str(parent_event.get("team1") or ""),
                        player2_name=str(parent_event.get("team2") or ""),
                        player1_odds=float(over_factor["v"]),
                        player2_odds=float(under_factor["v"]),
                        market_type="set_total_over_under",
                        raw={
                            **common_raw,
                            "player1_factor_id": over_factor_id,
                            "player2_factor_id": under_factor_id,
                            "player1_value": float(over_factor["v"]),
                            "player2_value": float(under_factor["v"]),
                            "player1_param": _factor_param(over_factor),
                            "player2_param": _factor_param(under_factor),
                            "param": _factor_param(over_factor),
                            "total_line": total_line_value,
                            "total_label_over": "over",
                            "total_label_under": "under",
                            "zone": "lv",
                        },
                    )
                )
    return markets


def extract_markets_from_payload(payload: Any, tennis_sport_id: str = "") -> list[LiveMarket]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("events") or payload.get("data") or payload.get("result") or []
    else:
        items = []

    markets: list[LiveMarket] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        sport_name = str(
            _first_non_empty(item, "sport", "sportName", "sport_name", "kind")
            or ""
        ).lower()
        sport_id = str(_first_non_empty(item, "sportId", "sport_id", "skId") or "")
        if "tennis" not in sport_name and tennis_sport_id and sport_id != tennis_sport_id:
            continue
        if "tennis" not in sport_name and not tennis_sport_id:
            continue

        player1_name = _first_non_empty(
            item,
            "player1_name",
            "player1",
            "team1",
            "competitor1",
            "home_name",
            "home",
        )
        player2_name = _first_non_empty(
            item,
            "player2_name",
            "player2",
            "team2",
            "competitor2",
            "away_name",
            "away",
        )
        player1_name = _extract_name(player1_name)
        player2_name = _extract_name(player2_name)
        if not player1_name or not player2_name:
            continue

        player1_odds = _extract_odds(item, "player1")
        player2_odds = _extract_odds(item, "player2")
        if not player1_odds or not player2_odds:
            continue

        market_id = str(_first_non_empty(item, "market_id", "eventId", "event_id", "id"))
        if not market_id:
            continue

        markets.append(
            LiveMarket(
                market_id=market_id,
                event_id=str(_first_non_empty(item, "event_id", "eventId", "id") or market_id),
                competition=str(_first_non_empty(item, "competition", "league", "tournament") or ""),
                surface=str(_first_non_empty(item, "surface", "court") or "Hard"),
                round_name=str(_first_non_empty(item, "round", "round_name", "stage") or "R32"),
                best_of=int(_first_non_empty(item, "best_of", "bestOf", "sets") or 3),
                tourney_level=str(_first_non_empty(item, "tourney_level", "level") or "tour"),
                player1_name=str(player1_name),
                player2_name=str(player2_name),
                player1_odds=float(player1_odds),
                player2_odds=float(player2_odds),
                raw=item,
            )
        )
    return markets


def extract_markets_from_spoyer_payload(payload: Any) -> list[LiveMarket]:
    if not isinstance(payload, dict):
        return []

    items = payload.get("games_pre") or payload.get("games_live") or payload.get("results") or payload.get("data") or []
    markets: list[LiveMarket] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        player1_name = _extract_name(_first_non_empty(item, "home", "player1", "team1"))
        player2_name = _extract_name(_first_non_empty(item, "away", "player2", "team2"))
        if not player1_name or not player2_name:
            continue

        player1_odds = _extract_odds(item, "player1")
        player2_odds = _extract_odds(item, "player2")
        if not player1_odds or not player2_odds:
            continue

        game_id = _first_non_empty(item, "game_id", "id")
        if game_id in (None, ""):
            continue

        league = item.get("league")
        if isinstance(league, dict):
            competition = str(_first_non_empty(league, "name", "title") or "")
        else:
            competition = str(league or "")

        markets.append(
            LiveMarket(
                market_id=str(game_id),
                event_id=str(game_id),
                competition=competition,
                surface=str(_first_non_empty(item, "surface", "court") or "Hard"),
                round_name=str(_first_non_empty(item, "round", "stage") or "R32"),
                best_of=int(_first_non_empty(item, "best_of", "bestOf", "sets") or 3),
                tourney_level=str(_first_non_empty(item, "tourney_level", "level") or "tour"),
                player1_name=player1_name,
                player2_name=player2_name,
                player1_odds=float(player1_odds),
                player2_odds=float(player2_odds),
                raw=item,
            )
        )
    return markets


def extract_fonbet_events(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    sports = payload.get("sports")
    events = payload.get("events")
    if not isinstance(sports, list) or not isinstance(events, list):
        return []

    sports_by_id = {
        int(item["id"]): item
        for item in sports
        if isinstance(item, dict) and item.get("id") not in (None, "")
    }

    def _root_sport(sport_id: int) -> dict[str, Any]:
        current = sports_by_id.get(sport_id, {})
        visited: set[int] = set()
        while isinstance(current, dict):
            parent_id = current.get("parentId")
            if parent_id in (None, ""):
                return current
            try:
                parent_key = int(parent_id)
            except (TypeError, ValueError):
                return current
            if parent_key in visited:
                return current
            visited.add(parent_key)
            parent = sports_by_id.get(parent_key)
            if not isinstance(parent, dict):
                return current
            current = parent
        return {}

    records: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        event_id = item.get("id")
        sport_id = item.get("sportId")
        if event_id in (None, "") or sport_id in (None, ""):
            continue

        try:
            event_id = int(event_id)
            sport_id = int(sport_id)
        except (TypeError, ValueError):
            continue

        sport = sports_by_id.get(sport_id, {})
        root_sport = _root_sport(sport_id)
        parent_id = item.get("parentId")
        try:
            parent_id = int(parent_id) if parent_id not in (None, "") else None
        except (TypeError, ValueError):
            parent_id = None

        records.append(
            {
                "event_id": event_id,
                "parent_id": parent_id,
                "sport_id": sport_id,
                "sport_name": str(sport.get("name") or ""),
                "sport_alias": str(sport.get("alias") or ""),
                "sport_kind": str(sport.get("kind") or ""),
                "root_sport_id": int(root_sport.get("id") or sport_id),
                "root_sport_name": str(root_sport.get("name") or sport.get("name") or ""),
                "event_name": str(item.get("name") or ""),
                "level": item.get("level"),
                "place": str(item.get("place") or ""),
                "team1": str(item.get("team1") or ""),
                "team2": str(item.get("team2") or ""),
                "priority": item.get("priority"),
                "sort_order": item.get("sortOrder"),
                "raw_event_json": item,
            }
        )
    return records


class FileMarketFeedClient:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch_live_markets(self) -> list[LiveMarket]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "customFactors" in payload and "events" in payload:
            return extract_markets_from_fonbet_catalog(payload)
        return extract_markets_from_payload(payload, tennis_sport_id=settings.fonbet_tennis_sport_id)


class EventMarketFeedClient:
    def __init__(self, event_id: int, client: "FonbetEventDetailsClient | None" = None, version: int = 0):
        self.event_id = int(event_id)
        self.client = client or FonbetEventDetailsClient()
        self.version = int(version)

    def fetch_live_markets(self) -> list[LiveMarket]:
        _url, payload, _headers = self.client.fetch_payload(self.event_id, self.version)
        return extract_markets_from_fonbet_catalog(payload)


class SnapshotMarketFeedClient:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.live_market_snapshots_path)

    def fetch_live_markets(self) -> list[LiveMarket]:
        if not self.path.exists():
            return []

        latest_by_market: dict[str, dict[str, Any]] = {}
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            market_id = str(payload.get("market_id") or "")
            event_id = str(payload.get("event_id") or "")
            if not market_id or not event_id:
                continue
            latest_by_market[market_id] = payload

        markets: list[LiveMarket] = []
        for payload in latest_by_market.values():
            try:
                player1_odds = float(payload["player1_odds"])
                player2_odds = float(payload["player2_odds"])
            except (KeyError, TypeError, ValueError):
                continue
            markets.append(
                LiveMarket(
                    market_id=str(payload["market_id"]),
                    event_id=str(payload["event_id"]),
                    competition=str(payload.get("competition") or ""),
                    surface=str(payload.get("surface") or "Hard"),
                    round_name=str(payload.get("round_name") or "R32"),
                    best_of=int(payload.get("best_of") or 3),
                    tourney_level=str(payload.get("tourney_level") or "tour"),
                    player1_name=str(payload.get("player1_name") or ""),
                    player2_name=str(payload.get("player2_name") or ""),
                    player1_odds=player1_odds,
                    player2_odds=player2_odds,
                    market_type=str(payload.get("market_type") or "match_winner"),
                    raw={
                        "score": payload.get("live_score"),
                        "comment": payload.get("live_comment"),
                        "liveDelay": payload.get("live_delay"),
                        "serveT": payload.get("serving_team"),
                        "player1_factor_id": payload.get("player1_factor_id"),
                        "player2_factor_id": payload.get("player2_factor_id"),
                        "player1_param": payload.get("player1_param"),
                        "player2_param": payload.get("player2_param"),
                        "scopeMarketId": payload.get("scope_market_id"),
                        "target_point_number": payload.get("target_point_number"),
                        "target_game_number": payload.get("target_game_number"),
                        "player1_value": payload.get("player1_odds"),
                        "player2_value": payload.get("player2_odds"),
                        "zone": payload.get("zone") or ("es" if payload.get("market_type") != "match_winner" else "lv"),
                    },
                )
            )
        return markets


class SnapshotRecorder:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.live_market_snapshots_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _record(self, market: LiveMarket) -> dict[str, Any]:
        raw = market.raw
        return {
            "timestamp_utc": raw.get("timestamp_utc"),
            "event_id": market.event_id,
            "market_id": market.market_id,
            "competition": market.competition,
            "surface": market.surface,
            "round_name": market.round_name,
            "best_of": market.best_of,
            "tourney_level": market.tourney_level,
            "player1_name": market.player1_name,
            "player2_name": market.player2_name,
            "player1_odds": market.player1_odds,
            "player2_odds": market.player2_odds,
            "market_type": market.market_type,
            "live_score": raw.get("score"),
            "live_comment": raw.get("comment"),
            "live_delay": raw.get("liveDelay"),
            "serving_team": raw.get("serveT"),
            "player1_factor_id": raw.get("player1_factor_id") or raw.get("factor"),
            "player2_factor_id": raw.get("player2_factor_id"),
            "player1_param": raw.get("player1_param") or raw.get("param"),
            "player2_param": raw.get("player2_param"),
            "scope_market_id": raw.get("scopeMarketId"),
            "target_game_number": raw.get("target_game_number"),
            "target_point_number": raw.get("target_point_number"),
            "zone": raw.get("zone"),
        }

    def write_markets(self, markets: list[LiveMarket]) -> int:
        latest_by_market = {market.market_id: self._record(market) for market in markets}
        with self.path.open("w", encoding="utf-8") as handle:
            for payload in latest_by_market.values():
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return len(latest_by_market)


class DatabaseSnapshotRecorder:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def _record(self, market: LiveMarket) -> dict[str, Any]:
        raw = market.raw
        return {
            "market_id": market.market_id,
            "timestamp_utc": raw.get("timestamp_utc"),
            "event_id": market.event_id,
            "competition": market.competition,
            "surface": market.surface,
            "round_name": market.round_name,
            "best_of": market.best_of,
            "tourney_level": market.tourney_level,
            "player1_name": market.player1_name,
            "player2_name": market.player2_name,
            "player1_odds": market.player1_odds,
            "player2_odds": market.player2_odds,
            "market_type": market.market_type,
            "live_score": raw.get("score"),
            "live_comment": raw.get("comment"),
            "live_delay": raw.get("liveDelay"),
            "serving_team": raw.get("serveT"),
            "player1_factor_id": raw.get("player1_factor_id") or raw.get("factor"),
            "player2_factor_id": raw.get("player2_factor_id"),
            "player1_param": raw.get("player1_param") or raw.get("param"),
            "player2_param": raw.get("player2_param"),
            "scope_market_id": raw.get("scopeMarketId"),
            "target_game_number": raw.get("target_game_number"),
            "target_point_number": raw.get("target_point_number"),
            "zone": raw.get("zone"),
        }

    def write_markets(self, markets: list[LiveMarket]) -> int:
        from sqlalchemy import text

        records = [self._record(market) for market in markets]
        if not records:
            return 0
        statement = text(
            """
            INSERT INTO live_market_snapshots (
                market_id, timestamp_utc, event_id, competition, surface, round_name, best_of, tourney_level,
                player1_name, player2_name, player1_odds, player2_odds, market_type, live_score, live_comment,
                live_delay, serving_team, player1_factor_id, player2_factor_id, player1_param, player2_param,
                scope_market_id, target_game_number, target_point_number, zone
            ) VALUES (
                :market_id, :timestamp_utc, :event_id, :competition, :surface, :round_name, :best_of, :tourney_level,
                :player1_name, :player2_name, :player1_odds, :player2_odds, :market_type, :live_score, :live_comment,
                :live_delay, :serving_team, :player1_factor_id, :player2_factor_id, :player1_param, :player2_param,
                :scope_market_id, :target_game_number, :target_point_number, :zone
            )
            ON CONFLICT (market_id) DO UPDATE SET
                timestamp_utc = EXCLUDED.timestamp_utc,
                event_id = EXCLUDED.event_id,
                competition = EXCLUDED.competition,
                surface = EXCLUDED.surface,
                round_name = EXCLUDED.round_name,
                best_of = EXCLUDED.best_of,
                tourney_level = EXCLUDED.tourney_level,
                player1_name = EXCLUDED.player1_name,
                player2_name = EXCLUDED.player2_name,
                player1_odds = EXCLUDED.player1_odds,
                player2_odds = EXCLUDED.player2_odds,
                market_type = EXCLUDED.market_type,
                live_score = EXCLUDED.live_score,
                live_comment = EXCLUDED.live_comment,
                live_delay = EXCLUDED.live_delay,
                serving_team = EXCLUDED.serving_team,
                player1_factor_id = EXCLUDED.player1_factor_id,
                player2_factor_id = EXCLUDED.player2_factor_id,
                player1_param = EXCLUDED.player1_param,
                player2_param = EXCLUDED.player2_param,
                scope_market_id = EXCLUDED.scope_market_id,
                target_game_number = EXCLUDED.target_game_number,
                target_point_number = EXCLUDED.target_point_number,
                zone = EXCLUDED.zone
            """
        )
        with self.engine.begin() as conn:
            conn.execute(statement, records)
        return len(records)


class DatabaseFonbetEventRecorder:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def write_payload(
        self,
        payload: dict[str, Any],
        requested_url: str,
        response_headers: dict[str, Any] | None = None,
        snapshot_utc: str | None = None,
    ) -> tuple[int, int]:
        from sqlalchemy import text

        events = extract_fonbet_events(payload)
        snapshot_record = {
            "snapshot_utc": snapshot_utc,
            "requested_url": requested_url,
            "response_headers_json": json.dumps(response_headers or {}, ensure_ascii=True),
            "payload_json": json.dumps(payload, ensure_ascii=True),
            "sports_count": len(payload.get("sports") or []) if isinstance(payload.get("sports"), list) else 0,
            "events_count": len(events),
        }
        snapshot_statement = text(
            """
            INSERT INTO fonbet_event_snapshots (
                snapshot_utc,
                requested_url,
                response_headers_json,
                payload_json,
                sports_count,
                events_count
            ) VALUES (
                COALESCE(CAST(:snapshot_utc AS TIMESTAMPTZ), CURRENT_TIMESTAMP),
                :requested_url,
                CAST(:response_headers_json AS JSONB),
                CAST(:payload_json AS JSONB),
                :sports_count,
                :events_count
            )
            RETURNING snapshot_id, snapshot_utc
            """
        )
        event_statement = text(
            """
            INSERT INTO fonbet_events (
                snapshot_id,
                snapshot_utc,
                event_id,
                parent_id,
                sport_id,
                sport_name,
                sport_alias,
                sport_kind,
                root_sport_id,
                root_sport_name,
                event_name,
                level,
                place,
                team1,
                team2,
                priority,
                sort_order,
                raw_event_json
            ) VALUES (
                :snapshot_id,
                :snapshot_utc,
                :event_id,
                :parent_id,
                :sport_id,
                :sport_name,
                :sport_alias,
                :sport_kind,
                :root_sport_id,
                :root_sport_name,
                :event_name,
                :level,
                :place,
                :team1,
                :team2,
                :priority,
                :sort_order,
                CAST(:raw_event_json AS JSONB)
            )
            """
        )

        with self.engine.begin() as conn:
            snapshot_row = conn.execute(snapshot_statement, snapshot_record).mappings().one()
            snapshot_id = int(snapshot_row["snapshot_id"])
            actual_snapshot_utc = snapshot_row["snapshot_utc"]
            if events:
                conn.execute(
                    event_statement,
                    [
                        {
                            **event,
                            "snapshot_id": snapshot_id,
                            "snapshot_utc": actual_snapshot_utc,
                            "raw_event_json": json.dumps(event["raw_event_json"], ensure_ascii=True),
                        }
                        for event in events
                    ],
                )
        return snapshot_id, len(events)


class DatabaseFonbetEventFeedClient:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def fetch_live_markets(self) -> list[LiveMarket]:
        from sqlalchemy import text

        query = text(
            """
            SELECT payload_json
            FROM fonbet_event_snapshots
            ORDER BY snapshot_utc DESC, snapshot_id DESC
            LIMIT 1
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().first()
        if not row:
            return []

        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict) and "customFactors" in payload and "events" in payload:
            return extract_markets_from_fonbet_catalog(payload)
        return []


class DatabaseMarketFeedClient:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()

    def fetch_live_markets(self) -> list[LiveMarket]:
        from sqlalchemy import text

        query = text(
            """
            SELECT
                market_id,
                event_id,
                competition,
                surface,
                round_name,
                best_of,
                tourney_level,
                player1_name,
                player2_name,
                player1_odds,
                player2_odds,
                market_type,
                live_score,
                live_comment,
                live_delay,
                serving_team,
                player1_factor_id,
                player2_factor_id,
                player1_param,
                player2_param,
                scope_market_id,
                target_game_number,
                target_point_number,
                zone,
                timestamp_utc
            FROM live_market_snapshots
            ORDER BY timestamp_utc DESC NULLS LAST, market_id
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        markets: list[LiveMarket] = []
        for row in rows:
            player1_odds = row["player1_odds"]
            player2_odds = row["player2_odds"]
            if player1_odds is None or player2_odds is None:
                continue
            markets.append(
                LiveMarket(
                    market_id=str(row["market_id"]),
                    event_id=str(row["event_id"]),
                    competition=str(row["competition"] or ""),
                    surface=str(row["surface"] or "Hard"),
                    round_name=str(row["round_name"] or "R32"),
                    best_of=int(row["best_of"] or 3),
                    tourney_level=str(row["tourney_level"] or "tour"),
                    player1_name=str(row["player1_name"] or ""),
                    player2_name=str(row["player2_name"] or ""),
                    player1_odds=float(player1_odds),
                    player2_odds=float(player2_odds),
                    market_type=str(row["market_type"] or "match_winner"),
                    raw={
                        "timestamp_utc": row["timestamp_utc"].isoformat() if row["timestamp_utc"] is not None else None,
                        "score": row["live_score"],
                        "comment": row["live_comment"],
                        "liveDelay": row["live_delay"],
                        "serveT": row["serving_team"],
                        "player1_factor_id": row["player1_factor_id"],
                        "player2_factor_id": row["player2_factor_id"],
                        "player1_param": row["player1_param"],
                        "player2_param": row["player2_param"],
                        "scopeMarketId": row["scope_market_id"],
                        "target_game_number": row["target_game_number"],
                        "target_point_number": row["target_point_number"],
                        "player1_value": row["player1_odds"],
                        "player2_value": row["player2_odds"],
                        "zone": row["zone"] or ("es" if row["market_type"] != "match_winner" else "lv"),
                    },
                )
            )
        return markets


class FonbetFeedClient:
    def __init__(self, feed_url: str | None = None, timeout_seconds: float | None = None):
        self.feed_url = feed_url or settings.fonbet_feed_url
        self.timeout_seconds = timeout_seconds or settings.fonbet_timeout_seconds

    def fetch_live_markets(self) -> list[LiveMarket]:
        if not self.feed_url:
            raise ValueError("FONBET_FEED_URL is not configured")

        for url in self._candidate_urls():
            request = Request(url, headers=self._headers())
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = _load_json_response(response)
            if isinstance(payload, dict) and "customFactors" in payload and "events" in payload:
                markets = extract_markets_from_fonbet_catalog(payload)
            else:
                markets = extract_markets_from_payload(payload, tennis_sport_id=settings.fonbet_tennis_sport_id)
            if markets:
                return markets
        return []

    def _candidate_urls(self) -> list[str]:
        urls: list[str] = []
        primary = self.feed_url
        if settings.fonbet_tennis_sport_id:
            separator = "&" if "?" in primary else "?"
            primary = f"{primary}{separator}{urlencode({'sportId': settings.fonbet_tennis_sport_id})}"
        urls.append(primary)

        parsed = urlparse(self.feed_url)
        if "/ma/line/liveEvents" in parsed.path:
            fallback = f"{parsed.scheme}://{parsed.netloc}/ma/events/list?lang={settings.fonbet_lang}&scopeMarket={settings.fonbet_scope_market_id}"
            if fallback not in urls:
                urls.append(fallback)
        return urls

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "*/*",
            "Accept-Language": f"{settings.fonbet_lang},en;q=0.9",
            "Origin": settings.fonbet_origin,
            "Referer": settings.fonbet_referer,
            "User-Agent": settings.fonbet_user_agent,
        }
        if settings.fonbet_cookie:
            headers["Cookie"] = settings.fonbet_cookie
        return headers


class FonbetEventsClient:
    def __init__(self, feed_url: str | None = None, timeout_seconds: float | None = None):
        self.feed_url = feed_url or settings.fonbet_feed_url
        self.timeout_seconds = timeout_seconds or settings.fonbet_timeout_seconds

    def fetch_payload(self) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if not self.feed_url:
            raise ValueError("FONBET_FEED_URL is not configured")

        url = self._catalog_url()
        request = Request(url, headers=self._headers())
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = _load_json_response(response)
            headers = dict(response.headers.items())
        if not isinstance(payload, dict):
            raise RuntimeError("Fonbet events payload is not a JSON object")
        return url, payload, headers

    def _catalog_url(self) -> str:
        parsed = urlparse(self.feed_url)
        if parsed.scheme and parsed.netloc and "/ma/events/list" in parsed.path:
            return self.feed_url
        if parsed.scheme and parsed.netloc:
            return (
                f"{parsed.scheme}://{parsed.netloc}/ma/events/list"
                f"?lang={settings.fonbet_lang}&scopeMarket={settings.fonbet_scope_market_id}"
            )
        return self.feed_url

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "*/*",
            "Accept-Language": f"{settings.fonbet_lang},en;q=0.9",
            "Origin": settings.fonbet_origin,
            "Referer": settings.fonbet_referer,
            "User-Agent": settings.fonbet_user_agent,
        }
        if settings.fonbet_cookie:
            headers["Cookie"] = settings.fonbet_cookie
        return headers


class FonbetEventDetailsClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.base_url = base_url or settings.fonbet_feed_url
        self.timeout_seconds = timeout_seconds or settings.fonbet_timeout_seconds

    def fetch_payload(self, event_id: int, version: int = 0) -> tuple[str, dict[str, Any], dict[str, Any]]:
        url = self._event_url(event_id=event_id, version=version)
        request = Request(url, headers=self._headers())
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = _load_json_response(response)
            headers = dict(response.headers.items())
        if not isinstance(payload, dict):
            raise RuntimeError("Fonbet event payload is not a JSON object")
        return url, payload, headers

    def _event_url(self, event_id: int, version: int = 0) -> str:
        parsed = urlparse(self.base_url)
        if parsed.scheme and parsed.netloc:
            return (
                f"{parsed.scheme}://{parsed.netloc}/ma/events/event"
                f"?lang={settings.fonbet_lang}&version={version}"
                f"&eventId={int(event_id)}&scopeMarket={settings.fonbet_scope_market_id}"
            )
        return self.base_url

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "*/*",
            "Accept-Language": f"{settings.fonbet_lang},en;q=0.9",
            "Origin": settings.fonbet_origin,
            "Referer": settings.fonbet_referer,
            "User-Agent": settings.fonbet_user_agent,
        }
        if settings.fonbet_cookie:
            headers["Cookie"] = settings.fonbet_cookie
        return headers


class SpoyerFeedClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        login: str | None = None,
        task: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.base_url = base_url or settings.spoyer_base_url
        self.token = token if token is not None else settings.spoyer_token
        self.login = login if login is not None else settings.spoyer_login
        self.task = task or settings.spoyer_task
        self.timeout_seconds = timeout_seconds or settings.fonbet_timeout_seconds

    def fetch_live_markets(self) -> list[LiveMarket]:
        if not self.token:
            raise ValueError("SPOYER_TOKEN is not configured")

        query = {"task": self.task, "bookmaker": settings.spoyer_bookmaker}
        if self.login:
            query["login"] = self.login
        else:
            query["token"] = self.token

        url = f"{self.base_url}?{urlencode(query)}"
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = _load_json_response(response)
        return extract_markets_from_spoyer_payload(payload)


class FonbetApiClient:
    def __init__(
        self,
        session_info_url: str | None = None,
        coupon_info_url: str | None = None,
        bet_request_id_url: str | None = None,
        bet_url: str | None = None,
        bet_result_url: str | None = None,
        auth_token: str | None = None,
        cookie: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.session_info_url = session_info_url or settings.fonbet_session_info_url
        self.coupon_info_url = coupon_info_url or settings.fonbet_coupon_info_url
        self.bet_request_id_url = bet_request_id_url or settings.fonbet_bet_request_id_url
        self.bet_url = bet_url or settings.fonbet_bet_url
        self.bet_result_url = bet_result_url or settings.fonbet_bet_result_url
        self.auth_token = auth_token if auth_token is not None else settings.fonbet_auth_token
        self.cookie = cookie if cookie is not None else settings.fonbet_cookie
        self.timeout_seconds = timeout_seconds or settings.fonbet_timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "*/*",
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": settings.fonbet_origin,
            "Referer": settings.fonbet_referer,
            "User-Agent": settings.fonbet_user_agent,
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    def _post_json_text(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not url:
            raise ValueError("Fonbet endpoint URL is not configured")
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        request = Request(url, data=body, headers=self._headers(), method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fonbet request failed with HTTP {exc.code}: {response_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Fonbet request failed: {exc.reason}") from exc

    def bet_slip_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json_text(self.coupon_info_url, payload)

    def session_info(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json_text(self.session_info_url, payload)

    def bet_request_id(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json_text(self.bet_request_id_url, payload)

    def bet(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json_text(self.bet_url, payload)

    def bet_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json_text(self.bet_result_url, payload)


class FonbetBetExecutor:
    def __init__(
        self,
        bet_url: str | None = None,
        auth_token: str | None = None,
        timeout_seconds: float | None = None,
        dry_run: bool | None = None,
        bet_result_retries: int = 4,
        bet_result_retry_delay_seconds: float = 1.0,
    ):
        self.bet_url = bet_url or settings.fonbet_bet_url
        self.auth_token = auth_token if auth_token is not None else settings.fonbet_auth_token
        self.timeout_seconds = timeout_seconds or settings.fonbet_timeout_seconds
        self.dry_run = settings.live_dry_run if dry_run is None else dry_run
        self.bet_result_retries = bet_result_retries
        self.bet_result_retry_delay_seconds = bet_result_retry_delay_seconds
        self.api_client = FonbetApiClient(
            bet_url=self.bet_url,
            auth_token=self.auth_token,
            timeout_seconds=self.timeout_seconds,
        )

    def _placement_status(self, bet_result_response: dict[str, Any]) -> str:
        if not isinstance(bet_result_response, dict):
            return "bet_result_unknown"
        if bet_result_response.get("result") != "couponResult":
            return "bet_result_pending"
        coupon = bet_result_response.get("coupon", {})
        result_code = coupon.get("resultCode")
        if result_code == 0:
            return "placed"
        if result_code == 2:
            return "odds_changed"
        if result_code == 100:
            return "temporarily_suspended"
        if result_code is None:
            return "bet_result_unknown"
        return f"rejected_{result_code}"

    def _next_request_id(self, candidate: ScoredSelection, payload_templates: dict[str, Any]) -> int:
        explicit = payload_templates.get("requestId")
        if explicit not in (None, ""):
            return int(explicit)
        raw_request_id = candidate.market.raw.get("requestId")
        if raw_request_id not in (None, ""):
            return int(raw_request_id)
        event_part = "".join(ch for ch in str(candidate.market.event_id) if ch.isdigit()) or "0"
        player_part = "".join(ch for ch in str(candidate.player_id) if ch.isdigit()) or "0"
        return int(f"{event_part}{player_part}"[-12:])

    def _session_payload_defaults(
        self,
        request_id: int | None = None,
        include_cdi: bool = True,
        include_device_id: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "lang": settings.fonbet_lang,
            "fsid": settings.fonbet_fsid or None,
            "sysId": settings.fonbet_sys_id,
            "clientId": settings.fonbet_client_id or None,
        }
        if include_cdi:
            payload["CDI"] = settings.fonbet_cdi
        if include_device_id:
            payload["deviceId"] = settings.fonbet_device_id or None
        if request_id is not None:
            payload["requestId"] = request_id
        return _compact_dict(payload)

    def _selection_payload_defaults(self, candidate: ScoredSelection, selection_number: int = 1) -> dict[str, Any]:
        raw = candidate.market.raw
        factor = self._selection_factor_id(candidate)
        if factor in (None, ""):
            raise ValueError(f"Missing Fonbet factor for market {candidate.market.market_id}")
        score = _first_non_empty(raw, "score", "scoreString", "score_value")
        value = self._selection_value(candidate)
        zone = str(_first_non_empty(raw, "zone", "eventZone") or "lv")
        return _compact_dict(
            {
                "num": selection_number,
                "event": int(candidate.market.event_id),
                "factor": int(factor),
                "value": float(value),
                "param": self._selection_param(candidate),
                "score": score,
                "zone": zone,
            }
        )

    def _bet_slip_info_selection_payload(self, candidate: ScoredSelection) -> dict[str, Any]:
        factor = self._selection_factor_id(candidate)
        if factor in (None, ""):
            raise ValueError(f"Missing Fonbet factor for market {candidate.market.market_id}")
        payload = {
            "eventId": int(candidate.market.event_id),
            "factorId": int(factor),
            "old": True,
        }
        param = self._selection_param(candidate)
        if param not in (None, ""):
            payload["param"] = param
        return payload

    def _selection_factor_id(self, candidate: ScoredSelection) -> Any:
        raw = candidate.market.raw
        if candidate.side == "player1":
            return _first_non_empty(raw, "player1_factor_id", "factor", "factorId", "bet_factor", "selectionFactor")
        return _first_non_empty(raw, "player2_factor_id", "opponent_factor_id", "factor2", "selectionFactor2", "factorId2")

    def _selection_value(self, candidate: ScoredSelection) -> float:
        raw = candidate.market.raw
        if candidate.side == "player1":
            value = _first_non_empty(raw, "player1_value", "value", "price", "currentOdd")
        else:
            value = _first_non_empty(raw, "player2_value", "opponent_value", "price2", "currentOdd2")
        if value in (None, ""):
            return float(candidate.odds)
        return float(value)

    def _selection_param(self, candidate: ScoredSelection) -> Any:
        raw = candidate.market.raw
        if candidate.side == "player1":
            return _coerce_numeric_param(_first_non_empty(raw, "player1_param", "param"))
        return _coerce_numeric_param(_first_non_empty(raw, "player2_param", "param2", "param"))

    def _coupon_defaults(self, candidate: ScoredSelection, selection_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "amount": round(candidate.stake, 2),
            "flexBet": settings.fonbet_flex_bet,
            "flexParam": settings.fonbet_flex_param,
            "mirror": settings.fonbet_mirror,
            "bets": [selection_payload],
        }

    def _coupon_defaults_many(self, stake: float, selections: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "amount": round(stake, 2),
            "flexBet": settings.fonbet_flex_bet,
            "flexParam": settings.fonbet_flex_param,
            "mirror": settings.fonbet_mirror,
            "bets": selections,
        }

    def _validate_configured_session(self) -> None:
        missing = []
        if not settings.fonbet_coupon_info_url:
            missing.append("FONBET_COUPON_INFO_URL")
        if not settings.fonbet_bet_request_id_url:
            missing.append("FONBET_BET_REQUEST_ID_URL")
        if not settings.fonbet_bet_url:
            missing.append("FONBET_BET_URL")
        if not settings.fonbet_bet_result_url:
            missing.append("FONBET_BET_RESULT_URL")
        if not settings.fonbet_fsid:
            missing.append("FONBET_FSID")
        if not settings.fonbet_client_id:
            missing.append("FONBET_CLIENT_ID")
        if not settings.fonbet_device_id:
            missing.append("FONBET_DEVICE_ID")
        if missing:
            raise ValueError("Missing required Fonbet session config: " + ", ".join(missing))

    def _build_request_payloads(self, candidate: ScoredSelection) -> dict[str, dict[str, Any]]:
        payload_templates = candidate.market.raw.get("fonbet_payloads", {})
        selection_payload = self._selection_payload_defaults(candidate)
        bet_slip_info_selection = self._bet_slip_info_selection_payload(candidate)
        coupon_payload = self._coupon_defaults(candidate, selection_payload)
        scope_market_id = _first_non_empty(candidate.market.raw, "scopeMarketId", "scope_market_id")
        request_id = self._next_request_id(candidate, payload_templates)

        bet_slip_info = _deep_merge(
            self._session_payload_defaults(),
            _deep_merge(
                payload_templates.get("bet_slip_info", {}),
                {
                    "bets": [bet_slip_info_selection],
                    "scopeMarketId": str(scope_market_id) if scope_market_id not in (None, "") else None,
                },
            ),
        )
        bet_request_id = _deep_merge(
            self._session_payload_defaults(),
            payload_templates.get("bet_request_id", {}),
        )
        bet = _deep_merge(
            self._session_payload_defaults(
                request_id=request_id,
                include_cdi=False,
                include_device_id=False,
            ),
            _deep_merge(
                payload_templates.get("bet", {}),
                {
                    "coupon": coupon_payload,
                },
            ),
        )
        bet_result = _deep_merge(
            self._session_payload_defaults(request_id=request_id),
            payload_templates.get("bet_result", {}),
        )
        return {
            "bet_slip_info": bet_slip_info,
            "bet_request_id": bet_request_id,
            "bet": bet,
            "bet_result": bet_result,
        }

    def _slip_selection(self, slip_info_response: dict[str, Any]) -> dict[str, Any] | None:
        bets = slip_info_response.get("bets")
        if not isinstance(bets, list) or not bets:
            return None
        slip_bet = bets[0]
        slip_event = slip_bet.get("event", {})
        slip_factor = slip_bet.get("factor", {})
        return {
            "score": slip_event.get("score"),
            "value": slip_factor.get("v"),
            "param": slip_factor.get("p"),
            "factor_id": slip_factor.get("id"),
        }

    def refresh_candidate(self, candidate: ScoredSelection) -> tuple[dict[str, Any], dict[str, Any]]:
        request_payloads = self._build_request_payloads(candidate)
        slip_info_response = self.api_client.bet_slip_info(request_payloads["bet_slip_info"])
        refreshed = self._slip_selection(slip_info_response) or {}
        return refreshed, slip_info_response

    def _apply_refresh_to_request_payloads(
        self,
        request_payloads: dict[str, dict[str, Any]],
        refreshed_selection: dict[str, Any],
    ) -> None:
        selection = request_payloads["bet"].get("coupon", {}).get("bets", [])
        if not selection:
            return
        bet_selection = selection[0]
        if refreshed_selection.get("score") not in (None, ""):
            bet_selection["score"] = refreshed_selection["score"]
        if refreshed_selection.get("value") not in (None, ""):
            bet_selection["value"] = refreshed_selection["value"]
        if refreshed_selection.get("param") not in (None, ""):
            bet_selection["param"] = refreshed_selection["param"]
        if refreshed_selection.get("factor_id") not in (None, ""):
            bet_selection["factor"] = refreshed_selection["factor_id"]

    def _poll_bet_result(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        for attempt_index in range(self.bet_result_retries + 1):
            response = self.api_client.bet_result(payload)
            attempts.append(response)
            if response.get("result") == "couponResult":
                return response, attempts
            if response.get("errorCode") != 200:
                return response, attempts
            if attempt_index < self.bet_result_retries:
                time.sleep(self.bet_result_retry_delay_seconds)
        return attempts[-1], attempts

    def place_bet(self, candidate: ScoredSelection) -> dict[str, Any]:
        payload = {
            "event_id": candidate.market.event_id,
            "market_id": candidate.market.market_id,
            "selection": candidate.side,
            "player_name": candidate.player_name,
            "odds": candidate.odds,
            "stake": candidate.stake,
            "model_probability": candidate.model_probability,
            "implied_probability": candidate.implied_probability,
            "edge": candidate.edge,
        }
        request_payloads = self._build_request_payloads(candidate)

        if self.dry_run:
            return {"status": "dry_run", "request": payload, "fonbet_requests": request_payloads}

        self._validate_configured_session()
        refreshed_selection, slip_info_response = self.refresh_candidate(candidate)
        self._apply_refresh_to_request_payloads(request_payloads, refreshed_selection)
        bet_request_id_response = self.api_client.bet_request_id(request_payloads["bet_request_id"])
        if bet_request_id_response.get("requestId") not in (None, ""):
            request_payloads["bet"]["requestId"] = bet_request_id_response["requestId"]
            request_payloads["bet_result"]["requestId"] = bet_request_id_response["requestId"]
        bet_response = self.api_client.bet(request_payloads["bet"])
        bet_result_response, bet_result_attempts = self._poll_bet_result(request_payloads["bet_result"])
        return {
            "status": self._placement_status(bet_result_response),
            "request": payload,
            "bet_slip_info_response": slip_info_response,
            "bet_request_id_response": bet_request_id_response,
            "bet_response": bet_response,
            "bet_result_response": bet_result_response,
            "bet_result_attempts": bet_result_attempts,
        }

    def place_prepared_bet(
        self,
        candidate: ScoredSelection,
        refreshed_selection: dict[str, Any],
        slip_info_response: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "event_id": candidate.market.event_id,
            "market_id": candidate.market.market_id,
            "selection": candidate.side,
            "player_name": candidate.player_name,
            "odds": candidate.odds,
            "stake": candidate.stake,
            "model_probability": candidate.model_probability,
            "implied_probability": candidate.implied_probability,
            "edge": candidate.edge,
        }
        request_payloads = self._build_request_payloads(candidate)
        self._apply_refresh_to_request_payloads(request_payloads, refreshed_selection)
        if self.dry_run:
            return {
                "status": "dry_run",
                "request": payload,
                "refreshed_selection": refreshed_selection,
                "bet_slip_info_response": slip_info_response,
                "fonbet_requests": request_payloads,
            }

        self._validate_configured_session()
        bet_request_id_response = self.api_client.bet_request_id(request_payloads["bet_request_id"])
        if bet_request_id_response.get("requestId") not in (None, ""):
            request_payloads["bet"]["requestId"] = bet_request_id_response["requestId"]
            request_payloads["bet_result"]["requestId"] = bet_request_id_response["requestId"]
        bet_response = self.api_client.bet(request_payloads["bet"])
        bet_result_response, bet_result_attempts = self._poll_bet_result(request_payloads["bet_result"])
        return {
            "status": self._placement_status(bet_result_response),
            "request": payload,
            "refreshed_selection": refreshed_selection,
            "bet_slip_info_response": slip_info_response,
            "bet_request_id_response": bet_request_id_response,
            "bet_response": bet_response,
            "bet_result_response": bet_result_response,
            "bet_result_attempts": bet_result_attempts,
        }

    def place_express_bet(self, candidates: list[ScoredSelection], stake: float | None = None) -> dict[str, Any]:
        if not candidates:
            raise ValueError("Express requires at least one selection")
        express_stake = float(stake if stake is not None else candidates[0].stake)
        payload_templates = {}
        selections = [
            self._selection_payload_defaults(candidate, selection_number=index + 1)
            for index, candidate in enumerate(candidates)
        ]
        slip_selections = [self._bet_slip_info_selection_payload(candidate) for candidate in candidates]
        request_id = self._next_request_id(candidates[0], payload_templates)
        scope_market_id = _first_non_empty(candidates[0].market.raw, "scopeMarketId", "scope_market_id")
        bet_slip_info = _deep_merge(
            self._session_payload_defaults(),
            {
                "bets": slip_selections,
                "scopeMarketId": str(scope_market_id) if scope_market_id not in (None, "") else None,
            },
        )
        bet_request_id = self._session_payload_defaults()
        bet = _deep_merge(
            self._session_payload_defaults(request_id=request_id),
            {"coupon": self._coupon_defaults_many(express_stake, selections)},
        )
        bet_result = self._session_payload_defaults(request_id=request_id)
        if self.dry_run:
            return {"status": "dry_run", "fonbet_requests": {"bet_slip_info": bet_slip_info, "bet_request_id": bet_request_id, "bet": bet, "bet_result": bet_result}}
        self._validate_configured_session()
        slip_info_response = self.api_client.bet_slip_info(bet_slip_info)
        bet_request_id_response = self.api_client.bet_request_id(bet_request_id)
        if bet_request_id_response.get("requestId") not in (None, ""):
            bet["requestId"] = bet_request_id_response["requestId"]
            bet_result["requestId"] = bet_request_id_response["requestId"]
        bet_response = self.api_client.bet(bet)
        bet_result_response, bet_result_attempts = self._poll_bet_result(bet_result)
        return {
            "status": self._placement_status(bet_result_response),
            "bet_slip_info_response": slip_info_response,
            "bet_request_id_response": bet_request_id_response,
            "bet_response": bet_response,
            "bet_result_response": bet_result_response,
            "bet_result_attempts": bet_result_attempts,
        }
