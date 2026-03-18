from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BettingAuditLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, payload: dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


class OddsHistoryRecorder:
    def __init__(self, engine: Any | None = None):
        self.engine = engine

    def write_many(self, records: list[dict[str, Any]]) -> int:
        if not records or self.engine is None:
            return 0
        from sqlalchemy import text

        statement = text(
            """
            INSERT INTO odds_history (
                match_id,
                event_id,
                market,
                market_id,
                selection,
                line_value,
                odds,
                bookmaker_prob,
                timestamp_utc,
                source,
                raw_json
            ) VALUES (
                :match_id,
                :event_id,
                :market,
                :market_id,
                :selection,
                :line_value,
                :odds,
                :bookmaker_prob,
                :timestamp_utc,
                :source,
                CAST(:raw_json AS JSONB)
            )
            """
        )
        prepared = []
        for record in records:
            prepared.append(
                {
                    **record,
                    "raw_json": json.dumps(record.get("raw_json") or {}, ensure_ascii=True),
                }
            )
        with self.engine.begin() as conn:
            conn.execute(statement, prepared)
        return len(prepared)


class DatabaseBetLogRecorder:
    def __init__(self, engine: Any | None = None):
        self.engine = engine

    def write(self, payload: dict[str, Any]) -> int:
        if self.engine is None:
            return 0
        from sqlalchemy import text

        statement = text(
            """
            INSERT INTO bet_log (
                settled_at,
                match_id,
                event_id,
                market,
                market_type,
                pick,
                odds,
                stake,
                result,
                profit,
                model_prob,
                bookmaker_prob,
                value,
                confidence,
                threshold_value,
                min_probability,
                data_quality_score,
                filter_surface,
                filter_tourney_level,
                filter_form_window,
                filter_passed,
                decision_reason,
                explanation_json,
                source_json
            ) VALUES (
                :settled_at,
                :match_id,
                :event_id,
                :market,
                :market_type,
                :pick,
                :odds,
                :stake,
                :result,
                :profit,
                :model_prob,
                :bookmaker_prob,
                :value,
                :confidence,
                :threshold_value,
                :min_probability,
                :data_quality_score,
                :filter_surface,
                :filter_tourney_level,
                :filter_form_window,
                :filter_passed,
                :decision_reason,
                CAST(:explanation_json AS JSONB),
                CAST(:source_json AS JSONB)
            )
            """
        )
        record = dict(payload)
        record["explanation_json"] = json.dumps(record.get("explanation_json") or {}, ensure_ascii=True)
        record["source_json"] = json.dumps(record.get("source_json") or {}, ensure_ascii=True)
        with self.engine.begin() as conn:
            conn.execute(statement, record)
        return 1
