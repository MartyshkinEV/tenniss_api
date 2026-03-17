try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from src.live.runtime import settle_outcome_record


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _parse_timestamp(value: Any) -> datetime:
    raw = str(value or "")
    if not raw:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _snapshot_paths(live_dir: Path) -> list[Path]:
    candidates = sorted(
        {
            *live_dir.glob("*rl_snapshots.jsonl"),
            live_dir / "market_snapshots.jsonl",
            live_dir / "point_trajectories.jsonl",
        }
    )
    return [path for path in candidates if path.exists()]


def _snapshot_candidates(
    live_dir: Path,
    tracker_state: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_market_id: dict[str, list[dict[str, Any]]] = {}
    by_event_id: dict[str, list[dict[str, Any]]] = {}

    def add(snapshot: dict[str, Any], source: str) -> None:
        event_id = str(snapshot.get("event_id") or "")
        market_id = str(snapshot.get("market_id") or "")
        if not event_id and not market_id:
            return
        enriched = dict(snapshot)
        enriched["_backfill_source"] = source
        if market_id:
            by_market_id.setdefault(market_id, []).append(enriched)
        if event_id:
            by_event_id.setdefault(event_id, []).append(enriched)

    for event_id, snapshot in (tracker_state.get("market_last_snapshot") or {}).items():
        if isinstance(snapshot, dict):
            add(snapshot, "tracker_state")
            if "event_id" not in snapshot:
                snapshot = {**snapshot, "event_id": event_id}
                add(snapshot, "tracker_state")

    for path in _snapshot_paths(live_dir):
        for row in _load_jsonl(path):
            if not isinstance(row, dict):
                continue
            add(row, path.name)

    for rows in by_market_id.values():
        rows.sort(key=lambda item: _parse_timestamp(item.get("timestamp_utc")), reverse=True)
    for rows in by_event_id.values():
        rows.sort(key=lambda item: _parse_timestamp(item.get("timestamp_utc")), reverse=True)
    return by_market_id, by_event_id


def _candidate_final_snapshots(
    row: dict[str, Any],
    tracker_state: dict[str, Any],
    by_market_id: dict[str, list[dict[str, Any]]],
    by_event_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    final_snapshot = row.get("final_snapshot")
    if isinstance(final_snapshot, dict) and final_snapshot:
        snapshots.append({**final_snapshot, "_backfill_source": "outcome_row"})

    market_id = str(row.get("market_id") or "")
    event_id = str(row.get("event_id") or "")
    snapshots.extend(by_market_id.get(market_id, []))
    snapshots.extend(by_event_id.get(event_id, []))

    tracker_snapshot = (tracker_state.get("market_last_snapshot") or {}).get(event_id)
    if isinstance(tracker_snapshot, dict) and tracker_snapshot:
        snapshots.append({**tracker_snapshot, "_backfill_source": "tracker_state_direct"})

    snapshots.sort(key=lambda item: _parse_timestamp(item.get("timestamp_utc")), reverse=True)
    unique: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for snapshot in snapshots:
        key = (
            str(snapshot.get("timestamp_utc") or ""),
            str(snapshot.get("market_id") or ""),
            str(snapshot.get("_backfill_source") or ""),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(snapshot)
    return unique


def _maybe_settle_row(
    row: dict[str, Any],
    tracker_state: dict[str, Any],
    by_market_id: dict[str, list[dict[str, Any]]],
    by_event_id: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if row.get("reward") is not None:
        return row, None

    bankroll_before = float(row.get("bankroll_before") or settings.live_bankroll or 1000.0)
    for snapshot in _candidate_final_snapshots(row, tracker_state, by_market_id, by_event_id):
        settled = settle_outcome_record(row, snapshot, bankroll_before=bankroll_before)
        if settled.get("reward") is None:
            continue
        settled["backfilled_at_utc"] = datetime.now(timezone.utc).isoformat()
        settled["backfill_source"] = snapshot.get("_backfill_source")
        audit = {
            "event_id": row.get("event_id"),
            "market_id": row.get("market_id"),
            "selection_id": row.get("selection_id"),
            "market_type": row.get("market_type"),
            "old_status": row.get("status"),
            "new_status": settled.get("status"),
            "profit": settled.get("profit"),
            "reward": settled.get("reward"),
            "backfill_source": settled.get("backfill_source"),
        }
        return settled, audit
    return row, None


def _rewrite_tracker_state(
    tracker_path: Path,
    tracker_state: dict[str, Any],
    settled_selection_ids: set[str],
) -> int:
    pending_by_event = tracker_state.get("pending_by_event") or {}
    removed = 0
    new_pending: dict[str, list[dict[str, Any]]] = {}
    for event_id, records in pending_by_event.items():
        kept: list[dict[str, Any]] = []
        for record in records:
            if str(record.get("selection_id") or "") in settled_selection_ids:
                removed += 1
                continue
            kept.append(record)
        if kept:
            new_pending[str(event_id)] = kept
    tracker_state["pending_by_event"] = new_pending
    tracker_path.write_text(json.dumps(tracker_state, ensure_ascii=True, indent=2), encoding="utf-8")
    return removed


def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", default=str(settings.live_rl_outcomes_path))
    parser.add_argument("--tracker-state", default=str(settings.live_rl_tracker_state_path))
    parser.add_argument("--live-dir", default=str(settings.artifacts_dir / "live_betting"))
    parser.add_argument("--audit-log", default=str(settings.artifacts_dir / "live_betting" / "rl_backfill_audit.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outcomes_path = Path(args.outcomes)
    tracker_path = Path(args.tracker_state)
    live_dir = Path(args.live_dir)
    audit_log_path = Path(args.audit_log)

    tracker_state = _load_json(tracker_path)
    by_market_id, by_event_id = _snapshot_candidates(live_dir, tracker_state)
    rows = _load_jsonl(outcomes_path)

    updated_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    settled_selection_ids: set[str] = set()
    changed = 0
    for row in rows:
        updated, audit = _maybe_settle_row(row, tracker_state, by_market_id, by_event_id)
        updated_rows.append(updated)
        if audit is not None:
            changed += 1
            settled_selection_ids.add(str(updated.get("selection_id") or ""))
            audits.append(audit)

    backup_suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outcomes_backup = outcomes_path.with_name(f"{outcomes_path.name}.bak.{backup_suffix}")
    tracker_backup = tracker_path.with_name(f"{tracker_path.name}.bak.{backup_suffix}")

    tracker_removed = 0
    if not args.dry_run:
        if outcomes_path.exists():
            shutil.copy2(outcomes_path, outcomes_backup)
        _write_jsonl(outcomes_path, updated_rows)
        if tracker_path.exists():
            shutil.copy2(tracker_path, tracker_backup)
        tracker_removed = _rewrite_tracker_state(tracker_path, tracker_state, settled_selection_ids)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": "rl_outcomes_backfill",
        "dry_run": bool(args.dry_run),
        "outcomes_path": str(outcomes_path),
        "tracker_state_path": str(tracker_path),
        "rows_total": len(rows),
        "rows_backfilled": changed,
        "tracker_pending_removed": tracker_removed if not args.dry_run else len(settled_selection_ids),
        "snapshot_sources_market_ids": len(by_market_id),
        "snapshot_sources_event_ids": len(by_event_id),
        "outcomes_backup": str(outcomes_backup) if not args.dry_run else None,
        "tracker_backup": str(tracker_backup) if not args.dry_run else None,
        "examples": audits[:10],
    }
    _append_jsonl(audit_log_path, summary)
    print(json.dumps(summary, ensure_ascii=True))
    return str(audit_log_path)


if __name__ == "__main__":
    main()
