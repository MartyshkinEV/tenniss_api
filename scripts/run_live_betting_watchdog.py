try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path("/opt/tennis_ai")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "live_betting"
PYTHON_BIN = PROJECT_ROOT / "venv" / "bin" / "python"


@dataclass(frozen=True)
class ManagedProcess:
    name: str
    script_path: Path
    pid_path: Path
    stdout_path: Path
    state_path: Path
    state_payload: dict

    @property
    def command(self) -> list[str]:
        return [str(PYTHON_BIN), str(self.script_path), "--send-bet"]

    @property
    def match_pattern(self) -> str:
        return str(self.script_path)


AUTO_LIVE = ManagedProcess(
    name="auto_live",
    script_path=PROJECT_ROOT / "scripts" / "run_auto_live_event_betting.py",
    pid_path=ARTIFACTS_DIR / "auto_live_event_betting_real.pid",
    stdout_path=ARTIFACTS_DIR / "auto_live_event_betting_real.out",
    state_path=ARTIFACTS_DIR / "auto_live_event_betting_state.json",
    state_payload={
        "placed_selection_keys": [],
        "active_bets": [],
        "event_cursor": 0,
        "recent_selection_keys": [],
    },
)

STATE_ONLY = ManagedProcess(
    name="state_only",
    script_path=PROJECT_ROOT / "scripts" / "run_state_only_live_batch.py",
    pid_path=ARTIFACTS_DIR / "state_only_live_batch.pid",
    stdout_path=ARTIFACTS_DIR / "state_only_live_batch.out",
    state_path=ARTIFACTS_DIR / "state_only_live_batch_state.json",
    state_payload={
        "recent": [],
        "blocked": [],
    },
)

MANAGED_PROCESSES = (AUTO_LIVE, STATE_ONLY)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _reset_state_files(processes: Iterable[ManagedProcess]) -> None:
    for process in processes:
        _save_json(process.state_path, process.state_payload)


def _read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _write_pid(pid_path: Path, pid: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")


def _pid_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _matching_pids(pattern: str, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid not in exclude:
            pids.append(pid)
    return pids


def _terminate_pid(pid: int, timeout_seconds: float) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _stop_processes(processes: Iterable[ManagedProcess], timeout_seconds: float) -> None:
    current_pid = os.getpid()
    seen: set[int] = set()
    for process in processes:
        pid = _read_pid(process.pid_path)
        if pid is not None and pid != current_pid:
            seen.add(pid)
        for matched_pid in _matching_pids(process.match_pattern, exclude={current_pid}):
            seen.add(matched_pid)
    for pid in sorted(seen):
        _terminate_pid(pid, timeout_seconds=timeout_seconds)


def _start_process(process: ManagedProcess) -> int:
    process.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = process.stdout_path.open("a", encoding="utf-8")
    child = subprocess.Popen(
        process.command,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    stdout_handle.close()
    _write_pid(process.pid_path, child.pid)
    return child.pid


def _restart_stack(
    processes: Iterable[ManagedProcess],
    *,
    reason: str,
    watchdog_log: Path,
    watchdog_state_path: Path,
    stop_timeout_seconds: float,
) -> dict:
    _append_jsonl(
        watchdog_log,
        {
            "timestamp_utc": _iso_now(),
            "action": "restart_started",
            "reason": reason,
        },
    )
    _stop_processes(processes, timeout_seconds=stop_timeout_seconds)
    _reset_state_files(processes)
    started: dict[str, int] = {}
    for process in processes:
        started[process.name] = _start_process(process)
    state = {
        "last_restart_utc": _iso_now(),
        "last_restart_reason": reason,
        "managed_pids": started,
    }
    _save_json(watchdog_state_path, state)
    _append_jsonl(
        watchdog_log,
        {
            "timestamp_utc": _iso_now(),
            "action": "restart_finished",
            "reason": reason,
            "managed_pids": started,
        },
    )
    return state


def _iter_recent_records(path: Path, since: datetime) -> Iterable[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp_raw = payload.get("timestamp_utc")
            if not timestamp_raw:
                continue
            try:
                timestamp = datetime.fromisoformat(str(timestamp_raw))
            except Exception:
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if timestamp >= since:
                records.append(payload)
    return records


def count_recent_placed_bets(journal_paths: Iterable[Path], *, window_minutes: float, now: datetime | None = None) -> int:
    now = now or _utc_now()
    since = now - timedelta(minutes=max(window_minutes, 0.1))
    total = 0
    for journal_path in journal_paths:
        for payload in _iter_recent_records(journal_path, since):
            if payload.get("status") == "placed":
                total += 1
    return total


def should_restart_for_low_bets(
    *,
    recent_bet_count: int,
    threshold: int,
    now: datetime,
    stack_started_at: datetime,
    grace_period_minutes: float,
    last_restart_at: datetime | None,
    cooldown_minutes: float,
) -> bool:
    if recent_bet_count > threshold:
        return False
    if now - stack_started_at < timedelta(minutes=max(grace_period_minutes, 0.0)):
        return False
    if last_restart_at is not None and now - last_restart_at < timedelta(minutes=max(cooldown_minutes, 0.0)):
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=30.0, help="Watchdog poll interval in seconds.")
    parser.add_argument("--bet-window-minutes", type=float, default=8.0, help="Window for recent placed-bet counting.")
    parser.add_argument("--min-recent-bets", type=int, default=2, help="Restart the stack if recent placed bets are at or below this number.")
    parser.add_argument("--startup-grace-minutes", type=float, default=6.0, help="Do not evaluate low-bet restarts immediately after a restart.")
    parser.add_argument("--restart-cooldown-minutes", type=float, default=4.0, help="Minimum time between full stack restarts.")
    parser.add_argument("--stop-timeout-seconds", type=float, default=10.0, help="How long to wait for child shutdown before SIGKILL.")
    parser.add_argument("--watchdog-log", default=str(ARTIFACTS_DIR / "live_betting_watchdog.jsonl"))
    parser.add_argument("--watchdog-state", default=str(ARTIFACTS_DIR / "live_betting_watchdog_state.json"))
    parser.add_argument("--auto-journal", default=str(ARTIFACTS_DIR / "auto_live_event_betting.jsonl"))
    parser.add_argument("--state-only-journal", default=str(ARTIFACTS_DIR / "state_only_live_batch.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    watchdog_log = Path(args.watchdog_log).resolve()
    watchdog_state_path = Path(args.watchdog_state).resolve()
    journal_paths = [Path(args.auto_journal).resolve(), Path(args.state_only_journal).resolve()]

    _append_jsonl(
        watchdog_log,
        {
            "timestamp_utc": _iso_now(),
            "action": "watchdog_started",
            "interval_seconds": float(args.interval),
            "bet_window_minutes": float(args.bet_window_minutes),
            "min_recent_bets": int(args.min_recent_bets),
            "startup_grace_minutes": float(args.startup_grace_minutes),
            "restart_cooldown_minutes": float(args.restart_cooldown_minutes),
        },
    )

    state = _restart_stack(
        MANAGED_PROCESSES,
        reason="watchdog_bootstrap",
        watchdog_log=watchdog_log,
        watchdog_state_path=watchdog_state_path,
        stop_timeout_seconds=float(args.stop_timeout_seconds),
    )
    stack_started_at = _utc_now()
    last_restart_at = _utc_now()

    while True:
        time.sleep(max(float(args.interval), 1.0))
        now = _utc_now()
        running = {}
        crashed: list[str] = []
        for process in MANAGED_PROCESSES:
            pid = _read_pid(process.pid_path)
            is_running = _pid_is_running(pid)
            running[process.name] = {"pid": pid, "running": is_running}
            if not is_running:
                crashed.append(process.name)

        recent_bet_count = count_recent_placed_bets(
            journal_paths,
            window_minutes=float(args.bet_window_minutes),
            now=now,
        )

        restart_reason: str | None = None
        if crashed:
            restart_reason = f"process_down:{','.join(crashed)}"
        elif should_restart_for_low_bets(
            recent_bet_count=recent_bet_count,
            threshold=int(args.min_recent_bets),
            now=now,
            stack_started_at=stack_started_at,
            grace_period_minutes=float(args.startup_grace_minutes),
            last_restart_at=last_restart_at,
            cooldown_minutes=float(args.restart_cooldown_minutes),
        ):
            restart_reason = f"low_recent_bets:{recent_bet_count}"

        _append_jsonl(
            watchdog_log,
            {
                "timestamp_utc": now.isoformat(),
                "action": "watchdog_heartbeat",
                "recent_placed_bets": recent_bet_count,
                "running": running,
                "restart_reason": restart_reason,
            },
        )

        if restart_reason is None:
            continue

        state = _restart_stack(
            MANAGED_PROCESSES,
            reason=restart_reason,
            watchdog_log=watchdog_log,
            watchdog_state_path=watchdog_state_path,
            stop_timeout_seconds=float(args.stop_timeout_seconds),
        )
        stack_started_at = _utc_now()
        last_restart_at = _utc_now()
        _save_json(watchdog_state_path, {**state, "stack_started_at_utc": stack_started_at.isoformat()})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
