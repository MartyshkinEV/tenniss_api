from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_live_betting_watchdog import count_recent_placed_bets, should_restart_for_low_bets


def test_count_recent_placed_bets_counts_only_recent_placed(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    now = datetime(2026, 3, 16, 18, 0, tzinfo=timezone.utc)
    recent = (now - timedelta(minutes=2)).isoformat()
    old = (now - timedelta(minutes=20)).isoformat()
    journal.write_text(
        "\n".join(
            [
                f'{{"timestamp_utc": "{recent}", "status": "placed"}}',
                f'{{"timestamp_utc": "{recent}", "status": "duplicate_skipped"}}',
                f'{{"timestamp_utc": "{old}", "status": "placed"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert count_recent_placed_bets([journal], window_minutes=5, now=now) == 1


def test_should_restart_for_low_bets_respects_grace_and_cooldown() -> None:
    now = datetime(2026, 3, 16, 18, 0, tzinfo=timezone.utc)
    stack_started_at = now - timedelta(minutes=10)
    last_restart_at = now - timedelta(minutes=10)

    assert should_restart_for_low_bets(
        recent_bet_count=2,
        threshold=2,
        now=now,
        stack_started_at=stack_started_at,
        grace_period_minutes=5,
        last_restart_at=last_restart_at,
        cooldown_minutes=4,
    )

    assert not should_restart_for_low_bets(
        recent_bet_count=2,
        threshold=2,
        now=now,
        stack_started_at=now - timedelta(minutes=2),
        grace_period_minutes=5,
        last_restart_at=last_restart_at,
        cooldown_minutes=4,
    )

    assert not should_restart_for_low_bets(
        recent_bet_count=2,
        threshold=2,
        now=now,
        stack_started_at=stack_started_at,
        grace_period_minutes=5,
        last_restart_at=now - timedelta(minutes=2),
        cooldown_minutes=4,
    )

    assert not should_restart_for_low_bets(
        recent_bet_count=3,
        threshold=2,
        now=now,
        stack_started_at=stack_started_at,
        grace_period_minutes=5,
        last_restart_at=last_restart_at,
        cooldown_minutes=4,
    )
