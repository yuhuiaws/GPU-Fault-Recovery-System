"""The acceptance runner's executor log window refusal names the Pod and window.

Live (ISO-001 on a576e9a6af9c) the refusal read "executor log window is
empty" and nothing else; the operator had to find which executor Pod on which
cluster had been silent for how long. The check stays fail-closed -- an empty
window is missing evidence, not a vacuous proof -- and now says what to look at.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.workload_acceptance_checks import (
    EXECUTOR_LOG_MAX_WINDOW_SECONDS,
    executor_logs_since,
)

NOW = datetime(2026, 10, 5, 21, 0, tzinfo=timezone.utc)


class FakeRegional:
    def __init__(self, logs: dict[str, str]) -> None:
        self.logs = logs
        self.calls: list[tuple[str, ...]] = []

    def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
        assert (plane, app) == ("gpu", "gpu-fault-cluster-executor")
        return [{"name": name, "uid": f"uid-{name}"} for name in sorted(self.logs)]

    def kubectl(self, plane: str, *arguments: str, timeout: int) -> str:
        self.calls.append((plane, *arguments))
        return self.logs[arguments[1]]


def test_the_window_is_bounded_and_read_per_ready_pod() -> None:
    regional = FakeRegional(
        {"executor-a": "claimed command=x\n", "executor-b": "heartbeat\n"}
    )

    logs = executor_logs_since(
        regional, since=NOW - timedelta(seconds=300), now=lambda: NOW
    )

    assert logs == {"executor-a": "claimed command=x\n", "executor-b": "heartbeat\n"}
    assert regional.calls == [
        ("gpu", "logs", "executor-a", "--since=360s"),
        ("gpu", "logs", "executor-b", "--since=360s"),
    ], "elapsed plus a minute of slack, per Pod"
    far = executor_logs_since(regional, since=NOW - timedelta(days=2), now=lambda: NOW)
    assert far, "two days back still reads logs"
    assert regional.calls[-1][-1] == f"--since={EXECUTOR_LOG_MAX_WINDOW_SECONDS}s", (
        "the window is capped"
    )
    executor_logs_since(regional, since=NOW, now=lambda: NOW)
    assert regional.calls[-1][-1] == "--since=60s", "never shorter than a minute"


def test_an_empty_window_is_refused_naming_the_pod_and_the_window() -> None:
    regional = FakeRegional({"executor-a": "claimed command=x\n", "executor-b": "  \n"})

    with pytest.raises(RegionalFixtureError) as refusal:
        executor_logs_since(
            regional, since=NOW - timedelta(seconds=120), now=lambda: NOW
        )

    message = str(refusal.value)
    assert "Pod executor-b logged nothing in the last 180s" in message
    assert NOW.isoformat()[:16] in message or "since" in message
    assert "GPU_FAULT_LOG_LEVEL" in message
    assert "heartbeat" in message


def test_no_ready_executor_is_its_own_refusal() -> None:
    with pytest.raises(RegionalFixtureError, match="no Ready executor"):
        executor_logs_since(FakeRegional({}), since=NOW, now=lambda: NOW)
