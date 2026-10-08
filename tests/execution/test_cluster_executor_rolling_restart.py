"""A control-plane rolling restart under continuous claim load costs the
executor no error line: a claim that got no verdict is retried as a warning,
and only a second consecutive failure is an error.

Live 2026-10-08: during the api-ha roll of release cc58602a7540 one claim's
connection was closed without a response; the executor logged it with a
traceback and E2E-001's preflight read the ERROR line as a runtime error."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterExecutorError
from scripts.e2e.regional.workload_acceptance_checks import suspicious_log_lines
from tests.execution.test_cluster_executor_lease_and_report import (
    FakeExecutorClient,
    RecordingAdapter,
)
from tests.execution.test_cluster_executor_report_retry import build

NO_VERDICT = ClusterExecutorError(
    "regional control plane request failed: URLError: <urlopen error Remote end "
    "closed connection without response>"
)


class RollingControlPlane(FakeExecutorClient):
    """Every claim answers, except the ones that meet a Pod leaving the roll."""

    def __init__(self, failing_calls: set[int]) -> None:
        super().__init__()
        self.failing_calls = failing_calls
        self.calls = 0

    def claim(self, executor_id: str, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls in self.failing_calls:
            raise NO_VERDICT
        return super().claim(executor_id, **kwargs)


def _run_cycles(monkeypatch: pytest.MonkeyPatch, executor: Any, cycles: int) -> None:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= cycles:
            raise StopIteration

    monkeypatch.setattr("gpu_fault.cluster_executor.executor.time.sleep", sleep)
    with pytest.raises(StopIteration):
        executor.run()


def test_a_rolling_restart_under_claim_load_leaves_no_error_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Any
) -> None:
    client = RollingControlPlane(failing_calls={3, 7})
    executor = build(client, [RecordingAdapter()], tmp_path)

    with caplog.at_level(logging.DEBUG, logger="gpu_fault.cluster_executor"):
        _run_cycles(monkeypatch, executor, cycles=10)

    assert client.calls >= 9 and len(client.claims) >= 7, (
        "claiming continued through and after both blips"
    )
    records = [
        r for r in caplog.records if r.name.startswith("gpu_fault.cluster_executor")
    ]
    assert not [r for r in records if r.levelno >= logging.ERROR], (
        "a single no-verdict claim is retried, not reported as an executor error"
    )
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert len([m for m in warnings if "got no verdict" in m]) == 2, warnings
    assert all("consecutive=1" in m for m in warnings if "got no verdict" in m), (
        warnings
    )
    rendered = "\n".join(f"{r.levelname} {r.getMessage()}" for r in records)
    assert suspicious_log_lines(rendered) == [], (
        "the E2E-001 preflight's own marker pattern finds no runtime error"
    )


def test_two_consecutive_no_verdict_claims_are_still_an_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Any
) -> None:
    client = RollingControlPlane(failing_calls={2, 3})
    executor = build(client, [RecordingAdapter()], tmp_path)

    with caplog.at_level(logging.DEBUG, logger="gpu_fault.cluster_executor"):
        _run_cycles(monkeypatch, executor, cycles=5)

    errors = [
        r
        for r in caplog.records
        if r.name.startswith("gpu_fault.cluster_executor")
        and r.levelno >= logging.ERROR
    ]
    assert len(errors) == 1 and "claim failed" in errors[0].getMessage(), (
        "the outage that persists is reported fail-closed, as before"
    )
    assert "consecutive=2" in errors[0].getMessage(), errors[0].getMessage()


def test_a_claim_closed_without_response_is_repeated_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The exact failure every api-ha roll produced at the old Pods' SIGTERM:
    the peer closed the connection before any status line. The client repeats
    the claim once and the executor sees a normal empty claim."""
    from http.client import RemoteDisconnected
    from urllib.error import URLError

    from gpu_fault.cluster_executor import regional_client
    from tests.execution.test_cluster_executor_report_retry import (
        CLUSTER,
        CONTROL_PLANE,
        TOKEN,
        _CannedResponse,
    )

    attempts: list[float] = []
    pauses: list[float] = []

    def fake_urlopen(*_args: Any, **_kwargs: Any) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise URLError(
                RemoteDisconnected("Remote end closed connection without response")
            )
        return _CannedResponse(b'{"commands": []}')

    monkeypatch.setattr(regional_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(regional_client.time, "sleep", pauses.append)
    client = regional_client.RegionalExecutorClient(CONTROL_PLANE, CLUSTER, TOKEN)

    with caplog.at_level(logging.DEBUG, logger="gpu_fault.cluster_executor"):
        commands = client.claim(
            "executor-1", max_commands=1, lease_seconds=60, wait_seconds=30
        )

    assert commands == [] and len(attempts) == 2, "the claim was repeated exactly once"
    assert pauses == [regional_client.CLOSED_WITHOUT_RESPONSE_RETRY_SECONDS], pauses
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "claiming again once" in warnings[0], warnings
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "no error line"
    )

    attempts.clear()
    pauses.clear()

    def timing_out(*_args: Any, **_kwargs: Any) -> Any:
        attempts.append(1)
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(regional_client, "urlopen", timing_out)
    with pytest.raises(ClusterExecutorError, match="request failed"):
        client.claim("executor-1", max_commands=1, lease_seconds=60)
    assert len(attempts) == 1 and pauses == [], (
        "a timeout is not a closed connection: the executor loop decides the retry"
    )
