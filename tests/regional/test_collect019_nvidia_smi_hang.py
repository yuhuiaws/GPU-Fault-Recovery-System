"""Contract tests for GF-REGIONAL-COLLECT-019 (nvidia-smi hang resilience)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from scripts.e2e.regional import collect019_verdicts as verdicts
from scripts.e2e.regional import run_collect019_nvidia_smi_hang as collect019
from scripts.e2e.regional.collector_window_fixture import open_window_or_rollback
from scripts.e2e.regional.regional_commands import RegionalFixtureError

CLUSTER = "cluster-a"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def _status(**overrides: Any) -> dict[str, Any]:
    status = {
        "collector": "HOST_TELEMETRY",
        "last_success_at": "2030-01-01T00:00:00+00:00",
        "last_error_at": "2030-01-01T00:03:00+00:00",
        "errors": [
            "nvidia-smi query timed out after 15s",
            "nvidia-smi circuit breaker open (3 consecutive timeouts)",
        ],
    }
    status.update(overrides)
    return status


def test_the_shipped_timing_keeps_the_erroring_window_inside_the_silence_threshold() -> (
    None
):
    assert verdicts.timing_errors(interval_seconds=15) == [], "shipped constants cohere"
    assert verdicts.HANG_SECONDS > verdicts.NVIDIA_SMI_TIMEOUT_SECONDS
    assert "does not exceed" in _text(
        verdicts.timing_errors(interval_seconds=15, hang_seconds=10)
    )
    assert "silence threshold" in _text(
        verdicts.timing_errors(interval_seconds=120, breaker_rounds=3)
    )


def test_the_erroring_status_contract_passes_on_timeout_then_breaker() -> None:
    assert verdicts.erroring_status_errors([_status()]) == [], (
        "timeout and breaker pass"
    )


def test_the_timeout_and_the_breaker_may_arrive_on_different_batches() -> None:
    """The breaker round replaces the timeout text; the runner remembers it.

    Live, the row read "timed out" for three rounds and then only "circuit
    breaker open"; requiring both on one row failed a hang that behaved
    exactly as specified.
    """

    seen: set[str] = set()
    first = [_status(errors=["_gpu_utilization: nvidia-smi timed out after 15s"])]
    assert "circuit breaker open" in _text(
        verdicts.erroring_status_errors(first, seen=seen)
    ), "the timeout alone is not yet the breaker"
    later = [
        _status(
            errors=[
                "_gpu_utilization: CollectorError: nvidia-smi circuit breaker open "
                "(3 consecutive timeouts); GPU queries skipped this round"
            ]
        )
    ]
    assert verdicts.erroring_status_errors(later, seen=seen) == [], (
        "a timeout read earlier and the breaker read now is the specified sequence"
    )
    assert any("timed out" in text for text in seen), "the runner keeps what it saw"


def test_the_erroring_status_contract_rejects_a_missing_breaker_or_timeout() -> None:
    assert "no nvidia-smi timeout" in _text(
        verdicts.erroring_status_errors(
            [
                _status(
                    errors=["nvidia-smi circuit breaker open (3 consecutive timeouts)"]
                )
            ]
        )
    )
    assert "circuit breaker open" in _text(
        verdicts.erroring_status_errors(
            [_status(errors=["nvidia-smi query timed out"])]
        )
    )
    assert "no HOST_TELEMETRY" in _text(
        verdicts.erroring_status_errors([{"collector": "GPU_METRICS"}])
    )


def test_the_service_contract_is_same_pid_no_restart() -> None:
    opened = {"ActiveState": "active", "NRestarts": "0", "MainPID": "4242"}
    assert verdicts.service_errors(opened, dict(opened)) == []
    assert "changed MainPID" in _text(
        verdicts.service_errors(opened, {**opened, "MainPID": "4300"})
    )
    assert "restarted during the hang" in _text(
        verdicts.service_errors(opened, {**opened, "NRestarts": "1"})
    )
    assert "not active" in _text(
        verdicts.service_errors(opened, {**opened, "ActiveState": "activating"})
    )


def _metrics(silent: int, erroring: int) -> str:
    return "\n".join(
        [
            f'gpu_fault_collector_silent_nodes{{cluster_id="{CLUSTER}",channel="HOST_TELEMETRY"}} {silent}',
            f'gpu_fault_collector_erroring_nodes{{cluster_id="{CLUSTER}",channel="HOST_TELEMETRY"}} {erroring}',
        ]
    )


def test_the_gauge_contract_wants_erroring_not_silence() -> None:
    assert (
        verdicts.gauge_errors([_metrics(0, 0)], [_metrics(0, 1)], cluster_id=CLUSTER)
        == []
    )
    assert "reported as silence" in _text(
        verdicts.gauge_errors([_metrics(0, 0)], [_metrics(1, 1)], cluster_id=CLUSTER)
    )
    assert "not at least 1" in _text(
        verdicts.gauge_errors([_metrics(0, 0)], [_metrics(0, 0)], cluster_id=CLUSTER)
    )


def test_the_window_recovery_and_close_contracts() -> None:
    opened = {
        "unit": verdicts.UNIT,
        "shadow": ["hang", "30", ""],
        "after": {"ActiveState": "active"},
        "deadman_timer": {"ActiveState": "active"},
    }
    assert verdicts.window_errors(opened) == []
    assert "window opened on" in _text(
        verdicts.window_errors({**opened, "unit": "gpu-fault-kernel-collector.service"})
    )
    assert "no active deadman" in _text(
        verdicts.window_errors({**opened, "deadman_timer": {}})
    )
    assert (
        verdicts.recovery_errors([_status(last_success_at="2030-01-01T00:10:00+00:00")])
        == []
    )
    assert "still erroring" in _text(verdicts.recovery_errors([_status()]))
    closed = {
        "dropin_removed": True,
        "window_root_removed": True,
        "after": {"ActiveState": "active"},
    }
    assert verdicts.closed_errors(closed) == []
    assert "drop-in survived" in _text(
        verdicts.closed_errors({**closed, "dropin_removed": False})
    )


def test_the_case_constants_and_plan_name_the_shadow(tmp_path: Any) -> None:
    assert collect019.CASE_ID == "GF-REGIONAL-COLLECT-019"
    assert collect019.CONFIRMATION == "COLLECT019_EXECUTE"
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-COLLECT-018"
    settings = type(
        "S",
        (),
        {"node": "node-a", "regional": type("R", (), {"cluster_id": CLUSTER})()},
    )()
    details = collect019.plan_details(settings, {"predecessor": {"valid": True}})
    assert details["risk"] == "live-non-destructive", details
    assert details["shadow"] == "hang:30" and details["unit"] == verdicts.UNIT, details
    assert "GPUs are untouched" in details["mutation"], details["mutation"]
    assert (
        details["rollback"]["window_deadman_seconds"] == verdicts.WINDOW_RESTORE_SECONDS
    )


class _HalfOpenProbe:
    """open-window that fails after writing the window, as the live probe did."""

    def __init__(self, *, close_fails: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.close_fails = close_fails

    def execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        self.calls.append(arguments)
        if arguments[0] == "open-window":
            raise RuntimeError("systemctl restart failed: control process exited")
        if self.close_fails:
            raise RuntimeError("no collector window is open for this run")
        return {"closed": True}


def test_a_failed_open_closes_the_half_open_window_and_reports_the_open_error() -> None:
    """The unit must not be left restart-looping behind a raised open."""

    probe = _HalfOpenProbe()
    try:
        open_window_or_rollback(probe, "c019-a1-1", "--unit", verdicts.UNIT)
    except RuntimeError as exc:
        assert "systemctl restart failed" in str(exc), "the open error is reported"
    else:
        raise AssertionError("a failed open must raise")
    assert [call[0] for call in probe.calls] == ["open-window", "close-window"], (
        probe.calls
    )
    assert probe.calls[1][1:3] == ("--run-id", "c019-a1-1"), probe.calls[1]

    stubborn = _HalfOpenProbe(close_fails=True)
    try:
        open_window_or_rollback(stubborn, "c019-a1-2", "--unit", verdicts.UNIT)
    except RuntimeError as exc:
        assert "systemctl restart failed" in str(exc), (
            "a failing rollback must not mask the open error"
        )
    else:
        raise AssertionError("a failed open must raise even when the close fails")


def test_the_gauge_source_is_the_ingress_role_that_publishes_the_census() -> None:
    # ``CollectorMetricsSnapshot`` is created with ``enabled=service_role in
    # {"all", "ingress"}`` (factory.py): a ``worker`` replica never renders the
    # census families. Reading them from control-workers and demanding a sample
    # from every replica failed the live case with "missing or invalid
    # host-channel samples" although the census was healthy on every ingress
    # replica, so the gauge verdict reads the publishing role only.
    from scripts.e2e.regional import collector_window_fixture as window

    calls: list[tuple[str, str]] = []

    class Regional:
        def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
            calls.append((plane, app))
            return [{"name": f"{app}-0"}]

        def kubectl(self, plane: str, *arguments: str, **kwargs: object) -> str:
            return '{"metrics": "gpu_fault_collector_silent_nodes{a=\\"b\\"} 0\\n"}'

    fixture = window.CollectorWindowFixture.__new__(window.CollectorWindowFixture)
    fixture.regional = Regional()
    texts = fixture.census_metrics()
    assert calls == [("cpu", window.API_APP)], (
        "the census is read from the ingress replicas only, never from workers"
    )
    assert texts and "gpu_fault_collector_silent_nodes" in texts[0], (
        "the ingress replica's metrics text is returned"
    )

    class Spy:
        def census_metrics(self) -> list[str]:
            return ["census text"]

        def control_plane_metrics(self) -> list[str]:
            raise AssertionError("the gauge verdict must not read worker replicas")

    assert collect019.census_texts(Spy()) == ["census text"], (
        "the runner's gauge texts come from the census-publishing role"
    )


NOW = datetime(2026, 9, 18, 6, 30, tzinfo=timezone.utc)


def _host_row(age_seconds: float | None) -> list[dict[str, Any]]:
    row: dict[str, Any] = {"collector": verdicts.HOST_CHANNEL, "errors": []}
    if age_seconds is not None:
        row["last_success_at"] = (NOW - timedelta(seconds=age_seconds)).isoformat()
    return [{"collector": "GPU_METRICS", "last_success_at": NOW.isoformat()}, row]


def test_the_freshness_contract_measures_silence_from_the_last_success() -> None:
    # Silence is judged from the host channel's last success, not from the
    # window open: COLLECT-015 had rebooted the target 15 min earlier and the
    # channel was 232 s behind, so the 420 s budget ran out mid-window and a
    # healthy erroring node was reported as silence.
    assert verdicts.freshness_errors(_host_row(100), now=NOW) == [], (
        "a success within the freshness limit lets the window open"
    )
    stale = verdicts.freshness_errors(_host_row(232), now=NOW)
    assert stale and "232s old" in stale[0] and "silence budget" in stale[0], stale
    assert verdicts.host_success_age_seconds(_host_row(232), now=NOW) == 232.0, (
        "the age is measured in seconds from the recorded success"
    )
    assert "no usable last_success_at" in _text(
        verdicts.freshness_errors(_host_row(None), now=NOW)
    )
    assert verdicts.HOST_SUCCESS_FRESHNESS_SECONDS <= (
        verdicts.HOST_SILENT_AFTER_SECONDS - 300
    ), "the limit leaves at least 300 s of silence budget for the erroring window"


def test_the_window_waits_for_a_fresh_host_success_and_then_gives_up() -> None:
    readings = iter([_host_row(300), _host_row(200), _host_row(40)])
    slept: list[float] = []

    class Fixture:
        def collector_statuses(self) -> list[dict[str, Any]]:
            return next(readings)

    fresh = collect019.wait_for_fresh_host_success(
        Fixture(), interval_seconds=15, now=lambda: NOW, sleep=slept.append
    )
    assert slept == [15, 15], "the runner polls once per collection interval"
    assert fresh["host_success_age_seconds"] == 40.0, (
        "the evidence records how fresh the channel was at the open"
    )
    assert fresh["records"] == _host_row(40), "the accepted statuses are returned"

    class Stale:
        def collector_statuses(self) -> list[dict[str, Any]]:
            return _host_row(500)

    with pytest.raises(RegionalFixtureError, match="precondition: .*500s old"):
        collect019.wait_for_fresh_host_success(
            Stale(),
            interval_seconds=15,
            wait_seconds=0.001,
            now=lambda: NOW,
            sleep=slept.append,
        )
