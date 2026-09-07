"""Contract tests for GF-REGIONAL-COLLECT-019 (nvidia-smi hang resilience)."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import collect019_verdicts as verdicts
from scripts.e2e.regional import run_collect019_nvidia_smi_hang as collect019

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
