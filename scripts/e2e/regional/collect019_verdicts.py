"""Pure verdict functions and constants of GF-REGIONAL-COLLECT-019.

The case proves ARCH-G5 on a real regional deployment: when ``nvidia-smi``
hangs past the host collector's per-call timeout for several rounds, the
collector stays up (no systemd restart loop), keeps delivering batches whose
``collection_errors`` name the timeout and then the open circuit breaker, and
the control plane shows the node as *erroring* on the host channel rather than
*silent*. When the hang ends the errors clear on the next accepted batch.

The hang is a PATH shadow visible to one collector unit only; the real binary
and the GPUs are untouched. Every function here judges documents the runner
wrote and touches no cluster.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from scripts.e2e.regional.collector_window_fixture import metric_max

CASE_ID = "GF-REGIONAL-COLLECT-019"
CONFIRMATION = "COLLECT019_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-018"
UNIT = "gpu-fault-host-collector.service"
HOST_CHANNEL = "HOST_TELEMETRY"
# The shipped host collector gives nvidia-smi 15 s per call and opens the
# breaker after 3 consecutive timeouts; the shadow sleeps past the timeout.
NVIDIA_SMI_TIMEOUT_SECONDS = 15
BREAKER_ROUNDS = 3
HANG_SECONDS = 30
SHADOW_MODE = f"hang:{HANG_SECONDS}"
TIMEOUT_TEXT = "timed out"
BREAKER_TEXT = "nvidia-smi circuit breaker open"
SILENT_METRIC = "gpu_fault_collector_silent_nodes"
ERRORING_METRIC = "gpu_fault_collector_erroring_nodes"
# Silence for the host channel is judged at 420 s by default; the whole
# erroring window must end well inside it or the case would page for silence.
HOST_SILENT_AFTER_SECONDS = 420
WINDOW_RESTORE_SECONDS = 600


def timing_errors(
    *,
    interval_seconds: int,
    hang_seconds: int = HANG_SECONDS,
    breaker_rounds: int = BREAKER_ROUNDS,
    silent_after_seconds: int = HOST_SILENT_AFTER_SECONDS,
) -> list[str]:
    """The hang must exceed the timeout and the erroring window must stay short."""

    errors: list[str] = []
    if hang_seconds <= NVIDIA_SMI_TIMEOUT_SECONDS:
        errors.append(
            f"hang {hang_seconds}s does not exceed the {NVIDIA_SMI_TIMEOUT_SECONDS}s "
            "nvidia-smi timeout"
        )
    rounds_to_breaker = breaker_rounds * (interval_seconds + NVIDIA_SMI_TIMEOUT_SECONDS)
    if rounds_to_breaker * 2 >= silent_after_seconds:
        errors.append(
            f"reaching the breaker takes {rounds_to_breaker}s per attempt; twice "
            f"that reaches the {silent_after_seconds}s silence threshold"
        )
    return errors


def window_errors(opened: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if opened.get("unit") != UNIT:
        errors.append(f"window opened on {opened.get('unit')!r}, not {UNIT!r}")
    shadow = opened.get("shadow") or []
    if not shadow or shadow[0] != "hang":
        errors.append(f"window shadow is {shadow!r}, expected a hang")
    if (opened.get("after") or {}).get("ActiveState") != "active":
        errors.append("the host collector is not active after the window opened")
    if not opened.get("deadman_timer", {}).get("ActiveState") == "active":
        errors.append("the window has no active deadman timer")
    return errors


def host_status(statuses: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in statuses:
        if record.get("collector") == HOST_CHANNEL:
            return record
    return None


def erroring_status_errors(statuses: list[dict[str, Any]]) -> list[str]:
    """The host channel reports the timeout and, after N rounds, the breaker."""

    status = host_status(statuses)
    if status is None:
        return ["the node has no HOST_TELEMETRY collector status row"]
    texts = [str(item) for item in status.get("errors") or []]
    errors: list[str] = []
    if not any("nvidia-smi" in text and TIMEOUT_TEXT in text for text in texts):
        errors.append(f"no nvidia-smi timeout collection error: {texts!r}")
    if not any(BREAKER_TEXT in text for text in texts):
        errors.append(f"no {BREAKER_TEXT!r} collection error: {texts!r}")
    if not status.get("last_error_at"):
        errors.append("the erroring batches did not stamp last_error_at")
    return errors


def service_errors(opened_after: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """The unit stayed up on the same PID: erroring is not crash-looping."""

    errors: list[str] = []
    if current.get("ActiveState") != "active":
        errors.append(f"{UNIT} is not active during the hang: {current}")
    if current.get("NRestarts") != opened_after.get("NRestarts"):
        errors.append(
            f"{UNIT} restarted during the hang: NRestarts "
            f"{opened_after.get('NRestarts')} -> {current.get('NRestarts')}"
        )
    if current.get("MainPID") != opened_after.get("MainPID"):
        errors.append(
            f"{UNIT} changed MainPID during the hang: "
            f"{opened_after.get('MainPID')} -> {current.get('MainPID')}"
        )
    return errors


def gauge_errors(before: list[str], during: list[str], *, cluster_id: str) -> list[str]:
    """Erroring rises for the host channel; silence does not."""

    errors: list[str] = []
    where = {"cluster_id": cluster_id, "channel": HOST_CHANNEL}
    if (metric_max(during, ERRORING_METRIC, where=where) or 0.0) < 1:
        errors.append(f"{ERRORING_METRIC} for the host channel is not at least 1")
    silent_before = metric_max(before, SILENT_METRIC, where=where) or 0.0
    silent_during = metric_max(during, SILENT_METRIC, where=where) or 0.0
    if silent_during > silent_before:
        errors.append(
            f"{SILENT_METRIC} for the host channel rose from {silent_before} to "
            f"{silent_during}; erroring was reported as silence"
        )
    return errors


def _stamp(value: Any) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def recovery_errors(statuses: list[dict[str, Any]]) -> list[str]:
    status = host_status(statuses)
    if status is None:
        return ["the node has no HOST_TELEMETRY collector status row"]
    success = _stamp(status.get("last_success_at"))
    error = _stamp(status.get("last_error_at"))
    if success is None:
        return ["the host channel never recorded a success after the window closed"]
    if error is not None and error >= success:
        return [
            "the host channel is still erroring after the window closed: "
            f"last_error_at={error.isoformat()} >= last_success_at={success.isoformat()}"
        ]
    return []


def closed_errors(closed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not closed.get("dropin_removed"):
        errors.append("the drop-in survived close-window")
    if not closed.get("window_root_removed"):
        errors.append("the window directory survived close-window")
    if (closed.get("after") or {}).get("ActiveState") != "active":
        errors.append("the host collector is not active after the window closed")
    return errors


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"
