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
# The silence clock starts at the host channel's LAST SUCCESS, not at the
# window open: a node whose host collector had gone 232 s without a success
# (COLLECT-015 had just rebooted it) was judged silent 3 min into the window
# although every batch was an erroring one. Opening only when the last success
# is at most this old leaves >= 300 s of the 420 s budget for the window.
HOST_SUCCESS_FRESHNESS_SECONDS = 120


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
    # Utilization and inventory are distinct queries, so a hung round pays the
    # timeout twice before it sleeps the interval.
    rounds_to_breaker = breaker_rounds * (
        interval_seconds + 2 * NVIDIA_SMI_TIMEOUT_SECONDS
    )
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


def host_success_age_seconds(
    statuses: list[dict[str, Any]], *, now: datetime
) -> float | None:
    """Seconds since the host channel's last successful batch; ``None`` if unknown."""

    status = host_status(statuses)
    value = status.get("last_success_at") if status else None
    if not value:
        return None
    try:
        last = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if last.tzinfo is None:
        return None
    return (now - last).total_seconds()


def freshness_errors(statuses: list[dict[str, Any]], *, now: datetime) -> list[str]:
    """The window may open only while the host channel's last success is fresh."""

    age = host_success_age_seconds(statuses, now=now)
    if age is None:
        return ["the host channel has no usable last_success_at before the window"]
    if age > HOST_SUCCESS_FRESHNESS_SECONDS:
        return [
            f"the host channel's last success is {age:.0f}s old before the window "
            f"(limit {HOST_SUCCESS_FRESHNESS_SECONDS}s): the {HOST_SILENT_AFTER_SECONDS}s "
            "silence budget would be spent by the erroring window"
        ]
    return []


def erroring_status_errors(
    statuses: list[dict[str, Any]], *, seen: set[str] | None = None
) -> list[str]:
    """The host channel reports the timeout and, after N rounds, the breaker.

    Every accepted batch replaces the row's ``errors``: the timed-out rounds
    say "timed out", the round that opens the breaker says "circuit breaker
    open" and no longer names the timeout. The two are never on the row at
    once, so the runner accumulates the texts it has read across its polls
    in ``seen``; the row itself only has to be erroring *now*.
    """

    status = host_status(statuses)
    if status is None:
        return ["the node has no HOST_TELEMETRY collector status row"]
    current = [str(item) for item in status.get("errors") or []]
    if seen is not None:
        seen.update(current)
    texts = sorted(set(current) | (seen or set()))
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
    for label, reading in (("opening", opened_after), ("current", current)):
        pid = str(reading.get("MainPID") or "")
        restarts = str(reading.get("NRestarts", ""))
        if not pid.isdecimal() or int(pid) <= 0:
            errors.append(f"{UNIT} has no valid MainPID in the {label} reading")
        if not restarts.isdecimal():
            errors.append(f"{UNIT} has no valid NRestarts in the {label} reading")
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
    for label, texts, family in (
        ("before", before, SILENT_METRIC),
        ("during", during, SILENT_METRIC),
        ("during", during, ERRORING_METRIC),
    ):
        if not texts or any(
            (value := metric_max([text], family, where=where)) is None or value < 0
            for text in texts
        ):
            errors.append(
                f"{label} {family} has missing or invalid host-channel samples"
            )
    if errors:
        return errors
    erroring = metric_max(during, ERRORING_METRIC, where=where)
    if erroring is None or erroring < 1:
        errors.append(f"{ERRORING_METRIC} for the host channel is not at least 1")
    silent_before = metric_max(before, SILENT_METRIC, where=where)
    silent_during = metric_max(during, SILENT_METRIC, where=where)
    if (
        silent_before is not None
        and silent_during is not None
        and silent_during > silent_before
    ):
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
