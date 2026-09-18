from __future__ import annotations

import pytest

from gpu_fault.hyperpod_spares import SPARE_POOL_STATE_ANNOTATION
from gpu_fault.models import WorkflowStatus
from gpu_fault.spare_health import (
    FAILURES_ANNOTATION,
    HEALTH_ANNOTATION,
    INCIDENT_ANNOTATION,
    LAST_ALERT_AT_ANNOTATION,
    UNAVAILABLE_AT_ANNOTATION,
)
from tests.hyperpod._cov95_runtime_spare_health import NODE, HealthHarness


@pytest.mark.parametrize(
    "field",
    [
        "failure_threshold",
        "unavailable_recheck_seconds",
        "unavailable_alert_seconds",
        "reservation_ttl_seconds",
    ],
)
def test_spare_controller_requires_bounded_positive_thresholds(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        HealthHarness(**{field: 0})


@pytest.mark.parametrize("stale", ["missing-incident", "missing-workflow"])
def test_stale_remediation_pointer_is_cleared_without_starting_a_new_reboot(
    stale: str,
) -> None:
    h = HealthHarness()
    if stale == "missing-incident":
        h.annotations[INCIDENT_ANNOTATION] = "gone-incident"
    else:
        h.incident()
        incident = h.store.get_incident("spare-repair")
        h.store.save_incident(
            incident.model_copy(update={"workflow_request_id": "gone-workflow"})
        )
    h.annotations[HEALTH_ANNOTATION] = "REBOOT_PENDING"
    result = h.scan()
    assert result["state"] == "HEALTHY" and result["incident_id"] is None, result
    assert INCIDENT_ANNOTATION not in h.annotations, h.annotations
    assert h.store.list_workflows() == [], (
        "stale bookkeeping must not manufacture hardware work"
    )


@pytest.mark.parametrize(
    "status",
    [None, WorkflowStatus.FAILED, WorkflowStatus.BLOCKED, WorkflowStatus.PENDING],
)
def test_reboot_followup_requires_a_successful_workflow_before_rechecking(
    status: WorkflowStatus | None,
) -> None:
    h = HealthHarness()
    h.incident(status)
    h.annotations[HEALTH_ANNOTATION] = "REBOOT_PENDING"
    result = h.scan()
    expected = "REBOOT_PENDING" if status is WorkflowStatus.PENDING else "UNAVAILABLE"
    assert result["state"] == expected, result
    assert h.annotations[HEALTH_ANNOTATION] == expected, h.annotations
    assert h.node["spec"]["unschedulable"] is True, h.node
    assert len(h.sent) == int(status is not WorkflowStatus.PENDING), h.sent


@pytest.mark.parametrize("counter", ["bad", "-1", "0"])
def test_recheck_requires_consecutive_hardware_failures_even_after_invalid_counter(
    counter: str,
) -> None:
    h = HealthHarness(ready=False, failure_threshold=2)
    h.incident(WorkflowStatus.SUCCEEDED)
    h.annotations.update(
        {HEALTH_ANNOTATION: "RECHECKING", FAILURES_ANNOTATION: counter}
    )
    first = h.scan()
    assert (
        first["state"] == "RECHECKING" and h.annotations[FAILURES_ANNOTATION] == "1"
    ), first
    second = h.scan()
    assert second["state"] == "UNAVAILABLE", second
    assert h.annotations[SPARE_POOL_STATE_ANNOTATION] == "UNAVAILABLE", h.annotations
    assert len(h.sent) == 1, h.sent


@pytest.mark.parametrize("phase", ["RECHECKING", "UNAVAILABLE"])
def test_configuration_failure_after_reboot_stays_advisory_not_hardware_escalation(
    phase: str,
) -> None:
    h = HealthHarness(failure_threshold=1, unavailable_recheck_seconds=60)
    h.incident(WorkflowStatus.SUCCEEDED)
    h.node["spec"]["unschedulable"] = False
    h.annotations[HEALTH_ANNOTATION] = phase
    h.annotations[UNAVAILABLE_AT_ANNOTATION] = h.clock().isoformat()
    h.annotations[LAST_ALERT_AT_ANNOTATION] = h.clock().isoformat()
    h.clock.advance(minutes=2)
    result = h.scan()
    assert result["state"] == "SUSPECT", result
    assert any(
        "unreserved spare is schedulable" in reason for reason in result["reasons"]
    ), result
    [workflow] = h.store.list_workflows()
    assert workflow.status is WorkflowStatus.SUCCEEDED, workflow
    assert len(h.store.list_notifications()) == 1, h.sent
    assert UNAVAILABLE_AT_ANNOTATION not in h.annotations, h.annotations


@pytest.mark.parametrize("timestamp", ["invalid", None])
def test_unavailable_node_without_a_valid_clock_starts_a_new_bounded_observation_window(
    timestamp: str | None,
) -> None:
    h = HealthHarness(unavailable_recheck_seconds=60)
    h.incident(WorkflowStatus.FAILED)
    h.annotations[HEALTH_ANNOTATION] = "UNAVAILABLE"
    if timestamp is not None:
        h.annotations[UNAVAILABLE_AT_ANNOTATION] = timestamp
    first = h.scan()
    assert first["state"] == "UNAVAILABLE", first
    assert h.annotations[UNAVAILABLE_AT_ANNOTATION] == h.clock().isoformat(), (
        h.annotations
    )
    assert len(h.sent) == 1, h.sent
    h.clock.advance(seconds=60)
    second = h.scan()
    assert second["state"] == "RECHECKING", second
    assert h.node["spec"]["unschedulable"] is True, h.node


def test_unknown_health_state_on_typed_node_recovers_only_after_actual_health_checks() -> (
    None
):
    h = HealthHarness(typed=True)
    h.annotations[HEALTH_ANNOTATION] = "future-state"
    result = h.scan()
    assert result["state"] == "HEALTHY" and result["reasons"] == [], result
    assert h.annotations[SPARE_POOL_STATE_ANNOTATION] == "AVAILABLE", h.annotations
    assert h.store.list_workflows() == [] and h.sent == [], result
    metrics = h.controller.metrics_snapshot()
    assert metrics["spare_reservations_observed_at"] == h.clock().isoformat(), metrics


def test_failed_marker_retirement_does_not_undo_observed_healthy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = HealthHarness()
    h.incident(WorkflowStatus.SUCCEEDED)
    h.annotations[HEALTH_ANNOTATION] = "RECHECKING"
    attempts = []

    def retire(store, incident_id, **kwargs):
        attempts.append((incident_id, kwargs))
        raise OSError("marker storage unavailable")

    monkeypatch.setattr("gpu_fault.spare_health.retire_markers_for_incident", retire)
    result = h.scan()
    assert result["state"] == "HEALTHY", result
    assert len(attempts) == 1 and attempts[0][0] == "spare-repair", attempts
    assert NODE in attempts[0][1]["reason"], attempts
    assert h.annotations[HEALTH_ANNOTATION] == "HEALTHY", h.annotations
    assert INCIDENT_ANNOTATION not in h.annotations, h.annotations
