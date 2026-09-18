from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.adapters import gpu_validation
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests.execution._cov95_runtime_validation import ValidationHarness

POWER = ("power_violation_total_us", "composite:POWER_LIMIT_THROTTLING")
MEMORY = ("ecc_sbe_volatile_total", "composite:CORRECTABLE_MEMORY_DEGRADATION")
THERMAL = ("gpu_temperature_c", "composite:THERMAL_STRESS")


def _findings(names, severity="WARNING"):
    return [
        SimpleNamespace(
            finding_id=f"finding-{index}",
            canonical_name=name,
            severity=SimpleNamespace(value=severity),
            gpu_uuid="GPU-a",
        )
        for index, name in enumerate(names)
    ]


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (POWER, WorkflowStepStatus.WAITING),
        (POWER[:1], WorkflowStepStatus.WAITING),
        (POWER[1:], WorkflowStepStatus.WAITING),
        (MEMORY, WorkflowStepStatus.WAITING),
        (THERMAL, WorkflowStepStatus.WAITING),
        ((*MEMORY, *POWER), WorkflowStepStatus.FAILED),
        ((*THERMAL, *POWER), WorkflowStepStatus.FAILED),
        ((*THERMAL, *MEMORY), WorkflowStepStatus.FAILED),
        ((*THERMAL, *MEMORY, *POWER), WorkflowStepStatus.FAILED),
        ((*POWER, "row_remap_failure"), WorkflowStepStatus.FAILED),
    ],
    ids=[
        "power",
        "power-counter",
        "power-composite",
        "memory",
        "thermal",
        "memory-power",
        "thermal-power",
        "thermal-memory",
        "all-classes",
        "unrecognized-warning",
    ],
)
def test_grace_requires_one_warning_class(monkeypatch, names, expected):
    harness = ValidationHarness(WorkflowOperation.VALIDATE_GPU)
    findings = _findings(names)
    monkeypatch.setattr(harness.feed, "findings", lambda *_args: findings)

    outcome = harness.execute()

    assert outcome.status is expected, outcome
    if expected is WorkflowStepStatus.WAITING:
        assert outcome.details["node_pending"]["node-a"][
            "transient_gpu_warning_cooldown"
        ] == [finding.finding_id for finding in findings]
    else:
        assert outcome.details["node_failures"] == {
            "node-a": ["active_gpu_health_findings"]
        }


@pytest.mark.parametrize("severity", ["CRITICAL", "UNKNOWN", None])
@pytest.mark.parametrize("mixed", [False, True], ids=["alone", "with-warning"])
def test_non_warning_power_findings_never_receive_grace(monkeypatch, severity, mixed):
    harness = ValidationHarness(WorkflowOperation.VALIDATE_GPU)
    findings = _findings(POWER[1:], severity)
    if mixed:
        findings.extend(_findings(POWER[:1]))
    monkeypatch.setattr(harness.feed, "findings", lambda *_args: findings)

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome


@pytest.mark.parametrize(
    ("age_seconds", "expected"),
    [
        (59, WorkflowStepStatus.WAITING),
        (60, WorkflowStepStatus.FAILED),
        (61, WorkflowStepStatus.FAILED),
    ],
    ids=["before-deadline", "at-deadline", "past-deadline"],
)
def test_power_grace_uses_the_original_workflow_deadline(
    monkeypatch, age_seconds, expected
):
    harness = ValidationHarness(
        WorkflowOperation.VALIDATE_GPU,
        transient_warning_grace=timedelta(seconds=60),
        temperature_warning_grace=timedelta(minutes=5),
    )
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        gpu_validation, "datetime", SimpleNamespace(now=lambda *_args: now)
    )
    harness.context = replace(
        harness.context,
        workflow=harness.context.workflow.model_copy(
            update={"created_at": now - timedelta(seconds=age_seconds)}
        ),
    )
    monkeypatch.setattr(harness.feed, "findings", lambda *_args: _findings(POWER))

    first = harness.execute()
    retried = harness.execute()

    assert first.status is retried.status is expected


def test_cleared_power_findings_can_succeed_without_waiting_out_the_grace(monkeypatch):
    harness = ValidationHarness(WorkflowOperation.VALIDATE_GPU)
    findings = _findings(POWER)
    monkeypatch.setattr(harness.feed, "findings", lambda *_args: findings)

    assert harness.execute().status is WorkflowStepStatus.WAITING
    findings.clear()
    assert harness.execute().status is WorkflowStepStatus.SUCCEEDED


def test_power_grace_does_not_apply_to_fabric_validation(monkeypatch):
    harness = ValidationHarness(WorkflowOperation.VALIDATE_FABRIC)
    monkeypatch.setattr(harness.feed, "findings", lambda *_args: _findings(POWER))

    outcome = harness.execute()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
