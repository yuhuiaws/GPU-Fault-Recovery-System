from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.kubernetes.primitives import kubernetes_request_timeout_seconds
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests.execution._cov95_runtime_restart import RestartHarness, changed_parameters
from tests.execution.test_restart_safety import UnusedApi


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("entry", ["environment", "constructor"])
def test_kubernetes_request_timeout_must_be_finite_before_any_transport(
    value: str, entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="positive"):
        if entry == "environment":
            monkeypatch.setenv("GPU_FAULT_KUBERNETES_REQUEST_TIMEOUT_SECONDS", value)
            kubernetes_request_timeout_seconds()
        else:
            KubernetesWorkflowAdapter(
                core_api=UnusedApi(),
                batch_api=UnusedApi(),
                custom_api=UnusedApi(),
                request_timeout_seconds=float(value),
            )


@pytest.mark.parametrize(
    ("kind", "status", "expected", "field"),
    [
        (
            "job",
            {"active": 0, "terminating": 1},
            WorkflowStepStatus.WAITING,
            "waiting_for_active_workloads",
        ),
        (
            "pytorchjob",
            {"replicaStatuses": {"Master": {"active": 1}, "ignored": "unknown"}},
            WorkflowStepStatus.WAITING,
            "waiting_for_active_workloads",
        ),
        (
            "pytorchjob",
            {
                "conditions": [
                    {"type": "Running", "status": "False"},
                    {"type": "Running", "status": "True"},
                ]
            },
            WorkflowStepStatus.WAITING,
            "waiting_for_active_workloads",
        ),
        (
            "pytorchjob",
            {"conditions": [{"type": "Ready", "status": "True"}]},
            WorkflowStepStatus.WAITING,
            "unknown_stop_state",
        ),
        (
            "jobset",
            {"replicatedJobs": ["unknown", {"active": 1}]},
            WorkflowStepStatus.WAITING,
            "waiting_for_active_workloads",
        ),
        (
            "jobset",
            {"replicatedJobs": [{"active": 0}]},
            WorkflowStepStatus.SUCCEEDED,
            "",
        ),
        (
            "jobset",
            {"replicatedJobs": [{"active": None}]},
            WorkflowStepStatus.WAITING,
            "unknown_stop_state",
        ),
    ],
)
def test_stop_verdict_uses_controller_state_without_guessing_unknown_activity(
    kind: str, status: dict[str, Any], expected: WorkflowStepStatus, field: str
) -> None:
    h = RestartHarness(kind)
    h.api.source["status"] = status
    step = h.context.step.model_copy(
        update={"operation": WorkflowOperation.STOP_WORKLOADS, "parameters": {}}
    )
    context = replace(
        h.context,
        step=step,
        workflow=h.context.workflow.model_copy(update={"official_steps": [step]}),
        idempotency_key="owned/0/STOP_WORKLOADS",
    )
    outcome = h.adapter.execute(context)
    assert outcome.status is expected, outcome
    assert len(h.api.patches) == 1 and h.api.created == {}, (
        h.api.patches,
        h.api.created,
    )
    assert outcome.details["suspended"] is True, outcome
    if field:
        assert outcome.details[field] == [f"training/{kind}/training-job"], outcome


@pytest.mark.parametrize("persistent_notifications", [False, True])
def test_storeless_unknown_source_gpu_count_never_creates_a_workload(
    persistent_notifications: bool,
) -> None:
    h = RestartHarness("job", source_gpu_count=0)
    sink = InMemoryStore() if persistent_notifications else None
    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(), batch_api=h.api, custom_api=h.api, notification_sink=sink
    )
    result = adapter.execute(h.context)
    assert result.status is (
        WorkflowStepStatus.WAITING
        if persistent_notifications
        else WorkflowStepStatus.FAILED
    ), result
    assert result.details["restart_submitted"] is False, result
    assert h.api.created == {} and h.api.patches == [], (h.api.created, h.api.patches)
    if sink is not None:
        assert len(sink.list_notifications()) == 1, sink.list_notifications()
        assert result.details["source_gpu_count"] == 0, result
    else:
        assert "persistent notification support" in (result.error or ""), result


@pytest.mark.parametrize("provider_state", ["unknown", "missing-state", "lookup-error"])
def test_restart_incident_premise_requires_a_known_provider_answer(
    provider_state: str,
) -> None:
    h = RestartHarness("job")
    queries: list[str] = []

    def ownership(incident_id: str) -> Any:
        queries.append(incident_id)
        if provider_state == "lookup-error":
            raise RuntimeError("fake ownership lookup unavailable")
        return SimpleNamespace(known=provider_state != "unknown", incident_state=None)

    adapter = KubernetesWorkflowAdapter(
        core_api=UnusedApi(),
        batch_api=h.api,
        custom_api=h.api,
        ownership_provider=SimpleNamespace(incident_ownership=ownership),
    )
    context = changed_parameters(
        h.context, requires_incident_state="RECOVERED", incident_id="repair-owner"
    )
    result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED, result
    assert result.details["reason"] == "INCIDENT_STATE_UNVERIFIABLE", result
    assert queries == ["repair-owner"], queries
    assert h.api.created == {} and h.api.patches == [], (h.api.created, h.api.patches)
