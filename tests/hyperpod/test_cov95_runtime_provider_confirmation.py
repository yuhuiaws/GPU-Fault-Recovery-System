from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.hyperpod import HyperPodAdapterError
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import NotFoundError
from tests.hyperpod._cov95_runtime_confirmation import NOW, ConfirmationHarness

REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE


@pytest.mark.parametrize(
    "defect",
    [
        "registry",
        "agent-missing",
        "not-ready",
        "boot",
        "incarnation",
        "provider-error",
        "provider-pending",
        "provider-count",
    ],
)
def test_reboot_confirmation_waits_without_complete_new_agent_and_provider_proof(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ConfirmationHarness(REBOOT)
    if defect == "registry":
        h.adapter.registry = None
    elif defect == "agent-missing":

        def missing(*args: Any) -> Any:
            raise NotFoundError("fake agent absence")

        monkeypatch.setattr(h.store, "get_agent", missing)
    elif defect == "not-ready":
        h.ready = False
    elif defect in {"boot", "incarnation"}:
        field = "boot_id" if defect == "boot" else "agent_incarnation_id"
        h.store.save_agent(
            h.record.model_copy(
                update={field: "boot-before" if defect == "boot" else "inc-before"}
            )
        )
    elif defect == "provider-error":
        h.provider.fail = HyperPodAdapterError("fake provider inventory unknown")
    elif defect == "provider-pending":
        h.provider.nodes[0] = h.provider.nodes[0].model_copy(
            update={"status": "Rebooting"}
        )
    else:
        h.provider.nodes = []
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert result.adapter_operation_id == "provider-owned", result
    assert h.provider.submissions == [], h.provider.submissions


@pytest.mark.parametrize("proof", ["retired", "no-source", "not-retired"])
def test_reboot_without_captured_baseline_requires_the_faults_retired_boot(
    proof: str,
) -> None:
    h = ConfirmationHarness(REBOOT)
    h.details["agent_baselines"] = {}
    h.context = replace(
        h.context,
        incident=h.incident.model_copy(
            update={"source_boot_id": None if proof == "no-source" else "boot-before"}
        ),
    )
    if proof == "not-retired":
        h.store.save_agent(h.record.model_copy(update={"retired_incarnation_ids": []}))
    result = h.execute()
    assert result.status is (
        WorkflowStepStatus.SUCCEEDED
        if proof == "retired"
        else WorkflowStepStatus.WAITING
    ), result
    if proof == "retired":
        assert result.details["externally_confirmed"] is True, result
    assert h.provider.submissions == [], h.provider.submissions


@pytest.mark.parametrize("operation", [REBOOT, REPLACE])
def test_provider_confirmation_requires_the_entire_stabilization_window(
    operation: WorkflowOperation,
) -> None:
    h = ConfirmationHarness(operation, stabilization=30)
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    h.advance(29)
    second = h.execute()
    assert second.status is WorkflowStepStatus.WAITING, second
    h.advance(1)
    final = h.execute()
    assert final.status is WorkflowStepStatus.SUCCEEDED, final
    field = (
        "post_reboot_stabilization_started_at"
        if operation is REBOOT
        else "post_replacement_stabilization_started_at"
    )
    assert final.details[field] == NOW.isoformat(), final
    assert final.details["externally_confirmed"] is True, final
    assert h.provider.submissions == [], h.provider.submissions


@pytest.mark.parametrize(
    "status",
    [
        WorkflowStepStatus.WAITING,
        WorkflowStepStatus.FAILED,
        WorkflowStepStatus.SUCCEEDED,
    ],
)
def test_post_reboot_snapshot_preserves_pending_or_optional_failure_evidence(
    status: WorkflowStepStatus,
) -> None:
    h = ConfirmationHarness(REBOOT)
    calls = []

    def snapshot(context: Any) -> WorkflowStepOutcome:
        calls.append(context)
        return WorkflowStepOutcome(
            status=status,
            error="fake snapshot unavailable"
            if status is WorkflowStepStatus.FAILED
            else None,
            details={"sample": "owned"},
        )

    h.adapter.node_action_adapter = SimpleNamespace(
        owner="gpu-fault-node-agent", execute=snapshot
    )
    result = h.execute()
    [context] = calls
    assert context.step.operation is WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT, context
    assert context.step.node_ids == ["node-a"], context
    assert context.step.parameters == {} and context.step.gpu_uuids == [], context
    if status is WorkflowStepStatus.WAITING:
        assert result.status is WorkflowStepStatus.WAITING, result
    else:
        assert result.status is WorkflowStepStatus.SUCCEEDED, result
        if status is WorkflowStepStatus.FAILED:
            assert (
                result.details["post_reboot_health_snapshot_error"]
                == "fake snapshot unavailable"
            ), result
        else:
            assert result.details["post_reboot_health_snapshot"] == {
                "sample": "owned"
            }, result
    assert h.provider.submissions == [], h.provider.submissions


@pytest.mark.parametrize(
    "defect",
    [
        "baseline",
        "provider-error",
        "provider-pending",
        "same-instance",
        "missing-instance",
        "not-ready",
        "duplicate-agent",
    ],
)
def test_legacy_replacement_confirmation_refuses_incomplete_or_ambiguous_proof(
    defect: str,
) -> None:
    h = ConfirmationHarness(REPLACE)
    if defect == "baseline":
        h.details["provider_baselines"] = {}
    elif defect == "provider-error":
        h.provider.fail = HyperPodAdapterError("fake unknown provider")
    elif defect in {"provider-pending", "same-instance", "missing-instance"}:
        update = (
            {"status": "Replacing"}
            if defect == "provider-pending"
            else {"instance_id": "i-old" if defect == "same-instance" else None}
        )
        h.provider.nodes[0] = h.provider.nodes[0].model_copy(update=update)
    elif defect == "not-ready":
        h.ready = False
    else:
        duplicate = h.record.model_copy(
            update={
                "node_id": "duplicate-agent",
                "agent_incarnation_id": "other-incarnation",
            }
        )
        h.store.save_agent(duplicate)
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert h.provider.submissions == [], h.provider.submissions


def test_legacy_replacement_rebinds_only_the_recorded_target_and_aliases() -> None:
    h = ConfirmationHarness(REPLACE)
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, result
    assert result.details["node_rebindings"] == {
        "node-a": "node-new",
        "i-old": "node-new",
    }, result
    assert (
        result.details["confirmation_source"]
        == "hyperpod-logical-node-new-instance-and-ready-agent"
    ), result
    assert h.provider.submissions == [], h.provider.submissions
