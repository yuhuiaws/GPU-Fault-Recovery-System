from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.hyperpod_spares import SpareAllocation
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests.execution._cov95_runtime_restart import ApiError
from tests.execution._support import FakeSpareCoordinator, RecordingNodeActionAdapter
from tests.hyperpod._cov95_provider_extra_safety import (
    provider_extra_isolation as provider_extra_isolation,
)
from tests.hyperpod._cov95_runtime_confirmation import ConfirmationHarness

REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE


@pytest.mark.parametrize("operation", [REBOOT, REPLACE])
@pytest.mark.parametrize("interruption", ["agent-readiness", "provider-state"])
def test_confirmation_restarts_stabilization_after_an_observed_health_interruption(
    operation, interruption
):
    h = ConfirmationHarness(operation, stabilization=30)
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, (
        "initial health did not start stabilization"
    )
    h.advance(15)
    if interruption == "agent-readiness":
        h.ready = False
    else:
        h.provider.nodes[0] = h.provider.nodes[0].model_copy(
            update={"status": "Rebooting"}
        )
    interrupted = h.execute()
    assert interrupted.status is WorkflowStepStatus.WAITING, (
        "unhealthy recovery was confirmed"
    )
    h.advance(15)
    h.ready = True
    h.provider.nodes[0] = h.provider.nodes[0].model_copy(update={"status": "Running"})
    restored = h.execute()
    assert restored.status is WorkflowStepStatus.WAITING, (
        "confirmation counted an observed unhealthy interval as successful stabilization"
    )
    h.advance(29)
    assert h.execute().status is WorkflowStepStatus.WAITING, (
        "the restarted stabilization window completed before its configured duration"
    )
    h.advance(1)
    assert h.execute().status is WorkflowStepStatus.SUCCEEDED, (
        "a complete healthy stabilization window did not confirm recovery"
    )
    assert h.provider.submissions == [], (
        "confirmation polling resubmitted a provider operation"
    )


@pytest.mark.parametrize("record", [None, {}, {"unschedulable": False}])
def test_reassertion_ignores_receipts_that_never_proved_isolation(record, monkeypatch):
    h = ConfirmationHarness(REBOOT)
    h.ready = False
    h.details["observed_isolation"] = {"node-a": record}
    reads = []

    def unexpected(node):
        reads.append(node)
        raise AssertionError("an unproved receipt authorized a scheduler read")

    monkeypatch.setattr(h.scheduler.core, "read_node", unexpected)
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, "an unready Agent was confirmed"
    assert reads == [], "unproved receipt entered the re-cordon path"
    assert h.provider.submissions == [], "an ignored receipt caused provider submission"


@pytest.mark.parametrize(
    ("error", "absent"),
    [
        pytest.param(KeyError("unit absent"), True, id="key-not-found"),
        pytest.param(ApiError(404), True, id="http-not-found"),
        pytest.param(RuntimeError("unit read failed"), False, id="unknown-failure"),
    ],
)
def test_reassertion_distinguishes_absence_from_unknown_scheduler_failure(
    error, absent, monkeypatch
):
    h = ConfirmationHarness(REBOOT)
    h.ready = False
    h.details["observed_isolation"] = {
        "node-a": {"unschedulable": True, "kubernetes_node": "node-a"}
    }

    def read(node):
        raise error

    monkeypatch.setattr(h.scheduler.core, "read_node", read)
    if not absent:
        with pytest.raises(RuntimeError, match="unit read failed"):
            h.execute()
    else:
        assert h.execute().status is WorkflowStepStatus.WAITING, (
            "temporary scheduler absence was mistaken for completed recovery"
        )
    assert h.provider.submissions == [], (
        "failed reassertion replayed the provider operation"
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_confirmation_persists_a_real_notification_without_sending_or_resubmitting(
    explicit,
):
    h = ConfirmationHarness(REBOOT)
    h.adapter.notification_sink = h.store
    if explicit:
        h.context = replace(
            h.context,
            request=h.context.request.model_copy(
                update={"confirmed_adapter_operation_ids": ["provider-owned"]}
            ),
        )
    result = h.execute()
    assert result.status is WorkflowStepStatus.SUCCEEDED, (
        "valid confirmation did not complete"
    )
    notification_id = result.details["notification_id"]
    assert [item.notification_id for item in h.store.list_notifications()] == [
        notification_id
    ], "confirmation lost or duplicated its persisted notification"
    assert result.details["confirmation_source"] == (
        "explicit-operation-id"
        if explicit
        else "hyperpod-running-and-new-agent-incarnation"
    ), "the recorded confirmation source did not match the observed proof"
    assert h.provider.submissions == [], (
        "notification persistence resubmitted the provider action"
    )


@pytest.mark.parametrize("usable", [False, True])
def test_legacy_waiting_spare_receipt_checks_allocation_before_rebinding(usable):
    h = ConfirmationHarness(REPLACE)
    h.adapter.spare_coordinator = FakeSpareCoordinator(
        SpareAllocation(
            applicable=usable,
            sufficient=usable,
            required=1,
            selected_node_ids=("spare-a",) if usable else (),
        )
    )
    h.adapter.node_action_adapter = RecordingNodeActionAdapter()
    result = h.execute()
    assert len(h.adapter.spare_coordinator.calls) == 1, (
        "legacy receipt did not perform exactly one local spare allocation check"
    )
    assert h.adapter.spare_coordinator.calls[0]["local_only"] is True, (
        "legacy spare handling attempted provider discovery for allocation"
    )
    if usable:
        assert result.details["action"] == "SPARE_FAILOVER", (
            "a healthy spare was not rebound"
        )
        assert result.details["node_rebindings"]["node-a"] == "spare-a", (
            "spare rebinding lost the original node identity"
        )
        assert result.details["provider_mutation_submitted"] is False, (
            "local failover claimed a provider replacement"
        )
    else:
        assert result.details["confirmation_source"] == (
            "hyperpod-logical-node-new-instance-and-ready-agent"
        ), "an unusable spare fabricated a warm-spare confirmation"
    assert h.provider.submissions == [], (
        "legacy confirmation submitted a provider action"
    )


def test_legacy_replacement_cannot_confirm_without_scheduler_isolation_proof():
    h = ConfirmationHarness(REPLACE)
    h.adapter.kubernetes_adapter = None
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, (
        "replacement was confirmed without a scheduler adapter to prove isolation"
    )
    assert h.provider.submissions == [], (
        "missing isolation caused provider resubmission"
    )


@pytest.mark.parametrize(
    "status", [WorkflowStepStatus.WAITING, WorkflowStepStatus.FAILED]
)
def test_legacy_replacement_does_not_confirm_an_unsuccessful_isolation(status):
    h = ConfirmationHarness(REPLACE)
    calls = []

    def isolate(context):
        calls.append(context)
        return WorkflowStepOutcome(status=status, error="unit isolation incomplete")

    class Scheduler:
        def _isolate(self, context):
            return isolate(context)

    h.adapter.kubernetes_adapter = Scheduler()
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, (
        "unconfirmed scheduler isolation was ignored"
    )
    assert len(calls) == 1, "replacement isolation was not attempted exactly once"
    assert calls[0].step.node_ids == ["node-new"], "isolation targeted the retired node"
    assert h.provider.submissions == [], "scheduler failure replayed a provider request"


def test_nonprovider_pending_receipt_is_not_automatically_confirmed():
    h = ConfirmationHarness(REBOOT)
    step = h.context.step.model_copy(
        update={"operation": WorkflowOperation.FREEZE_EVIDENCE}
    )
    h.workflow = h.workflow.model_copy(update={"official_steps": [step]})
    h.context = replace(h.context, step=step)
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, (
        "an unrelated operation borrowed HyperPod recovery confirmation"
    )
    assert h.provider.reads == [], (
        "a nonprovider operation read provider lifecycle state"
    )
    assert h.provider.submissions == [], (
        "an unrelated pending receipt submitted a provider action"
    )
