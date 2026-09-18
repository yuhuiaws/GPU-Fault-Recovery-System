from __future__ import annotations

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from tests.execution._cov95_runtime_workflows import FlowHarness


def test_warm_spare_completion_persists_one_notice_without_an_immediate_sender():
    h = FlowHarness([WorkflowOperation.REPLACE_NODE])
    h.adapter.outcomes[WorkflowOperation.REPLACE_NODE] = WorkflowStepOutcome.succeeded(
        operation_id="remote-replacement",
        details={
            "action": "SPARE_FAILOVER",
            "activated_spare_nodes": ["spare-a"],
            "node_rebindings": {"node-a": "spare-a"},
            "provider_mutation_submitted": False,
        },
    )
    first = h.execute()
    assert first.status is WorkflowStatus.SUCCEEDED, first
    [notice] = h.store.list_notifications()
    assert "node-a -> spare-a" in notice.body_text, notice
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.step_executions[0].details["notification_id"] == (
        notice.notification_id
    ), saved.step_executions
    replay = h.execute()
    assert replay.status is WorkflowStatus.SUCCEEDED, replay
    assert h.store.list_notifications() == [notice], (
        "terminal replay must preserve the single durable notification"
    )
    assert len(h.adapter.calls) == 1, "terminal replay must not repeat spare activation"
