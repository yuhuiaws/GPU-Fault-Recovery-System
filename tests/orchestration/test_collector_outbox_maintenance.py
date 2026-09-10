"""``gpu-fault-admin collector-outbox`` builds its incident and workflow in one
place, ``collector_outbox_maintenance``: an operator incident, a single
NODE_ACTION step carrying the validated outbox request, and the terminal
states the executor derives for it."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    NODE_EXCLUSIVE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
    OPERATION_RESOURCE_CLAIMS,
    OperationAdapter,
    operations_for_adapter,
)
from gpu_fault.orchestration import collector_outbox_maintenance as module
from tests._builders import active_workflow_executor, build_store, execute_workflow

NOW = datetime(2026, 9, 10, 8, 30, 15, tzinfo=timezone.utc)
OPERATOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"


def _build(**overrides):
    values = dict(
        collector="kernel",
        action="requeue-dead",
        confirm=True,
        path=None,
        operator=OPERATOR,
        reference="CHG-7",
        now=NOW,
    )
    values.update(overrides)
    return module.build_collector_outbox_workflow("gpu-a", "node-a", **values)


def test_the_operation_is_a_non_destructive_node_action_allowed_under_workloads() -> (
    None
):
    operation = WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE

    assert operation in operations_for_adapter(OperationAdapter.NODE_ACTION)
    assert operation not in DESTRUCTIVE_OPERATIONS
    assert operation not in NODE_WIDE_RECOVERY_OPERATIONS
    assert operation not in NODE_EXCLUSIVE_OPERATIONS, (
        "outbox maintenance shares the node with running recoveries"
    )
    assert OPERATION_RESOURCE_CLAIMS[operation] == frozenset(), (
        "no GPU or EFA runtime claim: nothing on the device is touched"
    )


def test_the_pair_is_an_operator_incident_and_a_single_node_action_step() -> None:
    incident, workflow = _build(path="/v1/kernel-logs")

    assert (
        incident.incident_id
        == "inc-operator-outbox-node-a-requeue-dead-20260910T083015Z"
    )
    assert incident.event_type == module.OPERATOR_EVENT_TYPE
    assert incident.event_source == "gpu-fault-admin"
    assert incident.policy_source == "OPERATOR"
    assert incident.policy_reference == "CHG-7"
    assert incident.cluster_id == "gpu-a" and incident.node_ids == ["node-a"]
    assert incident.state is IncidentState.ACTION_PENDING
    assert incident.fencing_token == 1
    assert incident.official_action == "COLLECTOR_OUTBOX_MAINTENANCE"
    assert incident.workflow_request_id == workflow.request_id
    assert incident.reasons == [
        "operator collector outbox requeue-dead on kernel path /v1/kernel-logs "
        f"requested by {OPERATOR} (CHG-7)"
    ]
    assert incident.created_at == incident.updated_at == NOW

    assert module.is_collector_outbox_workflow(workflow.request_id) is True
    assert workflow.incident_id == incident.incident_id
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.fencing_token == 1
    assert workflow.official_action == "COLLECTOR_OUTBOX_MAINTENANCE"
    assert len(workflow.official_steps) == 1
    step = workflow.official_steps[0]
    assert step.operation is WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE
    assert step.execution_owner == module.NODE_ACTION_OWNER == "gpu-fault-node-agent"
    assert step.node_ids == ["node-a"] and step.gpu_uuids == []
    assert step.parameters == {
        "collector": "kernel",
        "action": "requeue-dead",
        "confirm": True,
        "path": "/v1/kernel-logs",
        "operator": OPERATOR,
        "reference": "CHG-7",
    }
    assert workflow.created_at == workflow.updated_at == NOW


def test_read_only_actions_need_no_confirmation_and_no_reference() -> None:
    incident, workflow = _build(action="stats", confirm=False, reference=None)

    assert workflow.official_steps[0].parameters["confirm"] is False
    assert "path" not in workflow.official_steps[0].parameters
    assert incident.reasons[0].endswith("(no reference)"), incident.reasons
    assert incident.policy_reference is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"action": "requeue-dead", "confirm": False}, "confirm"),
        ({"collector": "sqs-hma"}, "unknown collector"),
        ({"action": "purge"}, "unknown outbox action"),
        ({"path": "v1/no-slash"}, "path filter"),
        ({"operator": " "}, "operator identity"),
    ],
)
def test_a_bad_request_is_refused_before_anything_is_built(
    overrides: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _build(**overrides)


def test_there_is_no_force_parameter() -> None:
    with pytest.raises(TypeError):
        _build(force=True)  # type: ignore[call-arg]


def test_the_builder_is_reachable_from_the_control_plane_wheel_closure() -> None:
    """``component_wheels`` builds wheels from imports; the in-Pod script imports
    the builder, so a control-plane module must name it (as ``incident_closure``
    names ``build_validated_restore_workflow``)."""

    from gpu_fault.orchestration import incident_closure

    assert (
        incident_closure.build_collector_outbox_workflow
        is module.build_collector_outbox_workflow
    )


class _NodeActionStub:
    """The node-action adapter as the executor sees it: owner and outcome."""

    owner = module.NODE_ACTION_OWNER

    def __init__(self, outcome: WorkflowStepOutcome) -> None:
        self.outcome = outcome
        self.contexts: list[WorkflowStepContext] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return (
            step.execution_owner == self.owner
            and step.operation is WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE
        )

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.contexts.append(context)
        return self.outcome


def _run_to_terminal(outcome: WorkflowStepOutcome):
    store = build_store()
    incident, workflow = _build()
    store.save_incident_and_workflow(incident, workflow)
    adapter = _NodeActionStub(outcome)
    executor = active_workflow_executor(
        store, [adapter], [WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE]
    )

    result = execute_workflow(executor, workflow.request_id, expected_fencing_token=1)

    assert adapter.contexts, "the step must reach the node-action adapter"
    return (
        result,
        store.get_workflow(workflow.request_id),
        store.get_incident(incident.incident_id),
    )


def test_a_succeeded_step_leaves_the_incident_recovered_with_the_node_result() -> None:
    node_result = {"node-a": {"action": "stats", "stats": {"depth": 2, "dead": 1}}}

    result, workflow, incident = _run_to_terminal(
        WorkflowStepOutcome.succeeded(details={"node_results": node_result})
    )

    assert result.status is WorkflowStatus.SUCCEEDED, result.error
    assert workflow.status is WorkflowStatus.SUCCEEDED
    assert incident.state is IncidentState.RECOVERED
    assert workflow.completed_operations == [
        WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE
    ]
    executions = [
        item
        for item in workflow.step_executions
        if item.operation is WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE
    ]
    assert executions[-1].details["node_results"] == node_result, (
        "the CLI reads the node result from the step execution details"
    )


def test_a_failed_step_ends_the_workflow_failed_and_records_the_refusal() -> None:
    """A lock refusal or an unreadable outbox fails the step. The workflow is
    FAILED; the incident is *not* parked ESCALATED: a workflow of one
    non-destructive, non-node-wide step is diagnostic-only to the executor,
    so the incident closes RECOVERED with the refusal on its reasons and the
    operator -- who read the failure on the CLI -- has nothing to close."""

    refusal = "node agent node-a: collector outbox lock is held: recorded holder pid 7"

    result, workflow, incident = _run_to_terminal(
        WorkflowStepOutcome.failed(
            refusal,
            details={
                "node_results": {},
                "lock_unavailable": True,
                "lock_holder": "recorded holder pid 7 (collector), alive",
            },
        )
    )

    assert result.status is WorkflowStatus.FAILED
    assert workflow.status is WorkflowStatus.FAILED
    assert incident.state is IncidentState.RECOVERED, incident.state
    assert incident.state is not IncidentState.ESCALATED, (
        "an operator maintenance failure must not open an awaiting-operator record"
    )
    assert any("recorded holder pid 7" in reason for reason in incident.reasons), (
        incident.reasons
    )
    failed = [
        item
        for item in workflow.step_executions
        if item.operation is WorkflowOperation.COLLECTOR_OUTBOX_MAINTENANCE
    ][-1]
    assert failed.error == refusal
    assert failed.details["lock_unavailable"] is True
