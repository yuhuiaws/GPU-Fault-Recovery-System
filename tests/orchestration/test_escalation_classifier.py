"""The hardware escalation covers releases and host validation, creates its
records atomically, inherits the source identity and the per-node GPU scope.

F-H1 (docs/review/FINAL-建议汇总.md). A failed RESTORE_SCHEDULING had no
follow-up, so the node stayed cordoned with a FAILED workflow nobody acted
on; VALIDATE_HOST was a step the escalation itself inserted yet its failure
was unclassifiable; the escalation wrote its incident and workflow as two
blind writes and lied about the policy that produced them; a multi-node
replacement plan lost the per-node GPU mapping the agents insist on.
"""

from __future__ import annotations

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService, next_rung
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
VALIDATE_HOST = WorkflowOperation.VALIDATE_HOST
DCGM = WorkflowOperation.RUN_DCGM_DIAGNOSTIC
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE


class _StubBuilder:
    """Compiles one step per operation with the requested scope."""

    def compile_steps(
        self, operations, profile, node_ids, gpu_uuids, workload_ids=None
    ):
        return (
            [
                workflow_step(
                    operation,
                    node_ids=list(node_ids),
                    gpu_uuids=list(gpu_uuids),
                    workload_ids=list(workload_ids or []),
                )
                for operation in operations
            ],
            [],
        )


def _failed(store, steps, executions, *, completed, gpu_uuids=("GPU-a", "GPU-b")):
    store.save_profile(default_simulated_profile())
    incident = fault_incident(
        "inc-src",
        "event-src",
        event_type="XID",
        state=IncidentState.ESCALATED,
        workflow_request_id="wf-src",
        node_ids=["node-a", "node-b"],
        gpu_uuids=list(gpu_uuids),
        job_id="train-9",
        attempt_id="train-9-a1",
        policy_version="610",
        policy_source="NVIDIA",
        fencing_token=1,
    )
    workflow = workflow_request(
        "wf-src",
        "inc-src",
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_steps=steps,
        completed_step_indexes=completed,
        completed_operations=[steps[index].operation for index in completed],
        step_executions=executions,
    )
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow


# ---------------------------------------------------------------- classifier


def test_a_failed_release_escalates_to_an_operator_instead_of_leaving_the_node_cordoned():
    steps = [
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
        workflow_step(RESET, node_ids=["node-a"], gpu_uuids=["GPU-a"]),
        workflow_step(RESTORE, node_ids=["node-a"]),
    ]
    workflow = workflow_request(
        "wf",
        "inc",
        status=WorkflowStatus.FAILED,
        official_steps=steps,
        completed_step_indexes=[0, 1],
        completed_operations=[steps[0].operation, RESET],
        step_executions=[
            workflow_step_execution(0, steps[0].operation),
            workflow_step_execution(1, RESET),
            workflow_step_execution(
                2, RESTORE, WorkflowStepStatus.FAILED, error="uncordon refused"
            ),
        ],
    )

    classification = HardwareEscalationService.classify(workflow)

    assert classification is not None
    stage, action, operation, failures = classification
    assert stage == "containment_or_release"
    assert action is RecoveryAction.ESCALATE_OPERATOR
    assert operation is WorkflowOperation.ESCALATE_SUPPORT
    assert [item.step_index for item in failures] == [2]


def test_a_failed_host_validation_is_classified_like_the_other_validations():
    steps = [
        workflow_step(REBOOT, node_ids=["node-a"]),
        workflow_step(VALIDATE_HOST, node_ids=["node-a"]),
    ]
    workflow = workflow_request(
        "wf",
        "inc",
        status=WorkflowStatus.FAILED,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[REBOOT],
        step_executions=[
            workflow_step_execution(0, REBOOT),
            workflow_step_execution(
                1, VALIDATE_HOST, WorkflowStepStatus.FAILED, error="host unhealthy"
            ),
        ],
    )

    classification = HardwareEscalationService.classify(workflow)

    assert classification is not None
    assert classification[0] == "reboot_validation"
    assert classification[2] is WorkflowOperation.REPLACE_NODE
    assert next_rung(VALIDATE_HOST, {REBOOT}) is WorkflowOperation.REPLACE_NODE


# ---------------------------------------------------------------- emit


def test_the_escalation_is_created_atomically_once_and_inherits_the_source_identity():
    store = build_store()
    steps = [workflow_step(RESET, node_ids=["node-a"], gpu_uuids=["GPU-a"])]
    incident, workflow = _failed(
        store,
        steps,
        [
            workflow_step_execution(
                0, RESET, WorkflowStepStatus.FAILED, error="reset refused"
            )
        ],
        completed=[],
    )
    creations: list[str] = []
    blind_writes: list[str] = []
    original_create = store.create_incident_workflow_if_absent
    original_save_workflow, original_save_incident = (
        store.save_workflow,
        store.save_incident,
    )

    def create(event_id, builder, **kwargs):
        creations.append(event_id)
        return original_create(event_id, builder, **kwargs)

    def save_workflow(value):
        blind_writes.append(value.request_id)
        return original_save_workflow(value)

    def save_incident(value):
        blind_writes.append(value.incident_id)
        return original_save_incident(value)

    store.create_incident_workflow_if_absent = create
    store.save_workflow, store.save_incident = save_workflow, save_incident
    service = HardwareEscalationService(store, _StubBuilder())

    first = service.escalate(workflow)
    second = service.escalate(workflow)

    assert first is not None and second is not None
    escalated, replacement = first
    assert second[0].incident_id == escalated.incident_id
    assert second[1].request_id == replacement.request_id
    assert creations == ["reboot-after-wf-src"]
    assert blind_writes == []
    # Identity comes from the incident being escalated, not from a hard-coded
    # node-health policy.
    assert escalated.policy_source == incident.policy_source
    assert escalated.policy_version == incident.policy_version
    assert escalated.event_type == incident.event_type
    assert escalated.job_id == "train-9" and escalated.attempt_id == "train-9-a1"
    assert escalated.state is IncidentState.ACTION_PENDING
    assert replacement.status is WorkflowStatus.PENDING
    assert REBOOT in {step.operation for step in replacement.official_steps}


def test_a_multi_node_replacement_keeps_the_per_node_gpu_mapping():
    store = build_store()
    mapping = {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
    steps = [
        workflow_step(
            DCGM,
            node_ids=["node-a", "node-b"],
            gpu_uuids=["GPU-a", "GPU-b"],
            parameters={"gpu_uuids_by_node": mapping},
        )
    ]
    _, workflow = _failed(
        store,
        steps,
        [
            workflow_step_execution(
                0,
                DCGM,
                WorkflowStepStatus.FAILED,
                error="thermal",
                details={"failed_nodes": ["node-a", "node-b"]},
            )
        ],
        completed=[],
    )

    result = HardwareEscalationService(store, _StubBuilder()).escalate(workflow)

    assert result is not None
    _, replacement = result
    bundle = next(
        step for step in replacement.official_steps if step.operation is BUNDLE
    )
    assert bundle.node_ids == ["node-a", "node-b"]
    assert bundle.parameters["gpu_uuids_by_node"] == mapping
    quarantine = next(
        step
        for step in replacement.official_steps
        if step.operation is WorkflowOperation.QUARANTINE
    )
    assert "gpu_uuids_by_node" not in quarantine.parameters
