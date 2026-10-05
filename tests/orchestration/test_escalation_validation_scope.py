"""A failed validation escalates the nodes the recovery action acted on.

Live on HyperPod (GF-REGIONAL-COLLECT-016): a RESET_GPU drill on one node ran
to RESTORE_GPU_SERVICES, but its VALIDATE_GPU had been widened by a merge to a
second node of the job that owned none of the step's GPUs. That node could
never report a sibling's GPU, the step sat WAITING until the 600 s per-step
cap failed it with empty details, and the ladder -- reading no ``failed_nodes``
-- escalated *both* nodes to a ``reboot-after`` successor that cordoned the
healthy sibling. The rung a validation fails into is a hardware action, and
it belongs to the nodes the reset/reboot/replacement acted on.
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
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.orchestration.test_escalation_classifier import _StubBuilder

RESET_NODE = "hyperpod-i-07e5"
SIBLING = "hyperpod-i-0246"
THIRD = "hyperpod-i-0cf5"
JOB_NODES = [SIBLING, RESET_NODE, THIRD]
JOB = "training/job/c016"
CAP_ERROR = (
    "step 7/VALIDATE_GPU stayed non-terminal for 601s, past the 600s per-step cap"
)


def _drill_steps():
    one = [RESET_NODE]
    return [
        workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=one),
        workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=one),
        workflow_step(
            WorkflowOperation.STOP_WORKLOADS, node_ids=JOB_NODES, workload_ids=[JOB]
        ),
        workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=one),
        workflow_step(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, node_ids=one),
        workflow_step(WorkflowOperation.RESET_GPU, node_ids=one, gpu_uuids=["GPU-a"]),
        workflow_step(WorkflowOperation.RESTORE_GPU_SERVICES, node_ids=one),
        # Widened by the merge: the sibling joined without owning GPU-a.
        workflow_step(
            WorkflowOperation.VALIDATE_GPU,
            node_ids=[SIBLING, RESET_NODE],
            gpu_uuids=["GPU-a", "GPU-b"],
        ),
        workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=JOB_NODES),
        workflow_step(
            WorkflowOperation.RESTART_WORKLOAD, node_ids=JOB_NODES, workload_ids=[JOB]
        ),
    ]


def _drill(store, *, validation_details: dict | None = None, reset_ran: bool = True):
    store.save_profile(default_simulated_profile())
    steps = _drill_steps()
    ran = list(range(7)) if reset_ran else [0]
    executions = [
        workflow_step_execution(index, steps[index].operation, phase="official")
        for index in ran
    ]
    executions.append(
        workflow_step_execution(
            7,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowStepStatus.FAILED,
            phase="official",
            error=CAP_ERROR,
            details=dict(validation_details or {}),
        )
    )
    incident = fault_incident(
        "inc-drill",
        "event-drill",
        state=IncidentState.QUARANTINED,
        workflow_request_id="wf-drill",
        node_ids=[RESET_NODE],
        gpu_uuids=["GPU-a"],
        job_id="c016",
        attempt_id="c016-a1",
        fencing_token=1,
    )
    workflow = workflow_request(
        "wf-drill",
        incident.incident_id,
        status=WorkflowStatus.FAILED,
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_steps=steps,
        completed_step_indexes=ran,
        completed_operations=[steps[index].operation for index in ran],
        step_executions=executions,
    )
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow


def _failed_validation(workflow):
    return [
        execution
        for execution in workflow.step_executions
        if execution.status is WorkflowStepStatus.FAILED
    ]


def test_a_capped_validation_escalates_only_the_node_the_reset_acted_on() -> None:
    incident, workflow = _drill(build_store())

    scope = HardwareEscalationService.collect_scope(
        workflow, incident, _failed_validation(workflow)
    )

    assert scope["ordered_failed_nodes"] == [RESET_NODE], (
        "the sibling the merge added to the validation had no reset; the reboot "
        "rung is not its business"
    )
    assert scope["gpu_uuids"] == ["GPU-a", "GPU-b"]


def test_pending_nodes_of_a_capped_step_count_as_its_failed_nodes() -> None:
    incident, workflow = _drill(
        build_store(), validation_details={"pending_nodes": [SIBLING]}
    )

    scope = HardwareEscalationService.collect_scope(
        workflow, incident, _failed_validation(workflow)
    )

    assert scope["ordered_failed_nodes"] == [RESET_NODE], (
        "a pending sibling narrows nothing past the acted node: the node that "
        "never answered for a GPU it does not own is not the node to reboot"
    )
    reset_only = _drill(
        build_store(), validation_details={"failed_nodes": [RESET_NODE]}
    )
    scope = HardwareEscalationService.collect_scope(
        reset_only[1], reset_only[0], _failed_validation(reset_only[1])
    )
    assert scope["ordered_failed_nodes"] == [RESET_NODE]


def test_a_validation_with_no_recovery_action_keeps_every_step_node() -> None:
    """A diagnostic validation (nothing acted on the nodes) is judged on all
    its nodes as before."""

    incident, workflow = _drill(build_store(), reset_ran=False)

    scope = HardwareEscalationService.collect_scope(
        workflow, incident, _failed_validation(workflow)
    )

    assert scope["ordered_failed_nodes"] == sorted([SIBLING, RESET_NODE])


def test_the_reboot_successor_isolates_and_reboots_only_the_reset_node() -> None:
    store = build_store()
    incident, workflow = _drill(store)

    result = HardwareEscalationService(store, _StubBuilder()).escalate(workflow)

    assert result is not None
    successor_incident, successor = result
    assert successor.request_id == f"workflow-reboot-after-{workflow.request_id}"
    assert successor_incident.effective_action is RecoveryAction.REBOOT_NODE
    assert successor_incident.node_ids == [RESET_NODE]
    by_operation = {step.operation: step for step in successor.official_steps}
    for operation in (
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.RESTORE_SCHEDULING,
    ):
        assert by_operation[operation].node_ids == [RESET_NODE], operation
    for operation in (
        WorkflowOperation.STOP_WORKLOADS,
        WorkflowOperation.RESTART_WORKLOAD,
    ):
        assert by_operation[operation].node_ids == JOB_NODES, (
            "workload steps keep the job's node scope"
        )
    assert successor_incident.reasons[0].startswith(
        "reset_validation remediation failed; failed operation or validation: "
        "VALIDATE_GPU; nodes: " + RESET_NODE
    ), successor_incident.reasons
