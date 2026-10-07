"""A failed node branch escalates in place; its siblings keep repairing.

F-N1 (docs/review/F-N1-设计与实施计划.md). In a multi-node job workflow one
node's RESET_GPU failing used to fail the whole workflow; the replacement
workflow then covered only that node, so its siblings never got their
RESTORE_SCHEDULING and the job did not restart. Now the failed branch walks
the escalation ladder inside the same workflow (reset -> reboot -> warm
spare), the other branches finish, and the job restarts once all of them
have. A branch that exhausts the ladder fails the workflow without
restarting the job.
"""

from __future__ import annotations

from gpu_fault.models import BlockedKind, WorkflowOperation, workflow_is_open
from gpu_fault.orchestration.escalation import next_rung
from tests._builders import build_store
from tests.execution._support import workflow_state

RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE


def test_model_defaults_and_the_escalation_ladder():
    store = build_store()
    _, workflow = workflow_state(store, [RESET])

    assert workflow.branch_escalation_counts == {}
    assert workflow.exhausted_branch_ids == []
    assert workflow.lifetime_deadline_at is None
    assert next_rung(RESET, set()) is REBOOT
    assert next_rung(WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, set()) is REBOOT
    assert next_rung(REBOOT, set()) is REPLACE
    assert next_rung(REPLACE, set()) is None
    assert next_rung(WorkflowOperation.RUN_FIELD_DIAGNOSTIC, set()) is None
    # A failed validation escalates from whatever recovery already ran.
    assert next_rung(WorkflowOperation.VALIDATE_GPU, {RESET}) is REBOOT
    assert next_rung(WorkflowOperation.VALIDATE_FABRIC, {REBOOT}) is REPLACE
    assert next_rung(WorkflowOperation.VALIDATE_GPU, {REPLACE}) is None
    assert next_rung(WorkflowOperation.VALIDATE_GPU, set()) is None


# ---------------------------------------------------------------- executor

import pytest  # noqa: E402

from gpu_fault.execution.branch_escalation import BranchEscalator  # noqa: E402
from gpu_fault.execution.models import WorkflowExecutionRequest  # noqa: E402
from gpu_fault.models import (  # noqa: E402
    IncidentState,
    RecoveryAction,
    WorkflowEventCode,
    WorkflowStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter  # noqa: E402
from gpu_fault.orchestration.dag_branching import DagBrancher  # noqa: E402
from gpu_fault.orchestration.escalation import (  # noqa: E402
    VALIDATION_UNOBSERVED_STAGE,
    HardwareEscalationService,
)
from tests._builders import (  # noqa: E402
    active_workflow_executor,
    copy_model,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import (  # noqa: E402
    RESTART_PARAMETERS,
    FakeAdapter,
    WorkflowStepOutcome,
)

STOP = WorkflowOperation.STOP_WORKLOADS
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
VALIDATIONS = (
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
    WorkflowOperation.RESTORE_SCHEDULING,
)
ALL_OPERATIONS = {STOP, RESET, REBOOT, REPLACE, RESTART_JOB, BUNDLE, *VALIDATIONS}


def _escalator(max_rungs: int = 2) -> BranchEscalator:
    def compile_steps(_workflow, operations, node_id, gpu_uuids):
        return [
            workflow_step(operation, node_ids=[node_id], gpu_uuids=list(gpu_uuids))
            for operation in operations
        ]

    return BranchEscalator(
        DagBrancher(RecoveryArbiter()), compile_steps, max_rungs=max_rungs
    )


def _job_dag(
    store, *, node_c_operation: WorkflowOperation = REBOOT, stop_done: bool = True
):
    """STOP (shared) -> {node-b RESET_GPU, node-c <op>} -> RESTART_WORKLOAD (join)."""

    incident, workflow = workflow_state(
        store, [STOP, RESET, node_c_operation, RESTART_JOB]
    )
    steps = [
        copy_model(
            workflow.official_steps[0],
            branch_id="shared",
            node_ids=["node-b", "node-c"],
        ),
        copy_model(
            workflow.official_steps[1],
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-b",
        ),
        copy_model(
            workflow.official_steps[2],
            node_ids=["node-c"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-c",
        ),
        copy_model(
            workflow.official_steps[3],
            depends_on_step_indexes=[1, 2],
            branch_id="join",
            node_ids=["node-b", "node-c"],
            parameters=dict(RESTART_PARAMETERS),
        ),
    ]
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        completed_step_indexes=[0] if stop_done else [],
        completed_operations=[STOP] if stop_done else [],
        step_executions=[workflow_step_execution(0, STOP)] if stop_done else [],
    )
    store.save_workflow(workflow)
    return incident, workflow


def _run(store, workflow, outcomes, *, escalator, operations=ALL_OPERATIONS):
    adapter = FakeAdapter(outcomes)
    executor = active_workflow_executor(store, [adapter], operations)
    executor.branch_escalator = escalator
    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    return result, adapter, store.get_workflow(workflow.request_id)


def _ok(*operations):
    return {operation: WorkflowStepOutcome.succeeded() for operation in operations}


def test_a_failed_node_branch_escalates_in_place_and_the_job_restarts_after_all():
    store = build_store()
    incident, workflow = _job_dag(store)
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused"),
        **_ok(REBOOT, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())

    assert result.status is WorkflowStatus.SUCCEEDED
    assert saved.branch_escalation_counts == {"node-b": 1}
    assert saved.exhausted_branch_ids == []
    assert 1 in saved.superseded_step_indexes  # the failed RESET step is retired
    assert "workflow-active/2/RESTART_NODE" in adapter.calls  # node-c's own reboot
    node_b_reboot = next(
        call
        for call in adapter.calls
        if call.endswith("/RESTART_NODE") and call != "workflow-active/2/RESTART_NODE"
    )
    restart = next(call for call in adapter.calls if call.endswith("/RESTART_WORKLOAD"))
    assert adapter.calls.index(node_b_reboot) < adapter.calls.index(restart)
    assert adapter.calls.index("workflow-active/2/RESTART_NODE") < adapter.calls.index(
        restart
    )
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED


def test_an_exhausted_branch_fails_the_workflow_without_restarting_the_job():
    store = build_store()
    incident, workflow = _job_dag(store, node_c_operation=BUNDLE)
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused"),
        REBOOT: WorkflowStepOutcome.failed("node did not come back"),
        REPLACE: WorkflowStepOutcome.failed("no healthy warm spare"),
        **_ok(BUNDLE, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())

    assert result.status is WorkflowStatus.FAILED
    # The record carries its reason, not only the TERMINAL audit event: the
    # API, the acceptance verdicts and the failure handler read the field.
    assert str(saved.terminal_failure_reason).startswith(
        "node branch escalation exhausted: branch:node-b"
    ), saved.terminal_failure_reason
    assert saved.branch_escalation_counts == {"node-b": 2}
    assert len(saved.exhausted_branch_ids) == 1
    assert saved.exhausted_branch_ids[0].startswith("branch:node-b"), (
        'expected saved.exhausted_branch_ids[0].startswith("branch:node-b") to be true'
    )
    assert all("RESTART_WORKLOAD" not in call for call in adapter.calls), (
        'expected all("RESTART_WORKLOAD" not in call for call in adapter.calls) to be true'
    )
    assert (
        "workflow-active/2/COLLECT_DIAGNOSTIC_BUNDLE" in adapter.calls
    )  # node-c finished
    assert [
        call.rsplit("/", 1)[1] for call in adapter.calls if "node" not in call
    ].count("REPLACE_NODE") == 1
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


def test_an_exhausted_ladder_restores_the_quiesce_it_climbed_past():
    """Live 2026-10-05 (gpu-eks-cluster DESTR-001): QUIESCE stopped the node's
    GPU services, RESET_GPU was busy, the ladder's RESTART_NODE was refused and
    no spare was declared. The branch replacement had superseded the chain's
    RESTORE_GPU_SERVICES, so the exhausted record ended FAILED with the node's
    services down until the agent's fail-safe timer. The record now appends and
    runs the owed restore before it ends, as a parked step failure does.
    """

    QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
    RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
    store = build_store()
    incident, workflow = _job_dag(store, node_c_operation=BUNDLE)
    steps = list(workflow.official_steps)
    # node-b: QUIESCE (1) -> RESET (2) -> RESTORE (3); node-c: BUNDLE (4); join (5)
    quiesce = copy_model(steps[1], operation=QUIESCE)
    reset = copy_model(steps[1], depends_on_step_indexes=[1])
    restore = copy_model(steps[1], operation=RESTORE, depends_on_step_indexes=[2])
    bundle = copy_model(steps[2])
    join = copy_model(steps[3], depends_on_step_indexes=[3, 4])
    workflow = copy_model(
        workflow, official_steps=[steps[0], quiesce, reset, restore, bundle, join]
    )
    store.save_workflow(workflow)
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused: In use by another client"),
        REBOOT: WorkflowStepOutcome.failed("submission cluster_name does not match"),
        REPLACE: WorkflowStepOutcome.failed("no healthy warm spare"),
        **_ok(QUIESCE, RESTORE, BUNDLE, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(
        store,
        workflow,
        outcomes,
        escalator=_escalator(),
        operations=ALL_OPERATIONS | {QUIESCE, RESTORE},
    )

    assert result.status is WorkflowStatus.FAILED, result
    assert str(saved.terminal_failure_reason).startswith(
        "node branch escalation exhausted: branch:node-b"
    ), saved.terminal_failure_reason
    operations = [call.rsplit("/", 1)[1] for call in adapter.calls]
    assert operations.count("RESTORE_GPU_SERVICES") == 1, adapter.calls
    assert operations.index("RESTORE_GPU_SERVICES") > operations.index(
        "REPLACE_NODE"
    ), "the restore is the compensation after the last rung, not the chain's own"
    assert 3 in saved.superseded_step_indexes, "the chain's restore was replaced"
    appended = saved.official_steps[-1]
    assert appended.operation is RESTORE and appended.node_ids == ["node-b"], appended
    assert appended.parameters == {"compensation_for_step_indexes": [1]}, appended
    assert len(saved.official_steps) - 1 in saved.completed_step_indexes, saved
    assert saved.pending_failure_step_index is None, saved
    assert all("RESTART_WORKLOAD" not in call for call in adapter.calls), adapter.calls
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


@pytest.mark.parametrize(
    "details",
    [
        pytest.param(
            {
                "node_action_interrupted": True,
                "manual_confirmation_required": True,
                "operation": RESET.value,
            },
            id="interrupted-node-action",
        ),
        pytest.param(
            {
                "node_action_error_code": "COMMAND_ID_REUSED",
                "node_action_retryable": False,
                "manual_confirmation_required": True,
                "http_status": 409,
            },
            id="command-id-reused",
        ),
    ],
)
def test_an_unknown_outcome_branch_failure_goes_to_an_operator_not_up_the_ladder(
    details,
):
    """The in-place ladder must stop where the whole-workflow ladder stops.

    An INTERRUPTED RESET_GPU (the agent may have reset the GPU) and a
    COMMAND_ID_REUSED (the step's body changed under an id whose attempt 1 may
    already have reset the old set) both carry ``manual_confirmation_required``.
    The branch escalator appended RESTART_NODE without reading it -- a reboot
    of a node whose GPU was possibly just reset, from the DAG path, while the
    whole-workflow classifier had already learned to hand the step over.
    """

    store = build_store()
    _, workflow = _job_dag(store, node_c_operation=BUNDLE)
    outcomes = {
        RESET: WorkflowStepOutcome.failed(
            "node agent node-b: refused", details=details
        ),
        **_ok(BUNDLE, REBOOT, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())

    assert all(not call.endswith("/RESTART_NODE") for call in adapter.calls), (
        "a reboot rung was appended for a step whose outcome nobody knows: "
        f"{adapter.calls}"
    )
    assert result.status is WorkflowStatus.BLOCKED, result.status
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "an unknown physical outcome must not release node ownership"
    )
    assert saved.branch_escalation_counts == {}, saved.branch_escalation_counts
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None, "the whole-workflow classifier must take it"
    assert classification[1] is RecoveryAction.ESCALATE_OPERATOR, classification[:3]


class _ConfirmableNodeAdapter(FakeAdapter):
    """Per-(operation, node) outcomes that still honour a confirmed WAITING."""

    def __init__(self, outcomes, per_node):
        super().__init__(outcomes)
        self.per_node = per_node

    def execute(self, context):
        key = (context.step.operation, tuple(context.step.node_ids))
        if key not in self.per_node:
            return super().execute(context)
        self.calls.append(context.idempotency_key)
        outcome = self.per_node[key]
        if (
            outcome.adapter_operation_id
            in context.request.confirmed_adapter_operation_ids
        ):
            return WorkflowStepOutcome.succeeded(
                operation_id=outcome.adapter_operation_id
            )
        return outcome


def _job_dag_with_release(store):
    """STOP -> {node-b RESET_GPU, node-c RESTART_NODE -> RESTORE_SCHEDULING} -> join."""

    RESTORE = WorkflowOperation.RESTORE_SCHEDULING
    incident, workflow = workflow_state(
        store, [STOP, RESET, REBOOT, RESTORE, RESTART_JOB]
    )
    base = workflow.official_steps
    steps = [
        copy_model(base[0], branch_id="shared", node_ids=["node-b", "node-c"]),
        copy_model(
            base[1],
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-b",
        ),
        copy_model(
            base[2],
            node_ids=["node-c"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-c",
        ),
        copy_model(
            base[3],
            node_ids=["node-c"],
            depends_on_step_indexes=[2],
            branch_id="branch:node-c",
        ),
        copy_model(
            base[4],
            depends_on_step_indexes=[1, 3],
            branch_id="join",
            node_ids=["node-b", "node-c"],
            parameters=dict(RESTART_PARAMETERS),
        ),
    ]
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[STOP],
        step_executions=[workflow_step_execution(0, STOP)],
    )
    store.save_workflow(workflow)
    return incident, workflow


def test_an_unknown_outcome_branch_is_exhausted_and_its_siblings_finish_first():
    """The unknown-outcome branch stops; the other node's repair is still wanted.

    Failing the whole workflow at the end of the batch left node-c's reboot
    WAITING forever and its RESTORE_SCHEDULING never ran -- the node stayed
    cordoned behind node-b's mystery. An unknown outcome is treated like an
    exhausted ladder instead: node-b's branch is retired with no rung, node-c
    finishes and is released, the join is skipped, and only then does the
    workflow fail to the classifier, which hands node-b to an operator.
    """

    store = build_store()
    _, workflow = _job_dag_with_release(store)
    adapter = _ConfirmableNodeAdapter(
        {
            RESET: WorkflowStepOutcome.failed(
                "node agent node-b: attempt closed without a result",
                details={
                    "node_action_interrupted": True,
                    "manual_confirmation_required": True,
                    "operation": RESET.value,
                },
            ),
            **_ok(REBOOT, RESTART_JOB, *VALIDATIONS),
        },
        {(REBOOT, ("node-c",)): WorkflowStepOutcome.waiting(operation_id="reboot-c")},
    )
    executor = active_workflow_executor(store, [adapter], ALL_OPERATIONS)
    executor.branch_escalator = _escalator()

    first = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    assert first.status is not WorkflowStatus.FAILED, (
        "the workflow was failed while node-c's reboot was still in flight: "
        f"{first.status} calls={adapter.calls}"
    )
    assert first.waiting_step_index == 2, first

    second = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(
            expected_fencing_token=workflow.fencing_token,
            confirmed_adapter_operation_ids=["reboot-c"],
        ),
    )
    saved = store.get_workflow(workflow.request_id)

    assert "workflow-active/3/RESTORE_SCHEDULING" in adapter.calls, (
        f"node-c was never released: {adapter.calls}"
    )
    assert second.status is WorkflowStatus.BLOCKED, second.status
    assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert workflow_is_open(saved.status, saved.blocked_kind), (
        "finished siblings do not settle the unknown node's physical outcome"
    )
    assert all(
        call == "workflow-active/2/RESTART_NODE" or not call.endswith("/RESTART_NODE")
        for call in adapter.calls
    ), f"a reboot rung was appended for node-b: {adapter.calls}"
    assert all("RESTART_WORKLOAD" not in call for call in adapter.calls), adapter.calls
    assert saved.branch_escalation_counts == {}, saved.branch_escalation_counts
    assert [item.split("/")[0] for item in saved.exhausted_branch_ids] == [
        "branch:node-b"
    ], saved.exhausted_branch_ids
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None, "the classifier must still see the failure"
    assert classification[1] is RecoveryAction.ESCALATE_OPERATOR, classification[:3]


def test_a_job_level_step_failure_still_fails_the_whole_workflow():
    store = build_store()
    _, workflow = _job_dag(store, stop_done=False)
    outcomes = {
        STOP: WorkflowStepOutcome.failed("stop refused"),
        **_ok(RESET, REBOOT, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())

    assert result.status is WorkflowStatus.FAILED
    assert saved.branch_escalation_counts == {}
    assert adapter.calls == ["workflow-active/0/STOP_WORKLOADS"]


def test_without_an_escalator_the_old_failure_path_is_kept():
    store = build_store()
    _, workflow = _job_dag(store)
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused"),
        **_ok(REBOOT, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=None)

    assert result.status is WorkflowStatus.FAILED
    assert saved.branch_escalation_counts == {}
    assert adapter.calls == [
        "workflow-active/1/RESET_GPU",
        "workflow-active/2/RESTART_NODE",
    ]


class _PerNodeAdapter(FakeAdapter):
    """Outcomes keyed by (operation, node): the two branches of one job
    disagree about the same rung, which is the DESTR-014 shape."""

    def __init__(self, outcomes, per_node):
        super().__init__(outcomes)
        self.per_node = per_node

    def execute(self, context):
        self.calls.append(context.idempotency_key)
        key = (context.step.operation, tuple(context.step.node_ids))
        if key in self.per_node:
            return self.per_node[key]
        return self.outcomes[context.step.operation]


def test_a_sibling_finishing_normally_does_not_restart_a_job_whose_other_branch_exhausted():
    """DESTR-014 (agreed 2026-09-06): node-b RESET fails, its reboot succeeds
    and the branch ends normally; node-c's reboot fails, its warm-spare
    replacement fails, the branch is exhausted. Nothing timed out. The
    workflow fails safely and the job is *not* restarted on the healthy
    node-b + anything else -- the operator owns the outcome."""

    store = build_store()
    incident, workflow = _job_dag(store, node_c_operation=REBOOT)
    adapter = _PerNodeAdapter(
        {
            **_ok(STOP, RESET, REBOOT, RESTART_JOB, *VALIDATIONS),
            REPLACE: WorkflowStepOutcome.failed("insufficient healthy HyperPod spares"),
        },
        {
            (RESET, ("node-b",)): WorkflowStepOutcome.failed(
                "node agent node-b: GPU device clients are still active"
            ),
            (REBOOT, ("node-b",)): WorkflowStepOutcome.succeeded(),
            (REBOOT, ("node-c",)): WorkflowStepOutcome.failed(
                "step waited 300s past its limit; last reported: awaiting boot id"
            ),
        },
    )
    executor = active_workflow_executor(store, [adapter], ALL_OPERATIONS)
    executor.branch_escalator = _escalator()

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    saved = store.get_workflow(workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    assert saved.branch_escalation_counts == {"node-b": 1, "node-c": 1}
    assert len(saved.exhausted_branch_ids) == 1
    assert saved.exhausted_branch_ids[0].startswith("branch:node-c"), (
        saved.exhausted_branch_ids
    )
    assert not any("RESTART_WORKLOAD" in call for call in adapter.calls), adapter.calls
    node_b_ops = [
        step.operation
        for index, step in enumerate(saved.official_steps)
        if step.node_ids == ["node-b"] and index in saved.completed_step_indexes
    ]
    assert node_b_ops == [REBOOT, *VALIDATIONS], node_b_ops
    node_c_replace = [
        item
        for item in saved.step_executions
        if item.operation is REPLACE and item.status.value == "FAILED"
    ]
    assert len(node_c_replace) == 1, saved.step_executions
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


# ------------------------------------------------- single-node linear plans


def _single_node_plan(store, *, nodes=("node-a",)):
    """A plain (non-DAG) node record: STOP -> RESET_GPU -> VALIDATE_GPU ->
    RESTORE_SCHEDULING -> RESTART_WORKLOAD, every step on the same node(s)."""

    incident, workflow = workflow_state(
        store,
        [
            STOP,
            RESET,
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.RESTORE_SCHEDULING,
            RESTART_JOB,
        ],
    )
    workflow = copy_model(
        workflow,
        official_steps=[
            copy_model(step, node_ids=list(nodes)) for step in workflow.official_steps
        ],
    )
    store.save_workflow(workflow)
    return incident, workflow


def _operations(adapter) -> list[str]:
    return [call.rsplit("/", 1)[1] for call in adapter.calls]


def test_a_single_node_plan_escalates_its_failed_reset_in_the_same_record():
    """The ladder is not a multi-node privilege: a plain node record whose
    RESET_GPU fails grows into a one-branch DAG and reboots in place, instead
    of ending FAILED and handing the reboot to a workflow-reboot-after record
    (live 2026-09-23: the same fault class showed two record shapes)."""

    store = build_store()
    incident, workflow = _single_node_plan(store)
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused"),
        **_ok(STOP, REBOOT, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())

    assert result.status is WorkflowStatus.SUCCEEDED
    assert saved.dag_enabled is True, "the record grew into a DAG at the escalation"
    assert saved.branch_escalation_counts == {"node-a": 1}
    assert saved.exhausted_branch_ids == []
    assert {1, 2, 3} <= set(saved.superseded_step_indexes), (
        "the failed reset and the validation tail behind it are retired"
    )
    operations = _operations(adapter)
    assert operations.count("RESTART_NODE") == 1
    assert operations.count("RESTART_WORKLOAD") == 1
    assert operations.index("RESTART_NODE") < operations.index("RESTART_WORKLOAD"), (
        "the job restarts only after the node's reboot"
    )
    assert (
        operations.index("RESTART_NODE")
        < operations.index("VALIDATE_HOST")
        < (operations.index("RESTART_WORKLOAD"))
    ), "the rung's validation tail runs between the reboot and the restart"
    rungs = [event for event in saved.events if event.code == "BRANCH_ESCALATED"]
    assert len(rungs) == 1
    assert rungs[0].details["from_operation"] == "RESET_GPU"
    assert rungs[0].details["to_operation"] == "RESTART_NODE"
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED


def test_a_multi_node_linear_plan_still_fails_as_a_whole():
    """Only a record that belongs to one node is adopted; a flat record whose
    steps name several nodes keeps the whole-workflow failure path."""

    store = build_store()
    _, workflow = _single_node_plan(store, nodes=("node-a", "node-b"))
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused"),
        **_ok(STOP, REBOOT, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())

    assert result.status is WorkflowStatus.FAILED
    assert saved.dag_enabled is False
    assert "RESTART_NODE" not in _operations(adapter)
    assert saved.branch_escalation_counts == {}


def test_a_single_node_plan_with_an_unknown_outcome_is_not_adopted():
    """The same stop as the DAG ladder: an interrupted reset is an operator's
    call, so the record holds for confirmation as before and no rung is
    appended."""

    store = build_store()
    _, workflow = _single_node_plan(store)
    outcomes = {
        RESET: WorkflowStepOutcome.failed(
            "node agent node-a: interrupted",
            details={
                "node_action_interrupted": True,
                "manual_confirmation_required": True,
                "operation": RESET.value,
            },
        ),
        **_ok(STOP, REBOOT, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(store, workflow, outcomes, escalator=_escalator())

    assert result.status is WorkflowStatus.BLOCKED
    assert saved.dag_enabled is False
    assert "RESTART_NODE" not in _operations(adapter)
    assert saved.exhausted_branch_ids == []


QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
RELEASE = WorkflowOperation.RESTORE_SCHEDULING
UNOBSERVED_VALIDATION = {
    "node_pending": {
        "node-a": {
            "missing_or_stale_collectors": ["GPU_METRICS"],
            "quiesce_taint_present": True,
            "quiesce_taint_holders": {"workflow-active": "incident-00000000"},
            "reason": (
                "the node still carries the gpu-fault quiesce taint (NoExecute) of "
                "workflow-active; the exporter cannot run until it is removed -- "
                "restore the owner incident (gpu-fault-admin submit-remediation "
                "--disposition restore)"
            ),
        }
    },
    "pending_nodes": ["node-a"],
    "failed_nodes_observed": [],
    "node_failures_observed": {},
    "step_waiting_seconds": 603,
    "step_waiting_timeout_seconds": 600,
}


def _single_node_reset_chain(store):
    """The DESTR-025 record: one node, QUIESCE -> RESET -> RESTORE -> VALIDATE -> release."""

    return workflow_state(store, [QUIESCE, RESET, RESTORE, VALIDATIONS[0], RELEASE])


def _operations(adapter) -> list[str]:
    return [call.rsplit("/", 1)[1] for call in adapter.calls]


def test_a_reboot_rung_that_released_the_quiesce_taint_lets_the_validation_run():
    """Live 2026-10-07 (gpu-eks-cluster DESTR-025, attempt 9): QUIESCE tainted the
    node (eks-managed), RESET_GPU failed on the busy device, the in-place ladder
    retired the chain's RESTORE_GPU_SERVICES and ran RESTART_NODE -- which left
    the NoExecute taint on the rebooted node, so VALIDATE_GPU never saw the
    exporter and the branch climbed to REPLACE_NODE. The provider lifecycle
    adapter now records the taint release on the confirmed reboot
    (``quiesce_taint_removed_nodes``); the record's accounting balances and the
    validation tail runs on the same record, as it does on HyperPod."""

    from gpu_fault.workflow_quarantine import quiesce_taint_nodes_held

    store = build_store()
    incident, workflow = _single_node_reset_chain(store)
    outcomes = {
        QUIESCE: WorkflowStepOutcome.succeeded(
            details={"quiesce_taint_nodes": ["node-a"]}
        ),
        RESET: WorkflowStepOutcome.failed(
            "node agent node-a: GpuClientsActive: GPU device clients are still active"
        ),
        REBOOT: WorkflowStepOutcome.succeeded(
            details={
                "action": "REBOOT",
                "confirmation_source": "eks-running-and-new-agent-incarnation",
                "quiesce_taint_removed_nodes": ["node-a"],
            }
        ),
        **_ok(RESTORE, RESTART_JOB, *VALIDATIONS),
    }

    result, adapter, saved = _run(
        store,
        workflow,
        outcomes,
        escalator=_escalator(),
        operations=ALL_OPERATIONS | {QUIESCE, RESTORE},
    )

    assert result.status is WorkflowStatus.SUCCEEDED, (result, saved.events[-3:])
    operations = _operations(adapter)
    assert operations.index("RESTART_NODE") < operations.index("VALIDATE_GPU"), (
        operations
    )
    assert "REPLACE_NODE" not in operations, operations
    assert "RESTORE_GPU_SERVICES" not in operations, (
        "the reboot settled the quiesce; no restore is owed or compensated"
    )
    assert quiesce_taint_nodes_held(saved) == set(), (
        "the reboot record lifted the taint the QUIESCE record applied"
    )
    assert saved.branch_escalation_counts == {"node-a": 1}, (
        saved.branch_escalation_counts
    )
    assert saved.exhausted_branch_ids == [], saved.exhausted_branch_ids
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED


def test_a_validation_starved_by_the_products_own_taint_takes_no_replace_rung():
    """The plan-time guard for the same record when the taint is still there
    (a reboot confirmed before this fix, or an untaint that failed): the
    validation saw nothing, so no REPLACE_NODE is appended -- DESTR-025
    authorises no replace -- and the whole-workflow hand-off is the operator's
    restore, not a quarantine."""

    store = build_store()
    incident, workflow = _single_node_reset_chain(store)
    outcomes = {
        QUIESCE: WorkflowStepOutcome.succeeded(
            details={"quiesce_taint_nodes": ["node-a"]}
        ),
        RESET: WorkflowStepOutcome.failed(
            "node agent node-a: GpuClientsActive: GPU device clients are still active"
        ),
        REBOOT: WorkflowStepOutcome.succeeded(details={"action": "REBOOT"}),
        VALIDATIONS[0]: WorkflowStepOutcome.failed(
            "step 6/VALIDATE_GPU stayed non-terminal for 603s, past the 600s per-step cap",
            details=dict(UNOBSERVED_VALIDATION),
        ),
        **_ok(RESTORE, REPLACE, RESTART_JOB, *VALIDATIONS[1:]),
    }

    result, adapter, saved = _run(
        store,
        workflow,
        outcomes,
        escalator=_escalator(),
        operations=ALL_OPERATIONS | {QUIESCE, RESTORE},
    )

    operations = _operations(adapter)
    assert "REPLACE_NODE" not in operations, operations
    assert result.status is WorkflowStatus.FAILED, result
    assert str(saved.terminal_failure_reason).startswith(
        "node branch escalation exhausted: branch:node-a"
    ), saved.terminal_failure_reason
    exhausted = next(
        event
        for event in saved.events
        if str(event.code) == str(WorkflowEventCode.BRANCH_EXHAUSTED.value)
    )
    assert "could not observe the node" in str(exhausted.reason), exhausted.reason
    assert "submit-remediation --disposition restore" in str(exhausted.reason), (
        exhausted.reason
    )
    assert saved.branch_escalation_counts == {"node-a": 1}, (
        saved.branch_escalation_counts
    )
    classification = HardwareEscalationService.classify(saved)
    assert classification is not None, "the operator is told"
    stage, action, operation, flagged = classification
    assert stage == VALIDATION_UNOBSERVED_STAGE, stage
    assert (action, operation) == (
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
    )
    assert [item.operation for item in flagged] == [WorkflowOperation.VALIDATE_GPU]
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED
