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

from gpu_fault.models import WorkflowOperation
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

from gpu_fault.execution.branch_escalation import BranchEscalator  # noqa: E402
from gpu_fault.execution.models import WorkflowExecutionRequest  # noqa: E402
from gpu_fault.models import IncidentState, WorkflowStatus  # noqa: E402
from gpu_fault.orchestration.arbitration import RecoveryArbiter  # noqa: E402
from gpu_fault.orchestration.dag_branching import DagBrancher  # noqa: E402
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


def _run(store, workflow, outcomes, *, escalator):
    adapter = FakeAdapter(outcomes)
    executor = active_workflow_executor(store, [adapter], ALL_OPERATIONS)
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
