from __future__ import annotations

import pytest

from gpu_fault.models import (
    BlockedKind,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.disposition import Disposition
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.orchestration._cov95_closure_extra_safety import (
    closure_extra_isolation as closure_extra_isolation,
)
from tests.orchestration._cov95_closure_extra_support import merger

RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
STOP = WorkflowOperation.STOP_WORKLOADS
RESTART = WorkflowOperation.RESTART_WORKLOAD
FREEZE = WorkflowOperation.FREEZE_EVIDENCE


def job_workflow(identity, *, operation=RESET, node="node-a", dag=False):
    steps = [
        workflow_step(
            STOP,
            node_ids=["node-a", "node-b"],
            workload_ids=["training/job/unit-job"],
            branch_id="shared" if dag else None,
        ),
        workflow_step(
            operation,
            node_ids=[node],
            gpu_uuids=["GPU-a"],
            depends_on_step_indexes=[0] if dag else [],
            branch_id="repair-a" if dag else None,
            branch_node_ids=[node] if dag else [],
        ),
        workflow_step(
            RESTART,
            node_ids=["node-a", "node-b"],
            workload_ids=["training/job/unit-job"],
            depends_on_step_indexes=[1] if dag else [],
            branch_id="join" if dag else None,
        ),
    ]
    return workflow_request(
        identity, "unit-incident", official_steps=steps, dag_enabled=dag
    )


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        ({}, Disposition.REPLACE_IN_PLACE),
        ({"completed_step_indexes": [0]}, Disposition.QUEUE_SUCCESSOR),
        ({"superseded_step_indexes": [0]}, Disposition.QUEUE_SUCCESSOR),
        ({"execution_owner_id": "unit-executor"}, Disposition.QUEUE_SUCCESSOR),
        (
            {
                "step_executions": [
                    workflow_step_execution(0, RESET, WorkflowStepStatus.WAITING)
                ]
            },
            Disposition.QUEUE_SUCCESSOR,
        ),
    ],
    ids=["never-ran", "completed", "superseded", "owned", "waiting"],
)
def test_operator_block_recompiles_in_place_only_before_any_execution(
    progress, expected
):
    existing = workflow_request(
        "blocked",
        "unit-incident",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        official_steps=[workflow_step(RESET, gpu_uuids=["GPU-a"])],
        **progress,
    )
    candidate = workflow_request(
        "candidate", "unit-incident", official_steps=[workflow_step(REBOOT)]
    )
    before = existing.model_dump(mode="json")
    verdict = merger().disposition(existing, candidate, "node-a", {"GPU-a"})
    assert verdict is expected, (
        f"operator-block execution history chose {verdict} instead of {expected}"
    )
    assert existing.model_dump(mode="json") == before, (
        "deciding whether a blocked plan can be replaced changed the incumbent"
    )


@pytest.mark.parametrize("missing_side", ["existing", "candidate"])
def test_parallel_merge_requires_a_physical_node_action_on_both_sides(missing_side):
    existing = job_workflow("existing")
    candidate = job_workflow("candidate", node="node-b", operation=REBOOT)
    if missing_side == "existing":
        existing = existing.model_copy(
            update={"official_steps": [workflow_step(STOP), workflow_step(RESTART)]}
        )
    else:
        candidate = candidate.model_copy(
            update={"official_steps": [workflow_step(STOP), workflow_step(RESTART)]}
        )
    assert merger().can_append_parallel_branch(existing, candidate) is False, (
        f"a workload-only {missing_side} created a physical recovery branch"
    )


def test_node_only_plan_cannot_accept_a_parallel_job_branch_without_a_restart_join():
    existing = workflow_request(
        "node-only", "unit-incident", official_steps=[workflow_step(RESET)]
    )
    candidate = job_workflow("candidate", node="node-b", operation=REBOOT)
    assert merger().can_append_parallel_branch(existing, candidate) is False, (
        "parallel recovery was admitted without a workload restart join"
    )


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
def test_preempting_an_unrepresented_node_preserves_the_old_branch_and_adds_the_new_one(
    dag,
):
    existing = job_workflow("existing", dag=dag).model_copy(
        update={
            "status": WorkflowStatus.RUNNING,
            "execution_owner_id": "unit-executor",
            "completed_step_indexes": [0],
        }
    )
    candidate = job_workflow("candidate", node="node-b", operation=REBOOT)
    service = merger()
    service.preemption_enabled = False
    before = existing.model_dump(mode="json")

    verdict = service.disposition(existing, candidate, "node-b", {"GPU-b"})
    combined = service.preempt_parallel_branch(existing, candidate, "node-b")

    assert verdict is Disposition.QUEUE_BRANCH_SUCCESSOR, (
        f"an unrepresented fault node vanished into the existing DAG: {verdict}"
    )
    assert combined.request_id == existing.request_id and combined.dag_enabled, (
        "adding the previously absent node lost the job workflow's identity"
    )
    assert any(
        step.operation is RESET and step.node_ids == ["node-a"]
        for step in combined.official_steps
    ), "the unrelated original node repair disappeared"
    assert any(
        step.operation is REBOOT and step.node_ids == ["node-b"]
        for step in combined.official_steps
    ), "the new node's required reboot was not scheduled"
    assert all(
        combined.official_steps[index].operation is not RESET
        for index in combined.superseded_step_indexes
    ), "a new node's branch superseded the unrelated incumbent repair"
    assert any(
        event.kind is WorkflowEventKind.PLAN_REWRITE for event in combined.events
    ), "the new recovery branch has no plan-rewrite audit"
    assert existing.model_dump(mode="json") == before, (
        "branch planning modified the executor's incumbent snapshot"
    )


def test_named_dag_branch_absorbs_a_weaker_same_node_action_without_rewriting():
    existing = job_workflow("existing", operation=REBOOT, dag=True).model_copy(
        update={"status": WorkflowStatus.RUNNING, "completed_step_indexes": [0]}
    )
    candidate = job_workflow("candidate", operation=RESET, node="node-a")
    before = existing.model_dump(mode="json")
    verdict = merger().disposition(existing, candidate, "node-a", {"GPU-a"})
    assert verdict is Disposition.ABSORB, (
        f"a live node-wide reboot did not absorb the same node's reset: {verdict}"
    )
    assert existing.model_dump(mode="json") == before, (
        "an absorbed fault rewrote the live DAG during classification"
    )


@pytest.mark.parametrize("progress", ["pending", "completed", "superseded", "waiting"])
def test_read_only_action_scope_uses_actual_step_history_to_decide_whether_it_started(
    progress,
):
    values = {}
    if progress == "completed":
        values["completed_step_indexes"] = [0]
    elif progress == "superseded":
        values["superseded_step_indexes"] = [0]
    elif progress == "waiting":
        values["step_executions"] = [
            workflow_step_execution(0, FREEZE, WorkflowStepStatus.WAITING)
        ]
    workflow = workflow_request(
        "read-only", "unit-incident", official_steps=[workflow_step(FREEZE)], **values
    )
    service = merger()
    assert service.recovery_action_has_started(workflow, [0]) is (
        progress != "pending"
    ), f"the read-only step's {progress} history was ignored when judging mutability"
    assert service.recovery_action_has_started(workflow, []) is False, (
        "an empty selected action scope inherited unrelated workflow progress"
    )
