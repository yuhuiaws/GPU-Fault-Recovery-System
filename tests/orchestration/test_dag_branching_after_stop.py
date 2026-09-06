"""A candidate step that lands after the shared STOP_WORKLOADS cannot capture
live process state, whichever branch attaches it (F-C4, log 60 item 3).

``_attach_candidate_branch`` marks such a step unavailable with a reason when
the existing STOP has already run; when the STOP was still open it appended the
suffix untouched, so a collection the candidate placed after its own STOP was
still told to capture processes that the shared STOP had ended.
"""

from __future__ import annotations

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepSpec
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from tests._builders import workflow_request, workflow_step

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
RESTART = WorkflowOperation.RESTART_WORKLOAD
COLLECT = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE


def _existing(*, stop_completed: bool):
    completed = [0, 1] if stop_completed else []
    return workflow_request(
        "workflow-existing",
        "incident-a",
        WorkflowStatus.RUNNING,
        1,
        runtime_profile_version="simulated-v1",
        official_action=RESET.value,
        official_steps=[
            workflow_step(FREEZE, node_ids=["node-a"]),
            workflow_step(STOP, node_ids=["node-a"], workload_ids=["job-a"]),
            workflow_step(RESET, node_ids=["node-a"]),
            workflow_step(RESTART, node_ids=["node-a"], workload_ids=["job-a"]),
        ],
        completed_step_indexes=completed,
        completed_operations=[FREEZE, STOP] if stop_completed else [],
    )


def _candidate_collecting_after_its_stop():
    return workflow_request(
        "workflow-candidate",
        "incident-a",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action=RESET.value,
        official_steps=[
            workflow_step(STOP, node_ids=["node-b"], workload_ids=["job-a"]),
            workflow_step(
                COLLECT, node_ids=["node-b"], parameters={"capture_process_state": True}
            ),
            workflow_step(RESET, node_ids=["node-b"]),
        ],
    )


def _appended_collection(combined, existing) -> WorkflowStepSpec:
    appended = combined.official_steps[len(existing.official_steps) :]
    return next(step for step in appended if step.operation is COLLECT)


@pytest.mark.parametrize(
    "stop_completed", [False, True], ids=["stop-still-open", "stop-already-ran"]
)
def test_a_collection_after_the_stop_is_marked_unavailable(
    stop_completed: bool,
) -> None:
    existing = _existing(stop_completed=stop_completed)
    brancher = DagBrancher(RecoveryArbiter())

    combined = brancher.append_parallel_job_branch(
        existing, _candidate_collecting_after_its_stop()
    )

    collection = _appended_collection(combined, existing)
    assert collection.parameters.get("capture_process_state") is False
    assert collection.parameters.get("running_process_evidence_unavailable") is True
    assert "STOP_WORKLOADS" in str(
        collection.parameters.get("running_process_evidence_reason")
    )
