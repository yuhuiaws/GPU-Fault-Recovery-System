"""A read-only event that a running node-mutating action already covers is
recorded on the incident, not queued behind the workflow.

F-N1 / F-B5 sub-item (docs/review/F-N1-设计与实施计划.md §5). After the
aggregation window closed, an evidence-only candidate for a node whose reboot
was already under way was queued as a successor workflow: it waited for the
reboot to finish and then collected evidence from a freshly rebooted node --
useless work that also kept a PENDING record alive behind every BLOCKED
predecessor. The event still lands on the incident; nothing is scheduled.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpu_fault.models import WorkflowOperation, WorkflowRequest, WorkflowStatus
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.disposition import DispositionApplier
from gpu_fault.orchestration.families.faults import NodeScopedFaultService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import fault_incident, workflow_request, workflow_step

PAST = datetime.now(timezone.utc) - timedelta(minutes=5)


def _merger() -> WorkflowMergeService:
    return WorkflowMergeService(
        RecoveryArbiter(),
        DagBrancher(RecoveryArbiter()),
        preemption_enabled=True,
        workload_scoped_operations=set(),
        node_exclusive_operations=set(),
        workflow_resource_claims_by_node=lambda _workflow: {},
    )


def _applier() -> DispositionApplier:
    arbiter = RecoveryArbiter()
    return DispositionApplier(
        arbiter=arbiter,
        brancher=DagBrancher(arbiter),
        aggregation_deadlines=lambda now, _workflow: (now, now),
        prepare_preempting_successor=lambda _existing, successor: successor,
        preempt_parallel_job_branch=lambda existing, _candidate, _node: existing,
        workflow_preemption_enabled=True,
    )


def _rebooting(node_id: str, gpu_uuids: list[str] | None = None) -> WorkflowRequest:
    return workflow_request(
        "wf-reboot",
        "inc-a",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(
                WorkflowOperation.RESTART_NODE,
                node_ids=[node_id],
                gpu_uuids=gpu_uuids or [],
            ),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=[node_id],
                depends_on_step_indexes=[0],
            ),
        ],
        aggregation_max_deadline=PAST,
    )


def _read_only(node_id: str, gpu_uuids: list[str] | None = None) -> WorkflowRequest:
    return workflow_request(
        "wf-evidence",
        "inc-b",
        status=WorkflowStatus.PENDING,
        official_steps=[
            workflow_step(
                WorkflowOperation.FREEZE_EVIDENCE,
                node_ids=[node_id],
                gpu_uuids=gpu_uuids or [],
            ),
            workflow_step(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE, node_ids=[node_id]
            ),
        ],
    )


def test_a_read_only_event_covered_by_a_running_reboot_is_recorded_not_queued():
    merger = _merger()

    disposition = merger.disposition(
        _rebooting("node-a"), _read_only("node-a"), "node-a", set()
    )

    assert disposition == "ABSORB_RECORD_ONLY"
    assert merger.absorbed_record_only_total == 1


def test_a_read_only_event_outside_the_mutating_scope_still_queues():
    merger = _merger()
    resetting = workflow_request(
        "wf-reset",
        "inc-a",
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow_step(
                WorkflowOperation.RESET_GPU, node_ids=["node-a"], gpu_uuids=["GPU-1"]
            )
        ],
        aggregation_max_deadline=PAST,
    )

    other_gpu = merger.disposition(
        resetting, _read_only("node-a", ["GPU-2"]), "node-a", {"GPU-2"}
    )
    other_node = merger.disposition(
        _rebooting("node-a"), _read_only("node-b"), "node-b", set()
    )
    same_gpu = merger.disposition(
        resetting, _read_only("node-a", ["GPU-1"]), "node-a", {"GPU-1"}
    )

    assert other_gpu == "QUEUE_SUCCESSOR"
    assert other_node == "QUEUE_SUCCESSOR"
    assert same_gpu == "ABSORB_RECORD_ONLY"


def test_the_faults_family_applies_record_only_without_touching_the_workflow():
    existing_workflow = _rebooting("node-a")
    existing_incident = fault_incident(
        "inc-a", "event-a", workflow_request_id="wf-reboot"
    )
    candidate_incident = fault_incident(
        "inc-b", "event-b", workflow_request_id="wf-evidence"
    )

    workflow, winner = NodeScopedFaultService._apply_disposition(
        SimpleNamespace(dispositions=_applier()),
        "ABSORB_RECORD_ONLY",
        "node-a",
        candidate_incident,
        _read_only("node-a"),
        existing_incident,
        existing_workflow,
        set(),
        False,
        datetime.now(timezone.utc),
    )

    assert workflow is existing_workflow
    assert winner is existing_incident
