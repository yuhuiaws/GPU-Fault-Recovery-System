"""A late fault after our own STOP_WORKLOADS still joins the job DAG.

The completion watcher records the product's own stop on its first missing pass
(37a83e5a): about a second after the stop the attempt's newest observation is a
STOPPED tombstone without containers. Attribution read PENDING/RUNNING
containers only, so a fault that arrived after it -- on another member node, or
a more severe one on the node under reset -- opened a node workflow of its own
instead of a branch of the job DAG that restarts the job onto that node
(PREEMPT-018). The eks-managed GF-REGIONAL-DESTR-014 split exactly so: node-b's
XID 46, 2.6 s after the tombstone, became a RESET_GPU workflow that waited out
the failure-domain budget the job workflow held. The workflow the tombstone
names supplies the allocation the tombstone dropped while it is PENDING/RUNNING,
its STOP covers the node and its join has not started (PREEMPT-020).

Timestamps are relative to the wall clock: the join check reads the workflow's
lifetime against it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest.faults import FaultIngestionService
from gpu_fault.app.ingest.workload_observations import ingest_workload_observation
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import (
    BlockedKind,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.orchestration.families.evidence import EvidenceOperationService
from gpu_fault.watcher import AttemptObservation, WorkloadPhase
from tests._builders import (
    attempt_observation,
    build_context,
    container_observation,
    copy_model,
    node_health_finding,
    workflow_step_execution,
)
from tests.orchestration._support import event

WORKLOAD = "training/pytorchjob/job-j"
NODES = ("node-b", "node-c")


def attempt(
    at: datetime,
    *,
    phase: WorkloadPhase = WorkloadPhase.RUNNING,
    containers: str = "live",
    initiator: str | None = None,
) -> AttemptObservation:
    """The two-node job as the watcher reports it: running, or stopped."""

    live = [
        container_observation(
            f"pod-{node}", f"worker-{node}", rank, node, gpu_uuids=[f"GPU-{node[-1]}"]
        )
        for rank, node in enumerate(NODES)
    ]
    return attempt_observation(
        "job-j",
        "attempt-a",
        at,
        expected_critical_ranks=len(NODES),
        workload_phase=phase,
        containers={
            "live": live,
            # The live watcher's tombstone (completion_observation._tombstone).
            "none": [],
            "terminated": [
                copy_model(item, terminated=True, exit_code=0) for item in live
            ],
        }[containers],
        workload_ids=[WORKLOAD],
        restart_budget=1,
        termination_initiator_incident_id=initiator,
        started_at=at - timedelta(minutes=3),
    )


def xid(
    context: ApplicationContext, code: int, node: str, at: datetime, event_id: str
) -> Any:
    """An XID as the kernel-log batch sends it: the workload state is the
    control plane's own resolution of the node at that instant, and it is
    ingested the moment it was observed."""

    resolved = context.topology.resolve("cluster-a", node, at)
    return copy_model(
        event(
            code,
            event_id=event_id,
            workload_state=WorkloadState(resolved.workload_state),
        ),
        node_id=node,
        gpu_uuid=f"GPU-{node[-1]}",
        observed_at=at,
        ingested_at=at,
        affected_workload_ids=list(resolved.workload_ids),
    )


def past_stop(
    context: ApplicationContext,
    workflow: WorkflowRequest,
    *,
    waiting: WorkflowOperation | None = None,
) -> WorkflowRequest:
    """The job workflow claimed and RUNNING with everything up to and
    including STOP_WORKLOADS SUCCEEDED, ``waiting`` still WAITING."""

    stop = next(
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    )
    done = list(range(stop + 1))
    executions = [
        workflow_step_execution(index, workflow.official_steps[index].operation)
        for index in done
    ]
    if waiting is not None:
        executions.append(
            workflow_step_execution(
                next(
                    index
                    for index, step in enumerate(workflow.official_steps)
                    if step.operation is waiting and index > stop
                ),
                waiting,
                WorkflowStepStatus.WAITING,
            )
        )
    running = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        not_before=None,
        execution_owner_id="executor-a",
        completed_step_indexes=done,
        step_executions=executions,
    )
    context.store.save_workflow(running)
    return running


def join(workflow: WorkflowRequest) -> tuple[int, list[str], Any, list[int]]:
    for index, step in enumerate(workflow.official_steps):
        if step.operation is WorkflowOperation.RESTART_WORKLOAD:
            return (
                index,
                list(step.node_ids),
                step.parameters.get("source_gpu_count"),
                list(step.depends_on_step_indexes),
            )
    raise AssertionError(f"no join in {workflow.request_id}")


def stopped_job(
    first: tuple[int, str],
    *,
    waiting: WorkflowOperation | None = None,
    tombstone: str = "none",
    started: timedelta = timedelta(seconds=40),
    tombstone_offset: timedelta = timedelta(seconds=12),
) -> tuple[ApplicationContext, FaultIngestionService, WorkflowRequest, datetime]:
    """The job runs on node-b and node-c since ``started`` ago, ``first``
    (XID, node) opens its job workflow, which gets past its shared
    STOP_WORKLOADS, and the watcher records the stop ``tombstone_offset`` after
    the start. Returns the stopped workflow and the tombstone's time."""

    context = build_context()
    service = FaultIngestionService(context)
    t0 = datetime.now(timezone.utc) - started
    ingest_workload_observation(context, attempt(t0))
    code, node = first
    decision = service.ingest_xid(
        xid(context, code, node, t0 + timedelta(seconds=5), f"xid{code}-{node}")
    )
    workflow = context.store.get_workflow(decision.workflow_request_id)
    assert join(workflow)[1] == list(NODES), workflow.official_steps
    workflow = past_stop(context, workflow, waiting=waiting)
    stopped_at = t0 + tombstone_offset
    ingest_workload_observation(
        context,
        attempt(
            stopped_at,
            phase=WorkloadPhase.STOPPED,
            containers=tombstone,
            initiator=workflow.incident_id,
        ),
    )
    return context, service, workflow, stopped_at


def test_a_late_fault_on_another_member_node_is_a_branch_of_the_job_dag() -> None:
    context, service, job, stopped_at = stopped_job((79, "node-c"))
    decision = service.ingest_xid(
        xid(context, 46, "node-b", stopped_at + timedelta(seconds=2.6), "xid46-b")
    )

    merged = context.store.get_workflow(job.request_id)
    assert decision.workflow_request_id == job.request_id, (
        "node-b's XID 46 opened a workflow of its own instead of joining the "
        "job DAG that stopped the job"
    )
    assert merged.dag_revision == job.dag_revision + 1
    branch = [
        index
        for index, step in enumerate(merged.official_steps)
        if step.branch_id == "branch:node-b"
    ]
    assert [merged.official_steps[index].operation for index in branch] == [
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.RESTORE_SCHEDULING,
    ]
    stop = next(
        index
        for index, step in enumerate(merged.official_steps)
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    )
    assert merged.official_steps[branch[0]].depends_on_step_indexes == [stop]
    # The join waits for both branches and still restarts the whole job: the
    # tombstone's empty allocation and GPU count 0 never reach it.
    _, nodes, gpu_count, depends = join(merged)
    assert nodes == list(NODES) and gpu_count == 2, merged.official_steps
    assert branch[-1] in depends and len(depends) == 2, depends
    assert context.store.get_incident(merged.incident_id).node_ids == sorted(NODES)
    # Batch 76 stays fixed: the stopped attempt owns no node.
    ownership = EvidenceOperationService(
        context.store, lambda *_args, **_kwargs: None
    ).ownership_metric_snapshot()
    assert ownership["current"].get(("cluster-a", "node-b"), 0) == 0, ownership


@pytest.mark.parametrize("tombstone", ["none", "terminated"])
def test_a_late_xid_79_on_the_node_under_reset_supersedes_the_reset(
    tombstone: str,
) -> None:
    context, service, job, stopped_at = stopped_job(
        (46, "node-b"),
        waiting=WorkflowOperation.QUIESCE_GPU_SERVICES,
        tombstone=tombstone,
    )
    reset = next(
        index
        for index, step in enumerate(job.official_steps)
        if step.operation is WorkflowOperation.RESET_GPU
    )
    decision = service.ingest_xid(
        xid(context, 79, "node-b", stopped_at + timedelta(seconds=1), "xid79-b")
    )

    merged = context.store.get_workflow(job.request_id)
    assert decision.workflow_request_id == job.request_id, (
        "the late XID 79 queued behind the reset instead of superseding it"
    )
    assert reset in merged.superseded_step_indexes, merged.superseded_step_indexes
    assert any(
        step.operation is WorkflowOperation.RESTART_NODE and step.node_ids == ["node-b"]
        for step in merged.official_steps
    ), merged.official_steps
    _, nodes, gpu_count, _ = join(merged)
    assert nodes == list(NODES) and gpu_count == 2, merged.official_steps


def _separate(
    context: ApplicationContext,
    service: FaultIngestionService,
    job: WorkflowRequest,
    late: Any,
) -> None:
    decision = service.ingest_xid(late)
    assert decision.workflow_request_id != job.request_id, decision
    assert context.store.get_workflow(job.request_id).dag_revision == job.dag_revision


@pytest.mark.parametrize(
    ("status", "blocked_kind"),
    [
        (WorkflowStatus.FAILED, None),
        (WorkflowStatus.BLOCKED, BlockedKind.NEEDS_OPERATOR),
    ],
)
def test_a_stopper_that_left_pending_or_running_owns_no_node(
    status: WorkflowStatus, blocked_kind: BlockedKind | None
) -> None:
    context, service, job, stopped_at = stopped_job((79, "node-c"))
    job = copy_model(job, status=status, blocked_kind=blocked_kind)
    context.store.save_workflow(job)
    _separate(
        context,
        service,
        job,
        xid(context, 46, "node-b", stopped_at + timedelta(seconds=2.6), "xid46-b"),
    )


def test_a_join_that_already_started_takes_no_branch() -> None:
    context, service, job, stopped_at = stopped_job((79, "node-c"))
    index = join(job)[0]
    job = copy_model(
        job,
        step_executions=[
            *job.step_executions,
            workflow_step_execution(
                index, WorkflowOperation.RESTART_WORKLOAD, WorkflowStepStatus.WAITING
            ),
        ],
    )
    context.store.save_workflow(job)
    _separate(
        context,
        service,
        job,
        xid(context, 46, "node-b", stopped_at + timedelta(seconds=2.6), "xid46-b"),
    )


def test_a_stop_that_never_ran_names_no_allocation() -> None:
    context, service, job, stopped_at = stopped_job((79, "node-c"))
    stop = next(
        index
        for index, step in enumerate(job.official_steps)
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    )
    job = copy_model(
        job,
        completed_step_indexes=[
            index for index in job.completed_step_indexes if index != stop
        ],
        step_executions=[
            item for item in job.step_executions if item.step_index != stop
        ],
    )
    context.store.save_workflow(job)
    _separate(
        context,
        service,
        job,
        xid(context, 46, "node-b", stopped_at + timedelta(seconds=2.6), "xid46-b"),
    )


def test_a_node_outside_the_stop_is_not_the_jobs() -> None:
    context, service, job, stopped_at = stopped_job((79, "node-c"))
    _separate(
        context,
        service,
        job,
        xid(context, 46, "node-x", stopped_at + timedelta(seconds=2.6), "xid46-x"),
    )


def test_an_event_another_job_owns_is_not_taken_by_the_stopped_one() -> None:
    context, service, job, stopped_at = stopped_job((79, "node-c"))
    late = copy_model(
        xid(context, 46, "node-b", stopped_at + timedelta(seconds=2.6), "xid46-b"),
        job_id="job-other",
        attempt_id="attempt-other",
    )
    assert (
        EvidenceOperationService(
            context.store, lambda *_args, **_kwargs: None
        ).stopped_attempt_observation(
            late, context.store.list_attempt_observations("cluster-a")
        )
        is None
    )
    _separate(context, service, job, late)


def test_a_tombstone_older_than_the_stopping_workflow_is_an_earlier_stop() -> None:
    # The job ran since 15 min ago and was stopped 10 min ago; the workflow
    # that names the stop's incident was compiled only now.
    context, service, job, _ = stopped_job(
        (79, "node-c"),
        started=timedelta(minutes=15),
        tombstone_offset=timedelta(minutes=5),
    )
    _separate(
        context,
        service,
        job,
        xid(
            context,
            46,
            "node-b",
            datetime.now(timezone.utc) - timedelta(seconds=10),
            "xid46-b",
        ),
    )


def test_health_findings_keep_their_own_attribution() -> None:
    context, _service, _job, stopped_at = stopped_job((79, "node-c"))
    finding = node_health_finding(
        "finding-node-b",
        "gpu-node-b",
        node_id="node-b",
        observed_at=stopped_at + timedelta(seconds=3),
        category=NodeHealthCategory.GPU,
        severity="critical",
        reason="replace node-b from warm spare",
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=["GPU-b"],
        runtime_profile_version="simulated-v1",
    )
    evidence = EvidenceOperationService(context.store, lambda *_args, **_kwargs: None)
    observations = context.store.list_attempt_observations("cluster-a")
    assert evidence.stopped_attempt_observation(finding, observations) is None
