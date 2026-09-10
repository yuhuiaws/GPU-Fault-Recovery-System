"""Mixed-class faults on one training attempt share one workflow, node by node.

The attempt group key is ``cluster + job + attempt``; an access SXID
(``RESET_GPU``), a trunk SXID (``RESET_ALL_GPUS_NVSWITCHES``), XID 95
(``RESET_GPU``) and XID 79 (``RESTART_NODE``) on different nodes of the same
attempt therefore meet in one workflow. The XID family applies the merge
verdict through ``DispositionApplier`` -- a second node becomes a
``branch:<node>`` beside ``branch:initial`` with one ``shared`` STOP and one
``join`` RESTART. The SXID family used to rewrite the existing plan in place
with only the new event's reset (window open) or mint a successor workflow
with a second STOP/RESTART (window closed), losing the first node's reset
either way. Every case here asserts the shape both families must produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.operation_registry import MULTI_NODE_BARRIER_OPERATIONS
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.policy import GpuFaultPolicyEngine, SxidEvent, SxidLinkScope, XidEvent
from gpu_fault.store import SqliteStore
from gpu_fault.watcher import AttemptObservation
from tests._builders import attempt_observation, container_observation, copy_model
from tests.orchestration.test_multi_node_sxid import NOW, sxid

RESET_GPU = WorkflowOperation.RESET_GPU
RESET_ALL = WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
RESTART_NODE = WorkflowOperation.RESTART_NODE
STOP = WorkflowOperation.STOP_WORKLOADS
RESTART = WorkflowOperation.RESTART_WORKLOAD
RESET_OPERATIONS = {RESET_GPU, RESET_ALL, RESTART_NODE}

ACCESS_ACTION = "RESET_PARTICIPATING_GPUS"
TRUNK_ACTION = "RESET_ALL_GPUS_AND_NVSWITCHES"
XID95_ACTION = "RESET_GPU"
XID79_ACTION = "RESTART_BM"

FLAT: frozenset[str | None] = frozenset({None})
BRANCHED = frozenset({"branch:initial", "shared", "join", "branch:node-1"})
SUCCESSOR = frozenset({"branch:initial", "shared", "join", "branch:node-0:successor:1"})


def observation(node_count: int = 2) -> AttemptObservation:
    return attempt_observation(
        "train",
        "train-a001",
        NOW,
        expected_critical_ranks=node_count,
        containers=[
            container_observation(
                f"pod-{rank}",
                f"trainer-{rank}",
                rank,
                f"node-{rank}",
                gpu_uuids=[f"GPU-{rank}-0", f"GPU-{rank}-1"],
            )
            for rank in range(node_count)
        ],
        workload_ids=["training/pytorchjob/train"],
        restart_budget=3,
    )


def trunk(rank: int) -> SxidEvent:
    return sxid(rank, event_id=f"trunk-{rank}")


def access(rank: int) -> SxidEvent:
    return copy_model(
        sxid(rank),
        event_id=f"access-{rank}",
        link_scope=SxidLinkScope.ACCESS,
        link_scope_source="TRUSTED_NVSWITCH_TOPOLOGY",
        fabric_partition=None,
    )


def xid(rank: int, code: int) -> XidEvent:
    return XidEvent(
        event_id=f"xid{code}-node-{rank}",
        cluster_id="cluster-a",
        node_id=f"node-{rank}",
        observed_at=NOW + timedelta(seconds=1),
        xid=code,
        gpu_uuid=f"GPU-{rank}-0",
        product="H200",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=["training/pytorchjob/train"],
    )


def event_gpus(event: XidEvent | SxidEvent) -> list[str]:
    if isinstance(event, XidEvent):
        return [event.gpu_uuid] if event.gpu_uuid else []
    return list(event.participating_gpu_uuids)


def ingest(orchestrator: IncidentOrchestrator, event: XidEvent | SxidEvent):
    engine = GpuFaultPolicyEngine()
    decision = (
        engine.evaluate_xid(event)
        if isinstance(event, XidEvent)
        else engine.evaluate_sxid(event)
    )
    return orchestrator.ingest(event, decision)


def make_running(
    store: SqliteStore,
    workflow: WorkflowRequest,
    *,
    through: WorkflowOperation,
    inflight: WorkflowOperation | None = None,
) -> WorkflowRequest:
    """Complete every step up to and including ``through``; ``inflight`` WAITING."""
    now = datetime.now(timezone.utc)
    executions: list[WorkflowStepExecution] = []
    done: list[int] = []
    for index, step in enumerate(workflow.official_steps):
        executions.append(
            WorkflowStepExecution(
                step_index=index,
                operation=step.operation,
                status=WorkflowStepStatus.SUCCEEDED,
                started_at=now,
                updated_at=now,
            )
        )
        done.append(index)
        if step.operation is through:
            break
    else:
        raise AssertionError(f"{through.value} is not in the plan")
    if inflight is not None:
        waiting = next(
            index
            for index, step in enumerate(workflow.official_steps)
            if step.operation is inflight
        )
        executions.append(
            WorkflowStepExecution(
                step_index=waiting,
                operation=inflight,
                status=WorkflowStepStatus.WAITING,
                started_at=now,
                updated_at=now,
            )
        )
    running = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-1",
        execution_lease_expires_at=now + timedelta(minutes=5),
        step_executions=executions,
        completed_step_indexes=done,
        not_before=now - timedelta(seconds=60),
        aggregation_max_deadline=now - timedelta(seconds=30),
    )
    store.save_workflow(running)
    return running


Resets = dict[str, frozenset[WorkflowOperation]]


def owes(**by_node: WorkflowOperation | tuple[WorkflowOperation, ...]) -> Resets:
    """``owes(node_0=RESET_GPU, node_1=RESET_ALL)`` -> the live reset steps per node."""
    return {
        node.replace("_", "-"): frozenset(
            operations if isinstance(operations, tuple) else (operations,)
        )
        for node, operations in by_node.items()
    }


@dataclass(frozen=True)
class Case:
    name: str
    events: tuple[XidEvent | SxidEvent, ...]
    official_action: str
    resets: Resets
    branches: frozenset[str | None]
    token_bump: int = 0
    node_count: int = 2
    # Window-closed cases: run the first event, then mark its workflow RUNNING
    # with the steps through ``through`` SUCCEEDED (and ``inflight`` WAITING).
    through: WorkflowOperation | None = None
    inflight: WorkflowOperation | None = None

    def __str__(self) -> str:
        return self.name


def _closed(
    name: str,
    first: XidEvent | SxidEvent,
    second: XidEvent | SxidEvent,
    official_action: str,
    resets: Resets,
    branches: frozenset[str | None],
    *,
    reset_waiting_resets: Resets | None = None,
) -> list[Case]:
    """One case per window-closed phase: after FREEZE, after STOP, reset WAITING."""
    first_reset = RESET_ALL if first.event_id.startswith("trunk") else RESET_GPU
    return [
        Case(
            f"{name}[after_freeze]",
            (first, second),
            official_action,
            resets,
            branches,
            through=WorkflowOperation.FREEZE_EVIDENCE,
        ),
        Case(
            f"{name}[after_stop]",
            (first, second),
            official_action,
            resets,
            branches,
            through=STOP,
        ),
        Case(
            f"{name}[reset_waiting]",
            (first, second),
            official_action,
            reset_waiting_resets or resets,
            branches,
            through=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            inflight=first_reset,
        ),
    ]


WINDOW_OPEN = [
    # Same class on two nodes is a branch per node too: the regional executor
    # has no barrier coordinator and fail-closed holds any reset step naming
    # more than one node, so no SXID merge may widen a reset across nodes.
    Case(
        "open:trunk0->trunk1",
        (trunk(0), trunk(1)),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL, node_1=RESET_ALL),
        BRANCHED,
    ),
    Case(
        "open:access0->access1",
        (access(0), access(1)),
        ACCESS_ACTION,
        owes(node_0=RESET_GPU, node_1=RESET_GPU),
        BRANCHED,
    ),
    Case(
        "open:access0->trunk1",
        (access(0), trunk(1)),
        TRUNK_ACTION,
        owes(node_0=RESET_GPU, node_1=RESET_ALL),
        BRANCHED,
    ),
    Case(
        "open:trunk0->access1",
        (trunk(0), access(1)),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL, node_1=RESET_GPU),
        BRANCHED,
    ),
    Case(
        "open:xid95_0->trunk1",
        (xid(0, 95), trunk(1)),
        TRUNK_ACTION,
        owes(node_0=RESET_GPU, node_1=RESET_ALL),
        BRANCHED,
    ),
    Case(
        "open:trunk0->xid95_1",
        (trunk(0), xid(1, 95)),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL, node_1=RESET_GPU),
        BRANCHED,
    ),
    Case(
        "open:xid79_0->trunk1",
        (xid(0, 79), trunk(1)),
        XID79_ACTION,
        owes(node_0=RESTART_NODE, node_1=RESET_ALL),
        BRANCHED,
    ),
    Case(
        "open:xid95_0->access1",
        (xid(0, 95), access(1)),
        XID95_ACTION,
        owes(node_0=RESET_GPU, node_1=RESET_GPU),
        BRANCHED,
    ),
    Case(
        "open:same-node xid95_0->trunk0 replaces in place",
        (xid(0, 95), trunk(0)),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL),
        FLAT,
        token_bump=1,
    ),
    Case(
        "open:same-node trunk0->xid95_0 is absorbed",
        (trunk(0), xid(0, 95)),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL),
        FLAT,
    ),
    Case(
        "open:access0->trunk1->access2",
        (access(0), trunk(1), access(2)),
        TRUNK_ACTION,
        owes(node_0=RESET_GPU, node_1=RESET_ALL, node_2=RESET_GPU),
        BRANCHED | {"branch:node-2"},
        node_count=3,
    ),
]

WINDOW_CLOSED = [
    *_closed(
        "closed:trunk0->trunk1",
        trunk(0),
        trunk(1),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL, node_1=RESET_ALL),
        BRANCHED,
    ),
    *_closed(
        "closed:trunk0->access1",
        trunk(0),
        access(1),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL, node_1=RESET_GPU),
        BRANCHED,
    ),
    *_closed(
        "closed:xid95_0->trunk1",
        xid(0, 95),
        trunk(1),
        TRUNK_ACTION,
        owes(node_0=RESET_GPU, node_1=RESET_ALL),
        BRANCHED,
    ),
    *_closed(
        "closed:trunk0->xid95_1",
        trunk(0),
        xid(1, 95),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL, node_1=RESET_GPU),
        BRANCHED,
    ),
    *_closed(
        "closed:same-node trunk0->xid79_0 preempts or queues behind the reset",
        trunk(0),
        xid(0, 79),
        XID79_ACTION,
        owes(node_0=RESTART_NODE),
        SUCCESSOR,
        reset_waiting_resets=owes(node_0=(RESET_ALL, RESTART_NODE)),
    ),
    *_closed(
        "closed:same-node trunk0->access0 is absorbed",
        trunk(0),
        access(0),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL),
        FLAT,
    ),
]


def _frozen_steps(workflow: WorkflowRequest) -> dict[int, tuple]:
    """Completed and in-flight steps, by index, in the shape that may not change."""
    indexes = set(workflow.completed_step_indexes) | {
        execution.step_index for execution in workflow.step_executions
    }
    return {
        index: (
            step.operation,
            tuple(step.node_ids),
            tuple(step.gpu_uuids),
            step.parameters,
        )
        for index, step in enumerate(workflow.official_steps)
        if index in indexes
    }


def _gpus_for(step, node_id: str) -> list[str]:
    mapping = step.parameters.get("gpu_uuids_by_node")
    if isinstance(mapping, dict):
        assert node_id in mapping, (
            f"{step.operation.value} names {node_id} without a GPU mapping: {mapping}"
        )
        return list(mapping[node_id])
    return list(step.gpu_uuids)


def _run(case: Case, tmp_path):
    store = SqliteStore(str(tmp_path / "dispositions.db"))
    store.save_profile(default_simulated_profile())
    store.save_attempt_observation(observation(case.node_count))
    orchestrator = IncidentOrchestrator(store, multi_node_aggregation_window_seconds=30)
    try:
        incident, workflow = ingest(orchestrator, case.events[0])
        if case.through is not None:
            workflow = make_running(
                store, workflow, through=case.through, inflight=case.inflight
            )
        frozen = _frozen_steps(workflow)
        token_before = workflow.fencing_token
        incident_ids = {incident.incident_id}
        for event in case.events[1:]:
            incident, _ = ingest(orchestrator, event)
            incident_ids.add(incident.incident_id)
        workflows = store.list_workflows()
        merged = store.get_workflow(incident.workflow_request_id)
        return incident_ids, workflows, merged, incident, frozen, token_before
    finally:
        store.close()


@pytest.mark.parametrize("case", [*WINDOW_OPEN, *WINDOW_CLOSED], ids=str)
def test_every_fault_node_keeps_its_own_reset_in_one_workflow(case, tmp_path):
    incident_ids, workflows, workflow, incident, frozen, token_before = _run(
        case, tmp_path
    )
    all_nodes = [f"node-{rank}" for rank in range(case.node_count)]
    gpus_by_node = {
        event.node_id: sorted(event_gpus(event)) for event in reversed(case.events)
    }
    for event in case.events:
        # A node hit twice keeps the union of both events' GPUs.
        gpus_by_node[event.node_id] = sorted(
            set(gpus_by_node[event.node_id]) | set(event_gpus(event))
        )

    assert incident_ids == {incident.incident_id}, "one attempt, one incident"
    assert [item.request_id for item in workflows] == [workflow.request_id], (
        "one attempt, one workflow: no successor record and no orphan"
    )

    reset_steps = [
        (index, step)
        for index, step in enumerate(workflow.official_steps)
        if step.operation in RESET_OPERATIONS
        and index not in workflow.superseded_step_indexes
    ]
    for node_id, operations in case.resets.items():
        for operation in operations:
            owned = [
                step
                for _, step in reset_steps
                if step.operation is operation and node_id in step.node_ids
            ]
            assert owned, (
                f"{node_id} lost its {operation.value}: "
                f"{[(s.operation.value, list(s.node_ids)) for _, s in reset_steps]}"
            )
            if operation is not RESTART_NODE:
                assert all(
                    set(gpus_by_node[node_id]) <= set(_gpus_for(step, node_id))
                    for step in owned
                ), f"{node_id} reset does not cover its own GPUs"
    for _, step in reset_steps:
        for node_id in step.node_ids:
            assert step.operation in case.resets.get(node_id, frozenset()), (
                f"{step.operation.value} targets {node_id}, which owes "
                f"{case.resets.get(node_id)}: a merge widened the wrong step"
            )

    for operation in (STOP, RESTART):
        steps = [
            step for step in workflow.official_steps if step.operation is operation
        ]
        assert len(steps) == 1, f"exactly one {operation.value}, got {len(steps)}"
        assert steps[0].node_ids == all_nodes, f"{operation.value} spans the attempt"

    assert _frozen_steps(workflow) | frozen == _frozen_steps(workflow), (
        "a completed or in-flight step changed under the agent that holds it"
    )
    for index, before in frozen.items():
        assert _frozen_steps(workflow)[index] == before, f"step {index} was rewritten"

    assert incident.node_ids == sorted({event.node_id for event in case.events})
    assert incident.official_action == case.official_action, (
        "the incident action is the higher recovery rank, never a later downgrade"
    )
    assert workflow.fencing_token == token_before + case.token_bump
    assert incident.fencing_token == workflow.fencing_token

    branch_ids = {step.branch_id for step in workflow.official_steps}
    assert branch_ids == case.branches, f"branch ids {branch_ids}"
    if case.branches != FLAT and not workflow.superseded_step_indexes:
        # Every live branch tail feeds the one join (a preempted branch chains
        # its successor behind what already ran; the join follows that tail).
        join = next(
            (index, step)
            for index, step in enumerate(workflow.official_steps)
            if step.branch_id == "join"
        )
        for branch_id in case.branches - {"shared", "join"}:
            tail = max(
                index
                for index, step in enumerate(workflow.official_steps)
                if step.branch_id == branch_id
                and index not in workflow.superseded_step_indexes
            )
            assert tail in join[1].depends_on_step_indexes, (
                f"the join does not wait for {branch_id}"
            )


def _case(name: str) -> Case:
    return next(case for case in [*WINDOW_OPEN, *WINDOW_CLOSED] if case.name == name)


def test_sxid_branch_carries_the_sxid_step_scope(tmp_path):
    """A branch the SXID family appends keeps the SXID-only parameters."""
    _, _, workflow, _, _, _ = _run(_case("open:access0->trunk1"), tmp_path)
    branch = [
        step for step in workflow.official_steps if step.branch_id == "branch:node-1"
    ]
    reset = next(step for step in branch if step.operation is RESET_ALL)
    assert reset.parameters["gpu_uuids_by_node"] == {"node-1": ["GPU-1-0", "GPU-1-1"]}
    assert reset.parameters["sxids_by_node"] == {"node-1": [11001]}
    assert reset.parameters["fabric_partitions_by_node"] == {
        "node-1": "cluster-a/node-1/local-nvswitch"
    }
    assert reset.parameters["fabric_partition"] == "cluster-a/node-1/local-nvswitch"
    assert reset.parameters["sxid"] == 11001
    assert reset.execution_owner is not None
    bundle = next(
        step
        for step in branch
        if step.operation is WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
    )
    assert bundle.parameters["classification"] == "FATAL"
    assert bundle.parameters["classification_source"] == (
        "NVIDIA_FABRIC_MANAGER_CATALOG"
    )
    assert bundle.parameters["sxids_by_node"] == {"node-1": [11001]}
    initial = [
        step for step in workflow.official_steps if step.branch_id == "branch:initial"
    ]
    first_reset = next(step for step in initial if step.operation is RESET_GPU)
    assert first_reset.parameters["gpu_uuids_by_node"] == {
        "node-0": ["GPU-0-0", "GPU-0-1"]
    }, "the first node's branch does not take the second node's GPUs"


XID_MIRRORS = {
    # SXID-family case -> the XID-family case with the same node/class layout.
    "open:trunk0->access1": Case(
        "mirror:trunk0->xid95_1",
        (trunk(0), xid(1, 95)),
        TRUNK_ACTION,
        owes(node_0=RESET_ALL, node_1=RESET_GPU),
        BRANCHED,
    ),
    "open:trunk0->trunk1": Case(
        "mirror:xid79_0->xid79_1",
        (xid(0, 79), xid(1, 79)),
        XID79_ACTION,
        owes(node_0=RESTART_NODE, node_1=RESTART_NODE),
        BRANCHED,
    ),
    "open:access0->access1": Case(
        "mirror:xid95_0->xid95_1",
        (xid(0, 95), xid(1, 95)),
        XID95_ACTION,
        owes(node_0=RESET_GPU, node_1=RESET_GPU),
        BRANCHED,
    ),
}


def _dag_shape(workflow: WorkflowRequest) -> tuple:
    by_branch: dict[str | None, list[str]] = {}
    for step in workflow.official_steps:
        by_branch.setdefault(step.branch_id, []).append(step.operation.value)
    resets_per_branch = {
        branch_id: sum(
            1
            for step in workflow.official_steps
            if step.branch_id == branch_id and step.operation in RESET_OPERATIONS
        )
        for branch_id in by_branch
        if branch_id not in {"shared", "join"}
    }
    return (
        set(by_branch),
        by_branch["shared"],
        by_branch["join"],
        resets_per_branch,
        workflow.dag_revision,
    )


@pytest.mark.parametrize("name", sorted(XID_MIRRORS), ids=str)
def test_sxid_and_xid_families_branch_symmetrically(name, tmp_path):
    """Mixed and same-class SXID pairs take the DAG shape the XID family gives."""
    mirror = XID_MIRRORS[name]
    _, _, sxid_workflow, _, _, _ = _run(_case(name), tmp_path / "sxid")
    _, _, xid_workflow, _, _, _ = _run(mirror, tmp_path / "xid")
    assert _dag_shape(sxid_workflow) == _dag_shape(xid_workflow), (
        f"{name} and {mirror.name} disagree on the DAG shape"
    )


def test_no_sxid_merge_leaves_a_reset_step_spanning_nodes(tmp_path):
    """The regional executor has no barrier coordinator: a barrier operation
    naming more than one node is held fail-closed until the workflow's lifetime
    runs out. No merge in the matrix may produce such a step."""
    for number, case in enumerate([*WINDOW_OPEN, *WINDOW_CLOSED]):
        _, _, workflow, _, _, _ = _run(case, tmp_path / f"case-{number}")
        for step in workflow.official_steps:
            if step.operation in MULTI_NODE_BARRIER_OPERATIONS:
                assert len(step.node_ids) == 1, (
                    f"{case.name}: {step.operation.value} names {list(step.node_ids)}"
                )
