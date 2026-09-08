"""The retired-generation plan: what it refuses, and how it bounds discovery.

``build_retired_generation_plan`` is the read-only report the DESTR-017 runner
records; the revocation itself is the dispatcher sweep's. ``cancellable`` was
decided by a prose prefix (P2-72H); discovery refused any fleet with more than a
thousand open workflows (P1-72G); a containment-only step was treated as
destructive (F-B4); a warning in ``blocked_reasons`` flipped the pending step
set (F-B4). Each case here fails on the old shape and pins the new one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.retired_generation import (
    OPEN_REMOTE_COMMANDS_CODE,
    build_retired_generation_plan,
    completed_destructive_operations,
    pending_destructive_operations,
    retired_generation_blockers,
    retired_generation_candidates,
)
from tests._builders import build_store, fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 6, 8, 0, tzinfo=timezone.utc)
CLUSTER = "cluster-a"
NODES = ["node-a", "node-b", "node-c"]
STEPS = [
    workflow_step(WorkflowOperation.FREEZE_EVIDENCE, node_ids=NODES),
    workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=NODES),
]


def retired_pair(
    store: Any,
    suffix: str,
    *,
    command_status: RemoteCommandStatus | None = RemoteCommandStatus.WAITING,
) -> tuple[str, str, str]:
    """One RUNNING generation-1 record under an incident at generation 4."""

    incident_id = f"inc-{suffix}"
    retired_id = f"workflow-retired-{suffix}"
    current_id = f"workflow-current-{suffix}"
    incident = fault_incident(
        incident_id,
        f"event-{suffix}",
        cluster_id=CLUSTER,
        node_ids=list(NODES),
        state=IncidentState.ACTION_PENDING,
        fencing_token=4,
        workflow_request_id=current_id,
    )
    store.save_incident(incident)
    retired = workflow_request(
        retired_id,
        incident_id,
        status=WorkflowStatus.RUNNING,
        fencing_token=1,
        official_action="RESTART_APP",
        official_steps=STEPS,
        execution_owner_id="executor-b",
        execution_lease_expires_at=NOW + timedelta(minutes=3),
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.FREEZE_EVIDENCE],
        updated_at=NOW - timedelta(minutes=1),
    )
    store.save_workflow(retired)
    store.save_workflow(
        workflow_request(
            current_id,
            incident_id,
            fencing_token=4,
            official_action="RUN_DIAGNOSTICS",
            official_steps=[workflow_step(WorkflowOperation.VALIDATE_GPU)],
        )
    )
    if command_status is not None:
        store.ensure_remote_command(
            RemoteActionCommand(
                command_id=f"command-{suffix}",
                cluster_id=CLUSTER,
                workflow_request_id=retired_id,
                incident_id=incident_id,
                step_index=1,
                fencing_token=1,
                idempotency_key=f"{retired_id}/1/STOP_WORKLOADS",
                step=STEPS[1],
                workflow=retired,
                incident=incident,
                status=command_status,
            )
        )
    return incident_id, retired_id, current_id


def test_cancellable_is_decided_by_a_structured_blocker_code() -> None:
    store = build_store()
    _incident_id, retired_id, _current_id = retired_pair(store, "a")
    plan = build_retired_generation_plan(store, [retired_id], now=NOW)
    item = plan["items"][0]

    assert item["blocker_codes"] == [OPEN_REMOTE_COMMANDS_CODE]
    assert item["cancellable"] is True

    retired = store.get_workflow(retired_id)
    store.save_workflow(
        retired.model_copy(
            update={"completed_operations": [WorkflowOperation.STOP_WORKLOADS]}
        )
    )
    blockers = retired_generation_blockers(
        store.get_workflow(retired_id),
        store.get_workflow("workflow-current-a"),
        store.list_remote_commands(workflow_request_ids=[retired_id]),
    )
    codes = [blocker.code for blocker in blockers]
    assert "completed_destructive_operations" in codes
    assert OPEN_REMOTE_COMMANDS_CODE in codes
    again = build_retired_generation_plan(store, [retired_id], now=NOW)
    assert again["items"][0]["cancellable"] is False


def test_discovery_bounds_the_scan_instead_of_refusing_a_busy_fleet() -> None:
    """The candidate status set is every open workflow, not the suspect ones.

    A busy fleet has more than a thousand PENDING rows in steady state, so the
    old ceiling made discovery permanently unavailable exactly where it mattered.
    """

    store = build_store()
    for index in range(1001):
        store.save_workflow(
            workflow_request(
                f"workflow-busy-{index:04d}",
                f"inc-busy-{index:04d}",
                status=WorkflowStatus.PENDING,
                fencing_token=1,
                updated_at=NOW - timedelta(minutes=2000 - index),
            )
        )
    _incident_id, retired_id, _current_id = retired_pair(
        store, "a", command_status=None
    )

    candidates = retired_generation_candidates(store, None)
    plan = build_retired_generation_plan(store, now=NOW)

    assert candidates == [retired_id]
    assert [item["request_id"] for item in plan["items"]] == [retired_id]
    assert plan["discovery"]["scanned"] == 1003
    assert plan["discovery"]["scan_truncated"] is False


def test_a_containment_only_step_does_not_forbid_revocation() -> None:
    """F-B4 (4): the refusal reads ``NODE_MUTATING_OPERATIONS``.

    A cordon changes what the scheduler may place on the node, not the node; the
    successor generation lifts it with its own RESTORE_SCHEDULING. Refusing on it
    kept the retired record -- and the starvation it causes -- alive for a step
    the successor was going to redo anyway.
    """

    store = build_store()
    _incident_id, retired_id, current_id = retired_pair(store, "a", command_status=None)
    retired = store.get_workflow(retired_id)
    store.save_workflow(
        retired.model_copy(
            update={
                "completed_operations": [
                    WorkflowOperation.FREEZE_EVIDENCE,
                    WorkflowOperation.MARK_UNSCHEDULABLE,
                ]
            }
        )
    )

    item = build_retired_generation_plan(store, [retired_id], now=NOW)["items"][0]

    assert completed_destructive_operations(store.get_workflow(retired_id)) == []
    assert item["completed_containment_operations"] == ["MARK_UNSCHEDULABLE"]
    assert item["eligible"] is True

    store.save_workflow(
        store.get_workflow(retired_id).model_copy(
            update={"completed_operations": [WorkflowOperation.STOP_WORKLOADS]}
        )
    )
    refused = build_retired_generation_plan(store, [retired_id], now=NOW)["items"][0]
    assert refused["completed_destructive_operations"] == ["STOP_WORKLOADS"]
    assert refused["eligible"] is False


def test_the_pending_step_set_follows_safety_only_not_blocked_reasons() -> None:
    """F-B4 (4): a warning in ``blocked_reasons`` must not flip the step set."""

    safety = [workflow_step(WorkflowOperation.QUIESCE_GPU_SERVICES, node_ids=NODES)]
    official = [workflow_step(WorkflowOperation.RESTART_NODE, node_ids=NODES)]
    warned = workflow_request(
        "workflow-warned",
        "inc-warned",
        status=WorkflowStatus.BLOCKED,
        fencing_token=1,
        official_steps=official,
        safety_steps=safety,
        blocked_reasons=["dispatcher internal error: ValueError: x"],
    )
    safety_only = warned.model_copy(update={"safety_only": True, "blocked_reasons": []})

    assert pending_destructive_operations(warned) == ["RESTART_NODE"]
    assert pending_destructive_operations(safety_only) == ["QUIESCE_GPU_SERVICES"]
