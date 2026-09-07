"""The retired-generation apply: approval binds the CAS, failures are reported.

``apply_retired_generation_plan`` runs two passes -- cancel open commands, then
re-plan and revoke. The second pass used to supply the compare-and-set value, so
the operator's approval no longer constrained the write (P1-72D); the restart
reservation release ran outside the ``try`` written to contain it (P1-72F); one
failing item threw away the record of every item already revoked and made the
rerun refuse (P1-60F); ``cancellable`` was decided by a prose prefix (P2-72H);
and discovery refused any fleet with more than a thousand open workflows
(P1-72G). Each case here fails on the old shape and pins the new one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.execution import restart_budget_preflight
from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.retired_generation import (
    OPEN_REMOTE_COMMANDS_CODE,
    apply_retired_generation_plan,
    build_retired_generation_plan,
    completed_destructive_operations,
    pending_destructive_operations,
    retired_generation_blockers,
    retired_generation_candidates,
)
from gpu_fault.store import SqliteStore
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


def test_the_cas_value_comes_from_the_approved_item_not_the_second_pass() -> None:
    """A re-plan between the passes must not become the thing being approved.

    Between cancelling the commands and revoking, the record is re-planned in
    place: same ``request_id``, generation 1 -> 2, a new step list the operator
    has never seen. The second pass still finds it eligible, and the old code took
    its compare-and-set value from that pass -- revoking a live record on the
    strength of an approval that described a different one.
    """

    store = build_store()
    _incident_id, retired_id, _current_id = retired_pair(store, "a")
    plan = build_retired_generation_plan(store, [retired_id], now=NOW)
    cancel = store.cancel_remote_commands_for_workflow

    def cancel_then_replan(workflow_request_id: str, *, reason: str) -> dict[str, int]:
        counts = cancel(workflow_request_id, reason=reason)
        workflow = store.get_workflow(workflow_request_id)
        store.save_workflow(
            workflow.model_copy(
                update={
                    "fencing_token": 2,
                    "official_steps": [
                        workflow_step(WorkflowOperation.RESTART_NODE, node_ids=NODES)
                    ],
                    "completed_step_indexes": [],
                    "completed_operations": [],
                }
            )
        )
        return counts

    store.cancel_remote_commands_for_workflow = cancel_then_replan  # type: ignore[method-assign]

    with pytest.raises(ValueError, match=r"fencing_token.*1.*2"):
        apply_retired_generation_plan(
            store,
            workflow_ids=[retired_id],
            expected_plan_sha256=plan["plan_sha256"],
            reference="pre-deploy-1",
            now=NOW,
        )

    assert store.get_workflow(retired_id).status is WorkflowStatus.RUNNING, (
        "a record the operator never reviewed was revoked"
    )


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


def test_a_failing_reservation_release_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = build_store()
    _incident_id, retired_id, _current_id = retired_pair(store, "a")
    plan = build_retired_generation_plan(store, [retired_id], now=NOW)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("serialization failure")

    monkeypatch.setattr(
        restart_budget_preflight, "release_unattempted_restart_reservations", explode
    )

    result = apply_retired_generation_plan(
        store,
        workflow_ids=[retired_id],
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-1",
        now=NOW,
    )

    assert store.get_workflow(retired_id).status is WorkflowStatus.SUPERSEDED
    assert result["applied_workflow_ids"] == [retired_id]
    assert len(result["restart_reservation_warnings"]) == 1
    assert "serialization failure" in result["restart_reservation_warnings"][0]


def test_one_failing_item_still_returns_what_the_others_did() -> None:
    store = build_store()
    _a, retired_a, _ = retired_pair(store, "a", command_status=None)
    _b, retired_b, _ = retired_pair(store, "b", command_status=None)
    plan = build_retired_generation_plan(store, [retired_a, retired_b], now=NOW)
    revoke = store.reconcile_retired_generation_workflow

    def revoke_unless_b(workflow_request_id: str, *args: Any, **kwargs: Any) -> Any:
        if workflow_request_id == retired_b:
            raise RuntimeError("row lock timeout")
        return revoke(workflow_request_id, *args, **kwargs)

    store.reconcile_retired_generation_workflow = revoke_unless_b  # type: ignore[method-assign]

    result = apply_retired_generation_plan(
        store,
        workflow_ids=[retired_a, retired_b],
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-1",
        now=NOW,
    )

    assert result["applied_workflow_ids"] == [retired_a]
    assert list(result["failed_workflow_ids"]) == [retired_b]
    assert "row lock timeout" in result["failures"][retired_b]
    assert store.get_workflow(retired_a).status is WorkflowStatus.SUPERSEDED
    assert store.get_workflow(retired_b).status is WorkflowStatus.RUNNING


def test_rerunning_after_a_partial_apply_skips_the_record_already_revoked() -> None:
    """The natural next move after a partial failure is to run the same command."""

    store = build_store()
    _a, retired_a, _ = retired_pair(store, "a", command_status=None)
    _b, retired_b, _ = retired_pair(store, "b", command_status=None)
    first = build_retired_generation_plan(store, [retired_a, retired_b], now=NOW)
    apply_retired_generation_plan(
        store,
        workflow_ids=[retired_a, retired_b],
        expected_plan_sha256=first["plan_sha256"],
        reference="pre-deploy-1",
        now=NOW,
    )
    # Undo the second one as if its write had failed after the first committed.
    store.save_workflow(
        store.get_workflow(retired_b).model_copy(
            update={"status": WorkflowStatus.RUNNING, "preempted_by_workflow_id": None}
        )
    )

    second = build_retired_generation_plan(store, [retired_a, retired_b], now=NOW)
    already = next(item for item in second["items"] if item["request_id"] == retired_a)
    assert already["already_revoked"] is True
    assert already["eligible"] is False

    result = apply_retired_generation_plan(
        store,
        workflow_ids=[retired_a, retired_b],
        expected_plan_sha256=second["plan_sha256"],
        reference="pre-deploy-2",
        now=NOW,
    )

    assert result["applied_workflow_ids"] == [retired_b]
    assert result["already_revoked_workflow_ids"] == [retired_a]
    assert result["failed_workflow_ids"] == []


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


ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/alice"


def _operator_events(store: Any, request_id: str) -> list[Any]:
    return [
        event
        for event in store.get_workflow(request_id).events
        if event.kind is WorkflowEventKind.OPERATOR_RETIRED_GENERATION
    ]


def test_the_revocation_leaves_an_attributed_event_on_the_workflow(
    tmp_path: Path,
) -> None:
    """The audit was a sentence in ``preemption_reason`` (I1).

    The revocation now also appends a ``WorkflowEvent`` in the same transaction:
    who (STS ARN), which approval (both plan digests), and the status it moved
    from and to. The SQLite store runs the shared transactional write; the memory
    store keeps its own copy, covered below.
    """

    store = SqliteStore(str(tmp_path / "retired.db"))
    try:
        _incident_id, retired_id, current_id = retired_pair(
            store, "a", command_status=None
        )
        plan = build_retired_generation_plan(store, [retired_id], now=NOW)

        apply_retired_generation_plan(
            store,
            workflow_ids=[retired_id],
            expected_plan_sha256=plan["plan_sha256"],
            reference="pre-deploy-1",
            now=NOW,
            actor=ACTOR,
            admin_plan_sha256="c" * 64,
        )

        events = _operator_events(store, retired_id)
        assert len(events) == 1, f"expected one operator event, found {events}"
        event = events[0]
        assert event.actor == ACTOR
        assert event.code == "OPERATOR_RETIRED_GENERATION"
        assert event.at == NOW
        assert event.status == WorkflowStatus.SUPERSEDED.value
        assert event.details["previous_status"] == WorkflowStatus.RUNNING.value
        assert event.details["new_status"] == WorkflowStatus.SUPERSEDED.value
        assert event.details["reference"] == "pre-deploy-1"
        assert event.details["successor_workflow_id"] == current_id
        assert event.details["plan_sha256"] == plan["plan_sha256"]
        assert event.details["admin_plan_sha256"] == "c" * 64
        revoked = store.get_workflow(retired_id)
        assert "pre-deploy-1" in (revoked.preemption_reason or ""), (
            "the human-readable audit sentence must stay alongside the event"
        )
    finally:
        store.close()


def test_the_memory_store_revocation_records_the_event_too() -> None:
    store = build_store()
    _incident_id, retired_id, current_id = retired_pair(store, "m", command_status=None)
    plan = build_retired_generation_plan(store, [retired_id], now=NOW)

    apply_retired_generation_plan(
        store,
        workflow_ids=[retired_id],
        expected_plan_sha256=plan["plan_sha256"],
        reference="pre-deploy-2",
        now=NOW,
        actor=ACTOR,
    )

    events = _operator_events(store, retired_id)
    assert len(events) == 1, f"expected one operator event, found {events}"
    assert events[0].details["reference"] == "pre-deploy-2"
    assert events[0].details["successor_workflow_id"] == current_id
    assert events[0].actor == ACTOR, "the memory store must record the real actor"
