"""The refusals and partial outcomes of the Pod-side ``workflow_reconcile`` exits.

Every exit there re-derives its rule on the record as it is now, writes one
item at a time and reports what it could not do instead of raising: the
never-dispatched close refuses a record whose approval no longer matches or
whose evidence proves nothing; the confirmed-parked planner lists every
condition still live; the never-changed apply isolates a failed write and turns
a failed reservation release into a warning; the storm-breaker abort hands
over what it can and names what moved. These cases pin those edges on a real
``InMemoryStore``; a failing store write is a wrapped store method.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.remediation_storm import StormBreakerState, closed_breaker, storm_scope
from gpu_fault.store import InMemoryStore
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.store.shared.remediation_budgets import blocked_by_remediation_budget
from gpu_fault.workflow_reconcile import (
    NEVER_DISPATCHED,
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
    close_never_dispatched_workflow,
    never_dispatched_plan_items,
    operator_confirmed_plan_items,
    remediation_storm_action,
)
from gpu_fault.workflow_resolution import (
    DEPARTED_NODE_PROOF_REASON,
    NODE_RESTORE_PROOF_REASON,
    OPERATOR_CONFIRMED,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.regional._regional_support import enqueue_remote_command

NOW = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)
NODE = "hyperpod-i-00000000000000007"
RESTART = WorkflowOperation.RESTART_WORKLOAD
REFERENCE = "CHG-2026-1101-01"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"


def _restart_step() -> Any:
    """A RESTART_WORKLOAD step whose job restart planning may have reserved."""

    return workflow_step(
        RESTART,
        node_ids=[NODE],
        parameters={"cluster_id": "cluster-a", "job_id": "job-a"},
    )


def _failing_reservation_release(store: InMemoryStore, monkeypatch) -> list[str]:
    """The restart-budget table refuses every release, as an outage would."""

    released: list[str] = []

    def release_job_restart(cluster_id: str, job_id: str, reservation: str) -> None:
        released.append(reservation)
        raise RuntimeError("restart budget table unavailable")

    monkeypatch.setattr(store, "release_job_restart", release_job_restart)
    return released


# ------------------------------------------------------- never dispatched


def _pending(
    store: InMemoryStore,
    request_id: str = "workflow-departed",
    *,
    incident_state: IncidentState = IncidentState.ACTION_PENDING,
    save_incident: bool = True,
) -> str:
    """A PENDING record nothing dispatched, older than the dispatch guard."""

    incident_id = f"incident-{request_id}"
    store.save_workflow(
        workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.PENDING,
            fencing_token=2,
            official_action=RESTART.value,
            official_steps=[_restart_step()],
            created_at=NOW - timedelta(hours=1),
            updated_at=NOW - timedelta(hours=1),
        )
    )
    if save_incident:
        store.save_incident(
            fault_incident(
                incident_id,
                f"event-{request_id}",
                node_ids=[NODE],
                state=incident_state,
                workflow_request_id=request_id,
                fencing_token=2,
                updated_at=NOW - timedelta(hours=1),
            )
        )
    return request_id


def _bridge_item(store: InMemoryStore, request_id: str, **changes: Any) -> dict:
    workflow = store.get_workflow(request_id)
    item = {
        "request_id": request_id,
        "fencing_token": workflow.fencing_token,
        "execution_epoch": workflow.execution_epoch,
        "workflow_created_at": workflow.created_at.isoformat(),
        "departed_node_evidence": [
            {
                "node_id": NODE,
                "provider": "hyperpod",
                "absent_from_kubernetes": True,
                "absent_from_provider": True,
            }
        ],
    }
    item.update(changes)
    return item


def _close(store: InMemoryStore, item: dict) -> Any:
    return close_never_dispatched_workflow(
        store,
        item,
        reference=REFERENCE,
        actor=ACTOR,
        admin_plan_sha256="b" * 64,
        now=NOW,
    )


def test_a_never_dispatched_record_without_its_incident_is_planned_as_unbound() -> None:
    store = InMemoryStore()
    request_id = _pending(store, save_incident=False)

    items, report = never_dispatched_plan_items(store, [request_id], now=NOW)

    assert report is None, "explicit ids are judged as given, no discovery"
    [item] = items
    assert (item["cluster_id"], item["node_ids"], item["incident_state"]) == (
        None,
        [],
        None,
    )
    assert item["terminalization"] == NEVER_DISPATCHED
    assert item["eligible"] is False
    assert "incident is missing" in item["reasons"], item["reasons"]
    assert item["reasons"][-1] == DEPARTED_NODE_PROOF_REASON


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"execution_epoch": 5},
            "workflow execution epoch changed: expected 5, found 0",
        ),
        (
            {"workflow_created_at": (NOW - timedelta(hours=2)).isoformat()},
            "workflow created_at changed: expected ",
        ),
        (
            {
                "departed_node_evidence": [
                    {
                        "node_id": NODE,
                        "absent_from_kubernetes": False,
                        "absent_from_provider": True,
                    }
                ]
            },
            "departed-node evidence does not prove every node gone",
        ),
    ],
    ids=["epoch-moved", "created-at-moved", "node-still-in-kubernetes"],
)
def test_a_never_dispatched_close_refuses_what_its_approval_no_longer_binds(
    changes, message
) -> None:
    store = InMemoryStore()
    request_id = _pending(store)
    before = store.get_workflow(request_id)

    with pytest.raises(ValueError, match=message):
        _close(store, _bridge_item(store, request_id, **changes))

    assert store.get_workflow(request_id) == before, "a refusal writes nothing"


def test_a_never_dispatched_close_of_an_escalated_incident_keeps_its_state(
    monkeypatch,
) -> None:
    """The incident already sits with the operator: the close adds the audit
    line only, and a reservation release that fails is a warning, never an
    undo of the terminal write."""

    store = InMemoryStore()
    request_id = _pending(store, incident_state=IncidentState.ESCALATED)
    released = _failing_reservation_release(store, monkeypatch)

    closed, incident, warnings = _close(store, _bridge_item(store, request_id))

    assert closed.status is WorkflowStatus.SUPERSEDED, closed
    saved = store.get_incident(incident.incident_id)
    assert saved.state is IncidentState.ESCALATED, saved
    assert saved.reasons[-1].startswith(f"operator reconciliation {REFERENCE}"), saved
    assert "Kubernetes and HyperPod" in saved.reasons[-1], saved.reasons[-1]
    assert len(released) == 1, "the unattempted restart's reservation was tried"
    assert warnings == [
        f"{request_id}: closed, but releasing its restart reservations failed "
        "(RuntimeError: restart budget table unavailable)"
    ]


# ------------------------------------------------- confirmed parked records


def _confirmed_execution(node: str = NODE) -> Any:
    return workflow_step_execution(
        0,
        WorkflowOperation.RESTART_NODE,
        WorkflowStepStatus.FAILED,
        details={
            "outcome_unknown": True,
            "operator_confirmed": {
                "actor": ACTOR,
                "reference": REFERENCE,
                "confirmed_at": (NOW - timedelta(hours=1)).isoformat(),
                "node_id": node,
                "operation": "RESTART_NODE",
            },
        },
    )


def test_a_confirmed_record_still_live_lists_every_condition_left_open() -> None:
    """Running, leased, waiting on a provider action, with an open command,
    and pointing at an incident and a plan the Store no longer holds."""

    store = InMemoryStore()
    request_id = "workflow-parked"
    store.save_workflow(
        workflow_request(
            request_id,
            "incident-gone",
            status=WorkflowStatus.RUNNING,
            source_plan_id="plan-gone",
            execution_owner_id="executor-a",
            execution_lease_expires_at=NOW + timedelta(minutes=5),
            official_steps=[
                workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[NODE]),
                workflow_step(WorkflowOperation.VALIDATE_GPU, node_ids=[NODE]),
            ],
            step_executions=[
                _confirmed_execution(),
                workflow_step_execution(
                    1, WorkflowOperation.VALIDATE_GPU, WorkflowStepStatus.WAITING
                ),
            ],
        )
    )
    enqueue_remote_command(store, "parked")

    [item] = operator_confirmed_plan_items(store, [request_id], now=NOW)

    assert item["terminalization"] == OPERATOR_CONFIRMED
    assert (item["cluster_id"], item["incident_state"], item["node_ids"]) == (
        None,
        None,
        [],
    )
    assert item["source_plan_status"] is None, "a plan the Store lost has no status"
    assert item["open_remote_commands"] == ["parked"]
    assert [entry["node_id"] for entry in item["confirmations"]] == [NODE]
    assert item["eligible"] is False
    assert item["reasons"] == [
        "workflow status is RUNNING, not BLOCKED",
        "workflow still has an execution owner",
        "workflow execution lease has not expired",
        "workflow has an unknown provider action",
        "workflow has open remote commands",
        "incident is missing",
        NODE_RESTORE_PROOF_REASON,
    ]


# ------------------------------------------------------ never-changed apply


def _never_changed(store: InMemoryStore, request_id: str) -> str:
    """A BLOCKED record that never ran its RESTART_WORKLOAD step."""

    incident_id = f"incident-{request_id}"
    store.save_workflow(
        workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.BLOCKED,
            blocked_kind=BlockedKind.NEEDS_OPERATOR,
            fencing_token=2,
            execution_epoch=1,
            official_action=RESTART.value,
            official_steps=[_restart_step()],
            blocked_reasons=["policy requires an operator"],
            updated_at=NOW - timedelta(hours=1),
        )
    )
    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{request_id}",
            node_ids=[NODE],
            state=IncidentState.ESCALATED,
            workflow_request_id=request_id,
            fencing_token=2,
            updated_at=NOW - timedelta(hours=1),
        )
    )
    return request_id


def test_an_apply_isolates_a_failed_write_and_warns_on_a_failed_release(
    monkeypatch,
) -> None:
    store = InMemoryStore()
    moved = _never_changed(store, "workflow-moved")
    closed = _never_changed(store, "workflow-closed")
    plan = build_workflow_reconcile_plan(store, [moved, closed], now=NOW)
    assert [item["eligible"] for item in plan["items"]] == [True, True], plan
    amend = store.amend_workflow

    def amend_workflow(request_id: str, *args: Any, **kwargs: Any) -> Any:
        if request_id == moved:
            raise StaleWriteError(f"workflow/{request_id} changed since it was read")
        return amend(request_id, *args, **kwargs)

    monkeypatch.setattr(store, "amend_workflow", amend_workflow)
    released = _failing_reservation_release(store, monkeypatch)

    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=[moved, closed],
        expected_plan_sha256=plan["plan_sha256"],
        reference=REFERENCE,
        now=NOW,
        actor=ACTOR,
    )

    assert result["applied_workflow_ids"] == [closed]
    assert result["failed_workflow_ids"] == [moved]
    assert result["failures"] == {
        moved: f"StaleWriteError: workflow/{moved} changed since it was read"
    }
    assert result["restart_reservation_warnings"] == [
        f"{closed}: closed, but releasing its restart reservations failed "
        "(RuntimeError: restart budget table unavailable)"
    ]
    assert len(released) == 1, "only the closed record's reservation was tried"
    assert store.get_workflow(closed).status is WorkflowStatus.SUPERSEDED
    assert store.get_workflow(moved).status is WorkflowStatus.BLOCKED


# ------------------------------------------------------- the storm breaker

CLUSTER = "gpu-a"


def _tripped(store: InMemoryStore, states: list[IncidentState]) -> list[str]:
    """A TRIPPED breaker of ``CLUSTER`` and one held workflow per state."""

    store.replace_remediation_storm_breaker(
        closed_breaker(CLUSTER, NOW).model_copy(
            update={
                "state": StormBreakerState.TRIPPED,
                "trip_id": f"{CLUSTER}@{NOW.isoformat()}",
                "tripped_at": NOW - timedelta(minutes=5),
                "trip_node_ids": ["node-0", "node-1", "node-2"],
                "trip_domains": 3,
                "trip_threshold": 3,
            }
        ),
        expected=None,
    )
    ids = []
    for index, state in enumerate(states):
        incident = fault_incident(
            f"incident-{index}",
            f"event-{index}",
            cluster_id=CLUSTER,
            node_ids=[f"node-{index + 3}"],
            state=state,
        )
        workflow = blocked_by_remediation_budget(
            workflow_request(
                f"workflow-{index}",
                incident.incident_id,
                official_steps=[
                    workflow_step(
                        WorkflowOperation.RESTART_NODE, node_ids=[f"node-{index + 3}"]
                    )
                ],
            ),
            "held",
            scope=storm_scope(CLUSTER),
            now=NOW,
        )
        store.save_incident_and_workflow(
            incident.model_copy(update={"workflow_request_id": workflow.request_id}),
            workflow,
        )
        ids.append(workflow.request_id)
    return ids


def _act(store: InMemoryStore, action: str) -> dict:
    return remediation_storm_action(
        store,
        CLUSTER,
        action,
        actor=ACTOR,
        reason="driver rollout reverted",
        reference=REFERENCE,
        now=NOW,
    )


def test_an_unknown_storm_action_is_refused_before_any_read() -> None:
    class NoReads:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"the refusal read the store ({name})")

    with pytest.raises(ValueError, match="unknown remediation storm action 'pause'"):
        remediation_storm_action(
            NoReads(),
            CLUSTER,
            "pause",
            actor=ACTOR,
            reason="r",
            reference=REFERENCE,
            now=NOW,
        )


def test_an_abort_hands_over_what_it_can_and_names_the_workflow_that_moved(
    monkeypatch,
) -> None:
    """workflow-0's incident is already QUARANTINED: it keeps that state and
    gets the audit line. workflow-1 was re-planned after the listing (its
    fencing token moved): it is left alone and named. The abort is partial."""

    store = InMemoryStore()
    first, second = _tripped(
        store, [IncidentState.QUARANTINED, IncidentState.ACTION_PENDING]
    )
    read = store.get_workflow
    listed = store.get_workflow(second)

    def get_workflow(request_id: str) -> Any:
        current = read(request_id)
        if request_id == second:
            return current.model_copy(
                update={"fencing_token": listed.fencing_token + 1}
            )
        return current

    monkeypatch.setattr(store, "get_workflow", get_workflow)

    result = _act(store, "abort")

    assert result["status"] == "partial", result
    assert result["state"] == "ABORTED"
    assert sorted(result["held_workflows"]) == [first, second]
    assert result["closed_workflows"] == [first]
    assert result["errors"] == [
        f"{second}: ValueError: the workflow moved since it was listed as held"
    ]
    handed = read(first)
    assert handed.status is WorkflowStatus.SUPERSEDED, handed
    assert handed.events[-1].code == WorkflowEventCode.REMEDIATION_STORM_ABORTED.value
    incident = store.get_incident("incident-0")
    assert incident.state is IncidentState.QUARANTINED, incident
    assert "aborted by an operator" in incident.reasons[-1], incident.reasons
    assert read(second).status is WorkflowStatus.PENDING, (
        "the moved workflow is not handed over"
    )
    assert store.get_incident("incident-1").state is IncidentState.ACTION_PENDING


def test_an_abort_reports_a_held_workflow_whose_incident_is_gone() -> None:
    store = InMemoryStore()
    [held] = _tripped(store, [IncidentState.ACTION_PENDING])
    workflow = store.get_workflow(held)
    store.save_workflow(
        workflow.model_copy(update={"incident_id": "incident-archived"}),
        expected=workflow,
    )

    result = _act(store, "abort")

    assert result["status"] == "partial", result
    assert result["closed_workflows"] == []
    [error] = result["errors"]
    assert error.startswith(f"{held}: NotFoundError"), error
    assert store.get_workflow(held).status is WorkflowStatus.PENDING, (
        "a held workflow that could not be handed over stays held"
    )
