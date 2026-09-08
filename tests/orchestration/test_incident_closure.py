"""An ESCALATED incident is closed when its node is restored, or by an operator.

DESTR-018 (2026-09-08) left a product gap: a reset incident that failed on its
lifetime goes ESCALATED (F-N1, handed to an operator) and ``reopen_if_terminal``
keeps it as the node's merge target, so every later XID on the node lands on it
as ABSORB_RECORD_ONLY. Even after the support escalation's validated restore
put the node back, the reset incident stayed ESCALATED and the node stayed
"record only" for that fault scope. The only exit was a second restore
workflow on the old incident.

``IncidentClosureService`` closes that gap twice over: a SUCCEEDED workflow
that restored the node (the executor's ``on_terminal`` hook) closes every
other ESCALATED incident on that cluster whose nodes it covers, and an
operator may close one directly. Both write RECOVERED through the store's CAS,
append the reason, record the audit event on the incident's last workflow and
retire the incident's markers; a RECOVERED pair is no longer a merge target,
so the next fault opens its own remediation.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.incident_closure import (
    AUTO_CLOSE_ACTOR,
    IncidentClosureService,
    IncidentNotClosable,
    restores_node,
)
from gpu_fault.store import NotFoundError, SqliteStore
from tests._builders import (
    build_context,
    build_store,
    copy_model,
    workflow_request,
    workflow_step,
)
from tests.orchestration._incident_closure_support import (
    COLLECT,
    FREEZE,
    MARK,
    OPERATOR,
    QUARANTINE,
    QUIESCE,
    RESET,
    RESTORE,
    SUPPORT,
    VALIDATE,
    _escalated_reset,
    _recovered,
    _restore_workflow,
)
from tests.orchestration._support import _node_event, ingest
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "closure.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


def _events(store, request_id: str):
    return [
        (event.kind, event.code, event.actor)
        for event in store.get_workflow(request_id).events
    ]


# ------------------------------------------------------------- operator close


def test_an_operator_closes_an_escalated_incident(store) -> None:
    incident, workflow = _escalated_reset(store)
    service = IncidentClosureService(store)

    closed, changed = service.close_incident(
        incident.incident_id, reason="node repaired", operator=OPERATOR
    )

    assert changed is True
    assert closed.state is IncidentState.RECOVERED
    stored = store.get_incident(incident.incident_id)
    assert stored.state is IncidentState.RECOVERED
    assert stored.reasons[-1] == f"operator closed: node repaired by {OPERATOR}"
    assert stored.reasons[0] == incident.reasons[0], "earlier reasons are kept"
    assert stored.workflow_request_id == workflow.request_id, (
        "the pointer to the last workflow stays for the history"
    )
    markers = store.list_markers_for_incident(incident.incident_id)
    assert markers and all(not marker.active for marker in markers), markers
    assert markers[0].retired_by == OPERATOR
    assert (
        WorkflowEventKind.OPERATOR_RECONCILED,
        WorkflowEventCode.INCIDENT_CLOSED.value,
        OPERATOR,
    ) in _events(store, workflow.request_id)
    assert service.operator_closed_total == 1
    assert service.auto_closed_by_restore_total == 0


def test_closing_a_recovered_incident_again_is_a_no_op(store) -> None:
    incident, workflow = _escalated_reset(store)
    service = IncidentClosureService(store)
    service.close_incident(incident.incident_id, reason="first", operator=OPERATOR)
    once = store.get_incident(incident.incident_id)

    again, changed = service.close_incident(
        incident.incident_id, reason="second", operator=OPERATOR
    )

    assert changed is False
    assert again == once
    assert store.get_incident(incident.incident_id) == once
    assert service.operator_closed_total == 1
    assert (
        sum(
            1
            for kind, code, _ in _events(store, workflow.request_id)
            if code == WorkflowEventCode.INCIDENT_CLOSED.value
        )
        == 1
    )


@pytest.mark.parametrize(
    "status",
    [WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING],
)
def test_an_incident_with_an_open_workflow_is_refused(store, status) -> None:
    incident, workflow = _escalated_reset(store, workflow_status=status)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable) as refused:
        service.close_incident(incident.incident_id, reason="x", operator=OPERATOR)

    assert workflow.request_id in str(refused.value)
    assert status.value in str(refused.value)
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED
    assert store.list_markers_for_incident(incident.incident_id)[0].active is True
    assert service.operator_closed_total == 0


def test_an_open_successor_on_the_node_also_refuses_the_close(store) -> None:
    incident, _ = _escalated_reset(store)
    successor = workflow_request(
        "wf-successor",
        incident.incident_id,
        status=WorkflowStatus.PENDING,
        official_steps=[workflow_step(RESET, node_ids=["node-a"])],
    )
    store.save_workflow(successor)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable, match="wf-successor"):
        service.close_incident(incident.incident_id, reason="x", operator=OPERATOR)


@pytest.mark.parametrize(
    "state",
    [
        IncidentState.DETECTED,
        IncidentState.ACTION_PENDING,
        IncidentState.SAFETY_PENDING,
        IncidentState.QUARANTINED,
    ],
)
def test_only_an_escalated_incident_may_be_closed_by_hand(store, state) -> None:
    incident, _ = _escalated_reset(store)
    store.save_incident(copy_model(incident, state=state), expected=incident)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable) as refused:
        service.close_incident(incident.incident_id, reason="x", operator=OPERATOR)

    assert state.value in str(refused.value)
    assert "ESCALATED" in str(refused.value)
    assert store.get_incident(incident.incident_id).state is state


def test_an_unknown_incident_is_not_found(store) -> None:
    with pytest.raises(NotFoundError):
        IncidentClosureService(store).close_incident(
            "inc-missing", reason="x", operator=OPERATOR
        )


def test_a_close_without_a_workflow_row_still_lands(store) -> None:
    incident, _ = _escalated_reset(store)
    store.save_incident(
        copy_model(incident, workflow_request_id=None), expected=incident
    )
    service = IncidentClosureService(store)

    closed, changed = service.close_incident(
        incident.incident_id, reason="x", operator=OPERATOR
    )

    assert changed is True and closed.state is IncidentState.RECOVERED


def test_a_closed_incident_stops_being_the_merge_target(store) -> None:
    incident, workflow = _escalated_reset(store)
    kept = NodeConflictService.reopen_if_terminal(incident, workflow)
    assert kept == (incident, workflow), "while ESCALATED the pair absorbs"

    IncidentClosureService(store).close_incident(
        incident.incident_id, reason="x", operator=OPERATOR
    )

    released = NodeConflictService.reopen_if_terminal(
        store.get_incident(incident.incident_id),
        store.get_workflow(workflow.request_id),
    )
    assert released == (None, None), "RECOVERED: the next fault opens its own record"


def test_dry_run_preview_names_the_verdict_without_writing(store) -> None:
    incident, workflow = _escalated_reset(store)
    running, _ = _escalated_reset(
        store, incident_id="inc-running", workflow_status=WorkflowStatus.RUNNING
    )
    service = IncidentClosureService(store)

    closable = service.preview(incident.incident_id)
    refused = service.preview(running.incident_id)
    missing = service.preview("inc-missing")

    assert closable == {
        "incident_id": incident.incident_id,
        "state": "ESCALATED",
        "closable": True,
        "refusal": None,
        "open_workflow_id": None,
    }
    assert refused["closable"] is False
    assert refused["open_workflow_id"] == "wf-inc-running"
    assert "RUNNING" in refused["refusal"]
    assert missing["closable"] is False and "not found" in missing["refusal"]
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


# ------------------------------------------------------- auto-close by restore


def test_a_validated_restore_closes_the_escalated_incident_on_its_node(store) -> None:
    reset, reset_workflow = _escalated_reset(store)
    restore = _restore_workflow("inc-support")
    support = _recovered("inc-support", restore)
    store.save_incident(support)
    store.save_workflow(restore)
    service = IncidentClosureService(store)

    closed = service.on_terminal(restore, support, list(restore.official_steps))

    assert closed == [reset.incident_id]
    stored = store.get_incident(reset.incident_id)
    assert stored.state is IncidentState.RECOVERED
    assert stored.reasons[-1] == "node restored via incident inc-support"
    assert all(
        not marker.active
        for marker in store.list_markers_for_incident(reset.incident_id)
    )
    assert (
        WorkflowEventKind.TERMINAL,
        WorkflowEventCode.INCIDENT_CLOSED.value,
        AUTO_CLOSE_ACTOR,
    ) in _events(store, reset_workflow.request_id)
    assert store.get_incident("inc-support") == support, "the restorer is untouched"
    assert service.auto_closed_by_restore_total == 1
    assert service.operator_closed_total == 0


def test_the_auto_close_is_idempotent_across_a_replayed_hook(store) -> None:
    reset, _ = _escalated_reset(store)
    restore = _restore_workflow("inc-support")
    support = _recovered("inc-support", restore)
    store.save_incident(support)
    service = IncidentClosureService(store)

    first = service.on_terminal(restore, support, list(restore.official_steps))
    second = service.on_terminal(restore, support, list(restore.official_steps))

    assert first == [reset.incident_id] and second == []
    assert service.auto_closed_by_restore_total == 1


def test_a_restore_covers_only_incidents_whose_every_node_it_restored(store) -> None:
    two_nodes, _ = _escalated_reset(
        store, incident_id="inc-pair", node_ids=("node-a", "node-b")
    )
    single, _ = _escalated_reset(store, incident_id="inc-single")
    elsewhere, _ = _escalated_reset(
        store, incident_id="inc-elsewhere", node_ids=("node-z",)
    )
    other_cluster, _ = _escalated_reset(
        store, incident_id="inc-other-cluster", cluster_id="cluster-b"
    )
    restore = _restore_workflow("inc-support")
    support = _recovered("inc-support", restore)
    store.save_incident(support)

    closed = IncidentClosureService(store).on_terminal(
        restore, support, list(restore.official_steps)
    )

    assert closed == [single.incident_id]
    for untouched in (two_nodes, elsewhere, other_cluster):
        assert store.get_incident(untouched.incident_id).state is (
            IncidentState.ESCALATED
        ), untouched.incident_id


def test_an_escalated_incident_with_an_open_workflow_is_left_alone(store) -> None:
    pending, _ = _escalated_reset(
        store, incident_id="inc-pending", workflow_status=WorkflowStatus.PENDING
    )
    restore = _restore_workflow("inc-support")
    support = _recovered("inc-support", restore)
    store.save_incident(support)

    closed = IncidentClosureService(store).on_terminal(
        restore, support, list(restore.official_steps)
    )

    assert closed == []
    assert store.get_incident(pending.incident_id).state is IncidentState.ESCALATED


@pytest.mark.parametrize(
    ("status", "state", "operations"),
    [
        pytest.param(
            WorkflowStatus.FAILED,
            IncidentState.ESCALATED,
            (VALIDATE, RESTORE),
            id="failed-restore",
        ),
        pytest.param(
            WorkflowStatus.SUCCEEDED,
            IncidentState.ESCALATED,
            (FREEZE, MARK, QUARANTINE, SUPPORT),
            id="support-handoff",
        ),
        pytest.param(
            WorkflowStatus.SUCCEEDED,
            IncidentState.QUARANTINED,
            (FREEZE, MARK, QUARANTINE),
            id="still-quarantined",
        ),
        pytest.param(
            WorkflowStatus.SUCCEEDED,
            IncidentState.RECOVERED,
            (FREEZE, COLLECT),
            id="diagnostic-only",
        ),
    ],
)
def test_a_workflow_that_did_not_restore_the_node_closes_nothing(
    store, status, state, operations
) -> None:
    reset, _ = _escalated_reset(store)
    workflow = _restore_workflow("inc-other", status=status, operations=operations)
    other = _recovered("inc-other", workflow, state=state)
    store.save_incident(other)
    service = IncidentClosureService(store)

    closed = service.on_terminal(workflow, other, list(workflow.official_steps))

    assert closed == []
    assert store.get_incident(reset.incident_id).state is IncidentState.ESCALATED
    assert service.auto_closed_by_restore_total == 0


def test_a_hook_without_an_incident_closes_nothing(store) -> None:
    _escalated_reset(store)
    restore = _restore_workflow("inc-support")

    assert IncidentClosureService(store).on_terminal(restore, None, []) == []


@pytest.mark.parametrize(
    ("status", "completed", "expected"),
    [
        pytest.param(
            WorkflowStatus.SUCCEEDED, (VALIDATE, RESTORE), True, id="validated-restore"
        ),
        pytest.param(
            WorkflowStatus.SUCCEEDED,
            (FREEZE, MARK, QUIESCE, RESET, RESTORE),
            True,
            id="reset-with-release",
        ),
        pytest.param(
            WorkflowStatus.SUCCEEDED,
            (WorkflowOperation.RESTART_NODE,),
            True,
            id="reboot-never-cordoned",
        ),
        pytest.param(WorkflowStatus.SUCCEEDED, (FREEZE, COLLECT), False, id="evidence"),
        pytest.param(
            WorkflowStatus.SUCCEEDED, (VALIDATE,), False, id="validation-alone"
        ),
        pytest.param(
            WorkflowStatus.SUCCEEDED, (FREEZE, MARK, QUARANTINE), False, id="isolated"
        ),
        pytest.param(
            WorkflowStatus.SUCCEEDED, (MARK, RESET), False, id="cordon-not-released"
        ),
        pytest.param(WorkflowStatus.FAILED, (VALIDATE, RESTORE), False, id="failed"),
        pytest.param(WorkflowStatus.SUCCEEDED, (), False, id="nothing-ran"),
    ],
)
def test_restores_node_is_judged_on_the_registry_classes(
    status, completed, expected
) -> None:
    workflow = workflow_request(
        "wf", "inc", status=status, completed_operations=list(completed)
    )

    assert restores_node(workflow) is expected


# --------------------------------------------------- the node is free again


def test_after_the_close_the_next_xid_opens_a_new_remediation() -> None:
    """End to end through the orchestrator: the lifetime-failed incident
    absorbs a second XID as record-only; once closed, a third XID compiles
    its own workflow instead of landing on the closed incident."""

    context = build_context()
    _, incident, workflow = ingest(
        context, _node_event(48, event_id="xid-1", gpu_uuid="GPU-a")
    )
    assert workflow.status is WorkflowStatus.PENDING
    failed = copy_model(
        workflow,
        status=WorkflowStatus.FAILED,
        lifetime_deadline_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        completed_operations=[FREEZE, MARK, QUIESCE],
    )
    context.store.save_workflow(failed, expected=workflow)
    current = context.store.get_incident(incident.incident_id)
    context.store.save_incident(
        copy_model(current, state=IncidentState.ESCALATED), expected=current
    )
    merger = context.orchestrator._workflow_merger

    _, absorbed_incident, absorbed_workflow = ingest(
        context, _node_event(48, event_id="xid-2", gpu_uuid="GPU-a")
    )
    assert absorbed_incident.incident_id == incident.incident_id
    assert absorbed_workflow.request_id == workflow.request_id
    assert merger.lifetime_record_only_total == 1, "ESCALATED: record only"

    IncidentClosureService(context.store).close_incident(
        incident.incident_id, reason="node repaired", operator=OPERATOR
    )
    _, fresh_incident, fresh_workflow = ingest(
        context, _node_event(48, event_id="xid-3", gpu_uuid="GPU-a")
    )

    assert fresh_incident.incident_id != incident.incident_id
    assert fresh_workflow.request_id != workflow.request_id
    assert fresh_workflow.status is WorkflowStatus.PENDING
    assert merger.lifetime_record_only_total == 1, "the third XID was planned"
    assert (
        context.store.get_incident(incident.incident_id).state
        is IncidentState.RECOVERED
    )
