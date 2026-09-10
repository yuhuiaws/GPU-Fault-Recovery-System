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

from gpu_fault.adapters.common import quarantine_taint_value
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
    NodeIsolationEvidence,
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
        "cluster_id": "cluster-a",
        "node_ids": ["node-a"],
        "evidence_required": False,
        "isolation_reasons": [],
    }
    assert refused["closable"] is False
    assert refused["open_workflow_id"] == "wf-inc-running"
    assert "RUNNING" in refused["refusal"]
    assert missing["closable"] is False and "not found" in missing["refusal"]
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


# ------------------------------------------- QUARANTINED close on node evidence


def _quarantined(store, incident_id: str = "inc-q", node_ids=("node-a",)):
    """A quarantine whose workflow ended; the incident still says QUARANTINED."""

    incident, workflow = _escalated_reset(
        store, incident_id=incident_id, node_ids=node_ids
    )
    quarantined = copy_model(incident, state=IncidentState.QUARANTINED)
    store.save_incident(quarantined, expected=incident)
    return quarantined, workflow


def _clean(node_id: str = "node-a") -> NodeIsolationEvidence:
    return NodeIsolationEvidence(node_id=node_id)


def test_a_quarantined_incident_closes_on_evidence_that_its_isolation_is_gone(
    store,
) -> None:
    incident, workflow = _quarantined(store)
    service = IncidentClosureService(store)

    closed, changed = service.close_incident(
        incident.incident_id,
        reason="node repaired by hand, isolation released by cleanup",
        operator=OPERATOR,
        reference="CHG-1",
        evidence=[_clean()],
    )

    assert changed is True and closed.state is IncidentState.RECOVERED
    stored = store.get_incident(incident.incident_id)
    assert stored.reasons[-2:] == [
        "operator closed: node repaired by hand, isolation released by cleanup "
        f"by {OPERATOR}",
        "isolation no longer present on node node-a",
    ]
    events = store.get_workflow(workflow.request_id).events
    close_event = [
        event
        for event in events
        if event.code == WorkflowEventCode.INCIDENT_CLOSED.value
    ][-1]
    assert close_event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert close_event.actor == OPERATOR
    assert close_event.details["previous_incident_state"] == "QUARANTINED"
    assert close_event.details["isolation_evidence"] == [
        {
            "node_id": "node-a",
            "exists": True,
            "unschedulable": False,
            "quarantine_taint_value": None,
            "isolation_annotations": {},
        }
    ], "the evidence the close rested on is on the audit event"
    assert all(
        not marker.active for marker in store.list_markers_for_incident("inc-q")
    ), "markers retire like an ESCALATED close"
    assert service.operator_closed_total == 1


def test_a_quarantined_incident_still_holding_its_own_taint_is_refused(store) -> None:
    incident, _ = _quarantined(store)
    service = IncidentClosureService(store)
    own_taint = NodeIsolationEvidence(
        node_id="node-a",
        quarantine_taint_value=quarantine_taint_value(incident.incident_id),
    )

    with pytest.raises(IncidentNotClosable) as refused:
        service.close_incident(
            incident.incident_id, reason="x", operator=OPERATOR, evidence=[own_taint]
        )

    assert "taint of incident inc-q" in str(refused.value)
    assert store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED
    assert service.operator_closed_total == 0


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        pytest.param(
            NodeIsolationEvidence(node_id="node-a", unschedulable=True),
            "still cordoned",
            id="cordoned",
        ),
        pytest.param(
            NodeIsolationEvidence(
                node_id="node-a",
                isolation_annotations={"gpu-fault.io/incident-id": "inc-q"},
            ),
            "isolation annotations of incident inc-q",
            id="own-annotation",
        ),
        pytest.param(
            NodeIsolationEvidence(node_id="node-a", exists=False),
            "not in the cluster",
            id="node-missing",
        ),
        pytest.param(
            NodeIsolationEvidence(node_id="node-other"),
            "no isolation evidence was supplied for node node-a",
            id="wrong-node",
        ),
        pytest.param(
            NodeIsolationEvidence(node_id="node-a", quarantine_taint_value="inc-q"),
            "taint of incident inc-q",
            id="raw-id-taint-value",
        ),
    ],
)
def test_evidence_that_does_not_clear_every_node_refuses_the_close(
    store, evidence, match
) -> None:
    incident, _ = _quarantined(store)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable, match=match):
        service.close_incident(
            incident.incident_id, reason="x", operator=OPERATOR, evidence=[evidence]
        )

    assert store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED


def test_a_taint_owned_by_another_incident_does_not_block_and_is_recorded(
    store,
) -> None:
    incident, _ = _quarantined(store)
    service = IncidentClosureService(store)
    other = NodeIsolationEvidence(
        node_id="node-a",
        quarantine_taint_value=quarantine_taint_value("inc-other"),
        isolation_annotations={
            "gpu-fault.io/incident-id": "inc-other",
            "gpu-fault.io/fencing-token": "4",
        },
    )

    preview = service.preview(incident.incident_id, evidence=[other])
    closed, changed = service.close_incident(
        incident.incident_id, reason="x", operator=OPERATOR, evidence=[other]
    )

    assert preview["closable"] is True and preview["evidence_required"] is False
    assert changed is True and closed.state is IncidentState.RECOVERED
    last = store.get_incident(incident.incident_id).reasons[-1]
    assert last.startswith("isolation no longer present on node node-a ("), last
    assert "owned by incident inc-other" in last, "the other owner is named, not judged"


def test_a_quarantined_incident_with_an_open_workflow_is_refused_despite_evidence(
    store,
) -> None:
    incident, _ = _quarantined(store)
    successor = workflow_request(
        "wf-successor",
        incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(RESTORE, node_ids=["node-a"])],
    )
    store.save_workflow(successor)
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable, match="wf-successor"):
        service.close_incident(
            incident.incident_id, reason="x", operator=OPERATOR, evidence=[_clean()]
        )


def test_every_node_of_a_multi_node_incident_needs_clearing_evidence(store) -> None:
    incident, _ = _quarantined(store, node_ids=("node-a", "node-b"))
    service = IncidentClosureService(store)

    with pytest.raises(IncidentNotClosable, match="node node-b"):
        service.close_incident(
            incident.incident_id, reason="x", operator=OPERATOR, evidence=[_clean()]
        )

    closed, _ = service.close_incident(
        incident.incident_id,
        reason="x",
        operator=OPERATOR,
        evidence=[_clean("node-b"), _clean("node-a")],
    )
    assert closed.reasons[-2:] == [
        "isolation no longer present on node node-a",
        "isolation no longer present on node node-b",
    ]


def test_without_evidence_a_quarantined_incident_is_refused_as_before(store) -> None:
    incident, _ = _quarantined(store)
    service = IncidentClosureService(store)

    preview = service.preview(incident.incident_id)
    with pytest.raises(IncidentNotClosable) as refused:
        service.close_incident(incident.incident_id, reason="x", operator=OPERATOR)

    assert preview["closable"] is False
    assert preview["evidence_required"] is True, (
        "the caller that can read the nodes is told to come back with evidence"
    )
    assert preview["cluster_id"] == "cluster-a" and preview["node_ids"] == ["node-a"]
    assert "QUARANTINED" in str(refused.value) and "ESCALATED" in str(refused.value)
    assert store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED


def test_evidence_is_ignored_for_an_escalated_incident(store) -> None:
    incident, _ = _escalated_reset(store)
    service = IncidentClosureService(store)
    tainted = NodeIsolationEvidence(
        node_id="node-a", quarantine_taint_value=quarantine_taint_value("inc-reset")
    )

    closed, changed = service.close_incident(
        incident.incident_id, reason="x", operator=OPERATOR, evidence=[tainted]
    )

    assert changed is True and closed.state is IncidentState.RECOVERED
    assert closed.reasons[-1] == f"operator closed: x by {OPERATOR}", (
        "ESCALATED closes on the signature alone; no isolation line is added"
    )


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


def test_a_succeeded_diagnostic_only_workflow_retires_its_recovered_incidents_markers(
    store,
) -> None:
    """ARCH-I4 on the success path (COLLECT-020).

    The isolating shape retires through RESTORE_SCHEDULING and a failed
    diagnostic through its inconclusive close; a diagnostic that *passed* left
    its marker active until the TTL while the incident read RECOVERED.
    """

    from gpu_fault.orchestration.incident_closure import RECOVERED_RETIREMENT_ACTOR
    from tests.orchestration._incident_closure_support import _marker

    diagnostic = _restore_workflow(
        "inc-identity",
        operations=(
            FREEZE,
            WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
            WorkflowOperation.VALIDATE_GPU,
        ),
    )
    incident = _recovered("inc-identity", diagnostic)
    store.save_incident(incident)
    store.save_workflow(diagnostic)
    store.add_marker(_marker("inc-identity", "node-a"))
    service = IncidentClosureService(store)

    closed = service.on_terminal(diagnostic, incident, list(diagnostic.official_steps))

    assert closed == [], (
        "a diagnostic-only workflow restores nothing and closes nothing"
    )
    (marker,) = store.list_markers_for_incident("inc-identity")
    assert marker.active is False, "a RECOVERED incident leaves no live marker"
    assert marker.retired_at is not None, "retirement is dated"
    assert marker.retired_reason == (
        f"workflow {diagnostic.request_id} SUCCEEDED: incident RECOVERED"
    )
    assert marker.retired_by == RECOVERED_RETIREMENT_ACTOR

    # Replaying the hook keeps the first retirement's stamp.
    stamp = marker.retired_at
    service.on_terminal(diagnostic, incident, list(diagnostic.official_steps))
    (again,) = store.list_markers_for_incident("inc-identity")
    assert again.retired_at == stamp, "retirement is idempotent"
