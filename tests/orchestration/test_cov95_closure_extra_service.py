from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from gpu_fault.adapters.common import ANNOTATION_FENCING, ANNOTATION_INCIDENT
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.incident_closure import (
    IncidentClosureService,
    NodeIsolationEvidence,
)
from gpu_fault.store.shared.errors import StaleWriteError
from tests._builders import workflow_step, workflow_step_execution
from tests.orchestration._cov95_closure_extra_safety import (
    closure_extra_isolation as closure_extra_isolation,
)
from tests.orchestration._cov95_closure_extra_support import (
    closure_store_fixture as closure_store_fixture,
)
from tests.orchestration._cov95_closure_extra_support import pair

RESTORE = WorkflowOperation.RESTORE_SCHEDULING
RESET = WorkflowOperation.RESET_GPU
STOP = WorkflowOperation.STOP_WORKLOADS


def test_isolation_mapping_roundtrip_preserves_only_recognized_evidence_fields():
    evidence = NodeIsolationEvidence.from_mapping(
        {
            "node_id": "node-a",
            "exists": True,
            "unschedulable": False,
            "quarantine_taint_value": "",
            "isolation_annotations": {
                ANNOTATION_INCIDENT: "other-incident",
                ANNOTATION_FENCING: 7,
                "unrelated": "not-isolation-evidence",
            },
        }
    )
    assert evidence.as_dict() == {
        "node_id": "node-a",
        "exists": True,
        "unschedulable": False,
        "quarantine_taint_value": None,
        "isolation_annotations": {
            ANNOTATION_INCIDENT: "other-incident",
            ANNOTATION_FENCING: "7",
        },
    }, "wire conversion changed node proof or included unrelated annotations"


def test_restore_hook_does_not_close_a_node_only_in_the_workload_stop_scope(
    closure_store,
):
    store = closure_store
    repaired, _ = pair(store, "repaired", nodes=("node-a",))
    untouched, _ = pair(store, "still-held", nodes=("node-b",))
    steps = [
        workflow_step(
            STOP, node_ids=["node-a", "node-b"], workload_ids=["training/job/unit"]
        ),
        workflow_step(RESET, node_ids=["node-a"]),
        workflow_step(RESTORE, node_ids=["node-a"]),
    ]
    restored, workflow = pair(
        store,
        "restoring-workflow",
        nodes=("node-a",),
        state=IncidentState.RECOVERED,
        status=WorkflowStatus.SUCCEEDED,
        workflow_values={
            "official_steps": steps,
            "completed_operations": [step.operation for step in steps],
            "completed_step_indexes": [0, 1, 2],
        },
    )
    closed = IncidentClosureService(store).on_terminal(workflow, restored, steps)
    assert set(closed) == {repaired.incident_id}, (
        "the restore hook treated workload-stop allocation nodes as restored hardware"
    )
    assert store.get_incident(untouched.incident_id).state is IncidentState.ESCALATED, (
        "an unrestored node lost its operator-held incident"
    )
    assert store.list_markers_for_incident(untouched.incident_id)[0].active is True, (
        "a marker was retired for a node this workflow never restored"
    )


def test_operator_close_cas_failure_preserves_the_concurrent_owner_and_marker(
    closure_store, monkeypatch
):
    store = closure_store
    incident, workflow = pair(store)
    original = store.save_incident
    raced = []

    def save(value, *, expected=None, **kwargs):
        if (
            value.incident_id == incident.incident_id
            and value.state is IncidentState.RECOVERED
        ):
            current = store.get_incident(value.incident_id)
            original(
                current.model_copy(
                    update={
                        "state": IncidentState.QUARANTINED,
                        "updated_at": current.updated_at + timedelta(seconds=1),
                    }
                ),
                expected=current,
            )
            raced.append(value.incident_id)
            raise StaleWriteError("unit close lost its compare-and-set")
        return original(value, expected=expected, **kwargs)

    monkeypatch.setattr(store, "save_incident", save)
    service = IncidentClosureService(store)
    with pytest.raises(StaleWriteError, match="compare-and-set"):
        service.close_incident(
            incident.incident_id, reason="unit repaired", operator="unit-operator"
        )
    assert raced == [incident.incident_id], (
        "the fixture did not create the intended CAS race"
    )
    assert (
        store.get_incident(incident.incident_id).state is IncidentState.QUARANTINED
    ), "operator closure overwrote the concurrent owner's state"
    assert store.get_workflow(workflow.request_id).events == [], (
        "a failed incident CAS still wrote a successful-close audit"
    )
    assert store.list_markers_for_incident(incident.incident_id)[0].active is True, (
        "a failed incident CAS still retired its marker"
    )
    assert service.operator_closed_total == 0, (
        "a rejected CAS incremented the close counter"
    )


@pytest.mark.parametrize("conflicts", [1, 2])
def test_close_audit_rereads_after_cas_conflict_and_bounds_its_retries(
    closure_store, monkeypatch, caplog, conflicts
):
    store = closure_store
    incident, workflow = pair(store)
    original = store.save_workflow
    attempts = []

    def save(value, *, expected=None, **kwargs):
        attempts.append(value)
        if len(attempts) <= conflicts:
            current = store.get_workflow(value.request_id)
            original(
                current.model_copy(
                    update={
                        "blocked_reasons": [
                            *current.blocked_reasons,
                            f"peer-update-{len(attempts)}",
                        ]
                    }
                ),
                expected=current,
            )
            raise StaleWriteError("unit concurrent audit update")
        return original(value, expected=expected, **kwargs)

    monkeypatch.setattr(store, "save_workflow", save)
    service = IncidentClosureService(store)
    with caplog.at_level(logging.WARNING):
        closed, changed = service.close_incident(
            incident.incident_id, reason="unit repaired", operator="unit-operator"
        )
    assert changed is True and closed.state is IncidentState.RECOVERED, (
        "audit contention rolled back a completed primary incident close"
    )
    assert len(attempts) == 2, "audit CAS retry was not bounded to two attempts"
    saved = store.get_workflow(workflow.request_id)
    assert saved.blocked_reasons == [
        f"peer-update-{index}" for index in range(1, conflicts + 1)
    ], "an audit retry overwrote another writer's workflow metadata"
    audits = [
        event
        for event in saved.events
        if event.code == WorkflowEventCode.INCIDENT_CLOSED.value
    ]
    assert len(audits) == (1 if conflicts == 1 else 0), (
        "audit persistence did not reflect the real CAS outcome"
    )
    if conflicts == 2:
        assert any("kept moving" in item.getMessage() for item in caplog.records), (
            "exhausted audit retries were not reported"
        )
    assert store.list_markers_for_incident(incident.incident_id)[0].active is False, (
        "audit contention skipped independent marker retirement"
    )


@pytest.mark.parametrize("failure", ["audit", "marker"])
def test_post_commit_failure_is_visible_without_undoing_the_primary_close(
    closure_store, monkeypatch, caplog, failure
):
    store = closure_store
    incident, workflow = pair(store)

    def unavailable(*args, **kwargs):
        raise OSError(f"unit {failure} persistence failed")

    monkeypatch.setattr(
        store, "save_workflow" if failure == "audit" else "add_marker", unavailable
    )
    with caplog.at_level(logging.ERROR):
        closed, changed = IncidentClosureService(store).close_incident(
            incident.incident_id, reason="unit repaired", operator="unit-operator"
        )
    assert changed is True and closed.state is IncidentState.RECOVERED, (
        "secondary failure hid the committed incident transition"
    )
    assert store.get_incident(incident.incident_id).state is IncidentState.RECOVERED, (
        "secondary failure reverted the primary close"
    )
    assert any(item.exc_info is not None for item in caplog.records), (
        "post-commit persistence failure lost its diagnostic"
    )
    marker = store.list_markers_for_incident(incident.incident_id)[0]
    assert marker.active is (failure == "marker"), (
        "independent retirement result was fabricated"
    )
    assert bool(store.get_workflow(workflow.request_id).events) is (
        failure != "audit"
    ), "independent audit result was fabricated"


def test_missing_workflow_pointer_does_not_prevent_closing_and_retiring_markers(
    closure_store,
):
    store = closure_store
    incident, workflow = pair(store)
    store.save_incident(
        incident.model_copy(update={"workflow_request_id": "unit-missing-workflow"}),
        expected=incident,
    )
    service = IncidentClosureService(store)
    closed, changed = service.close_incident(
        incident.incident_id, reason="unit repaired", operator="unit-operator"
    )
    assert changed is True and closed.state is IncidentState.RECOVERED, (
        "a missing history row prevented an otherwise authorized close"
    )
    assert store.get_workflow(workflow.request_id).events == [], (
        "closure wrote history to a workflow no longer named by the incident"
    )
    assert store.list_markers_for_incident(incident.incident_id)[0].active is False, (
        "a missing workflow row prevented incident marker retirement"
    )


@pytest.mark.parametrize("started", [False, True])
def test_operator_block_is_only_closable_when_it_never_changed_the_node(
    closure_store, started
):
    store = closure_store
    incident, workflow = pair(
        store,
        status=WorkflowStatus.BLOCKED,
        workflow_values={
            "blocked_kind": BlockedKind.NEEDS_OPERATOR,
            "step_executions": [workflow_step_execution(0, RESET)] if started else [],
        },
    )
    preview = IncidentClosureService(store).preview(incident.incident_id)
    assert preview["closable"] is (not started), (
        "operator-block progress did not gate closure"
    )
    if started:
        assert "BLOCKED/NEEDS_OPERATOR" in preview["refusal"], (
            "the blocking workflow kind was omitted from the refusal"
        )
        assert preview["open_workflow_id"] == workflow.request_id, (
            "refusal did not identify the workflow that still owns the node"
        )
    assert store.get_incident(incident.incident_id) == incident, (
        "preview mutated its incident"
    )


def test_terminal_hook_skips_a_cas_loser_and_closes_other_eligible_incidents(
    closure_store, monkeypatch
):
    store = closure_store
    raced, _ = pair(store, "raced")
    other, _ = pair(store, "other")
    steps = [workflow_step(RESTORE)]
    restored, workflow = pair(
        store,
        "restored",
        state=IncidentState.RECOVERED,
        status=WorkflowStatus.SUCCEEDED,
        workflow_values={"official_steps": steps, "completed_operations": [RESTORE]},
    )
    original = store.save_incident

    def save(value, *, expected=None, **kwargs):
        if (
            value.incident_id == raced.incident_id
            and value.state is IncidentState.RECOVERED
        ):
            raise StaleWriteError("unit concurrent incident owner")
        return original(value, expected=expected, **kwargs)

    monkeypatch.setattr(store, "save_incident", save)
    service = IncidentClosureService(store)
    closed = service.on_terminal(workflow, restored, steps)
    assert closed == [other.incident_id], (
        "one lost CAS prevented independent eligible closure"
    )
    assert store.get_incident(raced.incident_id).state is IncidentState.ESCALATED, (
        "the lost CAS was reported as a completed close"
    )
    assert store.list_markers_for_incident(raced.incident_id)[0].active is True, (
        "the lost CAS retired a marker still owned by another writer"
    )
    assert service.auto_closed_by_restore_total == 1, (
        "auto-close counter included a failed CAS"
    )


def test_terminal_hook_without_any_restored_node_cannot_close_another_incident(
    closure_store,
):
    store = closure_store
    pending, _ = pair(store, "unrelated")
    restored, workflow = pair(
        store,
        "empty-restore",
        nodes=(),
        state=IncidentState.RECOVERED,
        status=WorkflowStatus.SUCCEEDED,
        marker=False,
        workflow_values={"official_steps": [], "completed_operations": [RESTORE]},
    )
    closed = IncidentClosureService(store).on_terminal(workflow, restored, [])
    assert closed == [], "a scope-free terminal hook closed an unrelated incident"
    assert store.get_incident(pending.incident_id) == pending, (
        "a scope-free restoration mutated another incident"
    )
