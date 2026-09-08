"""``POST /v1/incidents/{incident_id}/close``: the operator exit for an
ESCALATED incident (DESTR-018 product gap, 2026-09-08).

The route is the API face of ``IncidentClosureService.close_incident``: it
requires the execution token like every other operator write on the incident
router, refuses (409, with the reason) an incident that is not ESCALATED or
still has an open workflow, and is idempotent -- closing a RECOVERED incident
again is a 200 that changed nothing. After a close the node is free: the next
fault on it compiles its own workflow instead of being recorded on the closed
incident.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
)
from tests._builders import asgi_client, build_store, copy_model
from tests.orchestration._incident_closure_support import _escalated_reset
from tests.orchestration._support import _node_event, ingest

TOKEN = "x" * 32
HEADERS = {"X-GPU-Fault-Execution-Token": TOKEN}
BODY = {"reason": "node repaired after vendor visit", "operator": "ops@example"}
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES


def _context(store=None) -> ApplicationContext:
    return ApplicationContext(store=store or build_store(), execution_token=TOKEN)


def _post(context: ApplicationContext, incident_id: str, *, headers=HEADERS, json=BODY):
    async def scenario():
        async with asgi_client(context) as client:
            return await client.post(
                f"/v1/incidents/{incident_id}/close", headers=headers, json=json
            )

    return asyncio.run(scenario())


def test_closing_needs_the_execution_token() -> None:
    context = _context()
    incident, _ = _escalated_reset(context.store)

    denied = _post(context, incident.incident_id, headers={})

    assert denied.status_code in {401, 403}, denied.text
    assert context.store.get_incident(incident.incident_id).state is (
        IncidentState.ESCALATED
    )


def test_an_operator_closes_an_escalated_incident_through_the_api() -> None:
    context = _context()
    incident, workflow = _escalated_reset(context.store)

    response = _post(context, incident.incident_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["closed"] is True
    assert body["incident"]["incident_id"] == incident.incident_id
    assert body["incident"]["state"] == "RECOVERED"
    assert body["incident"]["reasons"][-1] == (
        "operator closed: node repaired after vendor visit by ops@example"
    )
    stored = context.store.get_incident(incident.incident_id)
    assert stored.state is IncidentState.RECOVERED
    assert all(
        not marker.active
        for marker in context.store.list_markers_for_incident(incident.incident_id)
    ), "closing the incident must retire its markers"
    events = context.store.get_workflow(workflow.request_id).events
    assert [
        (event.code, event.actor)
        for event in events
        if event.code == WorkflowEventCode.INCIDENT_CLOSED.value
    ] == [(WorkflowEventCode.INCIDENT_CLOSED.value, "ops@example")]
    assert context.incident_closure.operator_closed_total == 1


def test_closing_again_is_an_idempotent_200() -> None:
    context = _context()
    incident, _ = _escalated_reset(context.store)
    first = _post(context, incident.incident_id)
    once = context.store.get_incident(incident.incident_id)

    second = _post(context, incident.incident_id)

    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["closed"] is False
    assert context.store.get_incident(incident.incident_id) == once
    assert context.incident_closure.operator_closed_total == 1


def test_an_incident_with_a_running_workflow_is_a_409_naming_it() -> None:
    context = _context()
    incident, workflow = _escalated_reset(
        context.store, workflow_status=WorkflowStatus.RUNNING
    )

    response = _post(context, incident.incident_id)

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert workflow.request_id in detail and "RUNNING" in detail
    assert context.store.get_incident(incident.incident_id).state is (
        IncidentState.ESCALATED
    )


def test_a_quarantined_incident_is_a_409_pointing_at_the_restore() -> None:
    context = _context()
    incident, _ = _escalated_reset(context.store)
    context.store.save_incident(
        copy_model(incident, state=IncidentState.QUARANTINED), expected=incident
    )

    response = _post(context, incident.incident_id)

    assert response.status_code == 409, response.text
    assert "QUARANTINED" in response.json()["detail"]


def test_an_unknown_incident_is_a_404() -> None:
    response = _post(_context(), "inc-missing")

    assert response.status_code == 404, response.text


def test_the_body_needs_a_reason_and_an_operator() -> None:
    context = _context()
    incident, _ = _escalated_reset(context.store)

    missing_operator = _post(context, incident.incident_id, json={"reason": "x"})
    empty_reason = _post(
        context, incident.incident_id, json={"reason": "", "operator": "ops"}
    )

    assert missing_operator.status_code == 422, missing_operator.text
    assert empty_reason.status_code == 422, empty_reason.text
    assert context.store.get_incident(incident.incident_id).state is (
        IncidentState.ESCALATED
    )


def test_after_the_api_close_the_next_xid_on_the_node_is_planned_again() -> None:
    context = _context()
    _, incident, workflow = ingest(
        context, _node_event(48, event_id="xid-1", gpu_uuid="GPU-a")
    )
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
    _, absorbed, _ = ingest(
        context, _node_event(48, event_id="xid-2", gpu_uuid="GPU-a")
    )
    assert absorbed.incident_id == incident.incident_id, "record-only while ESCALATED"

    response = _post(context, incident.incident_id)
    _, fresh, fresh_workflow = ingest(
        context, _node_event(48, event_id="xid-3", gpu_uuid="GPU-a")
    )

    assert response.status_code == 200, response.text
    assert fresh.incident_id != incident.incident_id
    assert fresh_workflow.status is WorkflowStatus.PENDING
    assert context.orchestrator._workflow_merger.lifetime_record_only_total == 1
