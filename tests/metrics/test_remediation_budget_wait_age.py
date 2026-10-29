"""How long one workflow has waited for remediation budget (design §9.5).

The refusal that begins a wait appends one ``REMEDIATION_BUDGET_WAIT`` hold
event; refusals after it append nothing, an admission ends the wait, and the
longest current wait is rendered per scope type of the refusing scope.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app.remediation_budget_metrics import (
    budget_tally,
    remediation_budget_lines,
)
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store import InMemoryStore, RemediationBudgetError
from gpu_fault.store.shared.remediation_budgets import (
    blocked_by_remediation_budget,
    budget_wait_started_at,
)
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
JOB_DOMAIN = "domain:cluster-a:job:job-7"
CLAIMS = {JOB_DOMAIN: 1, "node:cluster-a:node-a": 1}


def _seed(store: InMemoryStore, name: str):
    incident = fault_incident(f"incident-{name}", f"event-{name}", job_id="job-7")
    workflow = workflow_request(
        f"workflow-{name}",
        incident.incident_id,
        official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
        created_at=NOW - timedelta(minutes=10),
        updated_at=NOW - timedelta(minutes=10),
    )
    store.save_incident_and_workflow(
        incident.model_copy(update={"workflow_request_id": workflow.request_id}),
        workflow,
    )
    return workflow


def _claim(store: InMemoryStore, workflow, executor: str, at: datetime):
    return store.claim_workflow(
        workflow.request_id,
        executor,
        workflow.fencing_token,
        now=at,
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=CLAIMS,
    )


def _wait_events(workflow) -> list:
    code = WorkflowEventCode.REMEDIATION_BUDGET_WAIT.value
    return [
        event
        for event in workflow.events
        if event.kind is WorkflowEventKind.HOLD and event.code == code
    ]


def test_the_refusal_that_begins_a_wait_records_its_start_once() -> None:
    store = InMemoryStore()
    holder = _claim(store, _seed(store, "holder"), "executor-a", NOW)
    waiter = _seed(store, "waiter")

    for offset in (0, 60, 120):
        with pytest.raises(RemediationBudgetError):
            _claim(store, waiter, "executor-b", NOW + timedelta(seconds=offset))
    waiting = store.get_workflow(waiter.request_id)

    (event,) = _wait_events(waiting)
    assert event.at == NOW
    assert event.details["scope"] == JOB_DOMAIN
    assert waiting.remediation_budget_wait_count == 3
    assert budget_wait_started_at(waiting) == NOW

    # The holder ends; the admission ends the wait.
    store.save_workflow_if_leased(
        holder.model_copy(
            update={
                "status": WorkflowStatus.SUCCEEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        ),
        "executor-a",
        holder.execution_epoch,
        now=NOW + timedelta(seconds=10),
    )
    admitted = _claim(store, waiting, "executor-b", NOW + timedelta(seconds=180))
    assert budget_wait_started_at(admitted) is None

    # A later refusal of the same row begins a new wait.
    again = blocked_by_remediation_budget(
        admitted, "full", scope=JOB_DOMAIN, now=NOW + timedelta(hours=1)
    )
    assert [item.at for item in _wait_events(again)] == [NOW, NOW + timedelta(hours=1)]
    assert budget_wait_started_at(again) == NOW + timedelta(hours=1)


def test_a_row_an_older_release_refused_counts_from_dispatch_eligibility() -> None:
    legacy = workflow_request(
        "workflow-legacy",
        "incident-legacy",
        created_at=NOW - timedelta(hours=2),
        not_before=NOW - timedelta(minutes=90),
        remediation_budget_last_blocked_reason="full",
        remediation_budget_last_blocked_scope="cluster:cluster-a",
    )

    assert budget_wait_started_at(legacy) == NOW - timedelta(minutes=90)
    assert budget_wait_started_at(workflow_request("w", "i")) is None


def _waiter(name: str, scope: str, waited: timedelta, **values):
    return blocked_by_remediation_budget(
        workflow_request(f"workflow-{name}", f"incident-{name}", **values),
        "full",
        scope=scope,
        now=NOW - waited,
    )


def test_the_longest_wait_is_rendered_per_refusing_scope_type() -> None:
    rows = [
        _waiter("a", JOB_DOMAIN, timedelta(seconds=100)),
        _waiter("b", "domain:cluster-a:node:node-b", timedelta(seconds=4000)),
        _waiter("c", "node:cluster-a:node-c", timedelta(seconds=70)),
        _waiter("d", "cluster:cluster-a", timedelta(seconds=5000)),
        # Ended: refused once, then SUCCEEDED -- no longer a waiter.
        _waiter("e", JOB_DOMAIN, timedelta(hours=9), status=WorkflowStatus.SUCCEEDED),
    ]

    tally = budget_tally(rows, NOW)
    lines = remediation_budget_lines(tally, config=None)

    assert tally.longest_wait_by_scope_type == {
        "domain": 4000.0,
        "node": 70.0,
        "cluster": 5000.0,
    }
    assert (
        'gpu_fault_remediation_budget_longest_wait_seconds{scope_type="domain"} 4000'
    ) in lines
    assert (
        'gpu_fault_remediation_budget_longest_wait_seconds{scope_type="node"} 70'
    ) in lines
    assert "gpu_fault_remediation_budget_waiting_workflows 4" in lines


def test_no_waiter_renders_no_longest_wait_series() -> None:
    lines = remediation_budget_lines(budget_tally([], NOW), config=None)

    assert "# TYPE gpu_fault_remediation_budget_longest_wait_seconds gauge" in lines
    assert not [
        line
        for line in lines
        if line.startswith("gpu_fault_remediation_budget_longest_wait_seconds{")
    ], "a scope type without a waiter renders no series"
