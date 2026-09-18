"""A recovered sibling cannot remove a failed node's terminal ownership hold."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from gpu_fault.execution.terminal_state import terminal_decision
from gpu_fault.models import IncidentState, WorkflowStatus, WorkflowStepStatus
from gpu_fault.regional import RemoteIncidentOwnershipReport
from gpu_fault.store import InMemoryStore
from gpu_fault.workflow_quarantine import (
    TERMINAL_QUARANTINE_NODES,
    has_terminal_quarantine_hold,
)
from tests._builders import asgi_client, build_context, fault_incident
from tests.execution.test_terminal_quarantine_rebinding import (
    OP,
    OwnershipProvider,
    RecordingAdapter,
    RecordingCore,
    context_for,
    execute_plan,
    isolated_node,
    plan,
    scheduler,
    step,
)
from tests.regional._regional_support import TOKEN_A, registration

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def mixed_plan(*, dag: bool = False, explicit: bool = True):
    return plan(
        [
            step(OP.MARK_UNSCHEDULABLE),
            step(
                OP.QUARANTINE,
                ("node-b",),
                dependencies=(0,) if dag else (),
                parameters=(
                    {TERMINAL_QUARANTINE_NODES: ["node-b"]} if explicit else {}
                ),
            ),
            step(OP.RESTORE_SCHEDULING, dependencies=(1,) if dag else ()),
        ],
        dag=dag,
    )


@pytest.mark.parametrize("dag", [False, True], ids=["flat", "dag"])
@pytest.mark.parametrize("explicit", [False, True], ids=["legacy", "explicit"])
def test_executed_mixed_branch_retains_quarantined_incident(dag, explicit):
    workflow = mixed_plan(dag=dag, explicit=explicit)
    core = RecordingCore({name: isolated_node(name) for name in ("node-a", "node-b")})
    result, store = execute_plan(
        workflow, core, RecordingAdapter(), nodes=("node-a", "node-b")
    )
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert core.nodes["node-a"]["spec"]["unschedulable"] is False, (
        "the completed healthy branch must regain its original scheduling state"
    )
    assert core.nodes["node-b"]["spec"]["unschedulable"] is True, (
        "the terminal quarantine remains on the failed sibling"
    )
    incident = store.get_incident(workflow.incident_id)
    assert incident.state is IncidentState.QUARANTINED, (
        "successful readmission of A must not mark B's persistent hold recovered"
    )
    ended = store.get_workflow(workflow.request_id)
    assert ended.events[-1].details["incident_state"] == "QUARANTINED", (
        "terminal audit and persisted incident must agree"
    )


def ownership_report(store: InMemoryStore, incident_id: str):
    context = build_context(store=store)
    context.regional_mode = True
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))

    async def read():
        async with asgi_client(context) as client:
            response = await client.get(
                "/v1/regional/executors/incident-ownership",
                headers={
                    "Authorization": f"Bearer {TOKEN_A}",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
                params={"incident_id": incident_id},
            )
            assert response.status_code == 200, response.text
            return RemoteIncidentOwnershipReport.model_validate(response.json())

    return asyncio.run(read())


@pytest.mark.parametrize("local", [False, True], ids=["regional-api", "local-store"])
@pytest.mark.parametrize("preserving", [False, True], ids=["reset-only", "keep-hold"])
@pytest.mark.parametrize(
    "prior_state",
    [IncidentState.RECOVERED, IncidentState.ESCALATED, IncidentState.QUARANTINED],
)
def test_terminal_takeover_observes_node_hold_even_on_previously_misclassified_rows(
    local, preserving, prior_state
):
    old = mixed_plan().model_copy(
        update={
            "status": WorkflowStatus.SUCCEEDED,
            "completed_step_indexes": [0, 1, 2],
            "completed_operations": [
                OP.MARK_UNSCHEDULABLE,
                OP.QUARANTINE,
                OP.RESTORE_SCHEDULING,
            ],
        }
    )
    store = InMemoryStore()
    store.save_workflow(old)
    store.save_incident(
        fault_incident(
            old.incident_id,
            "old-event",
            node_ids=["node-a", "node-b"],
            workflow_request_id=old.request_id,
            state=prior_state,
        )
    )
    report = ownership_report(store, old.incident_id)
    assert report.known and report.terminal and report.quarantine_hold, (
        "the production ownership route must not trust an old global RECOVERED flag"
    )
    candidate = plan(
        [
            step(OP.MARK_UNSCHEDULABLE, ("node-b",)),
            step(OP.QUARANTINE if preserving else OP.RESTORE_SCHEDULING, ("node-b",)),
        ]
    ).model_copy(update={"request_id": "new-workflow", "incident_id": "new-incident"})
    core = RecordingCore({"node-b": isolated_node("node-b")})
    before = deepcopy(core.nodes)
    adapter = scheduler(
        core,
        store=store if local else None,
        provider=None if local else OwnershipProvider(report),
    )
    outcome = adapter.execute(context_for(candidate, 0))
    if preserving:
        assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
        assert core.nodes["node-b"]["spec"]["unschedulable"] is True, (
            "a preserving successor can take ownership without readmitting B"
        )
    else:
        assert outcome.status is WorkflowStepStatus.FAILED, outcome
        assert outcome.details["safety_rejection"] is True, outcome
        assert core.nodes == before, "refused takeover must not patch the node"


@pytest.mark.parametrize("safety", [False, True], ids=["official", "safety"])
def test_pending_restore_does_not_clear_completed_isolation(safety):
    steps = [
        step(OP.QUARANTINE, ("node-b",)),
        step(OP.RESTORE_SCHEDULING),
        step(OP.RESTORE_SCHEDULING, ("node-b",)),
    ]
    workflow = plan(
        [] if safety else steps,
        safety_steps=steps if safety else [],
        safety_only=safety,
        completed_step_indexes=[0, 1],
        completed_operations=[OP.QUARANTINE, OP.RESTORE_SCHEDULING],
    )
    assert has_terminal_quarantine_hold(workflow), (
        "a planned but unexecuted restore on B is not evidence of readmission"
    )


def test_retired_or_malformed_readmission_metadata_fails_closed():
    workflow = mixed_plan()
    workflow.official_steps.append(step(OP.RESTORE_SCHEDULING, ("node-b",)))
    workflow.superseded_step_indexes = [3]
    assert has_terminal_quarantine_hold(workflow), (
        "retiring a restore cannot release the failed node's hold"
    )
    workflow.official_steps[1].parameters[TERMINAL_QUARANTINE_NODES] = "node-b"
    assert has_terminal_quarantine_hold(workflow), (
        "malformed scope must refuse takeover instead of dropping the hold"
    )


@pytest.mark.parametrize(
    "status", [WorkflowStatus.SUCCEEDED, WorkflowStatus.SUPERSEDED]
)
def test_terminal_funnel_preserves_node_scope_and_input_records(status):
    workflow = mixed_plan().model_copy(
        update={
            "completed_step_indexes": [0, 1, 2],
            "completed_operations": [
                OP.MARK_UNSCHEDULABLE,
                OP.QUARANTINE,
                OP.RESTORE_SCHEDULING,
            ],
        }
    )
    incident = fault_incident(workflow.incident_id, "event")
    before = workflow.model_copy(deep=True)
    decision = terminal_decision(
        workflow, incident, status, 1, now=NOW, actor="alignment-test"
    )
    assert decision.incident is not None, "the existing incident must be retained"
    assert decision.incident.state is IncidentState.QUARANTINED, (
        "all normal terminal paths must preserve node-scoped quarantine"
    )
    assert workflow == before, "terminal derivation must not rewrite the input plan"
