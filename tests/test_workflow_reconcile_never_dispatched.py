"""The never-dispatched PENDING shape, as the Pod plans it.

On 2026-10-01 HyperPod reclaimed a spot node while its last telemetry was being
ingested; two workflows were created for its incidents and stayed PENDING with
no step execution, no remote command and no event -- the node's Agent was gone
and the fleet preflight never dispatched them. ``workflow-reconcile`` scanned
BLOCKED only, so the release preflight blocked every deploy on them.

The Pod side now plans the shape beside the BLOCKED backlog: PENDING, never
dispatched, no remote command, older than the dispatch guard, its incident
still naming it. It never makes the item eligible -- the departed-node proof is
the administrator's -- and the deployed apply keeps refusing it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.workflow_reconcile import (
    NEVER_DISPATCHED,
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
    never_dispatched_plan_items,
)
from gpu_fault.workflow_resolution import (
    DEPARTED_NODE_PROOF_REASON,
    NEVER_DISPATCHED_GUARD_AGE,
    never_dispatched_reconciliation_reasons,
    workflow_never_dispatched,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.regional._regional_support import enqueue_remote_command

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
PLUGIN = WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN
NODE = "hyperpod-i-00000000000000001"
DEPARTED_ID = "workflow-departed"
INCIDENT_ID = "incident-xid-46"


def _store_factories(tmp_path: Path) -> tuple[Callable[[], Any], ...]:
    return (
        InMemoryStore,
        lambda: SqliteStore(str(tmp_path / "never-dispatched.sqlite")),
    )


def _pending(
    store: Any,
    request_id: str = DEPARTED_ID,
    incident_id: str = INCIDENT_ID,
    *,
    age: timedelta = timedelta(hours=1),
    incident_points_at: str | None = None,
    **values: Any,
) -> None:
    """A PENDING record nothing ever dispatched, and the incident that names it."""

    fields: dict[str, Any] = {
        "status": WorkflowStatus.PENDING,
        "fencing_token": 2,
        "execution_epoch": 0,
        "official_action": PLUGIN.value,
        "official_steps": [workflow_step(PLUGIN, node_ids=[NODE])],
        "created_at": NOW - age,
        "updated_at": NOW - age,
        **values,
    }
    workflow = workflow_request(request_id, incident_id, **fields)
    store.save_workflow(workflow)
    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{incident_id}",
            node_ids=[NODE],
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=incident_points_at or request_id,
            fencing_token=2,
            updated_at=NOW - age,
        )
    )


def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        close()


# ------------------------------------------------------------------- the rule


def test_the_departed_node_proof_is_the_only_reason_left_for_the_shape() -> None:
    store = InMemoryStore()
    _pending(store)
    workflow = store.get_workflow(DEPARTED_ID)

    reasons = never_dispatched_reconciliation_reasons(
        workflow, store.get_incident(INCIDENT_ID), [], evaluated_at=NOW
    )

    assert workflow_never_dispatched(workflow), "no step was ever handed out"
    assert reasons == [DEPARTED_NODE_PROOF_REASON]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {
                "status": WorkflowStatus.BLOCKED,
                "blocked_kind": BlockedKind.NEEDS_OPERATOR,
            },
            "workflow status is BLOCKED, not PENDING",
        ),
        (
            {
                "step_executions": [
                    workflow_step_execution(0, PLUGIN, WorkflowStepStatus.FAILED)
                ]
            },
            "workflow has step executions",
        ),
        ({"completed_operations": [PLUGIN]}, "workflow completed or superseded a step"),
        ({"execution_owner_id": "executor-a"}, "workflow still has an execution owner"),
        (
            {"execution_lease_expires_at": NOW + timedelta(minutes=5)},
            "workflow execution lease has not expired",
        ),
        (
            {"remediation_budget_claims": ["claim-a"]},
            "workflow holds remediation budget claims",
        ),
        (
            {"age": timedelta(minutes=9)},
            "workflow is younger than the 10-minute dispatch guard",
        ),
        (
            {"incident_points_at": "workflow-successor"},
            "incident names another workflow; the dispatcher sweep owns a "
            "retired generation",
        ),
    ],
)
def test_every_other_condition_keeps_the_record_out(
    changes: dict[str, Any], expected: str
) -> None:
    store = InMemoryStore()
    _pending(store, **changes)
    workflow = store.get_workflow(DEPARTED_ID)

    reasons = never_dispatched_reconciliation_reasons(
        workflow, store.get_incident(INCIDENT_ID), [], evaluated_at=NOW
    )

    assert expected in reasons, reasons
    assert reasons[-1] == DEPARTED_NODE_PROOF_REASON


def test_a_settled_remote_command_means_an_agent_acted_on_it() -> None:
    store = InMemoryStore()
    _pending(store)
    # The helper keys the command's workflow off its id: ``workflow-departed``.
    enqueue_remote_command(store, "departed", status=RemoteCommandStatus.FAILED)
    commands = store.list_remote_commands(workflow_request_ids=[DEPARTED_ID])
    assert len(commands) == 1, "the fixture wrote the command"

    reasons = never_dispatched_reconciliation_reasons(
        store.get_workflow(DEPARTED_ID),
        store.get_incident(INCIDENT_ID),
        commands,
        evaluated_at=NOW,
    )

    assert "workflow has remote commands" in reasons


def test_the_guard_age_is_ten_minutes() -> None:
    assert NEVER_DISPATCHED_GUARD_AGE == timedelta(minutes=10)


# ------------------------------------------------------------ the Pod planner


@pytest.mark.parametrize("factory_index", [0, 1])
def test_discovery_selects_only_aged_never_dispatched_pending_records(
    tmp_path: Path, factory_index: int
) -> None:
    store = _store_factories(tmp_path)[factory_index]()
    try:
        _pending(store)
        _pending(store, "workflow-young", "incident-young", age=timedelta(minutes=5))
        _pending(
            store,
            "workflow-ran",
            "incident-ran",
            step_executions=[
                workflow_step_execution(0, PLUGIN, WorkflowStepStatus.FAILED)
            ],
        )
        _pending(
            store,
            "workflow-blocked",
            "incident-blocked",
            status=WorkflowStatus.BLOCKED,
            blocked_kind=BlockedKind.NEEDS_OPERATOR,
        )

        items, report = never_dispatched_plan_items(store, now=NOW)

        assert [item["request_id"] for item in items] == [DEPARTED_ID]
        assert report == {
            "scanned": 3,
            "selected": 1,
            "remaining": 0,
            "scan_truncated": False,
        }
        [item] = items
        assert item["terminalization"] == NEVER_DISPATCHED
        assert item["eligible"] is False, "the Pod never proves the node departed"
        assert item["reasons"] == [DEPARTED_NODE_PROOF_REASON]
        assert item["workflow_created_at"] == (NOW - timedelta(hours=1)).isoformat()
        assert item["node_ids"] == [NODE]
        assert item["incident_state"] == "ACTION_PENDING"
        assert item["step_execution_count"] == 0
        assert item["remote_command_count"] == 0
    finally:
        _close(store)


def test_discovery_honours_the_incident_filter_and_the_batch_size() -> None:
    store = InMemoryStore()
    _pending(store)
    _pending(store, "workflow-other", "incident-other", age=timedelta(hours=2))

    by_incident, _report = never_dispatched_plan_items(
        store, incident_ids=["incident-other"], now=NOW
    )
    oldest_first, report = never_dispatched_plan_items(store, max_items=1, now=NOW)

    assert [item["request_id"] for item in by_incident] == ["workflow-other"]
    assert [item["request_id"] for item in oldest_first] == ["workflow-other"], (
        "the batch walks the backlog oldest first"
    )
    assert report is not None and report["remaining"] == 1


def test_an_explicit_id_is_judged_whatever_its_status() -> None:
    store = InMemoryStore()
    _pending(
        store,
        "workflow-blocked",
        "incident-blocked",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
    )

    items, report = never_dispatched_plan_items(
        store, ["workflow-blocked", "workflow-missing"], now=NOW
    )

    assert report is None
    assert items[0]["reasons"][0] == "workflow status is BLOCKED, not PENDING"
    assert items[1] == {
        "request_id": "workflow-missing",
        "eligible": False,
        "reasons": ["workflow does not exist"],
    }


def test_the_blocked_plan_and_the_deployed_apply_are_unchanged_by_the_shape() -> None:
    """BLOCKED discovery still sees BLOCKED only, and the deployed apply never
    closes a PENDING record: the proof it would need is not in the Store."""

    store = InMemoryStore()
    _pending(store)

    plan = build_workflow_reconcile_plan(store, now=NOW)
    explicit = build_workflow_reconcile_plan(store, [DEPARTED_ID], now=NOW)

    assert plan["items"] == [], "BLOCKED discovery does not list PENDING records"
    [item] = explicit["items"]
    assert item["eligible"] is False
    assert "workflow status is PENDING, not BLOCKED" in item["reasons"]
    with pytest.raises(ValueError, match="ineligible records"):
        apply_workflow_reconcile_plan(
            store,
            workflow_ids=[DEPARTED_ID],
            expected_plan_sha256=explicit["plan_sha256"],
            reference="CHG-2026-1001-01",
            now=NOW,
        )
    assert store.get_workflow(DEPARTED_ID).status is WorkflowStatus.PENDING
