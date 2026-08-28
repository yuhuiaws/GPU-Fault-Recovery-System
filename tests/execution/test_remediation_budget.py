from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    remediation_budget_claims,
)
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import InMemoryStore, RemediationBudgetError, SqliteStore
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _workflow(store, suffix: str, *, node_id: str = "node-a"):
    incident = fault_incident(
        f"incident-{suffix}",
        f"event-{suffix}",
        node_ids=[node_id],
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        f"workflow-{suffix}",
        incident.incident_id,
        WorkflowStatus.PENDING,
        fencing_token=3,
        official_steps=[
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[node_id])
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    incident = incident.model_copy(update={"workflow_request_id": workflow.request_id})
    store.save_incident_and_workflow(incident, workflow)
    return incident, workflow


@pytest.mark.parametrize("durable", [False, True])
def test_remediation_budget_is_durable_and_released_by_lease_expiry(
    tmp_path, durable: bool
) -> None:
    store = SqliteStore(str(tmp_path / "budget.db")) if durable else InMemoryStore()
    _, first = _workflow(store, "first", node_id="node-a")
    _, second = _workflow(store, "second", node_id="node-b")
    claims = {"region": 1, "cluster:cluster-a": 1}

    store.claim_workflow(
        first.request_id,
        "executor-a",
        first.fencing_token,
        now=NOW,
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=claims,
    )
    with pytest.raises(RemediationBudgetError, match="scope=cluster:cluster-a"):
        store.claim_workflow(
            second.request_id,
            "executor-b",
            second.fencing_token,
            now=NOW,
            lease_duration=timedelta(seconds=30),
            remediation_budget_claims=claims,
        )

    blocked = store.get_workflow(second.request_id)
    assert blocked.remediation_budget_wait_count == 1
    assert "cluster:cluster-a" in (blocked.remediation_budget_last_blocked_reason or "")

    claimed = store.claim_workflow(
        second.request_id,
        "executor-b",
        second.fencing_token,
        now=NOW + timedelta(seconds=31),
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=claims,
    )
    assert claimed.remediation_budget_claims == sorted(claims)


def test_remediation_budget_claims_cover_all_required_scopes() -> None:
    store = InMemoryStore()
    incident, workflow = _workflow(store, "scopes")
    step = workflow.official_steps[0].model_copy(
        update={
            "parameters": {
                "fabric_partition": "fabric-a",
                "availability_zone": "us-west-2a",
            }
        }
    )
    workflow = workflow.model_copy(update={"official_steps": [step]})
    policy = RemediationBudgetPolicy(
        region_limit=10,
        cluster_limit=4,
        node_limit=1,
        failure_domain_limit=1,
        resource_class_limit=2,
    )

    claims = remediation_budget_claims(
        policy, workflow, incident, workflow.official_steps
    )

    assert claims["region"] == 10
    assert claims["cluster:cluster-a"] == 4
    assert claims["node:cluster-a:node-a"] == 1
    assert claims["domain:cluster-a:fabric-a"] == 1
    assert claims["domain:cluster-a:us-west-2a"] == 1
    assert claims["class:cluster-a:NODE_LIFECYCLE_MUTATION"] == 2


def test_restore_compensation_never_waits_for_a_remediation_budget() -> None:
    store = InMemoryStore()
    incident, workflow = _workflow(store, "restore")
    restore = workflow_step(WorkflowOperation.RESTORE_GPU_SERVICES, node_ids=["node-a"])
    workflow = workflow.model_copy(
        update={
            "official_steps": [restore],
            "pending_failure_step_index": 0,
            "pending_failure_error": "reset failed",
        }
    )

    assert (
        remediation_budget_claims(
            RemediationBudgetPolicy(), workflow, incident, workflow.official_steps
        )
        == {}
    )
