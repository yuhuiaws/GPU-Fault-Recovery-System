"""Workflow ownership, budget and CAS races in each PostgreSQL storage mode."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from gpu_fault.compile_blocked import close_compile_blocked_workflows
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.errors import (
    RemediationBudgetError,
    StaleWriteError,
    WorkflowLeaseError,
)
from tests._builders import active_workflow_executor, fault_incident, workflow_request
from tests.execution._cov95_runtime_workflows import RecordingAdapter
from tests.execution._support import workflow_state
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_postgres_workflow_state_tables import select_mode

database = state_tables.database
migration_database = state_tables.migration_database
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def store(migration_database, request):
    select_mode(migration_database, "workflow", request.param)
    assert POSTGRES_URL is not None, "workflow runtime tests require local PostgreSQL"
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()


def test_concurrent_claims_have_one_execution_owner(store) -> None:
    flow = workflow_request("claim-race", "claim-incident")
    store.save_workflow(flow)
    ready = Barrier(2)
    now = datetime.now(UTC)

    def claim(owner: str) -> WorkflowRequest | None:
        ready.wait(timeout=10)
        try:
            return store.claim_workflow(
                flow.request_id, owner, flow.fencing_token, now=now
            )
        except WorkflowLeaseError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(claim, ("executor-left", "executor-right")))
    winners = [result for result in results if result is not None]

    assert len(winners) == 1, "two executors acquired the same workflow lease"
    assert winners[0].execution_epoch == 1
    assert store.get_workflow(flow.request_id) == winners[0]


def test_concurrent_budget_claims_admit_only_one_workflow(store) -> None:
    flows = [
        workflow_request(f"budget-{side}", f"budget-incident-{side}")
        for side in ("left", "right")
    ]
    for flow in flows:
        store.save_workflow(flow)
    ready = Barrier(2)
    now = datetime.now(UTC)
    claims = {"region": 1, "cluster:cluster-a": 1}

    def claim(flow: WorkflowRequest) -> str:
        ready.wait(timeout=10)
        try:
            store.claim_workflow(
                flow.request_id,
                flow.request_id,
                flow.fencing_token,
                now=now,
                lease_duration=timedelta(seconds=30),
                remediation_budget_claims=claims,
            )
        except RemediationBudgetError:
            return "refused"
        return "admitted"

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(claim, flows))

    assert sorted(results) == ["admitted", "refused"], (
        "concurrent claims exceeded the durable remediation budget"
    )
    admitted = store.get_workflow(flows[results.index("admitted")].request_id)
    refused = store.get_workflow(flows[results.index("refused")].request_id)
    assert admitted.status is WorkflowStatus.RUNNING
    assert admitted.remediation_budget_claims == sorted(claims)
    assert refused.execution_owner_id is None, "a refused claim retained an executor"
    assert refused.execution_lease_expires_at is None, (
        "a refused claim retained a lease"
    )
    assert refused.remediation_budget_wait_count == 1
    assert refused.remediation_budget_last_blocked_scope == "cluster:cluster-a"

    acquired = store.claim_workflow(
        refused.request_id,
        "executor-after-expiry",
        refused.fencing_token,
        now=now + timedelta(seconds=31),
        remediation_budget_claims=claims,
    )
    assert acquired.remediation_budget_claims == sorted(claims)
    assert acquired.remediation_budget_last_blocked_scope is None, (
        "an expired competing lease kept the budget blocked"
    )


def test_budget_extension_is_atomic_and_observes_terminal_release(store) -> None:
    now = datetime.now(UTC)
    for name in ("holder", "extender"):
        store.save_workflow(workflow_request(name, f"incident-{name}"))
    holder = store.get_workflow("holder")
    extender = store.get_workflow("extender")
    leased = store.claim_workflow(
        holder.request_id,
        "holder-owner",
        holder.fencing_token,
        now=now,
        remediation_budget_claims={"region": 1},
    )
    before = store.claim_workflow(
        extender.request_id,
        "extender-owner",
        extender.fencing_token,
        now=now,
        remediation_budget_claims={},
    )

    with pytest.raises(RemediationBudgetError, match="scope=region"):
        store.extend_remediation_budget(
            extender.request_id, "extender-owner", {"region": 1}, now=now
        )
    assert store.get_workflow(extender.request_id) == before

    store.save_workflow_if_leased(
        leased.model_copy(
            update={
                "status": WorkflowStatus.SUCCEEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        ),
        "holder-owner",
        leased.execution_epoch,
        now=now,
    )
    extended = store.extend_remediation_budget(
        extender.request_id, "extender-owner", {"region": 1}, now=now
    )
    assert extended.remediation_budget_claims == ["region"]
    assert extended.execution_epoch == before.execution_epoch


def test_concurrent_full_cas_writes_cannot_both_succeed(store) -> None:
    flow = workflow_request("cas-race", "cas-incident")
    store.save_workflow(flow)
    expected = store.get_workflow(flow.request_id)
    ready = Barrier(2)

    def write(reason: str) -> str | None:
        ready.wait(timeout=10)
        try:
            store.save_workflow(
                expected.model_copy(update={"blocked_reasons": [reason]}),
                expected=expected,
            )
        except StaleWriteError:
            return None
        return reason

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(write, ("left-write", "right-write")))
    winners = [result for result in results if result is not None]

    assert len(winners) == 1, "a stale expected snapshot overwrote a concurrent write"
    assert store.get_workflow(flow.request_id).blocked_reasons == winners


@pytest.mark.parametrize("uncertain", [False, True])
def test_uncertain_execution_keeps_postgres_node_occupancy_in_every_mode(
    store, uncertain: bool
) -> None:
    operations = [
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    ]
    incident, workflow = workflow_state(store, operations)
    now = datetime.now(UTC)
    incident = incident.model_copy(update={"created_at": now, "updated_at": now})
    workflow = workflow.model_copy(update={"created_at": now, "updated_at": now})
    store.save_incident_and_workflow(incident, workflow)
    adapter = RecordingAdapter()
    adapter.outcomes[WorkflowOperation.RESET_GPU] = WorkflowStepOutcome.failed(
        "node execution result",
        details=(
            {"outcome_unknown": True, "manual_confirmation_required": True}
            if uncertain
            else {}
        ),
    )
    executor = active_workflow_executor(store, [adapter], operations)
    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    saved = store.get_workflow(workflow.request_id)
    occupying = NodeConflictService(
        store, RecoveryArbiter()
    ).active_node_exclusive_workflow(incident.cluster_id, set(incident.node_ids))

    if uncertain:
        assert result.status is WorkflowStatus.BLOCKED
        assert saved.blocked_kind is BlockedKind.NEEDS_OPERATOR
        assert occupying is not None and occupying.request_id == saved.request_id
        assert [call.step.operation for call in adapter.calls] == operations[:2]
        current_incident = store.get_incident(incident.incident_id)
        store.save_incident(
            current_incident.model_copy(update={"state": IncidentState.RECOVERED}),
            expected=current_incident,
        )
        assert close_compile_blocked_workflows(store, now=datetime.now(UTC)) == []
        assert store.get_workflow(saved.request_id) == saved, (
            "settlement released an unresolved physical action after incident closure"
        )
        retained = NodeConflictService(
            store, RecoveryArbiter()
        ).active_node_exclusive_workflow(incident.cluster_id, set(incident.node_ids))
        assert retained is not None and retained.request_id == saved.request_id
    else:
        assert result.status is WorkflowStatus.FAILED
        assert occupying is None, "a completed compensation must not retain the node"
        assert [call.step.operation for call in adapter.calls] == operations


def test_full_cas_accepts_missing_payload_defaults_in_every_mode(
    store, migration_database
) -> None:
    flow = workflow_request("legacy-default-cas", "legacy-default-incident")
    store.save_workflow(flow)
    mode = migration_database.execute(
        "SELECT mode FROM gpu_fault_control_state_modes WHERE kind='workflow'"
    ).fetchone()[0]
    if mode == "dedicated":
        query = (
            "UPDATE gpu_fault_workflows "
            "SET payload=payload - 'placement_hold' - 'superseded_step_indexes' "
            "WHERE request_id=%s"
        )
    else:
        query = (
            "UPDATE gpu_fault_objects "
            "SET payload=payload - 'placement_hold' - 'superseded_step_indexes' "
            "WHERE kind='workflow' AND key=%s"
        )
    migration_database.execute(query, (flow.request_id,))
    expected = store.get_workflow(flow.request_id)
    assert expected == flow, "missing default fields must decode to the same model"
    settled = flow.model_copy(update={"status": WorkflowStatus.SUPERSEDED})

    store.save_workflow(settled, expected=expected)

    assert store.get_workflow(flow.request_id) == settled
    with pytest.raises(StaleWriteError):
        store.save_workflow(flow, expected=expected)
    assert store.get_workflow(flow.request_id) == settled


@pytest.mark.parametrize("kind", ["incident", "plan"])
def test_generic_cas_accepts_model_defaults_without_overwriting_real_changes(
    store, migration_database, kind: str
) -> None:
    if kind == "incident":
        record = fault_incident("generic-cas", "generic-cas-event")
        save, get = store.save_incident, store.get_incident
        updated = record.model_copy(update={"state": IncidentState.RECOVERED})
        missing_field = "workflow_request_id"
    else:
        record = RecoveryPlan(
            plan_id="generic-cas",
            incident_id="generic-cas-incident",
            attempt_id="generic-cas-attempt",
            trigger="terminal-event",
            runtime_profile_version="profile-v1",
            steps=[],
        )
        save, get = store.save_plan, store.get_plan
        updated = record.model_copy(update={"status": PlanStatus.SUPERSEDED})
        missing_field = "resolved_by_restore_workflow_id"
    save(record)
    migration_database.execute(
        "UPDATE gpu_fault_objects SET payload=payload - %s WHERE kind=%s AND key=%s",
        (missing_field, kind, "generic-cas"),
    )
    expected = get("generic-cas")
    assert expected == record, "the generic model's default must preserve read identity"

    save(updated, expected=expected)

    assert get("generic-cas") == updated
    with pytest.raises(StaleWriteError):
        save(record, expected=expected)
    assert get("generic-cas") == updated, "a stale generic CAS overwrote the winner"
