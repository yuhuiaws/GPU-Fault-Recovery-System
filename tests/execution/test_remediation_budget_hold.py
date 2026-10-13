"""A remediation budget is held from admission until the workflow ends.

Batch 88 reproduced the defect end to end: once the first budgeted step
(STOP_WORKLOADS) completed, ``remediation_budget_claims`` returned ``{}`` and
the next claim cleared the held scopes, so with ``cluster_limit=1`` a second
workflow started in the cluster while the first one's RESTART_NODE was still
WAITING. These tests are those reproductions plus the hold's edges: a lapsed
lease, compensation, a preempting successor and a record that lost its claims
to an older release.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    remediation_budget_claims,
)
from gpu_fault.models import WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import InMemoryStore, RemediationBudgetError, SqliteStore
from tests._builders import (
    copy_model,
    execute_workflow,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._budget_hold_support import (
    QUIESCE,
    REBOOT,
    REBOOT_PLAN,
    RESET,
    RESTORE_GPU,
    STOP,
    VALIDATE,
    ScriptedAdapter,
    budget_executor,
    holders,
    seed,
)

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
CLUSTER = "cluster:cluster-a"
ONE_PER_CLUSTER = RemediationBudgetPolicy(region_limit=20, cluster_limit=1)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryStore()
        return
    sqlite = SqliteStore(str(tmp_path / "budget-hold.db"))
    try:
        yield sqlite
    finally:
        sqlite.close()


def _workflows(store, *request_ids: str):
    return [store.get_workflow(request_id) for request_id in request_ids]


def test_a_started_reboot_keeps_the_cluster_budget_until_the_workflow_ends(store):
    """budget_e2e.py: cluster_limit 1 holds B until A is terminal."""

    a = seed(store, "a", created_at=NOW)
    b = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    adapter = ScriptedAdapter(REBOOT_PLAN)
    executor = budget_executor(store, adapter, ONE_PER_CLUSTER)

    first = execute_workflow(executor, a.request_id)
    assert first.status is WorkflowStatus.RUNNING
    with pytest.raises(RemediationBudgetError) as refused:
        execute_workflow(executor, b.request_id)
    assert refused.value.scope == CLUSTER

    # A's next dispatch: STOP is complete, RESTART_NODE still WAITING. This is
    # the claim that used to drop every held scope.
    again = execute_workflow(executor, a.request_id)
    assert again.status is WorkflowStatus.RUNNING
    held = store.get_workflow(a.request_id)
    assert held.completed_step_indexes == [0]
    assert CLUSTER in held.remediation_budget_claims, (
        "a workflow lost its cluster claim while its reboot was in flight"
    )
    with pytest.raises(RemediationBudgetError):
        execute_workflow(executor, b.request_id)
    assert adapter.started() == {a.request_id}, adapter.calls

    adapter.released.add(a.request_id)
    assert execute_workflow(executor, a.request_id).status is WorkflowStatus.SUCCEEDED
    second = execute_workflow(executor, b.request_id)
    assert second.status is WorkflowStatus.RUNNING
    assert holders(_workflows(store, a.request_id, b.request_id), CLUSTER) == [
        b.request_id
    ]
    first_b_call = next(
        index for index, key in enumerate(adapter.calls) if key.startswith("workflow-b")
    )
    assert adapter.calls[first_b_call - 1] == f"{a.request_id}/1/{REBOOT.value}", (
        "B reached an adapter before A's reboot ended"
    )


def test_a_lapsed_lease_does_not_release_a_started_remediation(store):
    """The executor of A died mid-reboot: the lease lapses, the reboot does not.

    A dispatcher hand-over (every control-plane roll) lapses every WAITING
    lease the old holder kept; counting only live leases let a PENDING row of
    the same tick take the slot before A was re-claimed.
    """

    a = seed(store, "a", created_at=NOW)
    b = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    adapter = ScriptedAdapter(REBOOT_PLAN)
    execute_workflow(budget_executor(store, adapter, ONE_PER_CLUSTER), a.request_id)
    running = store.get_workflow(a.request_id)
    assert running.execution_lease_expires_at is not None, "WAITING keeps a lease"
    store.save_workflow(
        copy_model(
            running,
            execution_lease_expires_at=datetime.now(timezone.utc)
            - timedelta(seconds=1),
        )
    )

    claims = remediation_budget_claims(
        ONE_PER_CLUSTER, b, store.get_incident(b.incident_id), b.official_steps
    )
    with pytest.raises(RemediationBudgetError) as refused:
        store.claim_workflow(
            b.request_id,
            "executor-b",
            b.fencing_token,
            remediation_budget_claims=claims,
        )
    assert refused.value.scope == CLUSTER

    # Another executor takes A over after the lapse: A is never refused, its
    # hold survives the handover, and B still waits.
    taken = budget_executor(store, adapter, ONE_PER_CLUSTER, executor_id="executor-b")
    assert execute_workflow(taken, a.request_id).status is WorkflowStatus.RUNNING
    assert store.get_workflow(a.request_id).execution_owner_id == "executor-b"
    assert CLUSTER in store.get_workflow(a.request_id).remediation_budget_claims
    with pytest.raises(RemediationBudgetError):
        execute_workflow(taken, b.request_id)


def test_a_record_that_lost_its_claims_takes_them_back_without_a_refusal(store):
    """A row an older release left RUNNING with no claims after its STOP."""

    holder = seed(store, "holder", created_at=NOW)
    legacy = seed(store, "legacy", created_at=NOW + timedelta(seconds=1))
    adapter = ScriptedAdapter(REBOOT_PLAN)
    executor = budget_executor(store, adapter, ONE_PER_CLUSTER)
    execute_workflow(executor, holder.request_id)
    store.save_workflow(
        copy_model(
            store.get_workflow(legacy.request_id),
            status=WorkflowStatus.RUNNING,
            completed_step_indexes=[0],
            completed_operations=[STOP],
            step_executions=[workflow_step_execution(0, STOP)],
        )
    )

    resumed = execute_workflow(executor, legacy.request_id)

    assert resumed.status is WorkflowStatus.RUNNING
    row = store.get_workflow(legacy.request_id)
    assert CLUSTER in row.remediation_budget_claims, (
        "a started workflow must count again once this release dispatches it"
    )
    assert holders(
        _workflows(store, holder.request_id, legacy.request_id), CLUSTER
    ) == [holder.request_id, legacy.request_id]


def test_compensation_is_never_refused_and_keeps_its_hold(store):
    """A parked failure owes RESTORE_GPU_SERVICES; the budget cannot hold it."""

    full = seed(store, "full", created_at=NOW)
    parked = seed(store, "parked", created_at=NOW + timedelta(seconds=1))
    execute_workflow(
        budget_executor(store, ScriptedAdapter(REBOOT_PLAN), ONE_PER_CLUSTER),
        full.request_id,
    )
    resumed = copy_model(
        parked,
        completed_step_indexes=[0],
        completed_operations=[STOP],
        step_executions=[
            workflow_step_execution(0, STOP),
            workflow_step_execution(1, REBOOT, WorkflowStepStatus.FAILED),
        ],
        pending_failure_step_index=1,
        pending_failure_error="reboot failed",
        remediation_budget_claims=[CLUSTER, "region"],
        remediation_budget_limits={CLUSTER: 1, "region": 20},
    )
    store.save_workflow(resumed)
    incident = store.get_incident(parked.incident_id)

    claimed = store.claim_workflow(
        parked.request_id,
        "executor-a",
        parked.fencing_token,
        remediation_budget_claims=remediation_budget_claims(
            ONE_PER_CLUSTER, resumed, incident, resumed.official_steps
        ),
        remediation_budget_hold=True,
    )

    assert claimed.status is WorkflowStatus.RUNNING
    assert CLUSTER in claimed.remediation_budget_claims


# ---------------------------------------------------------------- preemption


def test_a_preempting_successor_takes_over_the_hold_without_a_gap(store):
    """Exactly one hold through the handoff: A's until S claims, then S's.

    A stopped its workload and is waiting on a read-only validation, a boundary
    a stronger action may take. S inherits the STOP. Between A's supersession
    and S's claim, B must not slip into the cluster's only slot, and S must not
    be refused for the slot its own predecessor reserved for it.
    """

    a = seed(store, "a", created_at=NOW, operations=[STOP, VALIDATE, RESET])
    adapter = ScriptedAdapter(
        [STOP, VALIDATE, RESET, REBOOT], waiting=[VALIDATE, REBOOT]
    )
    executor = budget_executor(store, adapter, ONE_PER_CLUSTER)
    assert execute_workflow(executor, a.request_id).status is WorkflowStatus.RUNNING

    successor = workflow_request(
        "workflow-s",
        a.incident_id,
        fencing_token=a.fencing_token,
        predecessor_workflow_id=a.request_id,
        preempt_predecessor=True,
        preemption_reason="strictly stronger recovery action",
        runtime_profile_version="active-v1",
        official_action="RESTART_NODE",
        official_steps=[
            workflow_step(STOP, node_ids=["node-a"]),
            workflow_step(REBOOT, node_ids=["node-a"]),
        ],
        completed_step_indexes=[0],
        completed_operations=[STOP],
        inherited_step_indexes=[0],
        step_executions=[workflow_step_execution(0, STOP)],
        created_at=NOW + timedelta(seconds=2),
        updated_at=NOW + timedelta(seconds=2),
    )
    store.save_workflow(successor)
    store.save_incident(
        copy_model(
            store.get_incident(a.incident_id), workflow_request_id=successor.request_id
        )
    )
    b = seed(store, "b", created_at=NOW + timedelta(seconds=1))

    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 1
    assert execute_workflow(executor, a.request_id).status is WorkflowStatus.SUPERSEDED
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 1, (
        "the superseded predecessor's hold must last until its successor claims"
    )
    with pytest.raises(RemediationBudgetError) as refused:
        execute_workflow(executor, b.request_id)
    assert refused.value.scope == CLUSTER

    assert execute_workflow(executor, successor.request_id).status is (
        WorkflowStatus.RUNNING
    )
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 1, (
        "the handoff counted the predecessor and the successor at once"
    )
    workflows = _workflows(store, a.request_id, b.request_id, successor.request_id)
    assert holders(workflows, CLUSTER) == [successor.request_id]
    with pytest.raises(RemediationBudgetError):
        execute_workflow(executor, b.request_id)
    assert b.request_id not in adapter.started(), adapter.calls


def test_a_reset_successor_adopting_a_live_quiesce_takes_over_the_hold(store):
    """The quiesce handoff window: A quiesced the GPUs and gave way to S without
    restoring them; S adopts the live quiesce right after its claim. The slot
    A held is S's alone from A's supersession to S's end."""

    a = seed(store, "a", created_at=NOW, operations=[QUIESCE, RESET, RESTORE_GPU])
    successor = workflow_request(
        "workflow-s",
        a.incident_id,
        fencing_token=a.fencing_token,
        predecessor_workflow_id=a.request_id,
        preempt_predecessor=True,
        preemption_reason="strictly stronger recovery action",
        runtime_profile_version="active-v1",
        official_steps=[
            workflow_step(op, node_ids=["node-a"])
            for op in (QUIESCE, RESET, RESTORE_GPU)
        ],
        created_at=NOW + timedelta(seconds=2),
        updated_at=NOW + timedelta(seconds=2),
    )
    store.save_workflow(successor)
    held = copy_model(
        store.get_workflow(a.request_id),
        status=WorkflowStatus.SUPERSEDED,
        preempted_by_workflow_id=successor.request_id,
        preemption_reason="strictly stronger recovery action",
        completed_step_indexes=[0],
        completed_operations=[QUIESCE],
        step_executions=[
            workflow_step_execution(
                0, QUIESCE, adapter_operation_id="remote/quiesce-proof"
            )
        ],
        remediation_budget_claims=[CLUSTER, "node:cluster-a:node-a", "region"],
    )
    store.save_workflow(held)
    store.save_incident(
        copy_model(
            store.get_incident(a.incident_id), workflow_request_id=successor.request_id
        )
    )
    b = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    adapter = ScriptedAdapter([QUIESCE, RESET, RESTORE_GPU, STOP, REBOOT])
    executor = budget_executor(store, adapter, ONE_PER_CLUSTER)

    with pytest.raises(RemediationBudgetError):
        execute_workflow(executor, b.request_id)
    assert execute_workflow(executor, successor.request_id).status is (
        WorkflowStatus.SUCCEEDED
    )

    finished = store.get_workflow(successor.request_id)
    assert finished.quiesce_handoff_from_workflow_id == a.request_id
    assert f"{successor.request_id}/0/{QUIESCE.value}" not in adapter.calls, (
        "the successor re-quiesced GPUs its predecessor left quiesced"
    )
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 0
    assert execute_workflow(executor, b.request_id).status is WorkflowStatus.RUNNING
