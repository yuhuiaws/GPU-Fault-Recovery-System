"""The regional dispatcher signs the preflight's reservation; it never reserves.

``reserve_restart_budgets`` is the only site that writes a restart budget. A
RESTART_WORKLOAD step reaching ``RegionalRemoteWorkflowAdapter.execute`` reads
that reservation back into the ``RestartAuthorization`` the remote command
carries, and a step without one fails closed instead of taking a second
reservation under its idempotency key.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.execution import WorkflowStepContext
from gpu_fault.execution.restart_budget_preflight import (
    issue_restart_authorization,
    reservation_id,
    reserve_restart_budgets,
)
from gpu_fault.models import RestartAuthorization, WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from gpu_fault.store import InMemoryStore, NotFoundError
from tests._builders import build_store, copy_model
from tests.regional._regional_support import TOKEN_A, registration, workflow_state

OWNER = "gpu-fault-kubernetes-adapter"
RESTART_PARAMETERS = {
    "cluster_id": "cluster-a",
    "job_id": "training-a",
    "source_attempt_id": "training-a-a001",
    "source_gpu_count": 8,
    "restart_budget": 1,
}


def _restart_context() -> WorkflowStepContext:
    base = workflow_state()
    step = copy_model(
        base.step,
        operation=WorkflowOperation.RESTART_WORKLOAD,
        workload_ids=["training/pytorchjob/training-a"],
        parameters=RESTART_PARAMETERS,
    )
    workflow = copy_model(base.workflow, official_steps=[step])
    return replace(
        base, workflow=workflow, step=step, idempotency_key=reservation_id(workflow, 0)
    )


def _adapter(store: InMemoryStore) -> RegionalRemoteWorkflowAdapter:
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    return RegionalRemoteWorkflowAdapter(store, owners={OWNER})


def test_remote_restart_command_carries_the_preflight_reservation() -> None:
    store = build_store()
    adapter = _adapter(store)
    context = _restart_context()
    assert (
        reserve_restart_budgets(
            store, context.workflow, context.incident, context.workflow.official_steps
        )
        is None
    )

    outcome = adapter.execute(context)

    commands = store.list_remote_commands()
    state = store.get_restart_budget("cluster-a", "training-a")
    assert outcome.status is WorkflowStepStatus.WAITING, outcome
    assert len(commands) == 1
    assert commands[0].restart_authorization == RestartAuthorization(
        cluster_id="cluster-a",
        job_id="training-a",
        source_attempt_id="training-a-a001",
        source_gpu_count=8,
        restart_budget=1,
        restart_count=1,
        reservation_id=context.idempotency_key,
    )
    assert commands[0].restart_authorization == issue_restart_authorization(
        store, context.incident, context.step, context.idempotency_key
    )
    # One reservation, made by the preflight; dispatch added none.
    assert state.reservation_ids == [context.idempotency_key]


def test_remote_restart_without_a_reservation_fails_closed_and_mints_nothing() -> None:
    store = build_store()
    adapter = _adapter(store)
    context = _restart_context()

    outcome = adapter.execute(context)

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["reason"] == "RESTART_RESERVATION_MISSING"
    assert outcome.details["reservation_id"] == context.idempotency_key
    assert store.list_remote_commands() == []
    # Failing closed reserved nothing either: there is still no budget row.
    with pytest.raises(NotFoundError):
        store.get_restart_budget("cluster-a", "training-a")
