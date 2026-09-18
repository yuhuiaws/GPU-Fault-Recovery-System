from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.execution import (
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    WorkflowExecutionError,
    WorkflowExecutionRequest,
)
from gpu_fault.models import (
    RestartAuthorization,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step_execution
from tests.execution._cov95_runtime_workflows import FlowHarness, RecordingAdapter

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
RESTART = WorkflowOperation.RESTART_WORKLOAD


@pytest.mark.parametrize("defect", ["timeout", "retry-limit"])
def test_executor_configuration_rejects_unbounded_or_negative_retry_inputs(
    defect: str,
) -> None:
    config = ProductionExecutorConfig(
        enabled=True,
        executor_id="unit-executor",
        allowed_operations=frozenset({FREEZE}),
        workflow_execution_timeout_seconds=0 if defect == "timeout" else 1800,
    )
    with pytest.raises(
        WorkflowExecutionError, match="must be positive|must not be negative"
    ):
        ProductionWorkflowExecutor(
            InMemoryStore(),
            [],
            config,
            step_transient_retry_limit=-1 if defect == "retry-limit" else 1,
        )


@pytest.mark.parametrize("defect", ["disabled", "empty"])
def test_executor_refusal_does_not_claim_or_mutate_workflow(defect: str) -> None:
    h = FlowHarness([FREEZE])
    if defect == "disabled":
        h.executor.config = replace(h.executor.config, enabled=False)
        expected = "executor is disabled"
    else:
        h.amend(official_steps=[])
        expected = "no executable steps"
    before = h.store.get_workflow(h.workflow.request_id)
    with pytest.raises(WorkflowExecutionError, match=expected):
        h.execute()
    assert h.store.get_workflow(h.workflow.request_id) == before, before
    assert h.adapter.calls == [], h.adapter.calls


def test_disallowed_operation_is_failed_without_calling_the_adapter() -> None:
    h = FlowHarness([FREEZE])
    h.executor.config = replace(h.executor.config, allowed_operations=frozenset())
    result = h.execute()
    assert result.status is WorkflowStatus.FAILED, result
    assert "not in GPU_FAULT_ALLOWED_OPERATIONS" in (result.error or ""), result
    assert h.adapter.calls == [], h.adapter.calls


@pytest.mark.parametrize("count", [0, 2])
def test_executor_requires_exactly_one_matching_adapter(count: int) -> None:
    h = FlowHarness([FREEZE])
    adapters = [RecordingAdapter() for _ in range(count)]
    h.executor.adapters = adapters
    result = h.execute()
    assert result.status is WorkflowStatus.FAILED, result
    assert f"found {count}" in (result.error or ""), result
    assert all(adapter.calls == [] for adapter in adapters), adapters


def test_supplied_restart_authorization_is_forwarded_without_duplicate_reservation() -> (
    None
):
    h = FlowHarness([RESTART])
    parameters = h.workflow.official_steps[0].parameters
    key = f"{h.workflow.request_id}/0/RESTART_WORKLOAD"
    authorization = RestartAuthorization(
        **parameters, reservation_id=key, restart_count=1
    )
    result = h.execute(
        WorkflowExecutionRequest(
            expected_fencing_token=3, restart_authorization=authorization
        )
    )
    assert result.status is WorkflowStatus.SUCCEEDED, result
    [context] = h.adapter.calls
    assert context.request.restart_authorization == authorization, context.request
    budget = h.store.get_restart_budget("cluster-a", "training-job")
    assert budget.restart_count == 1 and budget.reservation_ids == [key], budget


def test_graph_change_during_lease_renewal_defers_execution_until_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = FlowHarness([FREEZE])
    original = h.store.renew_workflow_lease
    changed = False

    def renew(*args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        workflow = original(*args, **kwargs)
        if not changed:
            changed = True
            workflow = workflow.model_copy(
                update={"dag_enabled": True, "dag_revision": 1}
            )
            h.store.save_workflow(workflow)
        return workflow

    monkeypatch.setattr(h.store, "renew_workflow_lease", renew)
    first = h.execute()
    assert first.status is WorkflowStatus.RUNNING and h.adapter.calls == [], first
    assert h.store.get_workflow(h.workflow.request_id).dag_enabled is True, first
    second = h.execute()
    assert second.status is WorkflowStatus.SUCCEEDED, second
    assert [context.step.operation for context in h.adapter.calls] == [FREEZE], (
        h.adapter.calls
    )


@pytest.mark.parametrize("release_needed", [False, True])
def test_already_withdrawn_workflow_runs_only_cleanup_for_touched_nodes(
    release_needed: bool,
) -> None:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    now = datetime.now(timezone.utc)
    h.amend(
        status=WorkflowStatus.RUNNING,
        workload_withdrawn_at=now,
        workload_withdrawn_reason="user stopped the workload",
        completed_step_indexes=[0] if release_needed else [],
        completed_operations=[QUIESCE] if release_needed else [],
        step_executions=(
            [workflow_step_execution(0, QUIESCE, WorkflowStepStatus.SUCCEEDED)]
            if release_needed
            else []
        ),
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUPERSEDED, result
    operations = [context.step.operation for context in h.adapter.calls]
    assert operations == ([RESTORE] if release_needed else []), operations
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.superseded_step_indexes == ([1] if release_needed else [0, 1, 2]), (
        saved
    )
    assert (
        saved.execution_owner_id is None and saved.execution_lease_expires_at is None
    ), saved


def test_already_withdrawn_resolved_workflow_does_not_replay_completed_step() -> None:
    h = FlowHarness([FREEZE])
    h.amend(
        status=WorkflowStatus.RUNNING,
        workload_withdrawn_at=datetime.now(timezone.utc),
        completed_step_indexes=[0],
        completed_operations=[FREEZE],
        step_executions=[
            workflow_step_execution(0, FREEZE, WorkflowStepStatus.SUCCEEDED)
        ],
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUPERSEDED, result
    assert h.adapter.calls == [], h.adapter.calls


def test_busy_lease_never_reaches_adapter_and_can_be_taken_over_after_expiry() -> None:
    h = FlowHarness([FREEZE])
    from gpu_fault.store import WorkflowLeaseError

    held = h.store.claim_workflow(
        h.workflow.request_id,
        "foreign-executor",
        3,
        lease_duration=timedelta(minutes=1),
    )
    with pytest.raises(WorkflowLeaseError):
        h.execute()
    assert h.adapter.calls == [], h.adapter.calls
    h.store.save_workflow(
        held.model_copy(
            update={
                "execution_lease_expires_at": datetime.now(timezone.utc)
                - timedelta(seconds=1)
            }
        )
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUCCEEDED, result
    assert (
        h.store.get_workflow(h.workflow.request_id).execution_epoch
        == held.execution_epoch + 1
    ), result
