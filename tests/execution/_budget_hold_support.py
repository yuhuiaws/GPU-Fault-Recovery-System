"""Builders for the held-remediation-budget tests (memory, SQLite, PostgreSQL).

A budget scope must stay held from a workflow's admission until it ends, so the
tests need workflows that stop part-way: ``ScriptedAdapter`` keeps a
RESTART_NODE (or any operation it is told to) WAITING until the test releases
that one workflow.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.node_action_confirmation import confirmed_execution
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from tests._builders import (
    active_workflow_executor,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)

STOP = WorkflowOperation.STOP_WORKLOADS
REBOOT = WorkflowOperation.RESTART_NODE
RESET = WorkflowOperation.RESET_GPU
MARK = WorkflowOperation.MARK_UNSCHEDULABLE
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
VERIFY = WorkflowOperation.VERIFY_NO_GPU_CLIENTS
RESTORE_GPU = WorkflowOperation.RESTORE_GPU_SERVICES
RESTORE_SCHEDULING = WorkflowOperation.RESTORE_SCHEDULING
VALIDATE = WorkflowOperation.VALIDATE_GPU
# The action-production request model's node branch (perf plan §4.3).
ACTION_MODEL = (
    MARK,
    STOP,
    QUIESCE,
    VERIFY,
    RESET,
    RESTORE_GPU,
    REBOOT,
    RESTORE_SCHEDULING,
)
REBOOT_PLAN = (STOP, REBOOT)


class ScriptedAdapter:
    """Every operation succeeds except the ``waiting`` ones (RESTART_NODE by
    default), which wait until the test releases that workflow."""

    def __init__(
        self,
        operations: Iterable[WorkflowOperation],
        *,
        waiting: Iterable[WorkflowOperation] = (WorkflowOperation.RESTART_NODE,),
    ) -> None:
        self.operations = frozenset(operations) | frozenset(waiting)
        self.waiting = frozenset(waiting)
        self.released: set[str] = set()
        self.calls: list[str] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a" and step.operation in self.operations

    def execute(self, context: Any) -> WorkflowStepOutcome:
        self.calls.append(context.idempotency_key)
        request_id = context.workflow.request_id
        operation = context.step.operation
        if operation in self.waiting and request_id not in self.released:
            return WorkflowStepOutcome.waiting(
                operation_id=f"{operation.value.lower()}/{request_id}"
            )
        return WorkflowStepOutcome.succeeded()

    def started(self) -> set[str]:
        """Workflows that reached an adapter at all."""

        return {key.split("/", 1)[0] for key in self.calls}


# What an executor that never saw a reboot's verdict records on the step.
UNKNOWN_OUTCOME = {"outcome_unknown": True, "manual_confirmation_required": True}


class UnknownOutcomeAdapter(ScriptedAdapter):
    """Like ``ScriptedAdapter``, but the reboot of every workflow in ``unknown``
    ends without a verdict, so ``terminal_state`` parks that record BLOCKED /
    NEEDS_OPERATOR with its node action unresolved."""

    def __init__(self, operations: Iterable[WorkflowOperation], unknown: set[str]):
        super().__init__(operations)
        self.unknown = unknown

    def execute(self, context: Any) -> WorkflowStepOutcome:
        if (
            context.step.operation is WorkflowOperation.RESTART_NODE
            and context.workflow.request_id in self.unknown
        ):
            self.calls.append(context.idempotency_key)
            return WorkflowStepOutcome.failed(
                "reboot outcome unknown", details=dict(UNKNOWN_OUTCOME)
            )
        return super().execute(context)


def operator_confirmed(
    workflow: WorkflowRequest, *, now: datetime, node_id: str
) -> WorkflowRequest:
    """``workflow`` as ``confirm-node-action`` leaves it: its reboot's
    uncertainty answered on the step, the record itself still BLOCKED."""

    confirmation = {
        "actor": "operator-1",
        "reference": "CHG-1",
        "confirmed_at": now.isoformat(),
        "node_id": node_id,
        "operation": WorkflowOperation.RESTART_NODE.value,
    }
    return workflow.model_copy(
        update={
            "step_executions": [
                confirmed_execution(item, confirmation, now=now)
                if item.operation is WorkflowOperation.RESTART_NODE
                else item
                for item in workflow.step_executions
            ]
        }
    )


def seed(
    store: Any,
    suffix: str,
    *,
    cluster_id: str = "cluster-a",
    node_id: str | None = None,
    operations: Sequence[WorkflowOperation] = REBOOT_PLAN,
    created_at: datetime,
    **values: Any,
) -> WorkflowRequest:
    node = node_id or f"node-{suffix}"
    incident = fault_incident(
        f"incident-{suffix}",
        f"event-{suffix}",
        cluster_id=cluster_id,
        node_ids=[node],
        official_action="RESTART_BM",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=created_at,
        updated_at=created_at,
    )
    workflow = workflow_request(
        f"workflow-{suffix}",
        incident.incident_id,
        runtime_profile_version="active-v1",
        official_action="RESTART_BM",
        official_steps=[workflow_step(op, node_ids=[node]) for op in operations],
        created_at=created_at,
        updated_at=created_at,
        **values,
    )
    store.save_incident(copy_model(incident, workflow_request_id=workflow.request_id))
    store.save_workflow(workflow)
    return workflow


def budget_executor(
    store: Any,
    adapter: ScriptedAdapter,
    policy: RemediationBudgetPolicy,
    *,
    executor_id: str = "executor-a",
) -> Any:
    executor = active_workflow_executor(
        store, [adapter], adapter.operations, executor_id=executor_id
    )
    executor.config = replace(executor.config, remediation_budget=policy)
    executor.waiting_lease_duration = timedelta(seconds=30)
    return executor


def budget_dispatcher(
    store: Any,
    adapter: ScriptedAdapter,
    policy: RemediationBudgetPolicy,
    *,
    batch_size: int = 100,
    max_workers: int = 8,
) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        budget_executor(store, adapter, policy),
        WorkflowDispatcherConfig(
            enabled=True, batch_size=batch_size, max_workers=max_workers
        ),
    )


def holders(workflows: Iterable[WorkflowRequest], scope: str) -> list[str]:
    """RUNNING workflows whose held claims name ``scope``."""

    return sorted(
        workflow.request_id
        for workflow in workflows
        if workflow.status is WorkflowStatus.RUNNING
        and scope in workflow.remediation_budget_claims
    )


def holders_by_cluster(workflows: Iterable[WorkflowRequest]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for workflow in workflows:
        if workflow.status is not WorkflowStatus.RUNNING:
            continue
        for scope in workflow.remediation_budget_claims:
            if scope.startswith("cluster:"):
                counts[scope.removeprefix("cluster:")] += 1
    return counts


def all_workflows(store: Any) -> list[WorkflowRequest]:
    return list(store.list_workflows(None, limit=100_000))
