from __future__ import annotations

import logging
import json
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Callable

from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    WorkflowExecutionRequest,
    WorkflowDispatchFailure,
    WorkflowDispatchReport,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store import (
    NotFoundError,
    WorkflowLeaseError,
)
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.execution.config import (
    WorkflowDispatcherConfig,
)
from gpu_fault.execution.executor import (
    ProductionWorkflowExecutor,
)
from gpu_fault.execution.models import (
    WorkflowExecutionError,
)
from gpu_fault.execution.transient_errors import (
    transient_store_error,
)

LOGGER = logging.getLogger(__name__)


class WorkflowDispatcher:
    """Durably scans executable workflows and drives the state machine.

    A provider submission that returns WAITING is only observed on later
    passes. The dispatcher never fabricates external completion evidence.
    """

    EXECUTABLE_STATUSES = {
        WorkflowStatus.PENDING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.RUNNING,
    }

    def __init__(
        self,
        store: ControlPlaneStore,
        executor: ProductionWorkflowExecutor,
        config: WorkflowDispatcherConfig,
        failure_handler: (Callable[[WorkflowRequest], object] | None) = None,
    ) -> None:
        if config.poll_interval_seconds <= 0:
            raise WorkflowExecutionError(
                "workflow dispatcher poll interval must be positive"
            )
        if config.batch_size <= 0:
            raise WorkflowExecutionError(
                "workflow dispatcher batch size must be positive"
            )
        if config.max_workers <= 0:
            raise WorkflowExecutionError("workflow dispatcher workers must be positive")
        self.store = store
        self.executor = executor
        self.config = config
        self.failure_handler = failure_handler
        self._stop = Event()
        self._wake = Event()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="workflow-dispatch",
        )

    def wake(self) -> None:
        """Request an early durable workflow scan."""
        if self.config.enabled:
            self._wake.set()

    def consume_wake(self) -> bool:
        requested = self._wake.is_set()
        if requested:
            self._wake.clear()
        return requested

    def run_once(self) -> WorkflowDispatchReport:
        self._reconcile_failed_workflows()
        now = datetime.now(timezone.utc)
        timed_out = self._expire_stuck_workflows(now)
        workflows = self.store.list_workflows(
            self.EXECUTABLE_STATUSES,
            limit=max(self.config.batch_size * 20, 100),
        )
        workflows = [
            workflow
            for workflow in workflows
            if workflow.not_before is None or workflow.not_before <= now
        ]
        workflows = [
            workflow
            for workflow in workflows
            if not self._processor_queue_blocks(workflow, now)
        ]
        workflows = [
            workflow
            for workflow in workflows
            if (
                workflow.predecessor_workflow_id is None
                or self.store.get_workflow(workflow.predecessor_workflow_id).status
                in {
                    WorkflowStatus.BLOCKED,
                    WorkflowStatus.SUCCEEDED,
                    WorkflowStatus.FAILED,
                    WorkflowStatus.SUPERSEDED,
                }
            )
        ]
        candidates = workflows
        workflows = []
        incident_ids = set()
        for workflow in candidates:
            if workflow.incident_id in incident_ids:
                continue
            incident_ids.add(workflow.incident_id)
            workflows.append(workflow)
            if len(workflows) >= self.config.batch_size:
                break
        executed = 0
        waiting = 0
        completed = 0
        failed = len(timed_out)
        failures: list[WorkflowDispatchFailure] = [
            WorkflowDispatchFailure(
                workflow_request_id=workflow.request_id,
                error="workflow execution deadline exceeded",
            )
            for workflow in timed_out
        ]
        futures = {
            self._worker_pool.submit(self._dispatch_workflow, workflow): workflow
            for workflow in workflows
        }
        for future in as_completed(futures):
            workflow = futures[future]
            try:
                outcome, failure = future.result()
                executed += outcome[0]
                waiting += outcome[1]
                completed += outcome[2]
                failed += outcome[3]
                if failure is not None:
                    failures.append(failure)
            except WorkflowLeaseError:
                # Another healthy replica owns this workflow. It remains
                # executable and will be observed on a later scan.
                waiting += 1
            except Exception as exc:
                if transient_store_error(exc):
                    LOGGER.warning(
                        "workflow dispatch hit a transient store "
                        "error and will retry: workflow=%s error=%s: %s",
                        workflow.request_id,
                        type(exc).__name__,
                        exc,
                    )
                    waiting += 1
                    continue
                blocked = self._block_after_internal_error(workflow, exc)
                self._sync_plan(blocked, blocked.status)
                failed += 1
                failures.append(
                    WorkflowDispatchFailure(
                        workflow_request_id=workflow.request_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return WorkflowDispatchReport(
            scanned=len(workflows) + len(timed_out),
            executed=executed,
            waiting=waiting,
            completed=completed,
            failed=failed,
            failures=failures,
        )

    def _dispatch_workflow(
        self, workflow: WorkflowRequest
    ) -> tuple[
        tuple[int, int, int, int],
        WorkflowDispatchFailure | None,
    ]:
        result = self.executor.execute(
            workflow.request_id,
            WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token,
                confirm_cluster_name=self.config.confirm_cluster_name,
            ),
        )
        self._sync_plan(workflow, result.status)
        if result.status in {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.SUPERSEDED,
        }:
            return (1, 0, 1, 0), None
        if result.status is WorkflowStatus.FAILED:
            if self.store.get_preempting_successor(workflow.request_id) is None:
                self._handle_failed_workflow(
                    self.store.get_workflow(workflow.request_id)
                )
            return (
                (1, 0, 0, 1),
                WorkflowDispatchFailure(
                    workflow_request_id=workflow.request_id,
                    error=result.error or "workflow execution failed",
                ),
            )
        return (1, 1, 0, 0), None

    def _block_after_internal_error(
        self,
        workflow: WorkflowRequest,
        error: Exception,
    ) -> WorkflowRequest:
        current = self.store.get_workflow(workflow.request_id)
        if current.status not in self.EXECUTABLE_STATUSES:
            return current
        owner = current.execution_owner_id
        if owner is None:
            current = self.store.claim_workflow(
                current.request_id,
                self.executor.config.executor_id,
                current.fencing_token,
                lease_duration=timedelta(seconds=30),
            )
            owner = current.execution_owner_id
        if owner != self.executor.config.executor_id:
            raise WorkflowLeaseError(
                "cannot block workflow after internal error because "
                f"it is leased by {owner}"
            )
        reason = f"dispatcher internal error: {type(error).__name__}: {error}"
        now = datetime.now(timezone.utc)
        blocked = current.model_copy(
            update={
                "status": WorkflowStatus.BLOCKED,
                "blocked_reasons": list(
                    dict.fromkeys([*current.blocked_reasons, reason])
                ),
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "updated_at": now,
            }
        )
        incident = self.store.get_incident(blocked.incident_id)
        if (
            incident.workflow_request_id is None
            or incident.workflow_request_id == blocked.request_id
        ):
            incident = incident.model_copy(
                update={
                    "state": IncidentState.ESCALATED,
                    "reasons": list(dict.fromkeys([*incident.reasons, reason])),
                    "updated_at": now,
                }
            )
            self.store.save_workflow_and_incident_if_leased(
                blocked,
                incident,
                owner,
                current.execution_epoch,
            )
        else:
            self.store.save_workflow_if_leased(
                blocked,
                owner,
                current.execution_epoch,
            )
        return blocked

    def _expire_stuck_workflows(self, now: datetime) -> list[WorkflowRequest]:
        expired = []
        for workflow in self.store.list_workflows(
            {WorkflowStatus.RUNNING},
            limit=1000,
        ):
            if workflow.execution_deadline is None or workflow.execution_deadline > now:
                continue
            try:
                claimed = self.store.claim_workflow(
                    workflow.request_id,
                    self.config.confirm_cluster_name or "workflow-deadline-watchdog",
                    workflow.fencing_token,
                    lease_duration=timedelta(seconds=30),
                    now=now,
                )
            except WorkflowLeaseError:
                continue
            cancellation = self.store.cancel_remote_commands_for_workflow(
                workflow.request_id,
                reason=("workflow execution deadline exceeded"),
            )
            failed = claimed.model_copy(
                update={
                    "status": WorkflowStatus.FAILED,
                    "blocked_reasons": list(
                        dict.fromkeys(
                            [
                                *claimed.blocked_reasons,
                                (
                                    "deadline watchdog remote command "
                                    f"cancellation: {cancellation}"
                                ),
                            ]
                        )
                    ),
                    "execution_owner_id": None,
                    "execution_lease_expires_at": None,
                    "updated_at": now,
                }
            )
            self.store.save_workflow_if_leased(
                failed,
                claimed.execution_owner_id,
                claimed.execution_epoch,
                now=now,
            )
            self._sync_plan(failed, WorkflowStatus.FAILED)
            self._handle_failed_workflow(failed)
            expired.append(failed)
        return expired

    def _processor_queue_blocks(self, workflow: WorkflowRequest, now: datetime) -> bool:
        if (
            workflow.aggregation_max_deadline is None
            or now >= workflow.aggregation_max_deadline
        ):
            return False
        try:
            incident = self.store.get_incident(workflow.incident_id)
        except NotFoundError:
            return False
        scope_keys = {
            json.dumps(
                [incident.cluster_id, "node", node_id],
                separators=(",", ":"),
            )
            for node_id in incident.node_ids
        }
        if incident.job_id and incident.attempt_id:
            scope_keys.add(
                json.dumps(
                    [
                        incident.cluster_id,
                        incident.job_id,
                        incident.attempt_id,
                    ],
                    separators=(",", ":"),
                )
            )
        return self.store.has_incomplete_processor_requests_for_scopes(
            incident.cluster_id, scope_keys
        )

    def _reconcile_failed_workflows(self) -> None:
        if self.failure_handler is None:
            return
        for workflow in self.store.list_unhandled_failed_workflows(limit=1000):
            try:
                self._handle_failed_workflow(workflow)
            except Exception:
                LOGGER.exception(
                    "failed workflow reconciliation failed: %s",
                    workflow.request_id,
                )

    def _handle_failed_workflow(self, workflow: WorkflowRequest) -> None:
        if self.failure_handler is not None:
            self.failure_handler(workflow)
            current = self.store.get_workflow(workflow.request_id)
            if (
                current.status is WorkflowStatus.FAILED
                and current.failure_handled_at is None
            ):
                self.store.save_workflow(
                    current.model_copy(
                        update={
                            "failure_handled_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                )

    def _sync_plan(
        self,
        workflow: WorkflowRequest,
        status: WorkflowStatus,
    ) -> None:
        if not workflow.source_plan_id:
            return
        plan = self.store.get_plan(workflow.source_plan_id)
        plan_status = {
            WorkflowStatus.PENDING: PlanStatus.PENDING,
            WorkflowStatus.SAFETY_PENDING: PlanStatus.PENDING,
            WorkflowStatus.RUNNING: PlanStatus.RUNNING,
            WorkflowStatus.SUCCEEDED: PlanStatus.SUCCEEDED,
            WorkflowStatus.FAILED: PlanStatus.FAILED,
            WorkflowStatus.BLOCKED: PlanStatus.FAILED,
            WorkflowStatus.SUPERSEDED: PlanStatus.FAILED,
        }[status]
        if plan.status is not plan_status:
            self.store.save_plan(plan.model_copy(update={"status": plan_status}))

    def run_forever(self) -> None:
        if not self.config.enabled:
            return
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:
                # A transient Aurora failover must not kill this daemon
                # thread while the Pod and /healthz remain healthy.
                # The next cycle reopens/refreshes pooled connections and
                # resumes the leased workflow from its persisted step.
                LOGGER.exception("workflow dispatcher cycle failed")
            self._wake.wait(self.config.poll_interval_seconds)
            self._wake.clear()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._worker_pool.shutdown(wait=True)
