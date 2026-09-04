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
from gpu_fault.execution import restart_budget_preflight
from gpu_fault.workflow_resolution import (
    RETIRED_GENERATION_STATUSES,
    abandoned_generation_successor,
    retired_generation_audit,
    retired_generation_reasons,
    retired_generation_successor,
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
        self._supersede_abandoned_generations(now)
        retired = self._revoke_retired_generations(now)
        timed_out = self._expire_stuck_workflows(now)
        workflows = self.store.list_workflows(
            self.EXECUTABLE_STATUSES,
            limit=max(self.config.batch_size * 20, 100),
        )
        workflows = [
            workflow
            for workflow in workflows
            if workflow.request_id not in retired
            and (workflow.not_before is None or workflow.not_before <= now)
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

    def _supersede_abandoned_generations(self, now: datetime) -> list[WorkflowRequest]:
        """Terminalize workflows their incident re-planned away from.

        See ``workflow_resolution.abandoned_generation_successor`` for why these
        records exist and why they are provably dead. Resolving them here rather
        than in the release gate is what makes it stick: the record becomes a
        real terminal record, so the per-incident admission slot below is freed
        for the successor the incident is waiting on, and nothing downstream
        needs an exception for it.

        Not counted into the dispatch report. A supersession is neither an
        execution nor a failure, and folding it into ``failed`` would alarm on
        cleanup.
        """

        superseded: list[WorkflowRequest] = []
        for workflow in self.store.list_workflows(
            {WorkflowStatus.PENDING},
            limit=1000,
        ):
            try:
                successor = abandoned_generation_successor(
                    self.store, workflow, now=now
                )
            except Exception:  # noqa: BLE001 - keep dispatching, do not terminalize
                LOGGER.exception(
                    "abandoned generation check failed, workflow left pending: %s",
                    workflow.request_id,
                )
                continue
            if successor is None:
                continue
            reason = (
                f"superseded by {successor.request_id}: incident "
                f"{workflow.incident_id} advanced to generation "
                f"{successor.fencing_token} before any step of generation "
                f"{workflow.fencing_token} ran"
            )
            try:
                claimed = self.store.claim_workflow(
                    workflow.request_id,
                    self.executor.config.executor_id,
                    workflow.fencing_token,
                    lease_duration=timedelta(seconds=30),
                    now=now,
                )
            except WorkflowLeaseError:
                continue
            replacement = claimed.model_copy(
                update={
                    "status": WorkflowStatus.SUPERSEDED,
                    "preempted_by_workflow_id": successor.request_id,
                    "preemption_reason": reason,
                    "superseded_at": now,
                    "blocked_reasons": list(
                        dict.fromkeys([*claimed.blocked_reasons, reason])
                    ),
                    "execution_owner_id": None,
                    "execution_lease_expires_at": None,
                    "updated_at": now,
                }
            )
            # Planning may already hold a restart reservation for a
            # RESTART_WORKLOAD step no adapter ever attempted. Terminalizing
            # without releasing it would spend that job's restart budget on a
            # workflow that never restarted anything.
            restart_budget_preflight.release_unattempted_restart_reservations(
                self.store,
                replacement,
            )
            self.store.save_workflow_if_leased(
                replacement,
                # The id the claim above succeeded with. Reading it back off the
                # record would be ``str | None``, and a claim that returns
                # without raising is a claim this executor holds.
                self.executor.config.executor_id,
                claimed.execution_epoch,
                now=now,
            )
            self._sync_plan(replacement, WorkflowStatus.SUPERSEDED)
            superseded.append(replacement)
            LOGGER.warning(
                "workflow superseded as an abandoned generation: "
                "workflow=%s generation=%s incident=%s successor=%s "
                "successor_generation=%s",
                workflow.request_id,
                workflow.fencing_token,
                workflow.incident_id,
                successor.request_id,
                successor.fencing_token,
            )
        return superseded

    def _revoke_retired_generations(self, now: datetime) -> set[str]:
        """Revoke a retired generation that had already started.

        ``_supersede_abandoned_generations`` above only clears the record that
        never ran. That is the cheap case. The one that actually cost a fleet was
        the opposite: on 2026-09-04 the retired generation was ``RUNNING``, held
        an execution owner, a renewing lease, six remediation budget claims and
        an unsettled ``STOP_WORKLOADS`` remote command against three nodes, and
        the fleet rollout fence was the only thing still standing between that
        command and a running 24-GPU job. Its incident had recovered at a higher
        generation four hours earlier.

        The order below is the whole point, and it is why this can take two
        ticks. A remote command outlives the workflow row, so revoking the
        workflow first would leave a destructive command behind whose next fence
        evaluation releases it. So: cancel the commands first -- ``PENDING`` and
        ``WAITING`` go terminal in the store immediately, and a ``LEASED`` one
        gets a cancellation request that both bars it from ever being claimed
        again and turns whatever the executor reports into a ``FAILED`` -- and
        revoke the workflow only once the store shows every one of them settled.

        No lease is taken. ``claim_workflow`` cannot help here: the lease holder
        is this very dispatch loop, which renews on every tick, so a
        lease-respecting revocation would wait forever on a record it is itself
        keeping alive. Safety comes from
        ``retired_generation_records`` re-deriving every condition inside the
        store transaction. In particular a record that has already *completed* a
        destructive operation is never revoked -- there is something real to
        compensate for, and the release gate goes on reporting it until an
        operator resolves it.

        Not counted into the dispatch report, for the same reason a supersession
        is not: cleanup is neither an execution nor a failure.

        Returns the ids of the retired generations that are still open, which
        ``run_once`` withholds from dispatch: a record waiting for its commands
        to settle is one ``_validate_fencing`` would reject, and blocking it on a
        fencing error would bury the revocation under a dispatch failure.
        """

        held: set[str] = set()
        for workflow in self.store.list_workflows(
            RETIRED_GENERATION_STATUSES,
            limit=1000,
        ):
            try:
                held |= self._revoke_retired_generation(workflow, now)
            except Exception:  # noqa: BLE001 - keep dispatching, do not revoke
                LOGGER.exception(
                    "retired generation revocation failed, workflow left open: %s",
                    workflow.request_id,
                )
                held.add(workflow.request_id)
        return held

    def _revoke_retired_generation(
        self,
        workflow: WorkflowRequest,
        now: datetime,
    ) -> set[str]:
        successor = retired_generation_successor(self.store, workflow)
        if successor is None:
            return set()
        # Asked of the record alone first -- an empty command list isolates the
        # conditions the workflow itself fails -- so a workflow that has already
        # mutated the fleet is refused before anything is cancelled.
        blocking = retired_generation_reasons(workflow, successor, [])
        if blocking:
            LOGGER.warning(
                "retired generation needs an operator: workflow=%s generation=%s "
                "incident=%s reasons=%s",
                workflow.request_id,
                workflow.fencing_token,
                workflow.incident_id,
                "; ".join(blocking),
            )
            return {workflow.request_id}
        commands = self.store.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
        if retired_generation_reasons(workflow, successor, commands):
            cancelled = self.store.cancel_remote_commands_for_workflow(
                workflow.request_id,
                reason=retired_generation_audit(
                    workflow,
                    self.store.get_incident(workflow.incident_id),
                    successor,
                ),
            )
            LOGGER.warning(
                "retired generation remote commands cancelled, revocation "
                "deferred: workflow=%s generation=%s incident=%s "
                "cancelled=%s cancellation_requested=%s",
                workflow.request_id,
                workflow.fencing_token,
                workflow.incident_id,
                cancelled.get("cancelled", 0),
                cancelled.get("cancellation_requested", 0),
            )
            return {workflow.request_id}
        revoked, _ = self.store.reconcile_retired_generation_workflow(
            workflow.request_id,
            successor.request_id,
            expected_fencing_token=workflow.fencing_token,
            reference=None,
            reconciled_at=now,
        )
        # Terminalizing releases the remediation budget claims on its own -- they
        # are only counted for a RUNNING workflow holding a live lease -- but a
        # restart reservation is a durable row and is not.
        restart_budget_preflight.release_unattempted_restart_reservations(
            self.store,
            revoked,
        )
        self._sync_plan(revoked, WorkflowStatus.SUPERSEDED)
        LOGGER.warning(
            "retired generation revoked: workflow=%s generation=%s "
            "incident=%s successor=%s successor_generation=%s",
            workflow.request_id,
            workflow.fencing_token,
            workflow.incident_id,
            successor.request_id,
            successor.fencing_token,
        )
        return set()

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
