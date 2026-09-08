from __future__ import annotations

import json
import logging
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Callable, Literal

from pydantic import ValidationError

from gpu_fault.compile_blocked import close_compile_blocked_workflows
from gpu_fault.execution import restart_budget_preflight
from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.executor import (
    DISPATCHER_ACTOR,
    DISPATCHER_INTERNAL_ERROR_ACTOR,
    DISPATCHER_WATCHDOG_ACTOR,
    HOLD_REASON_NODE_REMEDIATION_TIMEOUT,
    HOLD_REASON_NODE_UNDER_REMEDIATION,
    ProductionWorkflowExecutor,
    record_hold_event,
)
from gpu_fault.execution.models import (
    WorkflowExecutionError,
)
from gpu_fault.execution.transient_errors import (
    transient_store_error,
)
from gpu_fault.models import (
    EXECUTABLE_WORKFLOW_STATUSES,
    BlockedKind,
    FaultIncident,
    IncidentState,
    PlanStatus,
    WorkflowDispatchFailure,
    WorkflowDispatchReport,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
    bounded_reasons,
    execution_matches_step,
    execution_phase,
    record_workflow_event,
    workflow_is_open,
)
from gpu_fault.orchestration import WorkflowFencingError
from gpu_fault.orphaned_commands import cancel_orphaned_commands
from gpu_fault.store import (
    NotFoundError,
    WorkflowLeaseError,
)
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.store.shared.workflow_scan import dispatch_eligible_at
from gpu_fault.workflow_resolution import (
    RETIRED_GENERATION_STATUSES,
    abandoned_generation_successor,
    retired_generation_audit,
    retired_generation_reasons,
    retired_generation_successor,
)

LOGGER = logging.getLogger(__name__)

DISPATCHER_PLACEMENT_HOLD_ACTOR = "dispatcher-placement-hold"
HOLD_REASON_PLACEMENT_HOLD_DISSOLVED_TEXT = "placement hold dissolved: nodes freed"


class WorkflowDispatcher:
    """Durably scans executable workflows and drives the state machine.

    A provider submission that returns WAITING is only observed on later
    passes. The dispatcher never fabricates external completion evidence.
    """

    EXECUTABLE_STATUSES = set(EXECUTABLE_WORKFLOW_STATUSES)
    # Only an error that proves the record itself cannot be executed may
    # write it BLOCKED; everything else stays executable and is counted.
    BLOCKING_INTERNAL_ERRORS: tuple[type[BaseException], ...] = (ValidationError,)

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
        self.failure_handling_abandoned_total = 0
        self.plan_sync_misses_total = 0
        self.node_busy_timeouts_total = 0
        # Placement holds ended because the nodes they waited on were freed
        # inside the window (rule A, case 2); nothing reached the data plane.
        self.placement_holds_dissolved_total = 0
        self.internal_errors_total = 0
        self.deferred_total = 0
        self.preemption_pending_seen_total = 0
        # F-L1: rows every cycle scanned and set aside, by reason, summed over
        # the process lifetime; /metrics reads it, each ``run_once`` adds to it.
        self.filtered_total: dict[str, int] = {}
        # F-A5: the PENDING watchdog observes only. Re-derived every tick from
        # the rows the dispatch scan saw, in eligibility order.
        self.pending_age_seconds_max: float = 0.0
        self.oldest_pending_workflow_id: str | None = None
        self.pending_age_warnings_total = 0
        # F-A5: retired generations the sweep may not close without an
        # operator (``workflow_retired_generation_awaiting_operator``).
        self.retired_generation_awaiting_operator = 0
        # ARCH-E E3: proof the dispatch thread is still turning. Every other
        # gauge here is written by the loop itself, so a dead thread freezes
        # them at whatever healthy value they last had.
        self.last_cycle_timestamp_seconds = 0.0
        # Unix time of the newest event behind each counter. The counters are
        # per process and a multi-process Pod is scraped one process at a
        # time, so increase() over them misreads the interleaving as resets;
        # a timestamp survives that, because max() of it across samples is
        # the newest event whichever process reported it (ARCH-E E4).
        self.internal_error_last_seen_timestamp_seconds = 0.0
        self.failure_handling_abandoned_last_seen_timestamp_seconds = 0.0
        self._stop = Event()
        self._wake = Event()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="workflow-dispatch",
        )

    def wake(self) -> None:
        """Request an early durable workflow scan.

        Process-local: in the split deployment the ingress process that calls
        this is not the worker process that scans, so the call only shortens
        the poll interval when both roles share a process. The scan cadence
        itself is ``poll_interval_seconds`` (F-A8).
        """
        if self.config.enabled:
            self._wake.set()

    def consume_wake(self) -> bool:
        requested = self._wake.is_set()
        if requested:
            self._wake.clear()
        return requested

    DISPATCH_LEASE_KEY = "workflow-dispatch"
    MAX_SCAN_ROWS = 20_000

    def run_once(self) -> WorkflowDispatchReport:
        # Stamped first: a cycle that finds the lease held elsewhere, or fails
        # inside the store, is still a live thread.
        self.last_cycle_timestamp_seconds = time.time()
        if self.config.dispatch_lease_seconds > 0 and not self._holds_dispatch_lease():
            return WorkflowDispatchReport(
                scanned=0,
                executed=0,
                waiting=0,
                completed=0,
                failed=0,
                lease_held_by_other=True,
            )
        self._reconcile_failed_workflows()
        now = datetime.now(timezone.utc)
        retired = self.sweep_stuck_records(now)
        timed_out = self._expire_stuck_workflows(now)
        workflows, filtered, horizon_exhausted = self._scan_dispatchable(now, retired)
        executed = 0
        waiting = 0
        completed = 0
        failed = len(timed_out)
        internal_errors = 0
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
        deadline = (
            time.monotonic() + self.config.cycle_deadline_seconds
            if self.config.cycle_deadline_seconds > 0
            else None
        )
        deferred = 0
        pending: set[
            Future[tuple[tuple[int, int, int, int], WorkflowDispatchFailure | None]]
        ] = set(futures)
        completed_futures: list[
            Future[tuple[tuple[int, int, int, int], WorkflowDispatchFailure | None]]
        ] = []
        while pending:
            timeout = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            completed_futures.extend(done)
            if deadline is not None and pending and time.monotonic() >= deadline:
                # Past the cycle deadline: rows still queued stay PENDING for
                # the next scan; rows already running finish (F-C7).
                for future in list(pending):
                    if future.cancel():
                        pending.discard(future)
                        deferred += 1
                deadline = None
        self.deferred_total += deferred
        for future in completed_futures:
            workflow = futures[future]
            try:
                outcome, failure = future.result()
                executed += outcome[0]
                waiting += outcome[1]
                completed += outcome[2]
                failed += outcome[3]
                if failure is not None:
                    failures.append(failure)
            except (WorkflowLeaseError, WorkflowFencingError):
                # Another healthy replica owns this workflow, or its incident
                # has moved to a later generation and the successor link (or
                # the retired-generation sweep) will resolve it. Either way
                # the record is somebody's business: it stays executable and
                # is observed on a later scan, never written BLOCKED (F-J2).
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
                failures.append(
                    WorkflowDispatchFailure(
                        workflow_request_id=workflow.request_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                if isinstance(exc, self.BLOCKING_INTERNAL_ERRORS):
                    blocked = self._block_after_internal_error(workflow, exc)
                    self._sync_plan(blocked, blocked.status)
                    failed += 1
                    continue
                # Anything else says nothing about the record: an adapter
                # wiring bug on this replica, a missing capability, a
                # programming error. The lease the executor took is released,
                # the row goes back to the status it was scanned in with a
                # backoff, so it stays executable for a later tick (and the
                # other replicas) without a hot loop; it is counted and logged
                # with its traceback (F-B4 (3)). It used to be written BLOCKED.
                internal_errors += 1
                self.internal_errors_total += 1
                self.internal_error_last_seen_timestamp_seconds = time.time()
                self._release_after_internal_error(workflow)
                LOGGER.exception(
                    "workflow dispatch hit an internal error; workflow %s "
                    "stays %s: %s: %s",
                    workflow.request_id,
                    workflow.status.value,
                    type(exc).__name__,
                    exc,
                )
        for reason, count in filtered.items():
            self.filtered_total[reason] = self.filtered_total.get(reason, 0) + count
        return WorkflowDispatchReport(
            scanned=len(workflows) + len(timed_out),
            executed=executed,
            waiting=waiting,
            completed=completed,
            failed=failed,
            failures=failures,
            filtered=dict(filtered),
            horizon_exhausted=horizon_exhausted,
            deferred=deferred,
            internal_errors=internal_errors,
        )

    def _holds_dispatch_lease(self) -> bool:
        """Whether this process is the fleet's dispatcher for this tick (F-A1).

        Every process used to run a full scan and race the same rows; the lease
        makes the scan single-instance while ``claim_workflow`` keeps
        guaranteeing correctness if a lease ever overlaps.
        """

        lease = self.store.acquire_periodic_task_lease(
            self.DISPATCH_LEASE_KEY,
            self.executor.config.executor_id,
            now=datetime.now(timezone.utc),
            lease_duration=timedelta(seconds=self.config.dispatch_lease_seconds),
        )
        return lease.owner_id == self.executor.config.executor_id

    def _scan_dispatchable(
        self, now: datetime, retired: set[str]
    ) -> tuple[list[WorkflowRequest], dict[str, int], bool]:
        """Executable rows that pass every gate, counting what each gate held back.

        The store returns rows in eligibility order -- ``dispatch_eligible_at``
        = max(``created_at``, ``not_before``), which no merge rewrites (F-A2a)
        -- and the scan walks that order page by page, handing the last row of
        a page back as the cursor (F-A2c), until a full batch is found, the
        rows run out, or ``MAX_SCAN_ROWS`` have been walked in total. Rows that
        can never pass (an open predecessor, a fresh ``not_before``) therefore
        cannot hide the dispatchable rows behind them (P0-79B), and a page is
        never re-read. The batch is cut in the same order; ``filtered`` is the
        tick's answer to "why did nothing dispatch?".
        """

        page_size = max(self.config.batch_size * 20, 100)
        candidates: list[WorkflowRequest] = []
        filtered: dict[str, int] = {}
        scanned = 0
        after: WorkflowRequest | None = None
        first_page: list[WorkflowRequest] | None = None
        while True:
            # The permanent filters (future not_before, open predecessor,
            # retired generation) run in the store, below the LIMIT (F-A2b);
            # ``_eligible`` keeps them as a belt-and-braces pass.
            rows = self.store.list_workflows(
                self.EXECUTABLE_STATUSES,
                limit=page_size,
                dispatchable_at=now,
                exclude_request_ids=retired,
                after=after,
            )
            if first_page is None:
                first_page = rows
            scanned += len(rows)
            page_candidates, page_filtered = self._eligible(rows, now, retired)
            candidates.extend(page_candidates)
            for reason, count in page_filtered.items():
                filtered[reason] = filtered.get(reason, 0) + count
            if (
                len(candidates) >= self.config.batch_size
                or len(rows) < page_size
                or scanned >= self.MAX_SCAN_ROWS
            ):
                break
            after = rows[-1]
        horizon_exhausted = (
            len(rows) >= page_size
            and scanned >= self.MAX_SCAN_ROWS
            and len(candidates) < self.config.batch_size
        )
        deduped = self._one_per_incident(candidates)
        if len(deduped) < len(candidates):
            filtered["incident_dedupe"] = len(candidates) - len(deduped)
        batch = deduped[: self.config.batch_size]
        if len(batch) < len(deduped):
            filtered["batch_limit"] = len(deduped) - len(batch)
        if len(batch) < self.config.batch_size:
            # The pushed-down reasons no longer surface as rows; the report
            # still has to answer "why did nothing dispatch?", so ask for the
            # counts only when the tick came up short.
            for reason, count in self.store.count_held_workflows(
                self.EXECUTABLE_STATUSES,
                dispatchable_at=now,
                exclude_request_ids=retired,
            ).items():
                filtered[reason] = filtered.get(reason, 0) + count
        if horizon_exhausted:
            LOGGER.error(
                "workflow dispatch scan hit its ceiling of %d rows with only %d "
                "dispatchable; filtered=%s",
                scanned,
                len(candidates),
                dict(filtered),
            )
        self._observe_pending_age(first_page, now)
        return batch, filtered, horizon_exhausted

    def _observe_pending_age(
        self, first_page: list[WorkflowRequest], now: datetime
    ) -> None:
        """The PENDING watchdog: a gauge and a warning, never a terminal write.

        PENDING / SAFETY_PENDING rows had no watchdog at all -- ``execution_
        deadline`` is only stamped by a successful claim -- which is why
        "running for ever" had a backstop and "pending for ever" did not
        (F-A5). The first page is in eligibility order, so its first
        not-yet-running row is the oldest dispatchable one in the fleet.
        """

        oldest = next(
            (
                row
                for row in first_page
                if row.status in (WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING)
            ),
            None,
        )
        if oldest is None:
            self.pending_age_seconds_max = 0.0
            self.oldest_pending_workflow_id = None
            return
        age = max(0.0, (now - dispatch_eligible_at(oldest)).total_seconds())
        self.pending_age_seconds_max = age
        self.oldest_pending_workflow_id = oldest.request_id
        threshold = self.config.pending_age_warning_seconds
        if threshold > 0 and age > threshold:
            self.pending_age_warnings_total += 1
            LOGGER.warning(
                "workflow has been dispatchable but %s for %ds (threshold %ds); "
                "the dispatcher observes and does not terminalize it: "
                "workflow=%s incident=%s eligible_since=%s",
                oldest.status.value,
                int(age),
                int(threshold),
                oldest.request_id,
                oldest.incident_id,
                dispatch_eligible_at(oldest).isoformat(),
            )

    def _eligible(
        self, rows: list[WorkflowRequest], now: datetime, retired: set[str]
    ) -> tuple[list[WorkflowRequest], dict[str, int]]:
        filtered: dict[str, int] = {}

        def held(reason: str) -> None:
            filtered[reason] = filtered.get(reason, 0) + 1

        candidates: list[WorkflowRequest] = []
        for workflow in rows:
            if workflow.request_id in retired:
                held("retired")
                continue
            if workflow.not_before is not None and workflow.not_before > now:
                held("not_before")
                continue
            if (
                workflow.preemption_pending_by_workflow_id is not None
                and self._successor_is_open(workflow.preemption_pending_by_workflow_id)
            ):
                # The merge stamped this row because a stronger successor is
                # about to preempt it (F-C1). Observed, not held: the successor
                # is gated on this row reaching a terminal status, and the
                # executor is the one that supersedes it at its safe boundary
                # (a PENDING row's first step). Holding it here parked both.
                self.preemption_pending_seen_total += 1
            if self._processor_queue_blocks(workflow, now):
                held("processor_queue")
                continue
            if workflow.predecessor_workflow_id is not None:
                try:
                    predecessor = self.store.get_workflow(
                        workflow.predecessor_workflow_id
                    )
                except NotFoundError:
                    # The predecessor row is gone (retention, cleanup). A gone
                    # predecessor cannot be running; treat it as terminal
                    # instead of aborting the whole tick (P1-79F).
                    held("predecessor_missing")
                else:
                    # A BLOCKED predecessor only releases its successor when
                    # its safety plan settled; one waiting for an operator
                    # still owns the node (F-A4).
                    if workflow_is_open(predecessor.status, predecessor.blocked_kind):
                        held("predecessor")
                        continue
            busy = self._nodes_under_other_remediation(workflow)
            if not busy and self._is_unstarted_placement_hold(workflow):
                # The repair the hold waited on ended inside the window: the
                # job keeps running and the hold has nothing left to do.
                if self._dissolve_placement_hold(workflow):
                    held("placement_hold_dissolved")
                    continue
            if busy:
                wait = timedelta(seconds=self.config.node_busy_wait_seconds)
                if now < workflow.created_at + wait:
                    self._record_node_busy_hold(workflow, busy)
                    held("node_busy")
                    continue
                # F-N1 §8: waited long enough. The job workflow gives up by
                # stopping the job -- the signal its owner can see -- and
                # ends FAILED; the node remediation keeps its budget.
                self._fail_node_busy(workflow, busy)
                held("node_busy_timeout")
                continue
            candidates.append(workflow)
        return candidates, filtered

    def _successor_is_open(self, request_id: str) -> bool:
        try:
            successor = self.store.get_workflow(request_id)
        except NotFoundError:
            return False
        return successor.status in self.EXECUTABLE_STATUSES

    def _nodes_under_other_remediation(
        self, workflow: WorkflowRequest
    ) -> dict[str, str]:
        """Nodes of a not-yet-started job workflow that another open workflow
        is repairing, mapped to that workflow's id (F-N1 §8)."""

        if self._has_started(workflow) or not (
            # A placement hold has only a STOP; it waits like a job workflow.
            workflow.placement_hold
            or any(
                step.operation is WorkflowOperation.RESTART_WORKLOAD
                for step in workflow.official_steps
            )
        ):
            return {}
        nodes = {node for step in workflow.official_steps for node in step.node_ids}
        if not nodes:
            return {}
        try:
            incident = self.store.get_incident(workflow.incident_id)
        except NotFoundError:
            return {}
        busy: dict[str, str] = {}
        for other_incident, other in self.store.list_active_workflow_incidents(
            incident.cluster_id, node_ids=nodes
        ):
            if (
                other.request_id == workflow.request_id
                or other.incident_id == workflow.incident_id
            ):
                continue
            for node in sorted(nodes & set(other_incident.node_ids)):
                busy.setdefault(node, other.request_id)
        return busy

    @staticmethod
    def _has_started(workflow: WorkflowRequest) -> bool:
        """Whether the row is past the point where a rule A hold may act on it:
        a step ran or is leased, or the timeout already rewrote the plan."""

        return bool(
            workflow.completed_step_indexes
            or workflow.step_executions
            or workflow.execution_owner_id is not None
            or workflow.terminal_failure_reason
        )

    def _is_unstarted_placement_hold(self, workflow: WorkflowRequest) -> bool:
        return (
            workflow.placement_hold
            and workflow.status is WorkflowStatus.PENDING
            and not self._has_started(workflow)
        )

    def _dissolve_placement_hold(self, workflow: WorkflowRequest) -> bool:
        """End a placement hold whose nodes were freed inside the window.

        Claimed as this executor's id and ended through the executor's
        terminal funnel, like the watchdog reap: SUPERSEDED (nothing ran),
        incident RECOVERED (the job was never touched). A claim lost to a
        concurrent executor leaves the row alone; it is looked at again.
        """

        try:
            claimed = self.store.claim_workflow(
                workflow.request_id,
                self.executor.config.executor_id,
                workflow.fencing_token,
                lease_duration=timedelta(seconds=30),
            )
        except (NotFoundError, WorkflowLeaseError, WorkflowFencingError):
            return False
        except Exception as exc:  # noqa: BLE001 - a stale token is a store error type
            if transient_store_error(exc):
                raise
            LOGGER.warning(
                "placement hold %s could not be claimed for dissolution: %s",
                workflow.request_id,
                exc,
            )
            return False
        if claimed.execution_owner_id != self.executor.config.executor_id:
            return False
        incident: FaultIncident | None
        try:
            incident = self.store.get_incident(claimed.incident_id)
        except NotFoundError:
            incident = None
        reason = HOLD_REASON_PLACEMENT_HOLD_DISSOLVED_TEXT
        events = record_workflow_event(
            claimed,
            WorkflowEventKind.HOLD,
            code=WorkflowEventCode.PLACEMENT_HOLD_DISSOLVED.value,
            reason=reason,
            actor=DISPATCHER_PLACEMENT_HOLD_ACTOR,
            details={
                "reason": WorkflowEventCode.PLACEMENT_HOLD_DISSOLVED.value,
                "remediation_workflow_id": None,
                "node_ids": sorted(
                    {node for step in claimed.official_steps for node in step.node_ids}
                ),
            },
        ).events
        self.executor.terminalize_claimed(
            claimed,
            incident,
            WorkflowStatus.SUPERSEDED,
            claimed.execution_epoch,
            reason=reason,
            actor=DISPATCHER_PLACEMENT_HOLD_ACTOR,
            incident_state=IncidentState.RECOVERED,
            updates={"events": events},
        )
        self.placement_holds_dissolved_total += 1
        LOGGER.info(
            "placement hold %s dissolved: its nodes were freed inside the window",
            claimed.request_id,
        )
        return True

    @staticmethod
    def _remediation_ids(busy: dict[str, str]) -> tuple[str, list[str]]:
        """The remediation workflow(s) a busy-node hold waits on: the first
        sorted id as the one both wait paths name, and the full list."""

        ids = sorted(set(busy.values()))
        return ids[0], ids

    def _record_node_busy_hold(
        self, workflow: WorkflowRequest, busy: dict[str, str]
    ) -> None:
        """Write one HOLD event for a PENDING row the dispatcher does not own.

        The row has no lease holder, so the write goes through
        ``amend_workflow`` (it bumps ``merge_revision``, which is harmless
        here: nobody is executing it). ``record_hold_event`` dedupes against
        the last HOLD, so a 300 s wait polled every tick writes once, and once
        more only if the remediation it waits on changes.
        """

        primary, ids = self._remediation_ids(busy)
        held = record_hold_event(
            workflow,
            reason=HOLD_REASON_NODE_UNDER_REMEDIATION,
            remediation_workflow_id=primary,
            actor=DISPATCHER_ACTOR,
            details={
                "remediation_workflow_ids": ids,
                "node_ids": sorted(busy),
                "node_busy_wait_seconds": int(self.config.node_busy_wait_seconds),
                "held_since": workflow.created_at.isoformat(),
            },
        )
        if held is None:
            return
        try:
            self.store.amend_workflow(workflow.request_id, {"events": held.events})
        except NotFoundError:
            return

    def _fail_node_busy(self, workflow: WorkflowRequest, busy: dict[str, str]) -> None:
        primary, remediation_ids = self._remediation_ids(busy)
        reason = (
            f"{HOLD_REASON_NODE_REMEDIATION_TIMEOUT}: nodes still under "
            f"remediation after {int(self.config.node_busy_wait_seconds)}s: "
            + ", ".join(f"{node} ({other})" for node, other in sorted(busy.items()))
        )
        stop_steps = [
            step.model_copy(
                update={
                    "parameters": {
                        "termination_initiator_incident_id": workflow.incident_id,
                        **step.parameters,
                    },
                    "depends_on_step_indexes": [],
                    "branch_id": None,
                }
            )
            for step in workflow.official_steps
            if step.operation is WorkflowOperation.STOP_WORKLOADS
        ]
        kept_indexes = [
            index
            for index, step in enumerate(workflow.official_steps)
            if step.operation is WorkflowOperation.STOP_WORKLOADS
        ]
        superseded_indexes = [
            index
            for index in range(len(workflow.official_steps))
            if index not in kept_indexes
        ]
        updates: dict[str, object] = {"terminal_failure_reason": reason}
        if stop_steps:
            # Only the stop remains: the job owner sees a controller-initiated
            # stop, and the plan then ends FAILED with the reason above.
            updates.update(
                {
                    "official_steps": stop_steps,
                    "dag_enabled": False,
                    "superseded_step_indexes": [],
                    "not_before": None,
                }
            )
        else:
            updates["superseded_step_indexes"] = list(
                range(len(workflow.official_steps))
            )
        updates["events"] = record_workflow_event(
            workflow,
            WorkflowEventKind.PLAN_REWRITE,
            code=HOLD_REASON_NODE_REMEDIATION_TIMEOUT,
            reason=reason,
            actor=DISPATCHER_ACTOR,
            details={
                "reason": HOLD_REASON_NODE_REMEDIATION_TIMEOUT,
                "remediation_workflow_id": primary,
                "remediation_workflow_ids": remediation_ids,
                "node_ids": sorted(busy),
                "kept_step_indexes": kept_indexes,
                "superseded_step_indexes": superseded_indexes,
            },
        ).events
        try:
            self.store.amend_workflow(workflow.request_id, updates)
        except NotFoundError:
            return
        self.node_busy_timeouts_total += 1
        LOGGER.error(
            "job workflow %s gave up waiting for its nodes: %s",
            workflow.request_id,
            reason,
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

    @property
    def _restart_waiting_ttl(self) -> timedelta:
        """How long a RESTART_WORKLOAD may wait before a terminalizing path
        treats its reservation as never used (F-C9): the executor's own cap,
        so the watchdog, the internal-error BLOCK and the generation sweeps
        release exactly what the executor's terminal writes would."""

        return timedelta(
            seconds=self.executor.config.step_waiting_limit(
                WorkflowOperation.RESTART_WORKLOAD
            )
        )

    def _release_after_internal_error(self, workflow: WorkflowRequest) -> None:
        """Hand the row back after an error that is not the record's fault."""

        try:
            current = self.store.get_workflow(workflow.request_id)
        except NotFoundError:
            return
        owner = current.execution_owner_id
        if owner not in (None, self.executor.config.executor_id):
            return
        if current.status not in self.EXECUTABLE_STATUSES:
            return
        if owner is None:
            current = self.store.claim_workflow(
                current.request_id,
                self.executor.config.executor_id,
                current.fencing_token,
                lease_duration=timedelta(seconds=30),
            )
            owner = current.execution_owner_id
            if owner != self.executor.config.executor_id:
                return
        now = datetime.now(timezone.utc)
        restored = current.model_copy(
            update={
                # Back to the status the scan saw; RUNNING stays RUNNING so a
                # resumed workflow keeps its progress.
                "status": (
                    workflow.status
                    if workflow.status in self.EXECUTABLE_STATUSES
                    else current.status
                ),
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
                "not_before": now
                + timedelta(seconds=self.config.internal_error_backoff_seconds),
                "updated_at": now,
            }
        )
        self.store.save_workflow_if_leased(restored, owner, current.execution_epoch)

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
        incident: FaultIncident | None
        try:
            incident = self.store.get_incident(current.incident_id)
        except NotFoundError:
            incident = None
        if incident is not None:
            # The funnel decides whether the incident is still this
            # workflow's to end; the audit line is prepared either way.
            incident = incident.model_copy(
                update={"reasons": list(dict.fromkeys([*incident.reasons, reason]))}
            )
        # BLOCKED for an internal error is terminal until an operator acts;
        # the funnel releases the unattempted restart reservations, and the
        # preflight re-reserves if the record is ever unblocked (F-C9). The
        # incident is named ESCALATED, not the BLOCKED default QUARANTINED:
        # an internal error is an operator matter, not a settled safety phase.
        self.executor.terminalize_claimed(
            current,
            incident,
            WorkflowStatus.BLOCKED,
            current.execution_epoch,
            reason=reason,
            actor=DISPATCHER_INTERNAL_ERROR_ACTOR,
            incident_state=IncidentState.ESCALATED,
            updates={
                "blocked_kind": BlockedKind.INTERNAL_ERROR,
                "blocked_reasons": list(
                    dict.fromkeys([*current.blocked_reasons, reason])
                ),
            },
        )
        return self.store.get_workflow(current.request_id)

    def _expire_stuck_workflows(self, now: datetime) -> list[WorkflowRequest]:
        """Reap a workflow whose executor stopped touching it.

        This covers the abandoned record only, and cannot cover the looping one.
        It has to take the lease to act, and a workflow still being redispatched
        renews its lease every tick, so the claim below loses every time against
        exactly the workflow the deadline was written for. That case is enforced
        by the lease holder instead, in ``step_bounds.workflow_deadline_failure``;
        the pair is what makes ``execution_deadline`` real. The lost claim is
        logged rather than passed over in silence, because for the four hours
        before that fix the only visible symptom was a workflow that never ended.

        PENDING / SAFETY_PENDING rows are deliberately not reaped here; they are
        observed by ``_observe_pending_age`` (F-A5).
        """

        expired = []
        for workflow in self.store.list_workflows(
            {WorkflowStatus.RUNNING},
            limit=1000,
        ):
            if workflow.execution_deadline is None or workflow.execution_deadline > now:
                continue
            try:
                failed = self._reap_overdue_workflow(workflow, now)
            except Exception:  # noqa: BLE001 - one bad record must not stop the sweep
                LOGGER.exception(
                    "deadline watchdog could not reap workflow %s; continuing",
                    workflow.request_id,
                )
                continue
            if failed is not None:
                expired.append(failed)
        return expired

    def _reap_overdue_workflow(
        self, workflow: WorkflowRequest, now: datetime
    ) -> WorkflowRequest | None:
        deadline = workflow.execution_deadline
        if deadline is None:
            return None
        lease_active = (
            workflow.execution_owner_id is not None
            and workflow.execution_lease_expires_at is not None
            and workflow.execution_lease_expires_at > now
        )
        if lease_active:
            # A live lease -- another replica's or this process's own executor
            # thread -- means the lease holder enforces the deadline
            # (``step_bounds.workflow_deadline_failure``). Claiming as our own
            # executor id would succeed against our own lease and reap a
            # workflow mid-execution (F-A5).
            LOGGER.warning(
                "workflow is past its execution deadline but held by its "
                "executor, so the watchdog cannot reap it; the lease holder "
                "enforces the deadline: workflow=%s deadline=%s "
                "overdue_seconds=%s owner=%s lease_expires_at=%s",
                workflow.request_id,
                deadline.isoformat(),
                int((now - deadline).total_seconds()),
                workflow.execution_owner_id,
                (
                    workflow.execution_lease_expires_at.isoformat()
                    if workflow.execution_lease_expires_at is not None
                    else None
                ),
            )
            return None
        try:
            # The watchdog acts as this executor, not as the cluster: the
            # cluster name is one identity shared by every replica in the
            # region, so the store could not tell two watchdogs apart (F-A5).
            claimed = self.store.claim_workflow(
                workflow.request_id,
                self.executor.config.executor_id,
                workflow.fencing_token,
                lease_duration=timedelta(seconds=30),
                now=now,
            )
        except WorkflowLeaseError:
            LOGGER.warning(
                "workflow is past its execution deadline but held by its "
                "executor, so the watchdog cannot reap it; the lease holder "
                "enforces the deadline: workflow=%s deadline=%s "
                "overdue_seconds=%s owner=%s lease_expires_at=%s",
                workflow.request_id,
                deadline.isoformat(),
                int((now - deadline).total_seconds()),
                workflow.execution_owner_id,
                (
                    workflow.execution_lease_expires_at.isoformat()
                    if workflow.execution_lease_expires_at is not None
                    else None
                ),
            )
            return None
        cancellation = self.store.cancel_remote_commands_for_workflow(
            workflow.request_id,
            reason=("workflow execution deadline exceeded"),
        )
        # The verdict is recorded as a FAILED execution of the step the
        # deadline caught, not appended to ``blocked_reasons``: that field
        # is the switch that makes the executor read ``safety_steps``
        # (P0-61A), and a FAILED workflow without a failed step execution
        # is one the hardware escalation classifier cannot classify
        # (P0-46C).
        overdue = int((now - deadline).total_seconds())
        error = (
            "workflow execution deadline exceeded at "
            f"{deadline.isoformat()} "
            f"({overdue}s overdue); reaped by the dispatcher watchdog"
        )
        incident: FaultIncident | None
        try:
            incident = self.store.get_incident(claimed.incident_id)
        except NotFoundError:
            incident = None
        # The claim above succeeded as ``executor.config.executor_id``, so the
        # executor's own terminal funnel owns the write: it drops the owner,
        # derives the incident state from what the workflow completed (F-C9
        # release of unattempted restart reservations included), and records
        # the TERMINAL event under the watchdog's name.
        self.executor.terminalize_claimed(
            claimed,
            incident,
            WorkflowStatus.FAILED,
            claimed.execution_epoch,
            reason=error,
            actor=DISPATCHER_WATCHDOG_ACTOR,
            updates={
                "step_executions": self._deadline_step_executions(
                    claimed,
                    now,
                    error=error,
                    details={
                        "workflow_execution_deadline": (deadline.isoformat()),
                        "workflow_deadline_overdue_seconds": overdue,
                        "workflow_deadline_remote_command_cancellation": (cancellation),
                    },
                ),
            },
        )
        failed = self.store.get_workflow(workflow.request_id)
        self._sync_plan(failed, WorkflowStatus.FAILED)
        self._handle_failed_workflow(failed)
        return failed

    def _one_per_incident(
        self, candidates: list[WorkflowRequest]
    ) -> list[WorkflowRequest]:
        """One dispatchable workflow per incident, preferring the one it names.

        Picking the first (oldest-eligible) candidate let a stale record
        of the same incident win every tick and starve the workflow the
        incident was actually waiting on -- for five hours on 2026-09-04. The
        incident's ``workflow_request_id`` is the tie-breaker; a missing or
        unreadable incident falls back to the oldest candidate.
        """

        by_incident: dict[str, list[WorkflowRequest]] = {}
        for workflow in candidates:
            by_incident.setdefault(workflow.incident_id, []).append(workflow)
        chosen: list[WorkflowRequest] = []
        for incident_id, group in by_incident.items():
            selected = group[0]
            if len(group) > 1:
                try:
                    pointer = self.store.get_incident(incident_id).workflow_request_id
                except NotFoundError:
                    pointer = None
                # When the pointer does not resolve inside the group, a
                # successor flagged to preempt its predecessor is the one the
                # merge meant to run; the oldest row is the last resort (F-C1).
                flagged = [item for item in group if item.preempt_predecessor]
                selected = next(
                    (item for item in group if item.request_id == pointer),
                    flagged[0] if len(flagged) == 1 else group[0],
                )
            chosen.append(selected)
        return chosen

    @staticmethod
    def _deadline_step_executions(
        workflow: WorkflowRequest,
        now: datetime,
        *,
        error: str,
        details: dict[str, object],
    ) -> list[WorkflowStepExecution]:
        """Step executions with a FAILED record for the step the deadline caught."""

        steps = (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
        if not steps:
            return list(workflow.step_executions)
        completed = set(workflow.completed_step_indexes)
        index = next(
            (i for i in range(len(steps)) if i not in completed), len(steps) - 1
        )
        phase = execution_phase(workflow)
        kept = [
            item
            for item in workflow.step_executions
            if not execution_matches_step(item, index, steps[index].operation, phase)
        ]
        kept.append(
            WorkflowStepExecution(
                step_index=index,
                operation=steps[index].operation,
                status=WorkflowStepStatus.FAILED,
                phase=phase,
                error=error,
                details=dict(details),
                started_at=now,
                updated_at=now,
            )
        )
        return sorted(kept, key=lambda item: item.step_index)

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

    def sweep_stuck_records(self, now: datetime) -> set[str]:
        """Close the records nothing else will -- by Store predicate, never age.

        Abandoned and retired generations, compile-time BLOCKED no-ops and remote
        commands a terminal workflow left open; not counted into the dispatch
        report. Returns the still-settling retired generations the scan withholds.
        """

        self._supersede_abandoned_generations(now)
        retired = self._revoke_retired_generations(now)
        close_compile_blocked_workflows(self.store, now=now)
        cancel_orphaned_commands(self.store, now=now)
        return retired

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
                waiting_ttl=self._restart_waiting_ttl,
            )
            # The incident is what an operator reads first; without a line on
            # it the record closed under it is invisible there. Loaded only for
            # a workflow actually superseded, so the sweep costs the rest
            # nothing more. Bounded and deduplicated, so a rerun writes it once.
            audit = (
                f"abandoned generation {workflow.fencing_token} workflow "
                f"{workflow.request_id} superseded by {successor.request_id} "
                f"at generation {successor.fencing_token}"
            )
            incident = self.store.get_incident(workflow.incident_id)
            self.store.save_workflow_and_incident_if_leased(
                replacement,
                incident.model_copy(
                    update={
                        "reasons": bounded_reasons([*incident.reasons, audit]),
                        "updated_at": now,
                    }
                ),
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
        never ran. The one that actually cost a fleet was the opposite: on
        2026-09-04 the retired generation was ``RUNNING``, held an owner, a
        renewing lease, six budget claims and an unsettled ``STOP_WORKLOADS``
        remote command against three nodes of a running 24-GPU job, four hours
        after its incident had recovered at a higher generation.

        The order below is the whole point, and why this can take two ticks. A
        remote command outlives the workflow row, so revoking the workflow first
        would leave a destructive command behind whose next fence evaluation
        releases it. So: cancel the commands first -- ``PENDING`` and ``WAITING``
        go terminal at once, a ``LEASED`` one gets a cancellation request that
        bars it from being claimed again and turns whatever the executor reports
        into a ``FAILED`` -- and revoke only once the store shows them settled.

        No lease is taken: the lease holder is this very dispatch loop, which
        renews on every tick, so a lease-respecting revocation would wait
        forever on a record it is itself keeping alive. Safety comes from
        ``retired_generation_records`` re-deriving every condition inside the
        store transaction; a record that has already *completed* a destructive
        operation is never revoked -- there is something real to compensate
        for, and the release gate goes on reporting it until an operator acts.
        """

        held: set[str] = set()
        awaiting_operator = 0
        for workflow in self.store.list_workflows(
            RETIRED_GENERATION_STATUSES,
            limit=1000,
        ):
            try:
                verdict = self._revoke_retired_generation(workflow, now)
            except Exception:  # noqa: BLE001 - keep dispatching, do not revoke
                LOGGER.exception(
                    "retired generation revocation failed, workflow left open: %s",
                    workflow.request_id,
                )
                held.add(workflow.request_id)
                continue
            if verdict is None:
                continue
            held.add(workflow.request_id)
            if verdict == "awaiting_operator":
                awaiting_operator += 1
        # A gauge, not a counter: the records still waiting on an operator
        # this tick (F-A5, ``workflow_retired_generation_awaiting_operator``).
        self.retired_generation_awaiting_operator = awaiting_operator
        return held

    def _revoke_retired_generation(
        self,
        workflow: WorkflowRequest,
        now: datetime,
    ) -> Literal["awaiting_operator", "commands_settling"] | None:
        """Revoke one retired generation, or say why it is still held.

        ``None``: nothing to hold (not retired, or revoked now).
        ``"awaiting_operator"``: the record itself refuses revocation and only
        an operator can close it. ``"commands_settling"``: its remote commands
        were cancelled this tick; the next tick revokes it.
        """

        successor = retired_generation_successor(self.store, workflow)
        if successor is None:
            return None
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
            return "awaiting_operator"
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
            return "commands_settling"
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
            waiting_ttl=self._restart_waiting_ttl,
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
        return None

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
        if self.failure_handler is None:
            return
        try:
            self.failure_handler(workflow)
        except Exception:
            # Bounded (F-A6): a handler that fails deterministically used to
            # be re-invoked every tick, forever, by every process.
            current = self.store.get_workflow(workflow.request_id)
            attempts = current.failure_handling_attempts + 1
            update: dict[str, object] = {
                "failure_handling_attempts": attempts,
                "updated_at": datetime.now(timezone.utc),
            }
            abandoned = attempts >= self.config.failure_handling_max_attempts
            if abandoned:
                update["failure_handled_at"] = datetime.now(timezone.utc)
            # Compare-and-set on ``current``: a row another process moved since
            # the read raises ``StaleWriteError`` up to the per-workflow guard
            # in ``_reconcile_failed_workflows``; the next tick re-reads. The
            # counter moves after the write so a refused write is not counted
            # as an abandonment (store review 2026-09-07, item B).
            self.store.save_workflow(
                current.model_copy(update=update), expected=current
            )
            if abandoned:
                self.failure_handling_abandoned_total += 1
                self.failure_handling_abandoned_last_seen_timestamp_seconds = (
                    time.time()
                )
            LOGGER.exception(
                "failed workflow handler raised (attempt %d/%d%s): %s",
                attempts,
                self.config.failure_handling_max_attempts,
                ", abandoned" if abandoned else "",
                workflow.request_id,
            )
            return
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
                ),
                expected=current,
            )

    def _sync_plan(
        self,
        workflow: WorkflowRequest,
        status: WorkflowStatus,
    ) -> None:
        if not workflow.source_plan_id:
            return
        try:
            plan = self.store.get_plan(workflow.source_plan_id)
        except NotFoundError:
            # The plan row is gone; the workflow outcome stands on its own.
            # Raising here aborted the tick after the workflow was already
            # saved (P2-80D).
            self.plan_sync_misses_total += 1
            LOGGER.warning(
                "workflow %s names plan %s which no longer exists; status %s "
                "not mirrored",
                workflow.request_id,
                workflow.source_plan_id,
                status.value,
            )
            return
        plan_status = {
            WorkflowStatus.PENDING: PlanStatus.PENDING,
            WorkflowStatus.SAFETY_PENDING: PlanStatus.PENDING,
            WorkflowStatus.RUNNING: PlanStatus.RUNNING,
            WorkflowStatus.SUCCEEDED: PlanStatus.SUCCEEDED,
            WorkflowStatus.FAILED: PlanStatus.FAILED,
            WorkflowStatus.BLOCKED: PlanStatus.BLOCKED,
            WorkflowStatus.SUPERSEDED: PlanStatus.SUPERSEDED,
        }[status]
        if plan.status is not plan_status:
            try:
                self.store.save_plan(
                    plan.model_copy(update={"status": plan_status}), expected=plan
                )
            except StaleWriteError:
                # Another writer moved the plan between the read above and
                # this mirror write; the workflow outcome stands and the plan
                # is left as that writer wrote it (ARCH-D5).
                self.plan_sync_misses_total += 1
                LOGGER.warning(
                    "plan %s moved while mirroring workflow %s status %s; "
                    "not overwritten",
                    plan.plan_id,
                    workflow.request_id,
                    status.value,
                )

    def run_forever(self) -> None:
        if not self.config.enabled:
            return
        while not self._stop.is_set():
            # Clear before the scan, never after the wait: a wake that lands
            # while a cycle runs stays set and the wait returns at once, and a
            # wake that lands between the wait and the clear is answered by
            # the scan that immediately follows (F-A8).
            self._wake.clear()
            try:
                self.run_once()
            except Exception:
                # A transient Aurora failover must not kill this daemon
                # thread while the Pod and /healthz remain healthy.
                # The next cycle reopens/refreshes pooled connections and
                # resumes the leased workflow from its persisted step.
                LOGGER.exception("workflow dispatcher cycle failed")
            self._wake.wait(self.config.poll_interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._worker_pool.shutdown(wait=True)
