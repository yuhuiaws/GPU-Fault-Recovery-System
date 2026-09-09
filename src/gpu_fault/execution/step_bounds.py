"""Time bounds on a single workflow step, and on the workflow around it.

Both bounds live outside ``ProductionWorkflowExecutor`` because they are one
concern -- how long a thing is allowed to stay non-terminal -- and neither needs
anything from the executor beyond its store and its config.

The workflow bound is enforced by the lease holder rather than by
``WorkflowDispatcher._expire_stuck_workflows``. That watchdog has to
``claim_workflow`` to act, and the store refuses a claim from another owner while
the lease is live, so a workflow that is being redispatched every few seconds
renews the lease faster than the watchdog can take it. The only records the
watchdog can reap are the ones whose executor has already stopped touching them;
the looping workflow -- the one the deadline exists for -- was immune to its own
deadline. Observed live on 2026-09-05: a workflow ten minutes past
``execution_deadline``, still RUNNING, its lease renewed three minutes into the
future on every pass.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.execution.config import OPERATOR_ACKNOWLEDGEMENT_OPERATIONS
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    EXECUTABLE_WORKFLOW_STATUSES,
    StepPhase,
    WorkflowEvent,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    execution_matches_step,
    execution_phase,
    record_workflow_event,
)
from gpu_fault.store.shared.errors import NotFoundError

LOGGER = logging.getLogger(__name__)

# The outcome-detail keys a STEP_ATTEMPT event may carry. An allow-list, not a
# copy: ``outcome.details`` can hold payloads (a bundle manifest, a command's
# last lines) and the audit trail is small and bounded. ``reason`` is the
# adapter's own machine code for a wait or a refusal.
ATTEMPT_DETAIL_KEYS = (
    "reason",
    "gpu_reset_commit_attempt",
    "gpu_client_quiesce_attempt",
    "step_waiting_seconds",
    "step_waiting_slow",
)
_ATTEMPT_DETAIL_STRING_LIMIT = 128
# D-10: a WAITING attempt that differs from the previous one only in how long
# it has waited says nothing new; the record carries the age, the event does
# not need repeating. Every other allow-listed key is part of the identity.
_ATTEMPT_VOLATILE_DETAIL_KEYS = frozenset({"step_waiting_seconds"})


# Compensation that restores what the workflow itself stopped. It never
# escalates and it is the only thing allowed to start after the deadline.
DEADLINE_EXEMPT_COMPENSATION = frozenset({WorkflowOperation.RESTORE_GPU_SERVICES})


def workflow_deadline_failure(
    executor: Any,
    workflow: WorkflowRequest,
    step: WorkflowStepSpec,
    index: int,
) -> WorkflowStepOutcome | None:
    """Turn an exceeded workflow deadline into this step's failure.

    Returned as a step failure rather than terminalized on the spot so the
    established failure path runs: an unrestored quiesce still gets its
    compensation, the incident still lands in the right state, and the dispatcher
    still hands the record to the failure handler. The blunter alternative --
    terminalizing from a watchdog -- skips all three.
    """

    if step.operation in DEADLINE_EXEMPT_COMPENSATION:
        # The undo of a quiesce that already ran is owed even past the
        # deadline; failing it here would leave GPU services stopped on a
        # node handed to an operator, and count the same workflow twice.
        return None
    now = datetime.now(timezone.utc)
    lifetime = workflow.lifetime_deadline_at
    lifetime_hit = lifetime is not None and now >= lifetime
    deadline = workflow.execution_deadline
    if lifetime_hit:
        # The remediation's hard lifetime (F-N1) is the stronger verdict: it is
        # reported as such so neither the branch escalator nor the hardware
        # escalation plans another rung -- this record goes to an operator.
        deadline = lifetime
    if deadline is None or now < deadline:
        return None
    overdue = int((now - deadline).total_seconds())
    what = "workflow lifetime" if lifetime_hit else "workflow execution deadline"
    if lifetime_hit:
        counter = getattr(executor, "lifetime_exceeded_total", None)
        if counter is not None:
            executor.lifetime_exceeded_total = counter + 1
    LOGGER.error(
        "%s exceeded, failing step: workflow=%s step=%s/%s deadline=%s "
        "overdue_seconds=%s",
        what,
        workflow.request_id,
        index,
        step.operation.value,
        deadline.isoformat(),
        overdue,
    )
    # A remote command outlives the workflow record, so a deadline that
    # terminalizes the workflow without cancelling first leaves a command behind
    # that the next fence evaluation is free to release.
    cancellation: dict[str, int] | str
    try:
        cancellation = executor.store.cancel_remote_commands_for_workflow(
            workflow.request_id,
            reason=f"{what} exceeded",
        )
    except Exception as exc:  # noqa: BLE001 - the deadline still has to land
        LOGGER.exception(
            "workflow deadline could not cancel remote commands: workflow=%s",
            workflow.request_id,
        )
        cancellation = f"{type(exc).__name__}: {exc}"
    return WorkflowStepOutcome.failed(
        f"{what} exceeded at "
        f"{deadline.isoformat()} ({overdue}s overdue) before step "
        f"{index}/{step.operation.value}",
        details={
            "workflow_execution_deadline": deadline.isoformat(),
            "workflow_deadline_overdue_seconds": overdue,
            "workflow_deadline_remote_command_cancellation": cancellation,
            "workflow_lifetime_exceeded": lifetime_hit,
        },
    )


def bounded_waiting_outcome(
    executor: Any,
    workflow: WorkflowRequest,
    step: WorkflowStepSpec,
    index: int,
    outcome: WorkflowStepOutcome,
) -> WorkflowStepOutcome:
    """Cap how long one step may stay non-terminal, and say so on the way.

    The cap is deliberately not another retry counter. Every existing bound is a
    count an adapter reads back out of the step execution record, so a record
    whose key never advances -- a synthetic check borrowing its parent's
    ``step_index``, an outcome that is never persisted -- takes the counter's
    ceiling with it. This measures elapsed time off the record instead, which no
    adapter can freeze.
    """

    if outcome.status is not WorkflowStepStatus.WAITING:
        return outcome
    previous = previous_execution(workflow, step, index)
    if previous is None:
        return outcome
    since = step_elapsed_since(executor, workflow, previous)
    waited = int((datetime.now(timezone.utc) - since).total_seconds())
    # Per-operation, because a delegated node replacement waits on a provider
    # for as long as that provider takes while a quiesce that has not answered
    # in ten minutes is already wrong. One ceiling for both can only be the
    # larger, which is the same as having none for everything else.
    limit = executor.config.step_waiting_limit(step.operation)
    details = dict(outcome.details or {})
    # Rule A: a restart held on its ``requires_incident_state`` premise is a
    # job waiting on a node another remediation is repairing, and that wait
    # has one window -- the same ``node_busy_wait_seconds`` the dispatcher
    # applies before the workflow starts. Past it the job is not restarted --
    # unless the remediation it names is still open, in which case the
    # remediation's own lifetime is the bound (``_premise_hold_limit``).
    premise_hold = details.get("reason") == WorkflowEventCode.NODE_UNDER_REMEDIATION
    if premise_hold:
        limit = _premise_hold_limit(executor, details, limit, since)
    details["step_waiting_seconds"] = waited
    if waited >= limit:
        LOGGER.error(
            "workflow step exceeded its waiting cap: workflow=%s step=%s/%s "
            "waited_seconds=%s limit_seconds=%s last_detail=%s",
            workflow.request_id,
            index,
            step.operation.value,
            waited,
            limit,
            outcome.details or {},
        )
        details["step_waiting_timeout_seconds"] = limit
        if premise_hold:
            details["reason"] = WorkflowEventCode.NODE_REMEDIATION_TIMEOUT.value
            return WorkflowStepOutcome.failed(
                f"{WorkflowEventCode.NODE_REMEDIATION_TIMEOUT.value}: step "
                f"{index}/{step.operation.value} waited {waited}s for the node "
                f"remediation, past the {limit}s window; the job is not restarted"
                + (f"; last reported: {outcome.error}" if outcome.error else ""),
                details=details,
            )
        return WorkflowStepOutcome.failed(
            f"step {index}/{step.operation.value} stayed non-terminal for "
            f"{waited}s, past the {limit}s per-step cap"
            + (f"; last reported: {outcome.error}" if outcome.error else ""),
            details=details,
        )
    warning = executor.config.step_waiting_warning_limit(step.operation)
    if waited >= warning:
        details["step_waiting_slow"] = True
        # Latched on the record so the log fires on the crossing rather than on
        # every one of the redispatches that follow it.
        if not (previous.details or {}).get("step_waiting_slow"):
            LOGGER.warning(
                "workflow step is waiting far longer than expected: "
                "workflow=%s step=%s/%s waited_seconds=%s "
                "warning_seconds=%s cap_seconds=%s",
                workflow.request_id,
                index,
                step.operation.value,
                waited,
                warning,
                limit,
            )
    return replace(outcome, details=details)


def _premise_hold_limit(
    executor: Any,
    details: dict[str, Any],
    step_limit: int,
    since: datetime,
) -> int:
    """How long a restart may wait on its ``NODE_UNDER_REMEDIATION`` premise.

    The node-busy window (240 s by default) is the right bound for a job
    workflow that has not started: the dispatcher can still rewrite it into a
    stop. It is the wrong bound once the ``after_incident`` restart is running
    and the repair it waits for is real work in progress. A GPU reset chain
    holds a maintenance window of up to seven minutes, then needs fresh
    telemetry for VALIDATE_GPU and a RESTORE_SCHEDULING; a reboot chain waits
    on HyperPod for up to forty-five minutes. Cut at 240 s, the restart failed
    minutes before the repair landed and the job stayed down after a repair
    that succeeded (逻辑 6).

    So while the remediation the premise names is still open
    (``EXECUTABLE_WORKFLOW_STATUSES``) the bound is that remediation's own
    ``lifetime_deadline_at``, measured from when this step began waiting; a
    remediation not yet claimed has no lifetime stamped and gets the restart's
    per-operation cap as the fallback. Neither ever shortens the window the
    dispatcher already granted. Once the remediation is terminal the adapter
    settles the premise on its own (SUCCEEDED -> RECOVERED -> restart;
    FAILED/BLOCKED -> INCIDENT_NOT_RECOVERABLE), so a WAITING that still
    names it keeps today's window -- as does a hold with no
    ``remediation_workflow_id``, or one whose record cannot be read. Failing
    closed to the old cap is deliberate: a store outage must not turn a
    bounded wait into an unbounded one.
    """

    window = int(executor.config.node_busy_wait_seconds)
    remediation = _open_remediation(executor, details.get("remediation_workflow_id"))
    if remediation is None:
        return min(step_limit, window)
    lifetime = remediation.lifetime_deadline_at
    if lifetime is None:
        return max(step_limit, window)
    details["remediation_lifetime_deadline_at"] = lifetime.isoformat()
    return max(window, int((lifetime - since).total_seconds()))


def _open_remediation(executor: Any, request_id: Any) -> WorkflowRequest | None:
    """The still-executable remediation workflow ``request_id`` names, or ``None``.

    ``None`` covers every case the caller must treat as "keep today's cap": no
    id on the hold, no store on this executor, a record the store does not
    have (``NotFoundError`` -- the id was stamped from an incident whose
    workflow has since been pruned), a store that cannot answer, or a record
    that is already terminal.
    """

    if not request_id:
        return None
    store = getattr(executor, "store", None)
    if store is None:
        return None
    try:
        remediation: WorkflowRequest = store.get_workflow(str(request_id))
    except NotFoundError:
        return None
    except Exception:  # noqa: BLE001 - the bound must still land on the old cap
        LOGGER.warning(
            "could not read the remediation workflow a restart premise waits on; "
            "keeping the node-busy window: remediation_workflow=%s",
            request_id,
            exc_info=True,
        )
        return None
    if remediation.status not in EXECUTABLE_WORKFLOW_STATUSES:
        return None
    return remediation


def record_attempt(
    workflow: WorkflowRequest,
    step: WorkflowStepSpec,
    index: int,
    outcome: WorkflowStepOutcome,
) -> WorkflowRequest:
    """Put this attempt on the workflow, keeping the step's original start time.

    ``started_at`` used to be re-defaulted on every attempt, which made it mean
    "when this step was last touched" -- a value indistinguishable from
    ``updated_at`` and read by nothing. Keeping the first one is what gives
    ``bounded_waiting_outcome`` something to measure: how long this step has gone
    without reaching a terminal outcome. Recording it is part of the same
    function because that is the only way the preserved value reaches the store.
    """

    phase = execution_phase(workflow)
    previous = previous_execution(workflow, step, index)
    # The record below replaces this step's last one; the event is what keeps
    # the attempts that came before it (RF-2) -- unless this attempt is the
    # same wait as the last one, polled again (D-10): a step waiting on a
    # remote command is redispatched every tick, and one event per tick filled
    # the bounded event list in twenty minutes and evicted the history the
    # events exist to keep.
    if not _same_wait(previous, outcome):
        workflow = _record_attempt_event(workflow, step, index, outcome, phase)
    execution = WorkflowStepExecution(
        step_index=index,
        operation=step.operation,
        status=outcome.status,
        phase=phase,
        adapter_operation_id=outcome.adapter_operation_id,
        error=outcome.error,
        details=outcome.details or {},
        started_at=(
            previous.started_at if previous is not None else datetime.now(timezone.utc)
        ),
    )
    # A step's identity is (phase, index, operation) (F-C2 / P0-62D). What gets
    # replaced is exactly this step's own record -- a legacy record without a
    # phase counts as this step's. Records at this index for another phase or
    # another legitimate operation are different steps and stay. A record whose
    # operation matches neither phase at this index is a stale rebound (the
    # index now hosts another operation) and is dropped.
    legitimate = {
        steps[index].operation
        for steps in (workflow.official_steps, workflow.safety_steps)
        if index < len(steps)
    }
    executions = [
        item
        for item in workflow.step_executions
        if item.step_index != index
        or (
            not execution_matches_step(item, index, step.operation, phase)
            and item.operation in legitimate
        )
    ]
    executions.append(execution)
    return workflow.model_copy(
        update={
            "step_executions": sorted(executions, key=lambda item: item.step_index),
            "updated_at": datetime.now(timezone.utc),
        }
    )


def _same_wait(
    previous: WorkflowStepExecution | None, outcome: WorkflowStepOutcome
) -> bool:
    """Is ``outcome`` the wait ``previous`` already recorded, seen again?"""

    if (
        previous is None
        or previous.status is not WorkflowStepStatus.WAITING
        or outcome.status is not WorkflowStepStatus.WAITING
        or previous.error != outcome.error
        or previous.adapter_operation_id != outcome.adapter_operation_id
    ):
        return False
    before = previous.details or {}
    after = outcome.details or {}
    return all(
        before.get(key) == after.get(key)
        for key in ATTEMPT_DETAIL_KEYS
        if key not in _ATTEMPT_VOLATILE_DETAIL_KEYS
    )


def attempt_event_code(outcome: WorkflowStepOutcome) -> WorkflowEventCode:
    """The machine code for one attempt's outcome.

    A failure the bounds in this module produced (deadline, lifetime, waiting
    cap) is told apart from an adapter's own failure by the detail keys those
    bounds stamp, so a report can count "ran out of time" separately from
    "the node refused".
    """

    if outcome.status is WorkflowStepStatus.WAITING:
        return WorkflowEventCode.STEP_WAITING
    if outcome.status is WorkflowStepStatus.SUCCEEDED:
        return WorkflowEventCode.STEP_SUCCEEDED
    details = outcome.details or {}
    if details.get("workflow_lifetime_exceeded"):
        return WorkflowEventCode.LIFETIME_EXCEEDED
    if "workflow_execution_deadline" in details:
        return WorkflowEventCode.DEADLINE_EXCEEDED
    if details.get("reason") == WorkflowEventCode.NODE_REMEDIATION_TIMEOUT:
        # Rule A's give-up, spelt the same way the dispatcher spells its own.
        return WorkflowEventCode.NODE_REMEDIATION_TIMEOUT
    if "step_waiting_timeout_seconds" in details:
        return WorkflowEventCode.STEP_WAITING_TIMEOUT
    return WorkflowEventCode.STEP_FAILED


def step_attempt_history(
    workflow: WorkflowRequest,
    step_index: int,
    operation: WorkflowOperation,
    phase: StepPhase | None = None,
) -> list[WorkflowEvent]:
    """Every recorded attempt at one step identity, oldest first.

    Same dual read as ``execution_matches_step``: an event or a caller without
    a phase matches either phase. Bounded by ``WORKFLOW_EVENTS_LIMIT``, so on a
    workflow that outlived its budget the middle of a long retry run is gone
    and the ``attempt`` numbers restart from what survived.
    """

    return [
        event
        for event in workflow.events
        if event.kind is WorkflowEventKind.STEP_ATTEMPT
        and event.step_index == step_index
        and event.operation is operation
        and (phase is None or event.phase is None or event.phase == phase)
    ]


def _record_attempt_event(
    workflow: WorkflowRequest,
    step: WorkflowStepSpec,
    index: int,
    outcome: WorkflowStepOutcome,
    phase: StepPhase,
) -> WorkflowRequest:
    source = outcome.details or {}
    details: dict[str, Any] = {
        "attempt": len(step_attempt_history(workflow, index, step.operation, phase))
        + 1,
    }
    if outcome.adapter_operation_id is not None:
        details["adapter_operation_id"] = outcome.adapter_operation_id
    for key in ATTEMPT_DETAIL_KEYS:
        value = source.get(key)
        if isinstance(value, str):
            details[key] = value[:_ATTEMPT_DETAIL_STRING_LIMIT]
        elif isinstance(value, (bool, int, float)):
            details[key] = value
    return record_workflow_event(
        workflow,
        WorkflowEventKind.STEP_ATTEMPT,
        code=attempt_event_code(outcome),
        step_index=index,
        operation=step.operation,
        phase=phase,
        status=outcome.status.value,
        reason=outcome.error,
        details=details,
    )


def previous_execution(
    workflow: WorkflowRequest,
    step: WorkflowStepSpec,
    index: int,
) -> WorkflowStepExecution | None:
    """The record this step already has, if it is the same step.

    Reversed because a remote executor appends a synthetic prior record for the
    step it is about to run without replacing the existing one, so an index can
    carry more than one entry and the last is the live one. The operation has to
    match: a workflow that fell back to ``safety_steps`` reuses the same indexes
    for entirely different operations.
    """

    phase = execution_phase(workflow)
    return next(
        (
            item
            for item in reversed(workflow.step_executions)
            if execution_matches_step(item, index, step.operation, phase)
        ),
        None,
    )


def step_elapsed_since(
    executor: Any,
    workflow: WorkflowRequest,
    previous: WorkflowStepExecution,
) -> datetime:
    """When this step's clock started, never earlier than this window.

    ``started_at`` can predate the current execution window: a merged or branched
    workflow inherits the executions of the record it absorbed. So the window's
    own start (``execution_window_start``) is the floor. Clamping can only push
    the cap later, never fire it early on inherited time.
    """

    since = previous.started_at
    window_start = execution_window_start(executor, workflow)
    if window_start is None:
        return since
    # A window cannot start in the future: whatever stamped it did so no later
    # than now, so the clamp only ever restores a wait that the arithmetic
    # below would otherwise deny.
    window_start = min(window_start, datetime.now(timezone.utc))
    return max(since, window_start)


def execution_window_start(executor: Any, workflow: WorkflowRequest) -> datetime | None:
    """When this record's execution window opened, or ``None`` if unknown.

    Three anchors say when the record was first claimed, and none of them can
    be earlier than that moment, so the earliest one is the closest answer:

    * the first ``CLAIM`` event -- written by the claim that leased the record
      and kept at the head of the bounded event list; on a row that was already
      running when CLAIM events were introduced it names a later claim;
    * ``lifetime_deadline_at`` minus the lifetime ``claim_deadlines`` stamped it
      with (F-N1) -- set once and inherited, except that an
      operator-acknowledgement workflow re-floors it on every claim;
    * ``execution_deadline`` minus the execution budget -- stable for a
      sequential workflow, but re-stamped on every claim of a DAG workflow, so
      for a DAG it is not read at all: it put the window start at the current
      tick and ``step_waiting_seconds`` read 0 for every step of every
      multi-node job workflow (control-plane review 2026-09-08, D-1).

    A workflow holding an operator-acknowledgement step (CHECK_MECHANICALS) has
    both deadlines floored at ``now + acknowledgement timeout`` at claim time;
    subtracting only the plain budget put the window start ~24 h in the future
    and ``step_waiting_seconds`` went negative (-84599 observed live), so the
    same floor is applied to the budget here. A workflow holding a node install
    has its execution deadline floored at the install ceiling plus the
    containment allowance (F2), so that floor is applied to the execution
    budget too: subtracting only the 1800 s timeout put the window start 700 s
    in the future and the install's wait read 0 for as long.
    """

    anchors: list[datetime] = []
    first_claim = next(
        (
            event.at
            for event in workflow.events
            if event.kind is WorkflowEventKind.CLAIM
        ),
        None,
    )
    if first_claim is not None:
        anchors.append(first_claim)
    acknowledgement = (
        float(
            getattr(
                executor.config,
                "operator_acknowledgement_timeout_seconds",
                0,
            )
        )
        if any(
            step.operation in OPERATOR_ACKNOWLEDGEMENT_OPERATIONS
            for step in workflow.official_steps
        )
        else 0.0
    )
    if workflow.lifetime_deadline_at is not None:
        is_job = workflow.dag_enabled or any(
            step.operation is WorkflowOperation.RESTART_WORKLOAD
            for step in workflow.official_steps
        )
        lifetime_seconds = float(
            executor.config.job_workflow_lifetime_seconds
            if is_job
            else executor.config.node_workflow_lifetime_seconds
        )
        anchors.append(
            workflow.lifetime_deadline_at
            - timedelta(seconds=max(lifetime_seconds, acknowledgement))
        )
    if workflow.execution_deadline is not None and not workflow.dag_enabled:
        # The budget ``claim_deadlines`` stamped: the plain timeout, floored at
        # the acknowledgement window and -- for a workflow holding a node
        # install -- at the install ceiling plus the containment allowance
        # (F2). ``ProductionExecutorConfig.execution_budget_seconds`` is that
        # arithmetic; a config double without it gets the acknowledgement
        # floor alone.
        execution_budget = getattr(executor.config, "execution_budget_seconds", None)
        if execution_budget is not None:
            budget_seconds = float(
                execution_budget(step.operation for step in workflow.official_steps)
            )
        else:
            budget_seconds = max(
                float(executor.config.workflow_execution_timeout_seconds),
                acknowledgement,
            )
        anchors.append(workflow.execution_deadline - timedelta(seconds=budget_seconds))
    return min(anchors, default=None)
