"""Pure verdict functions and constants of GF-REGIONAL-DESTR-018.

Split out of ``run_destr018_lifetime_deadline.py`` so the runner stays a
driver: everything here is unit-tested against synthetic control-plane, Node
Agent ledger and kernel-journal snapshots and touches no cluster.

The case proves the hard workflow lifetime (F-N1): a RESET_GPU remediation
that cannot leave ``VERIFY_NO_GPU_CLIENTS`` because a device holder keeps one
GPU open must lose the whole workflow to its lifetime deadline rather than
grind through the 60-attempt verify budget, must cancel its in-flight remote
command, must still run its one deadline-exempt compensation, and must escalate
only to a human.

Two contracts here are subtle enough to name:

* The step that waits is ``VERIFY_NO_GPU_CLIENTS``, not ``RESET_GPU``. The
  reset step has no WAITING branch, so the lifetime has to fire while the
  workflow is still verifying that the device is free. That is what makes the
  timing arithmetic in :func:`lifetime_margin_errors` load-bearing: if the
  compressed lifetime is long enough for the verify step to burn
  ``GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS`` attempts first, the workflow
  fails for the wrong reason and the case proves nothing.
* The data-plane verdicts are written around one Node Agent fact: the ledger
  only ever gets a ``started_at`` from ``mark_in_progress``, which the Agent
  executor calls on every execution it owns. A row with no ``started_at`` is
  therefore undecidable, never a pass.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

CASE_ID = "GF-REGIONAL-DESTR-018"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-010"
EXPECTED_XID = 46
ABSORB_XID = 79

# The step that waits, the step that must never run, and the one compensation
# the deadline is allowed to let through (execution/step_bounds.py:42).
WAITING_STEP = "VERIFY_NO_GPU_CLIENTS"
RESET_STEP = "RESET_GPU"
COMPENSATION_STEP = "RESTORE_GPU_SERVICES"

# Node-mutating Agent operations that must not start after the deadline. The
# compensation is deliberately absent: it is the only deadline-exempt step.
POST_CANCEL_FORBIDDEN_OPERATIONS = (
    "QUIESCE_GPU_SERVICES",
    WAITING_STEP,
    RESET_STEP,
    "RESET_ALL_GPUS_NVSWITCHES",
    "VALIDATE_GPU",
)
# Control-plane operations that a lifetime failure must never reach: the
# deadline retires the remediation to a human, it does not climb the ladder.
FORBIDDEN_OPERATIONS = frozenset(
    {
        "RESTART_NODE",
        "REPLACE_NODE",
        "RESTART_VM",
        "RESET_ALL_GPUS_NVSWITCHES",
        "VALIDATE_GPU",
        "VALIDATE_HOST",
        "VALIDATE_FABRIC",
        "RESTORE_SCHEDULING",
        "RESTART_WORKLOAD",
    }
)
# The support escalation a lifetime failure is allowed to raise, in order
# (orchestration/escalation.py builds these for RecoveryAction.ESCALATE_OPERATOR
# on an idle node -- no STOP_WORKLOADS, because nothing is running).
SUPPORT_ESCALATION_OPERATIONS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUARANTINE",
    "ESCALATE_SUPPORT",
)
SUPPORT_ESCALATION_STAGE = "lifetime_exceeded"
SUPPORT_ESCALATION_ACTION = "ESCALATE_OPERATOR"

# The two status sources a cancelled remote command may carry
# (store/*/remote_commands.py): WAITING at cancellation, or LEASED and
# reporting afterwards.
CANCELLED_BY_TIMEOUT = "workflow-timeout"
COMPLETED_AFTER_CANCELLATION = "completed-after-cancellation"
CANCELLATION_STATUS_SOURCES = (CANCELLED_BY_TIMEOUT, COMPLETED_AFTER_CANCELLATION)

QUARANTINE_TAINT = "gpu-fault.io/quarantined"
LIFETIME_METRIC = "gpu_fault_workflow_lifetime_exceeded_total"
RECORD_ONLY_METRIC = (
    'gpu_fault_workflow_merge_record_only_total{reason="lifetime_exceeded"}'
)

# ---------------------------------------------------------------------------
# Timing arithmetic
# ---------------------------------------------------------------------------
# The compressed window written into the control worker for the drill. The
# node lifetime is the deadline the case exists to prove, and every other knob
# is set in lockstep so the control plane will actually boot with it (its
# start-up guard in execution/config.py refuses a lifetime below the step
# ceilings, the managed-recovery window, or at/below the lease):
#
# * execution timeout == lifetime -- claim_deadlines stamps
#   ``min(execution, lifetime)``; below the lifetime the execution deadline
#   fires first and the failure would not carry workflow_lifetime_exceeded,
#   above it the value is silently truncated to the lifetime.
# * step timeout == lifetime -- validate_timing_relationships forbids a step
#   waiting ceiling *above* the node lifetime, so this is the largest value the
#   control plane will accept; keeping it *at* the lifetime (rather than below)
#   means the WAITING step's own cap cannot fire before the lifetime does,
#   because executor._execute_step checks the workflow deadline before it
#   dispatches the step and the step's wait only starts after containment.
# * managed recovery == lifetime -- a per-operation override may not sit below
#   the default step timeout (from_mapping) and may not exceed the lifetime
#   (validate_timing_relationships), so with the step timeout compressed to the
#   lifetime the managed-recovery window is pinned to the same value.
# * step warning < step timeout, and lease < execution timeout -- the two
#   ordering rules from_mapping and validate_timing_relationships impose that
#   the equal knobs above would otherwise violate.
LIFETIME_SECONDS = 180
EXECUTION_TIMEOUT_SECONDS = 180
STEP_TIMEOUT_SECONDS = 180
MANAGED_RECOVERY_SECONDS = 180
STEP_WARNING_SECONDS = 150
LEASE_DURATION_SECONDS = 120
# The env-window helper's own bounds; repeated here so the arithmetic refuses
# a value the window would refuse anyway.
MINIMUM_WINDOW_SECONDS = 60
MAXIMUM_WINDOW_SECONDS = 3600
# Deployed defaults the arithmetic is judged against.
VERIFY_MAX_ATTEMPTS = 60
# The per-step waiting cap the drill actually runs under: the window compresses
# it to the lifetime (see STEP_TIMEOUT_SECONDS above), so the margin arithmetic
# is judged against the lifetime, not the shipped 600s default.
STEP_WAITING_CAP_SECONDS = STEP_TIMEOUT_SECONDS
# The shipped default step waiting cap, recorded as pre-window evidence.
SHIPPED_STEP_WAITING_CAP_SECONDS = 600
# The dispatch poll interval of the deployed control worker: the fastest
# cadence a WAITING step can be redispatched at. A measured cadence below it
# means the site is not the one this arithmetic was computed for.
CADENCE_FLOOR_SECONDS = 5
# Wall clock the workflow spends before it first reaches the verify step
# (MARK_UNSCHEDULABLE + QUIESCE_GPU_SERVICES on an idle node). Subtracted from
# the lifetime, the remainder is the window the verify step can burn attempts
# in. Deliberately small, because a *shorter* containment means *more* verify
# attempts, so the worst case is the pessimistic one.
CONTAINMENT_ALLOWANCE_SECONDS = 90
# Attempts that must still be unspent when the lifetime fires. Without a
# margin an attempt-budget failure and a lifetime failure are indistinguishable.
ATTEMPT_MARGIN = 15
# Cadence gaps a sample needs before the runner may re-assert the arithmetic
# against it.
MINIMUM_CADENCE_GAPS = 2


def parse_time(value: Any) -> datetime | None:
    """A UTC datetime from an ISO-8601 field, or ``None`` if unusable."""

    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def worst_case_verify_attempts(
    *,
    lifetime_seconds: int,
    cadence_seconds: float,
    containment_allowance_seconds: int = CONTAINMENT_ALLOWANCE_SECONDS,
) -> int:
    """Verify attempts the drill can burn before the lifetime fires.

    Pessimistic on purpose: containment is assumed to take its whole
    allowance's *complement*, i.e. the verify step is assumed to own every
    second of the lifetime that containment did not need.
    """

    window = max(lifetime_seconds - containment_allowance_seconds, 0)
    if cadence_seconds <= 0:
        raise ValueError("cadence must be positive")
    return math.ceil(window / cadence_seconds)


def lifetime_margin_errors(
    *,
    lifetime_seconds: int,
    execution_timeout_seconds: int,
    cadence_seconds: float | None,
    verify_max_attempts: int = VERIFY_MAX_ATTEMPTS,
    step_waiting_cap_seconds: int = STEP_WAITING_CAP_SECONDS,
    containment_allowance_seconds: int = CONTAINMENT_ALLOWANCE_SECONDS,
    attempt_margin: int = ATTEMPT_MARGIN,
) -> list[str]:
    """Refuse a compressed window that would fail for the wrong reason.

    ``GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS`` is never lowered for this
    drill -- lowering it would move the failure into the Agent's attempt budget
    and hide the deadline. Instead the lifetime is chosen so that
    ``lifetime < attempts x cadence``, and the runner recomputes this against a
    measured cadence rather than an assumed one.
    """

    errors: list[str] = []
    for name, value in (
        ("lifetime", lifetime_seconds),
        ("execution timeout", execution_timeout_seconds),
    ):
        if not MINIMUM_WINDOW_SECONDS <= value <= MAXIMUM_WINDOW_SECONDS:
            errors.append(
                f"{name} {value}s is outside the env window's bounds "
                f"[{MINIMUM_WINDOW_SECONDS}, {MAXIMUM_WINDOW_SECONDS}]"
            )
    if execution_timeout_seconds < lifetime_seconds:
        errors.append(
            f"execution timeout {execution_timeout_seconds}s is below the "
            f"lifetime {lifetime_seconds}s; the execution deadline would fire "
            "first and the failure would not carry workflow_lifetime_exceeded"
        )
    if step_waiting_cap_seconds < lifetime_seconds:
        errors.append(
            f"the per-step waiting cap {step_waiting_cap_seconds}s is below the "
            f"lifetime {lifetime_seconds}s; the step's own bound could end the "
            "wait before the workflow lifetime does"
        )
    if cadence_seconds is None:
        errors.append(
            "redispatch cadence is unknown; the attempt margin cannot be "
            "computed, so the drill is refused rather than guessed"
        )
        return errors
    if cadence_seconds < CADENCE_FLOOR_SECONDS:
        errors.append(
            f"measured redispatch cadence {cadence_seconds:.3f}s is below the "
            f"deployed floor {CADENCE_FLOOR_SECONDS}s; recompute the window "
            "for this site instead of reusing this one"
        )
        return errors
    window = lifetime_seconds - containment_allowance_seconds
    if window <= 0:
        errors.append(
            f"lifetime {lifetime_seconds}s leaves no room after the "
            f"containment allowance {containment_allowance_seconds}s"
        )
        return errors
    attempts = worst_case_verify_attempts(
        lifetime_seconds=lifetime_seconds,
        cadence_seconds=cadence_seconds,
        containment_allowance_seconds=containment_allowance_seconds,
    )
    if attempts + attempt_margin > verify_max_attempts:
        errors.append(
            f"worst case {attempts} verify attempts plus the {attempt_margin} "
            f"attempt margin exceeds GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS="
            f"{verify_max_attempts}; the step could fail on its attempt budget "
            "instead of on the workflow lifetime"
        )
    return errors


def timing_evidence(
    *,
    lifetime_seconds: int,
    execution_timeout_seconds: int,
    cadence_seconds: float | None,
) -> dict[str, Any]:
    """The arithmetic, recorded so a run can be re-judged from evidence."""

    attempts = None
    if cadence_seconds is not None and cadence_seconds > 0:
        attempts = worst_case_verify_attempts(
            lifetime_seconds=lifetime_seconds,
            cadence_seconds=cadence_seconds,
        )
    return {
        "lifetime_seconds": lifetime_seconds,
        "execution_timeout_seconds": execution_timeout_seconds,
        "cadence_seconds": cadence_seconds,
        "cadence_floor_seconds": CADENCE_FLOOR_SECONDS,
        "containment_allowance_seconds": CONTAINMENT_ALLOWANCE_SECONDS,
        "verify_max_attempts": VERIFY_MAX_ATTEMPTS,
        "attempt_margin": ATTEMPT_MARGIN,
        "step_waiting_cap_seconds": STEP_WAITING_CAP_SECONDS,
        "worst_case_verify_attempts": attempts,
    }


def cadence_sample(
    rows: list[dict[str, Any]],
    *,
    operation: str = WAITING_STEP,
    baseline_command_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Observed redispatch cadence of a WAITING step, from ledger rows.

    One ledger row per verify attempt (``command_suffix`` carries the attempt
    number), so the gaps between consecutive ``started_at`` values are the
    cadence the control plane redispatched at.
    """

    baseline = baseline_command_ids or set()
    stamps = sorted(
        stamp
        for row in rows
        if row.get("operation") == operation
        and row.get("command_id") not in baseline
        and (stamp := parse_time(row.get("started_at"))) is not None
    )
    gaps = [
        (second - first).total_seconds()
        for first, second in zip(stamps, stamps[1:])
        if (second - first).total_seconds() > 0
    ]
    return {
        "operation": operation,
        "row_count": len(stamps),
        "gap_count": len(gaps),
        "min_gap_seconds": min(gaps) if gaps else None,
        "max_gap_seconds": max(gaps) if gaps else None,
    }


def cadence_errors(
    sample: dict[str, Any],
    *,
    minimum_gaps: int = MINIMUM_CADENCE_GAPS,
) -> list[str]:
    errors: list[str] = []
    gap_count = int(sample.get("gap_count") or 0)
    if gap_count < minimum_gaps:
        errors.append(
            f"the redispatch cadence sample has {gap_count} gaps, fewer than "
            f"the {minimum_gaps} needed to judge the attempt margin"
        )
        return errors
    minimum = sample.get("min_gap_seconds")
    if minimum is None or float(minimum) <= 0:
        errors.append(f"the redispatch cadence sample has no usable gap: {sample}")
    return errors


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------
def cancellation_moment(commands: list[dict[str, Any]]) -> datetime | None:
    """T_cancel: when the deadline cancelled the workflow's remote commands.

    A command that was WAITING is failed outright with
    ``status_source=workflow-timeout`` and no cancellation stamp, so its
    ``updated_at`` is the moment; a LEASED one carries
    ``cancellation_requested_at``. The earliest of those is T_cancel.
    """

    moments: list[datetime] = []
    for command in commands:
        requested = parse_time(command.get("cancellation_requested_at"))
        if requested is not None:
            moments.append(requested)
            continue
        if command.get("status_source") == CANCELLED_BY_TIMEOUT:
            updated = parse_time(command.get("updated_at"))
            if updated is not None:
                moments.append(updated)
    return min(moments) if moments else None


def step_operations(workflow: dict[str, Any]) -> list[str]:
    return [
        str(step.get("operation"))
        for step in (workflow.get("official_steps") or [])
        if isinstance(step, dict)
    ]


def executions_of(workflow: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    return [
        item
        for item in (workflow.get("step_executions") or [])
        if item.get("operation") == operation
    ]


def lifetime_failed_executions(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in (workflow.get("step_executions") or [])
        if (item.get("details") or {}).get("workflow_lifetime_exceeded") is True
    ]


def workflow_errors(
    workflow: dict[str, Any],
    incident: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """The wf-A half of the terminal contract."""

    errors: list[str] = []
    if workflow.get("status") != "FAILED":
        errors.append(f"workflow status is not FAILED: {workflow.get('status')}")
    deadline = parse_time(workflow.get("lifetime_deadline_at"))
    if deadline is None:
        errors.append(
            "workflow carries no lifetime_deadline_at; the env window did not "
            "reach the worker that claimed it"
        )
    lifetime_failures = lifetime_failed_executions(workflow)
    if not lifetime_failures:
        errors.append(
            "no step execution carries details.workflow_lifetime_exceeded=true; "
            "the workflow failed for another reason"
        )
    else:
        operations = {item.get("operation") for item in lifetime_failures}
        if WAITING_STEP not in operations:
            errors.append(
                f"the lifetime failure did not land on {WAITING_STEP}: "
                f"{sorted(str(item) for item in operations)}"
            )
        for item in lifetime_failures:
            if item.get("status") != "FAILED":
                errors.append(
                    "a lifetime-exceeded step execution is not FAILED: "
                    f"{item.get('operation')}/{item.get('status')}"
                )
            details = item.get("details") or {}
            for key in (
                "workflow_execution_deadline",
                "workflow_deadline_overdue_seconds",
                "workflow_deadline_remote_command_cancellation",
            ):
                if key not in details:
                    errors.append(
                        f"the lifetime failure of {item.get('operation')} is "
                        f"missing details.{key}"
                    )
    for execution in executions_of(workflow, RESET_STEP):
        if execution.get("status") == "SUCCEEDED":
            errors.append(
                "a RESET_GPU step execution SUCCEEDED after the deadline; the "
                "control plane turned a late node result into success"
            )
    compensations = [
        item
        for item in executions_of(workflow, COMPENSATION_STEP)
        if item.get("status") == "SUCCEEDED"
    ]
    if len(compensations) != 1:
        errors.append(
            f"expected exactly one successful {COMPENSATION_STEP} step "
            f"execution, found {len(compensations)}"
        )
    reached = {
        str(item.get("operation")) for item in (workflow.get("step_executions") or [])
    }
    forbidden = sorted(reached & FORBIDDEN_OPERATIONS)
    if forbidden:
        errors.append(
            f"the workflow reached operations the deadline forbids: {forbidden}"
        )
    if incident.get("state") not in {"ESCALATED", "QUARANTINED"}:
        errors.append(
            f"incident state is not ESCALATED or QUARANTINED: {incident.get('state')}"
        )
    if list(incident.get("node_ids") or []) != [node]:
        errors.append(
            f"incident covers nodes {incident.get('node_ids')}, not just {node}"
        )
    return errors


def remote_command_errors(
    commands: list[dict[str, Any]],
    *,
    t_cancel: datetime,
) -> list[str]:
    """Every in-flight command must end FAILED with a cancellation source."""

    errors: list[str] = []
    waiting = [
        command
        for command in commands
        if (command.get("step") or {}).get("operation") == WAITING_STEP
    ]
    if not waiting:
        errors.append(f"no remote command was issued for {WAITING_STEP}")
    cancelled = [
        command
        for command in commands
        if command.get("status_source") in CANCELLATION_STATUS_SOURCES
    ]
    if not cancelled:
        errors.append(
            "no remote command carries a cancellation status source "
            f"{list(CANCELLATION_STATUS_SOURCES)}; the deadline cancelled nothing"
        )
    for command in cancelled:
        operation = (command.get("step") or {}).get("operation")
        if command.get("status") != "FAILED":
            errors.append(
                f"cancelled remote command {operation} is not FAILED: "
                f"{command.get('status')}"
            )
        if command.get("status_source") == COMPLETED_AFTER_CANCELLATION:
            details = command.get("result_details") or {}
            if not details.get("post_cancellation_status"):
                errors.append(
                    f"remote command {operation} completed after cancellation "
                    "without result_details.post_cancellation_status"
                )
    for command in commands:
        operation = (command.get("step") or {}).get("operation")
        completed = parse_time(command.get("updated_at"))
        if command.get("status") != "SUCCEEDED":
            continue
        if completed is not None and completed > t_cancel:
            if operation != COMPENSATION_STEP:
                errors.append(
                    f"remote command {operation} SUCCEEDED after the deadline; "
                    f"only {COMPENSATION_STEP} is deadline-exempt"
                )
    compensations = [
        command
        for command in commands
        if (command.get("step") or {}).get("operation") == COMPENSATION_STEP
        and command.get("status") == "SUCCEEDED"
    ]
    if len(compensations) != 1:
        errors.append(
            f"expected exactly one successful {COMPENSATION_STEP} remote "
            f"command, found {len(compensations)}"
        )
    return errors


def escalation_errors(
    escalation: dict[str, Any],
    *,
    node: str,
) -> list[str]:
    """The escalation half: one human handoff, and it must not self-chain.

    ``HardwareEscalationService.emit`` copies the source workflow's
    ``lifetime_deadline_at`` into the support workflow ("the chain shares one
    lifetime"), and ``claim_deadlines`` never re-stamps an existing lifetime.
    A support workflow that inherits an already-expired lifetime therefore
    fails on its own first step, never reaches QUARANTINE, and is escalated
    again -- one new workflow per reconcile pass. This verdict is the assertion
    that closes that hole: the handoff must be reachable and terminal, and
    there must be no second-order escalation.
    """

    errors: list[str] = []
    incident = escalation.get("incident") or {}
    workflow = escalation.get("workflow") or {}
    if not incident or not workflow:
        errors.append(
            "the lifetime failure raised no support escalation; "
            f"{SUPPORT_ESCALATION_STAGE} must hand the node to a human"
        )
        return errors
    if incident.get("effective_action") != SUPPORT_ESCALATION_ACTION:
        errors.append(
            "support incident action is not "
            f"{SUPPORT_ESCALATION_ACTION}: {incident.get('effective_action')}"
        )
    reasons = [str(item) for item in (incident.get("reasons") or [])]
    if not reasons or not reasons[0].startswith(SUPPORT_ESCALATION_STAGE):
        errors.append(
            "support incident does not name the "
            f"{SUPPORT_ESCALATION_STAGE} stage: {reasons[:1]}"
        )
    if list(incident.get("node_ids") or []) != [node]:
        errors.append(
            f"support incident covers nodes {incident.get('node_ids')}, not just {node}"
        )
    operations = step_operations(workflow)
    if operations != list(SUPPORT_ESCALATION_OPERATIONS):
        errors.append(
            f"support workflow steps are {operations}, not "
            f"{list(SUPPORT_ESCALATION_OPERATIONS)}"
        )
    inherited = lifetime_failed_executions(workflow)
    if inherited:
        errors.append(
            "the support workflow failed on the lifetime it inherited from the "
            f"workflow it replaces ({[item.get('operation') for item in inherited]}); "
            "the node was never quarantined and the escalation will repeat"
        )
    if workflow.get("status") not in {"SUCCEEDED", "PENDING", "RUNNING"}:
        errors.append(
            f"support workflow status is {workflow.get('status')}; the human "
            "handoff neither completed nor is still running"
        )
    second = escalation.get("second_order_workflow") or {}
    if second:
        errors.append(
            "a second-order escalation exists "
            f"({second.get('request_id')}); the lifetime failure escalated its "
            "own escalation"
        )
    return errors


SUPPORT_TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"})


def quarantine_settled(
    node_snapshot: dict[str, Any],
    support_workflow: dict[str, Any] | None,
) -> bool:
    """Whether the quarantine verdict can be judged yet.

    The support workflow's QUARANTINE step runs asynchronously after the
    lifetime failure, so a node read the instant the reset workflow went FAILED
    has not been tainted *yet*. The verdict is ready once the taint is there, or
    once the support workflow is terminal (then whatever the node shows is what
    it will keep showing).
    """

    taints = [
        str(item.get("key"))
        for item in (node_snapshot.get("taints") or [])
        if isinstance(item, dict)
    ]
    if QUARANTINE_TAINT in taints:
        return True
    status = (support_workflow or {}).get("status")
    return status in SUPPORT_TERMINAL_STATUSES


def absorb_settled(snapshot: dict[str, Any]) -> bool:
    """Whether the absorbed event has been merged, not merely ingested.

    The event row appears first; the merge onto an incident is what
    :func:`absorb_errors` judges, so a snapshot with the event but no incident
    is still in flight.
    """

    return bool(snapshot.get("event")) and bool(snapshot.get("incident"))


def quarantine_errors(node_snapshot: dict[str, Any]) -> list[str]:
    """The node must end isolated: unschedulable and carrying the taint."""

    errors: list[str] = []
    taints = [
        str(item.get("key"))
        for item in (node_snapshot.get("taints") or [])
        if isinstance(item, dict)
    ]
    if QUARANTINE_TAINT not in taints:
        errors.append(
            f"node does not carry the {QUARANTINE_TAINT} taint after the "
            f"lifetime escalation: {taints}"
        )
    if node_snapshot.get("unschedulable") is not True:
        errors.append(
            "node is still schedulable after the lifetime escalation: "
            f"unschedulable={node_snapshot.get('unschedulable')}"
        )
    return errors


def absorb_errors(
    snapshot: dict[str, Any],
    *,
    incident_id: str,
    workflow_request_id: str,
    official_step_count: int,
) -> list[str]:
    """XID 79 on the same node is recorded, never planned.

    ``WorkflowMergeService.disposition`` checks ``lifetime_exceeded(existing)``
    before anything else and returns ABSORB_RECORD_ONLY, so the second fault
    lands on the incident that already exists and adds no step. On a node whose
    remediation was retired to a human, the support incident may be the merge
    target, so either incident of the pair is a pass -- a *new* executable
    workflow is not.
    """

    errors: list[str] = []
    event = snapshot.get("event") or {}
    if not event:
        errors.append("the second XID never reached the control plane")
        return errors
    if int(event.get("xid") or 0) != ABSORB_XID:
        errors.append(f"the second event is not XID {ABSORB_XID}: {event.get('xid')}")
    incident = snapshot.get("incident") or {}
    workflow = snapshot.get("workflow") or {}
    if not incident:
        errors.append("the second XID produced no incident record")
        return errors
    allowed_incidents = {incident_id, f"inc-support-after-{workflow_request_id}"}
    if incident.get("incident_id") not in allowed_incidents:
        errors.append(
            f"the second XID opened incident {incident.get('incident_id')}, "
            f"not one of {sorted(allowed_incidents)}"
        )
    allowed_workflows = {
        workflow_request_id,
        f"workflow-support-after-{workflow_request_id}",
    }
    request_id = workflow.get("request_id")
    if request_id is not None and request_id not in allowed_workflows:
        errors.append(
            f"the second XID opened workflow {request_id}, not one of "
            f"{sorted(allowed_workflows)}"
        )
    if request_id == workflow_request_id:
        if workflow.get("status") != "FAILED":
            errors.append(
                "the absorbed event revived the failed workflow: "
                f"{workflow.get('status')}"
            )
        if len(step_operations(workflow)) != official_step_count:
            errors.append(
                "the absorbed event added steps to the failed workflow: "
                f"{len(step_operations(workflow))} != {official_step_count}"
            )
    return errors


def new_executable_workflow_errors(
    workflows: list[dict[str, Any]],
    *,
    known_request_ids: set[str],
    started_after: datetime | None = None,
    node: str | None = None,
) -> list[str]:
    """No PENDING or RUNNING workflow the drill could have caused may exist
    beyond the known pair.

    The store lists workflows cluster-wide. A workflow another node's incident
    opened, or one that predates this case, is not evidence about this drill,
    so the read is narrowed to workflows created at or after ``started_after``
    and, when ``node`` is given, to those that name the node. A workflow whose
    ``created_at`` cannot be read is kept: unknown is not "before".
    """

    def relevant(item: dict[str, Any]) -> bool:
        if started_after is not None:
            created = parse_time(item.get("created_at"))
            if created is not None and created < started_after:
                return False
        if node is not None and node not in (item.get("node_ids") or []):
            return False
        return True

    executable = [
        item
        for item in workflows
        if item.get("status") in {"PENDING", "RUNNING"}
        and item.get("request_id") not in known_request_ids
        and relevant(item)
    ]
    if not executable:
        return []
    return [
        "the drill left executable workflows behind: "
        f"{sorted(str(item.get('request_id')) for item in executable)}"
    ]


# ---------------------------------------------------------------------------
# Data plane: the Node Agent ledger and the kernel journal
# ---------------------------------------------------------------------------
def new_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_command_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("command_id") not in baseline_command_ids]


def late_row_errors(
    rows: list[dict[str, Any]],
    *,
    t_cancel: datetime,
    baseline_command_ids: set[str],
    operations: tuple[str, ...] = POST_CANCEL_FORBIDDEN_OPERATIONS,
) -> list[str]:
    """Verdict 1 (strict): nothing node-mutating starts after T_cancel.

    ``started_at`` is written by ``mark_in_progress`` alone, so a row of one of
    these operations with no ``started_at`` cannot be placed against T_cancel;
    that is a failure, not a pass.
    """

    errors: list[str] = []
    for row in new_rows(rows, baseline_command_ids=baseline_command_ids):
        operation = str(row.get("operation"))
        if operation not in operations:
            continue
        started = parse_time(row.get("started_at"))
        if started is None:
            errors.append(
                f"ledger row {row.get('command_id')} ({operation}) has no "
                "started_at, so it cannot be placed against the cancellation; "
                "the drill cannot prove the node stopped"
            )
            continue
        if started > t_cancel:
            errors.append(
                f"the node started {operation} at {started.isoformat()}, after "
                f"the cancellation at {t_cancel.isoformat()}: "
                f"{row.get('command_id')}"
            )
    return errors


def straddling_rows(
    rows: list[dict[str, Any]],
    *,
    t_cancel: datetime,
    baseline_command_ids: set[str],
    operations: tuple[str, ...] = POST_CANCEL_FORBIDDEN_OPERATIONS,
) -> list[dict[str, Any]]:
    """Rows that started before T_cancel and finished after it."""

    result: list[dict[str, Any]] = []
    for row in new_rows(rows, baseline_command_ids=baseline_command_ids):
        if str(row.get("operation")) not in operations:
            continue
        started = parse_time(row.get("started_at"))
        completed = parse_time(row.get("completed_at"))
        if started is None or completed is None:
            continue
        if started < t_cancel < completed:
            result.append(row)
    return result


def straddling_row_errors(
    rows: list[dict[str, Any]],
    *,
    t_cancel: datetime,
    baseline_command_ids: set[str],
    kernel_journal: dict[str, Any],
    commands: list[dict[str, Any]],
    expect_forced_failure: bool = True,
) -> list[str]:
    """Verdict 2: at most one straddling row, and it must add up.

    A command the node had already leased can legitimately finish after the
    deadline cancelled it. Exactly one such row is tolerated, and it has to
    agree with the kernel: a SUCCEEDED reset means exactly one kernel-origin
    reset for the target BDF, a FAILED one means none. The control plane must
    not turn that late result into success -- the remote command ends FAILED
    with a cancellation status source, and a ``completed-after-cancellation``
    command must report the row's own state in
    ``result_details.post_cancellation_status``.

    In this drill the holder makes every verify and reset attempt fail, so a
    straddling row must be FAILED with an empty kernel journal.
    """

    errors: list[str] = []
    candidates = straddling_rows(
        rows,
        t_cancel=t_cancel,
        baseline_command_ids=baseline_command_ids,
    )
    if len(candidates) > 1:
        errors.append(
            "more than one ledger row straddles the cancellation: "
            f"{[row.get('command_id') for row in candidates]}"
        )
        return errors
    if not candidates:
        return errors
    row = candidates[0]
    operation = str(row.get("operation"))
    state = str(row.get("state"))
    if state not in {"SUCCEEDED", "FAILED"}:
        errors.append(
            f"the straddling ledger row {row.get('command_id')} is {state}; a "
            "row that never reached a terminal state cannot be judged"
        )
        return errors
    if operation == RESET_STEP:
        resets = int(kernel_journal.get("target_reset_count") or 0)
        if state == "SUCCEEDED" and resets != 1:
            errors.append(
                "the straddling RESET_GPU row claims SUCCEEDED but the kernel "
                f"journal shows {resets} resets for the target device"
            )
        if state == "FAILED" and resets != 0:
            errors.append(
                "the straddling RESET_GPU row claims FAILED but the kernel "
                f"journal shows {resets} resets for the target device"
            )
    if expect_forced_failure and state != "FAILED":
        errors.append(
            f"the straddling {operation} row is {state}; the device holder was "
            "meant to make every attempt fail"
        )
    # The ledger row and the remote command share the command id; that is the
    # only match that says "the same dispatch". Matching on the operation would
    # pick any VERIFY attempt of the workflow and judge the wrong command.
    row_id = row.get("command_id")
    if row_id:
        command = next(
            (item for item in commands if item.get("command_id") == row_id), None
        )
    else:
        command = next(
            (
                item
                for item in commands
                if (item.get("step") or {}).get("operation") == operation
            ),
            None,
        )
        if command is not None:
            errors.append(
                f"the straddling {operation} ledger row has no command_id; the "
                f"remote command {command.get('command_id')} was matched by "
                "operation only, which cannot prove it is the same dispatch"
            )
    if command is None:
        errors.append(
            f"no remote command matches the straddling ledger row "
            f"{row.get('command_id')}"
        )
        return errors
    if command.get("status") != "FAILED":
        errors.append(
            f"the remote command for the straddling {operation} row is "
            f"{command.get('status')}; a cancelled command stays FAILED"
        )
    source = command.get("status_source")
    if source not in CANCELLATION_STATUS_SOURCES:
        errors.append(
            f"the remote command for the straddling {operation} row carries "
            f"status_source={source}, not one of "
            f"{list(CANCELLATION_STATUS_SOURCES)}"
        )
    elif source == COMPLETED_AFTER_CANCELLATION:
        reported = (command.get("result_details") or {}).get("post_cancellation_status")
        if reported != state:
            errors.append(
                "result_details.post_cancellation_status is "
                f"{reported}, not the straddling row's own state {state}"
            )
    return errors


def compensation_row_errors(
    rows: list[dict[str, Any]],
    *,
    t_cancel: datetime,
    baseline_command_ids: set[str],
) -> list[str]:
    """Verdict 3: exactly one successful compensation after T_cancel."""

    candidates = []
    for row in new_rows(rows, baseline_command_ids=baseline_command_ids):
        if str(row.get("operation")) != COMPENSATION_STEP:
            continue
        completed = parse_time(row.get("completed_at"))
        if completed is not None and completed > t_cancel:
            candidates.append(row)
    errors: list[str] = []
    if len(candidates) != 1:
        errors.append(
            f"expected exactly one {COMPENSATION_STEP} ledger row after the "
            f"cancellation, found {len(candidates)}: "
            f"{[row.get('command_id') for row in candidates]}"
        )
        return errors
    row = candidates[0]
    if str(row.get("state")) != "SUCCEEDED":
        errors.append(
            f"the {COMPENSATION_STEP} ledger row after the cancellation is "
            f"{row.get('state')}; the one deadline-exempt compensation failed"
        )
    started = parse_time(row.get("started_at"))
    if started is None:
        errors.append(
            f"the {COMPENSATION_STEP} ledger row has no started_at, so the "
            "compensation cannot be placed after the cancellation"
        )
    return errors


def data_plane_errors(
    rows: list[dict[str, Any]],
    *,
    t_cancel: datetime,
    baseline_command_ids: set[str],
    kernel_journal: dict[str, Any],
    commands: list[dict[str, Any]],
    expect_forced_failure: bool = True,
) -> list[str]:
    """The three refined data-plane verdicts, in order."""

    return [
        *late_row_errors(
            rows,
            t_cancel=t_cancel,
            baseline_command_ids=baseline_command_ids,
        ),
        *straddling_row_errors(
            rows,
            t_cancel=t_cancel,
            baseline_command_ids=baseline_command_ids,
            kernel_journal=kernel_journal,
            commands=commands,
            expect_forced_failure=expect_forced_failure,
        ),
        *compensation_row_errors(
            rows,
            t_cancel=t_cancel,
            baseline_command_ids=baseline_command_ids,
        ),
    ]


# ---------------------------------------------------------------------------
# Optional metric evidence
# ---------------------------------------------------------------------------
def counter_value(metrics: str, name: str) -> float | None:
    """The value of one Prometheus counter line, or ``None`` if absent."""

    for line in (metrics or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        head, _, tail = stripped.rpartition(" ")
        if head != name:
            continue
        try:
            return float(tail)
        except ValueError:
            return None
    return None


def counter_total(samples: list[str], name: str) -> float | None:
    """``name`` summed over every replica that reports it, or ``None``.

    The counters live in the worker process that failed the workflow, and a
    Deployment runs several. Summing is the only reading that survives not
    knowing which replica held the dispatch lease; the window opens before the
    drill precisely so no rollout resets them in the middle.
    """

    values = [counter_value(text, name) for text in samples]
    present = [item for item in values if item is not None]
    return sum(present) if present else None


def metric_evidence(before: list[str], after: list[str]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key, name in (
        ("lifetime_exceeded_total", LIFETIME_METRIC),
        ("merge_record_only_total", RECORD_ONLY_METRIC),
    ):
        first = counter_total(before, name)
        second = counter_total(after, name)
        evidence[key] = {
            "before": first,
            "after": second,
            "delta": (
                second - first if first is not None and second is not None else None
            ),
        }
    return evidence


def metric_errors(evidence: dict[str, Any]) -> list[str]:
    """Counters are optional evidence: judged only when both samples exist.

    A worker restart between the samples resets the counters, and the scrape
    that answers may not be the replica that failed the workflow, so a missing
    or negative delta is recorded rather than failed.
    """

    errors: list[str] = []
    for key, name in (
        ("lifetime_exceeded_total", LIFETIME_METRIC),
        ("merge_record_only_total", RECORD_ONLY_METRIC),
    ):
        sample = evidence.get(key) or {}
        delta = sample.get("delta")
        if delta is None:
            continue
        if delta < 0:
            continue
        if delta == 0:
            errors.append(
                f"{name} did not move across the drill; the counter was "
                f"readable and stayed at {sample.get('after')}"
            )
    return errors
