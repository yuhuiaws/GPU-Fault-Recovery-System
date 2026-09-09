"""Seeding and verdicts for GF-REGIONAL-PREEMPT-036, without any I/O of its own.

The case proves that the dispatcher closes a wedged workflow on its own. Three
production incidents each held a release for hours because the record that
blocked it had no close path: a workflow BLOCKED at compile time with no
executable owner, a FAILED workflow whose remote command stayed WAITING for
thirteen hours, and a re-planned-away generation that kept being dispatched.
Their first fix was an operator command (``workflow-reconcile --mode ...``);
since 2026-09-08 every one of them is a Store predicate the dispatcher runs on
``WorkflowDispatcher.sweep_stuck_records`` every tick, so the case seeds one
store per shape and drives that method against it.

Everything here is pure in the sense that matters: the seeding functions take a
Store and only call its public API -- no SQL, no DDL -- the sweep is the product's
own dispatcher built from its public constructors, and the verdict functions take
read-backs of the Store and the JSON the release probes printed and return a list
of human-readable failures. The runner owns what cannot be unit tested:
provisioning a throwaway database, running the shipped probe sources in a
subprocess, and writing evidence.

Two conventions the verdicts encode, because the product decides them and prose
could drift from it:

* the retired generation takes two passes -- its remote command is cancelled
  first and the row is revoked only once the Store shows it settled -- so each
  seed says how many ``passes`` it needs, and a pass after that changes nothing;
* the audit lands where each sweep puts it. The compile-blocked close and the
  orphan cancel append an ``OPERATOR_RECONCILED`` event with actor
  ``dispatcher`` to the workflow they act on; the retired-generation revoke has
  no operator to name, writes its audit into ``preemption_reason`` and the
  incident's ``reasons``, and appends no operator event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from gpu_fault.compile_blocked import CLOSE_MARKER, DISPATCHER_ACTOR
from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
)
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.execution.executor import ProductionWorkflowExecutor
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.orphaned_commands import AUDIT_ACTION
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus

CASE_ID = "GF-REGIONAL-PREEMPT-036"
COMPILE_BLOCKED_SHAPE = "compile-blocked"
ORPHANED_COMMANDS_SHAPE = "orphaned-commands"
RETIRED_GENERATION_SHAPE = "retired-generation"
SHAPES = (COMPILE_BLOCKED_SHAPE, ORPHANED_COMMANDS_SHAPE, RETIRED_GENERATION_SHAPE)
CLUSTER_ID = "p036-cluster-a"
NODE_ID = "p036-node-a"
STEP_OWNER = "cluster-executor"
POLICY_VERSION = "610"
POLICY_SOURCE = "NVIDIA"
CANCELLED_STATUS_SOURCE = "workflow-timeout"
SWEEPER_EXECUTOR_ID = "p036-sweeper"
OPERATOR_RECONCILED_KIND = "OPERATOR_RECONCILED"
OPERATOR_EVENT_KINDS = frozenset(
    {OPERATOR_RECONCILED_KIND, "OPERATOR_RETIRED_GENERATION"}
)
OPEN_COMMAND_STATUSES = (
    RemoteCommandStatus.PENDING.value,
    RemoteCommandStatus.LEASED.value,
    RemoteCommandStatus.WAITING.value,
)


@dataclass(frozen=True)
class ShapeSeed:
    """What one shape's store holds, and what the sweep must do to it."""

    shape: str
    workflow_ids: tuple[str, ...]
    incident_ids: tuple[str, ...]
    closed_ids: tuple[str, ...]
    untouched_ids: tuple[str, ...]
    passes: int
    statuses_before: Mapping[str, str]
    statuses_after: Mapping[str, str]
    reason_substrings: Mapping[str, tuple[str, ...]]
    preempted_by: Mapping[str, str]
    incident_reason_substrings: Mapping[str, tuple[str, ...]]
    event_kind: str | None
    event_details: Mapping[str, Mapping[str, Any]]
    cancelled_command_ids: tuple[str, ...]
    untouched_command_ids: tuple[str, ...]
    cancel_reason_substring: str
    blockers_before: tuple[str, ...]
    blockers_after: tuple[str, ...]
    resolved_blocked: tuple[str, ...]
    compile_blocked_before: tuple[str, ...]
    compile_blocked_after: tuple[str, ...]
    held_after: tuple[str, ...] = field(default=())


def _step(
    operation: WorkflowOperation,
    *,
    node_ids: Sequence[str] = (NODE_ID,),
) -> WorkflowStepSpec:
    return WorkflowStepSpec(
        operation=operation,
        execution_owner=STEP_OWNER,
        node_ids=list(node_ids),
    )


def _hardware_steps(action: WorkflowOperation) -> list[WorkflowStepSpec]:
    """A containment / remediation / restore triple, the shape releases gate on.

    ``workflow_safety`` only counts a workflow that has a destructive step, so a
    record seeded without one would not be a release blocker and the before/after
    measurement would prove nothing.
    """

    return [
        _step(WorkflowOperation.MARK_UNSCHEDULABLE),
        _step(action),
        _step(WorkflowOperation.RESTORE_SCHEDULING),
    ]


def _incident(
    incident_id: str,
    *,
    state: IncidentState,
    workflow_request_id: str,
    fencing_token: int,
    created_at: datetime,
) -> FaultIncident:
    return FaultIncident(
        incident_id=incident_id,
        event_id=f"{incident_id}-event",
        event_type="XID",
        cluster_id=CLUSTER_ID,
        node_ids=[NODE_ID],
        policy_version=POLICY_VERSION,
        policy_source=POLICY_SOURCE,
        state=state,
        workflow_request_id=workflow_request_id,
        fencing_token=fencing_token,
        created_at=created_at,
        updated_at=created_at,
    )


def _command(
    command_id: str,
    *,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    step_index: int,
    status: RemoteCommandStatus,
    created_at: datetime,
) -> RemoteActionCommand:
    step = workflow.official_steps[step_index]
    return RemoteActionCommand(
        command_id=command_id,
        cluster_id=CLUSTER_ID,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=step_index,
        fencing_token=workflow.fencing_token,
        idempotency_key=f"{workflow.request_id}/{step_index}/{step.operation.value}",
        step=step,
        workflow=workflow,
        incident=incident,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )


def seed_compile_blocked(store: Any, *, now: datetime | None = None) -> ShapeSeed:
    """A workflow the compiler refused, next to one the restore reconcile owns.

    Closed: BLOCKED with ``no executable owner for efaDriverRemediation``, no
    step execution, no command, no source plan, incident settled -- the record
    observed live on 2026-09-04. The release preflight sets it aside as
    ``compile_blocked`` (so the release carrying the sweep can roll past it) and
    the sweep closes it on the first pass.

    Untouched twin: also BLOCKED, but it *was* plan-driven, so it carries a
    ``source_plan_id`` and belongs to ``workflow-reconcile``. Its incident already
    RECOVERED through a same-generation successor that completed
    ``RESTORE_SCHEDULING``, which is why the preflight reports it as
    ``resolved_blocked``: the twin must survive the sweep byte for byte.
    """

    stamp = now or datetime.now(timezone.utc)
    earlier = stamp - timedelta(hours=6)
    blocked_id = "p036cb-blocked"
    twin_id = "p036cb-restore-twin"
    successor_id = "p036cb-restored"
    blocked_incident = _incident(
        "p036cb-incident-efa",
        state=IncidentState.ESCALATED,
        workflow_request_id=blocked_id,
        fencing_token=3,
        created_at=earlier,
    )
    blocked_reason = "no executable owner for efaDriverRemediation"
    blocked = WorkflowRequest(
        request_id=blocked_id,
        incident_id=blocked_incident.incident_id,
        status=WorkflowStatus.BLOCKED,
        official_action="REMEDIATE_EFA_DRIVER",
        fencing_token=3,
        official_steps=_hardware_steps(WorkflowOperation.REMEDIATE_EFA_DRIVER),
        blocked_reasons=[blocked_reason],
        created_at=earlier,
        updated_at=earlier,
    )
    twin_incident = _incident(
        "p036cb-incident-restore",
        state=IncidentState.RECOVERED,
        workflow_request_id=successor_id,
        fencing_token=4,
        created_at=earlier,
    )
    twin = WorkflowRequest(
        request_id=twin_id,
        incident_id=twin_incident.incident_id,
        status=WorkflowStatus.BLOCKED,
        official_action="RESET_GPU",
        # Same generation as its successor: an older generation under an
        # incident that moved on would be a retired generation, which the
        # dispatcher revokes on its own -- a different shape from this one.
        fencing_token=4,
        source_plan_id="p036cb-plan-1",
        official_steps=_hardware_steps(WorkflowOperation.RESET_GPU),
        blocked_reasons=["safety settled by a later workflow"],
        created_at=earlier,
        updated_at=earlier,
    )
    successor = WorkflowRequest(
        request_id=successor_id,
        incident_id=twin_incident.incident_id,
        status=WorkflowStatus.SUCCEEDED,
        official_action="RESET_GPU",
        fencing_token=4,
        official_steps=_hardware_steps(WorkflowOperation.RESET_GPU),
        completed_operations=[
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
        completed_step_indexes=[0, 1, 2],
        created_at=earlier,
        updated_at=earlier,
    )
    store.save_incident_and_workflow(blocked_incident, blocked)
    store.save_incident_and_workflow(twin_incident, successor)
    store.save_workflow(twin)
    return ShapeSeed(
        shape=COMPILE_BLOCKED_SHAPE,
        workflow_ids=(blocked_id, twin_id, successor_id),
        incident_ids=(blocked_incident.incident_id, twin_incident.incident_id),
        closed_ids=(blocked_id,),
        untouched_ids=(twin_id, successor_id),
        passes=1,
        statuses_before={
            blocked_id: WorkflowStatus.BLOCKED.value,
            twin_id: WorkflowStatus.BLOCKED.value,
            successor_id: WorkflowStatus.SUCCEEDED.value,
        },
        statuses_after={
            blocked_id: WorkflowStatus.SUPERSEDED.value,
            twin_id: WorkflowStatus.BLOCKED.value,
            successor_id: WorkflowStatus.SUCCEEDED.value,
        },
        reason_substrings={
            blocked_id: (
                f"{DISPATCHER_ACTOR} reconciliation",
                CLOSE_MARKER,
                blocked_reason,
            ),
        },
        preempted_by={},
        incident_reason_substrings={},
        event_kind=OPERATOR_RECONCILED_KIND,
        event_details={
            blocked_id: {
                "terminalization": CLOSE_MARKER,
                "blocked_reasons": [blocked_reason],
            }
        },
        cancelled_command_ids=(),
        untouched_command_ids=(),
        cancel_reason_substring="",
        blockers_before=(),
        blockers_after=(),
        resolved_blocked=(twin_id,),
        compile_blocked_before=(blocked_id,),
        compile_blocked_after=(),
    )


def seed_orphaned_commands(store: Any, *, now: datetime | None = None) -> ShapeSeed:
    """A command a FAILED workflow left WAITING, next to a live one that must not move.

    Closed: the 2026-09-05 shape -- ``CHECK_MECHANICALS`` still ``WAITING``
    thirteen hours after its workflow FAILED. The release upgrade refuses to
    start while any command is open, so the orphan blocked the release that
    stops orphans from forming; the sweep cancels it on the first pass and
    leaves the FAILED row as it was, with the cancel on its audit trail.

    Untouched twin: a RUNNING workflow with a ``WAITING`` ``RESET_GPU`` command.
    That command is live work only the executor holding its lease may settle, so
    it must come out of the sweep byte for byte unchanged, however old.
    """

    stamp = now or datetime.now(timezone.utc)
    earlier = stamp - timedelta(hours=13)
    failed_id = "p036oc-failed"
    running_id = "p036oc-running"
    orphan_command = "p036oc-command-orphan"
    live_command = "p036oc-command-live"
    failed_incident = _incident(
        "p036oc-incident-failed",
        state=IncidentState.ESCALATED,
        workflow_request_id=failed_id,
        fencing_token=2,
        created_at=earlier,
    )
    failed = WorkflowRequest(
        request_id=failed_id,
        incident_id=failed_incident.incident_id,
        status=WorkflowStatus.FAILED,
        official_action="RESTART_NODE",
        fencing_token=2,
        official_steps=[
            _step(WorkflowOperation.MARK_UNSCHEDULABLE),
            _step(WorkflowOperation.CHECK_MECHANICALS),
        ],
        terminal_failure_reason="step 1 exceeded its waiting cap",
        created_at=earlier,
        updated_at=earlier,
    )
    running_incident = _incident(
        "p036oc-incident-running",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=running_id,
        fencing_token=7,
        created_at=earlier,
    )
    running = WorkflowRequest(
        request_id=running_id,
        incident_id=running_incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_action="RESET_GPU",
        fencing_token=7,
        official_steps=_hardware_steps(WorkflowOperation.RESET_GPU),
        execution_owner_id="p036-executor-b",
        execution_lease_expires_at=stamp + timedelta(minutes=2),
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
        completed_step_indexes=[0],
        step_executions=[
            WorkflowStepExecution(
                step_index=0,
                operation=WorkflowOperation.MARK_UNSCHEDULABLE,
                status=WorkflowStepStatus.SUCCEEDED,
                started_at=earlier,
                updated_at=earlier,
            ),
            WorkflowStepExecution(
                step_index=1,
                operation=WorkflowOperation.RESET_GPU,
                status=WorkflowStepStatus.WAITING,
                adapter_operation_id=f"remote/{live_command}",
                started_at=earlier,
                updated_at=earlier,
            ),
        ],
        created_at=earlier,
        updated_at=earlier,
    )
    store.save_incident_and_workflow(failed_incident, failed)
    store.save_incident_and_workflow(running_incident, running)
    store.ensure_remote_command(
        _command(
            orphan_command,
            workflow=failed,
            incident=failed_incident,
            step_index=1,
            status=RemoteCommandStatus.WAITING,
            created_at=earlier,
        )
    )
    store.ensure_remote_command(
        _command(
            live_command,
            workflow=running,
            incident=running_incident,
            step_index=1,
            status=RemoteCommandStatus.WAITING,
            created_at=earlier,
        )
    )
    return ShapeSeed(
        shape=ORPHANED_COMMANDS_SHAPE,
        workflow_ids=(failed_id, running_id),
        incident_ids=(failed_incident.incident_id, running_incident.incident_id),
        closed_ids=(failed_id,),
        untouched_ids=(running_id,),
        passes=1,
        statuses_before={
            failed_id: WorkflowStatus.FAILED.value,
            running_id: WorkflowStatus.RUNNING.value,
        },
        statuses_after={
            failed_id: WorkflowStatus.FAILED.value,
            running_id: WorkflowStatus.RUNNING.value,
        },
        reason_substrings={},
        preempted_by={},
        incident_reason_substrings={},
        event_kind=OPERATOR_RECONCILED_KIND,
        event_details={
            failed_id: {
                "action": AUDIT_ACTION,
                "command_ids": [orphan_command],
                "cancelled_remote_commands": {
                    "cancelled": 1,
                    "cancellation_requested": 0,
                },
            }
        },
        cancelled_command_ids=(orphan_command,),
        untouched_command_ids=(live_command,),
        cancel_reason_substring=f"{DISPATCHER_ACTOR} reconciliation",
        blockers_before=(running_id,),
        blockers_after=(running_id,),
        resolved_blocked=(),
        compile_blocked_before=(),
        compile_blocked_after=(),
    )


def seed_retired_generation(store: Any, *, now: datetime | None = None) -> ShapeSeed:
    """A generation the incident re-planned away from, next to one that changed a node.

    Closed: the 2026-09-04 shape -- an older generation still RUNNING with an
    execution owner, a renewing lease and a remediation budget claim, being
    dispatched on a loop while the successor starves. It holds one ``WAITING``
    remote command, so the sweep takes two passes: the first cancels the
    command, the second revokes the row once the Store shows it settled. That
    hinge is the reason the case seeds a command here at all.

    Untouched twin: an even older generation that already completed
    ``RESET_GPU``. Effect a later workflow would have to compensate for is the
    one blocker no sweep may clear, so it must stay held for a human
    (``held_after``) and must still be a release blocker afterwards.
    """

    stamp = now or datetime.now(timezone.utc)
    earlier = stamp - timedelta(hours=3)
    current_id = "p036rg-current"
    retired_id = "p036rg-retired"
    mutated_id = "p036rg-mutated"
    command_id = "p036rg-command-retired"
    incident = _incident(
        "p036rg-incident",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=current_id,
        fencing_token=5,
        created_at=earlier,
    )
    current = WorkflowRequest(
        request_id=current_id,
        incident_id=incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_action="RESET_GPU",
        fencing_token=5,
        official_steps=_hardware_steps(WorkflowOperation.RESET_GPU),
        created_at=earlier,
        updated_at=earlier,
    )
    retired = WorkflowRequest(
        request_id=retired_id,
        incident_id=incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_action="RESET_GPU",
        fencing_token=2,
        official_steps=_hardware_steps(WorkflowOperation.RESET_GPU),
        execution_owner_id="p036-executor-a",
        execution_lease_expires_at=stamp + timedelta(minutes=2),
        remediation_budget_claims=[f"{CLUSTER_ID}/RESET_GPU"],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
        completed_step_indexes=[0],
        step_executions=[
            WorkflowStepExecution(
                step_index=0,
                operation=WorkflowOperation.MARK_UNSCHEDULABLE,
                status=WorkflowStepStatus.SUCCEEDED,
                started_at=earlier,
                updated_at=earlier,
            ),
            WorkflowStepExecution(
                step_index=1,
                operation=WorkflowOperation.RESET_GPU,
                status=WorkflowStepStatus.WAITING,
                adapter_operation_id=f"remote/{command_id}",
                started_at=earlier,
                updated_at=earlier,
            ),
        ],
        created_at=earlier,
        updated_at=earlier,
    )
    mutated = WorkflowRequest(
        request_id=mutated_id,
        incident_id=incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_action="RESET_GPU",
        fencing_token=1,
        official_steps=_hardware_steps(WorkflowOperation.RESET_GPU),
        completed_operations=[
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.RESET_GPU,
        ],
        completed_step_indexes=[0, 1],
        created_at=earlier,
        updated_at=earlier,
    )
    store.save_incident_and_workflow(incident, current)
    store.save_workflow(retired)
    store.save_workflow(mutated)
    store.ensure_remote_command(
        _command(
            command_id,
            workflow=retired,
            incident=incident,
            step_index=1,
            status=RemoteCommandStatus.WAITING,
            created_at=earlier,
        )
    )
    return ShapeSeed(
        shape=RETIRED_GENERATION_SHAPE,
        workflow_ids=(retired_id, mutated_id, current_id),
        incident_ids=(incident.incident_id,),
        closed_ids=(retired_id,),
        untouched_ids=(mutated_id, current_id),
        passes=2,
        statuses_before={
            retired_id: WorkflowStatus.RUNNING.value,
            mutated_id: WorkflowStatus.RUNNING.value,
            current_id: WorkflowStatus.RUNNING.value,
        },
        statuses_after={
            retired_id: WorkflowStatus.SUPERSEDED.value,
            mutated_id: WorkflowStatus.RUNNING.value,
            current_id: WorkflowStatus.RUNNING.value,
        },
        reason_substrings={
            retired_id: (
                f"revoked retired generation {retired_id}",
                f"in favour of {current_id}",
                "advanced to generation 5",
            ),
        },
        preempted_by={retired_id: current_id},
        incident_reason_substrings={
            incident.incident_id: (f"revoked retired generation {retired_id}",),
        },
        event_kind=None,
        event_details={},
        cancelled_command_ids=(command_id,),
        untouched_command_ids=(),
        cancel_reason_substring=f"revoked retired generation {retired_id}",
        blockers_before=(current_id, mutated_id, retired_id),
        blockers_after=(current_id, mutated_id),
        resolved_blocked=(),
        compile_blocked_before=(),
        compile_blocked_after=(),
        held_after=(mutated_id,),
    )


SEEDERS = {
    COMPILE_BLOCKED_SHAPE: seed_compile_blocked,
    ORPHANED_COMMANDS_SHAPE: seed_orphaned_commands,
    RETIRED_GENERATION_SHAPE: seed_retired_generation,
}


def seed_shape(shape: str, store: Any, *, now: datetime | None = None) -> ShapeSeed:
    try:
        seeder = SEEDERS[shape]
    except KeyError:
        raise ValueError(f"unsupported stuck-workflow shape: {shape}") from None
    return seeder(store, now=now)


# --------------------------------------------------------------------------- #
# The product's own sweep, built from its public constructors
# --------------------------------------------------------------------------- #
def build_sweeper(store: Any) -> WorkflowDispatcher:
    """A dispatcher with no adapters: it can sweep, and it cannot execute a step.

    ``sweep_stuck_records`` is what the production dispatch loop runs before its
    scan; calling it directly, rather than ``run_once``, keeps the seeded live
    twins from being dispatched into an executor that has nothing to run them.
    """

    executor = ProductionWorkflowExecutor(
        store,
        [],
        ProductionExecutorConfig(
            enabled=True,
            executor_id=SWEEPER_EXECUTOR_ID,
            allowed_operations=frozenset(),
        ),
    )
    return WorkflowDispatcher(store, executor, WorkflowDispatcherConfig(enabled=True))


def sweep(store: Any, *, passes: int) -> list[list[str]]:
    """Run the sweep ``passes`` times; per pass, the retired generations still held."""

    sweeper = build_sweeper(store)
    return [
        sorted(sweeper.sweep_stuck_records(datetime.now(timezone.utc)))
        for _ in range(passes)
    ]


# --------------------------------------------------------------------------- #
# Read-backs
# --------------------------------------------------------------------------- #
def workflow_snapshot(store: Any, request_ids: Iterable[str]) -> dict[str, Any]:
    """The fields the verdicts read, per workflow, from a fresh Store read."""

    snapshot: dict[str, Any] = {}
    for request_id in request_ids:
        workflow = store.get_workflow(request_id)
        snapshot[request_id] = {
            "status": workflow.status.value,
            "preemption_reason": workflow.preemption_reason or "",
            "preempted_by_workflow_id": workflow.preempted_by_workflow_id,
            "execution_owner_id": workflow.execution_owner_id,
            "fencing_token": workflow.fencing_token,
            "events": [
                {
                    "kind": event.kind.value,
                    "actor": event.actor,
                    "status": event.status,
                    "reason": event.reason,
                    "details": dict(event.details),
                }
                for event in workflow.events
                if event.kind.value in OPERATOR_EVENT_KINDS
            ],
        }
    return snapshot


def incident_snapshot(store: Any, incident_ids: Iterable[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for incident_id in incident_ids:
        incident = store.get_incident(incident_id)
        snapshot[incident_id] = {
            "state": incident.state.value,
            "workflow_request_id": incident.workflow_request_id,
            "fencing_token": incident.fencing_token,
            "reasons": list(incident.reasons),
        }
    return snapshot


def command_snapshot(store: Any, request_ids: Iterable[str]) -> dict[str, Any]:
    """The fields the verdicts read, per remote command, from a fresh Store read."""

    snapshot: dict[str, Any] = {}
    for command in store.list_remote_commands(
        workflow_request_ids=sorted(set(request_ids))
    ):
        snapshot[command.command_id] = {
            "status": command.status.value,
            "status_source": command.status_source,
            "error": command.error or "",
            "lease_owner": command.lease_owner,
            "workflow_request_id": command.workflow_request_id,
        }
    return snapshot


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def record_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ShapeSeed,
) -> list[str]:
    """Every workflow ended where the shape promises; the twins did not move at all."""

    errors: list[str] = []
    for request_id, status in seed.statuses_after.items():
        observed = (after.get(request_id) or {}).get("status")
        if observed != status:
            errors.append(f"{request_id}: status is {observed!r}, expected {status!r}")
    for request_id, substrings in seed.reason_substrings.items():
        reason = str((after.get(request_id) or {}).get("preemption_reason") or "")
        for substring in substrings:
            if substring not in reason:
                errors.append(
                    f"{request_id}: preemption_reason does not carry {substring!r}"
                )
    for request_id, successor in seed.preempted_by.items():
        observed = (after.get(request_id) or {}).get("preempted_by_workflow_id")
        if observed != successor:
            errors.append(
                f"{request_id}: preempted_by_workflow_id is {observed!r}, "
                f"expected {successor!r}"
            )
        if (after.get(request_id) or {}).get("execution_owner_id") is not None:
            errors.append(f"{request_id}: revoked record still holds an owner")
    for request_id in seed.untouched_ids:
        if before.get(request_id) != after.get(request_id):
            errors.append(
                f"{request_id}: an untouched record changed from "
                f"{before.get(request_id)!r} to {after.get(request_id)!r}"
            )
    return errors


def incident_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ShapeSeed,
) -> list[str]:
    """The retired-generation audit lands on the incident; every other incident is untouched."""

    errors: list[str] = []
    for incident_id in seed.incident_ids:
        substrings = seed.incident_reason_substrings.get(incident_id, ())
        if not substrings:
            if before.get(incident_id) != after.get(incident_id):
                errors.append(
                    f"{incident_id}: incident changed from {before.get(incident_id)!r} "
                    f"to {after.get(incident_id)!r}"
                )
            continue
        reasons = " | ".join((after.get(incident_id) or {}).get("reasons") or [])
        for substring in substrings:
            if substring not in reasons:
                errors.append(
                    f"{incident_id}: incident reasons do not carry {substring!r}"
                )
        for key in ("state", "workflow_request_id", "fencing_token"):
            if (before.get(incident_id) or {}).get(key) != (
                after.get(incident_id) or {}
            ).get(key):
                errors.append(f"{incident_id}: the sweep moved incident {key}")
    return errors


def event_errors(snapshot: Mapping[str, Any], seed: ShapeSeed) -> list[str]:
    """Every record the sweep wrote carries exactly one dispatcher event; no other does.

    The event names the actor (``dispatcher``), the status the write moved the
    record from and to, and the shape's own details (what was closed, which
    commands were cancelled). The retired-generation revoke declares no event
    kind: its audit is judged by ``record_errors`` and ``incident_errors``.
    """

    errors: list[str] = []
    if seed.event_kind is not None:
        for request_id in seed.closed_ids:
            events = [
                item
                for item in (snapshot.get(request_id) or {}).get("events") or []
                if item.get("kind") == seed.event_kind
            ]
            if len(events) != 1:
                errors.append(
                    f"{request_id}: {len(events)} {seed.event_kind} event(s), "
                    "expected exactly 1"
                )
                continue
            event = events[0]
            details = event.get("details") or {}
            if event.get("actor") != DISPATCHER_ACTOR:
                errors.append(
                    f"{request_id}: event actor is {event.get('actor')!r}, "
                    f"expected {DISPATCHER_ACTOR!r}"
                )
            if details.get("previous_status") != seed.statuses_before[request_id]:
                errors.append(
                    f"{request_id}: event previous_status is "
                    f"{details.get('previous_status')!r}, expected "
                    f"{seed.statuses_before[request_id]!r}"
                )
            expected_after = seed.statuses_after[request_id]
            if details.get("new_status") != expected_after or event.get("status") != (
                expected_after
            ):
                errors.append(
                    f"{request_id}: event new_status is {details.get('new_status')!r}, "
                    f"expected {expected_after!r}"
                )
            for key, value in seed.event_details.get(request_id, {}).items():
                if details.get(key) != value:
                    errors.append(
                        f"{request_id}: event {key} is {details.get(key)!r}, "
                        f"expected {value!r}"
                    )
    for request_id in seed.untouched_ids:
        stray = (snapshot.get(request_id) or {}).get("events") or []
        if stray:
            errors.append(
                f"{request_id}: an untouched record carries {len(stray)} "
                "operator event(s)"
            )
    return errors


def command_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ShapeSeed,
) -> list[str]:
    """The orphans were failed with the expected source and the live one untouched."""

    errors: list[str] = []
    for command_id in seed.cancelled_command_ids:
        observed = after.get(command_id)
        if observed is None:
            errors.append(f"{command_id}: command is missing after the sweep")
            continue
        if observed.get("status") != RemoteCommandStatus.FAILED.value:
            errors.append(
                f"{command_id}: status is {observed.get('status')!r}, expected FAILED"
            )
        if observed.get("status_source") != CANCELLED_STATUS_SOURCE:
            errors.append(
                f"{command_id}: status_source is {observed.get('status_source')!r}, "
                f"expected {CANCELLED_STATUS_SOURCE!r}"
            )
        if seed.cancel_reason_substring not in str(observed.get("error") or ""):
            errors.append(
                f"{command_id}: error does not carry {seed.cancel_reason_substring!r}"
            )
        if observed.get("lease_owner") is not None:
            errors.append(f"{command_id}: cancelled command still holds a lease")
    for command_id in seed.untouched_command_ids:
        if before.get(command_id) != after.get(command_id):
            errors.append(
                f"{command_id}: live command changed from {before.get(command_id)!r} "
                f"to {after.get(command_id)!r}"
            )
    return errors


def held_errors(held_per_pass: Sequence[Sequence[str]], seed: ShapeSeed) -> list[str]:
    """What the last pass still withheld from dispatch is exactly the operator's record."""

    if len(held_per_pass) != seed.passes:
        return [f"sweep ran {len(held_per_pass)} pass(es), expected {seed.passes}"]
    held = sorted(str(value) for value in held_per_pass[-1])
    if held != sorted(seed.held_after):
        return [f"the last pass still holds {held}, expected {sorted(seed.held_after)}"]
    return []


def safety_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ShapeSeed,
) -> list[str]:
    """The release preflight stopped naming the records the sweep closed, and no other.

    ``blockers_after`` is not always empty and must not be asserted as empty. A
    successor generation that is genuinely running, and a record that already
    changed a node and therefore needs a human, are real blockers both before and
    after -- reporting them as cleared would be the bug this case exists to
    prevent. The compile-time record is never a blocker at all: the preflight
    sets it aside as ``compile_blocked`` until the sweep closes it.
    """

    errors: list[str] = []
    for name, values, expected in (
        ("blockers before", before.get("blockers"), seed.blockers_before),
        ("blockers after", after.get("blockers"), seed.blockers_after),
        (
            "compile_blocked before",
            before.get("compile_blocked"),
            seed.compile_blocked_before,
        ),
        (
            "compile_blocked after",
            after.get("compile_blocked"),
            seed.compile_blocked_after,
        ),
        (
            "resolved_blocked before",
            before.get("resolved_blocked"),
            seed.resolved_blocked,
        ),
        (
            "resolved_blocked after",
            after.get("resolved_blocked"),
            seed.resolved_blocked,
        ),
    ):
        observed = sorted(str(value) for value in values or [])
        if observed != sorted(expected):
            errors.append(f"{name} are {observed}, expected {sorted(expected)}")
    for name, values in (("before", before), ("after", after)):
        for key, count_key in (
            ("blockers", "blocker_count"),
            ("compile_blocked", "compile_blocked_count"),
            ("resolved_blocked", "resolved_blocked_count"),
        ):
            count = values.get(count_key)
            listed = len(values.get(key) or [])
            if count != listed:
                errors.append(f"{count_key} {name} is {count!r} for {listed} listed")
    cleared = set(seed.blockers_before) - set(seed.blockers_after)
    cleared |= set(seed.compile_blocked_before) - set(seed.compile_blocked_after)
    if cleared and not cleared <= set(seed.closed_ids):
        errors.append(
            "the seed expects records the sweep does not close to be cleared: "
            f"{sorted(cleared - set(seed.closed_ids))}"
        )
    return errors


def _open_commands(stats: Mapping[str, Any]) -> int:
    """The counter the release upgrade refuses to start against, from ``by_status``."""

    counters = stats.get("by_status")
    if not isinstance(counters, Mapping):
        raise ValueError("remote command stats carry no by_status counters")
    return sum(int(counters.get(name, 0)) for name in OPEN_COMMAND_STATUSES)


def open_command_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ShapeSeed,
) -> list[str]:
    """The Store's open-command counters, the release gate, dropped by the orphans."""

    expected = len(seed.cancelled_command_ids)
    dropped = _open_commands(before) - _open_commands(after)
    if dropped != expected:
        return [f"open remote commands dropped by {dropped}, expected {expected}"]
    return []


def rerun_errors(
    settled: Mapping[str, Mapping[str, Any]],
    rerun: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """One more pass after the shape settled changes nothing, anywhere.

    ``settled`` and ``rerun`` each map ``workflows`` / ``incidents`` /
    ``commands`` to the snapshots taken after the last required pass and after
    the extra one. Any difference -- a status, an appended event, an incident
    reason -- means the sweep is not idempotent on that shape.
    """

    errors: list[str] = []
    for kind in ("workflows", "incidents", "commands"):
        earlier = settled.get(kind) or {}
        later = rerun.get(kind) or {}
        for key in sorted(set(earlier) | set(later)):
            if earlier.get(key) != later.get(key):
                errors.append(
                    f"{kind}: the rerun changed {key} from {earlier.get(key)!r} "
                    f"to {later.get(key)!r}"
                )
    return errors


def case_verdict(stages: Mapping[str, Sequence[str]]) -> str:
    return "PASS" if not any(errors for errors in stages.values()) else "FAIL"


def all_errors(stages: Mapping[str, Sequence[str]]) -> list[str]:
    return [f"{stage}: {error}" for stage in sorted(stages) for error in stages[stage]]
