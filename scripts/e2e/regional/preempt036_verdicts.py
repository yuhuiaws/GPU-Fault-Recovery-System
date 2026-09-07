"""Seeding and verdicts for GF-REGIONAL-PREEMPT-036, without any I/O of its own.

The case proves the operator path that closes a wedged workflow:
``gpu-fault-admin workflow-reconcile --mode {compile-blocked,orphaned-commands,
retired-generation}`` in ``--plan`` then ``--apply``. Three production incidents
each held a release for hours because no such path existed, and each one has a
different shape, so the case seeds one store per mode.

Everything here is pure. The seeding functions take a Store and only call its
public API -- no SQL, no DDL -- and the verdict functions take the JSON the real
plan/apply entry points printed plus a read-back of the Store, and return a list
of human-readable failures. The runner owns the parts that cannot be unit
tested: provisioning a throwaway database, shipping the module source into a
subprocess, and writing evidence.

Two conventions the verdicts encode, because the reconcile code decides them and
prose could drift from it:

* ``actionable`` is ``eligible or cancellable``, not ``eligible``. A retired
  generation with an open remote command is planned as *cancellable*: the apply
  is allowed to cancel the command itself and then revoke. Reading only
  ``eligible`` would call that record refused when the code applies it.
* rerunning after a successful apply is not one contract but two.
  ``compile-blocked`` and ``retired-generation`` re-plan the record as
  ``already_closed`` / ``already_revoked`` and the apply is a no-op;
  ``orphaned-commands`` writes no workflow, so the re-plan finds no open command
  and the apply *refuses*. Both are correct; a single "rerun is idempotent"
  assertion would be wrong for one of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

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
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus

CASE_ID = "GF-REGIONAL-PREEMPT-036"
COMPILE_BLOCKED_MODE = "compile-blocked"
ORPHANED_COMMANDS_MODE = "orphaned-commands"
RETIRED_GENERATION_MODE = "retired-generation"
MODES = (COMPILE_BLOCKED_MODE, ORPHANED_COMMANDS_MODE, RETIRED_GENERATION_MODE)
REFERENCE = "CHG-PREEMPT-036"
CLUSTER_ID = "p035-cluster-a"
NODE_ID = "p035-node-a"
STEP_OWNER = "cluster-executor"
POLICY_VERSION = "610"
POLICY_SOURCE = "NVIDIA"
CANCELLED_STATUS_SOURCE = "workflow-timeout"
RERUN_NO_OP = "no-op"
RERUN_REFUSED = "refused"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Every mode refuses an apply whose rebuilt digest differs from the approval,
# but each spells it its own way, so the tamper verdict is per mode.
DRIFT_REFUSALS = {
    COMPILE_BLOCKED_MODE: "compile-blocked reconcile plan changed before apply",
    ORPHANED_COMMANDS_MODE: "orphaned-commands reconcile plan changed before apply",
    RETIRED_GENERATION_MODE: ("retired generation reconcile plan changed before apply"),
}
INELIGIBLE_REFUSALS = {
    COMPILE_BLOCKED_MODE: (
        "compile-blocked reconcile plan contains ineligible records"
    ),
    ORPHANED_COMMANDS_MODE: (
        "orphaned-commands reconcile plan contains ineligible records"
    ),
    RETIRED_GENERATION_MODE: (
        "retired generation reconcile plan contains ineligible records"
    ),
}


@dataclass(frozen=True)
class ModeSeed:
    """What one mode's store holds, and what the mode must do to it."""

    mode: str
    plan_mode: str
    apply_mode: str
    workflow_ids: tuple[str, ...]
    actionable_ids: tuple[str, ...]
    refused_ids: tuple[str, ...]
    plan_flags: Mapping[str, Mapping[str, bool]]
    plan_reasons: Mapping[str, tuple[str, ...]]
    refusal_substrings: Mapping[str, str]
    approved_digest_field: str
    settled_digest_differs: bool
    statuses_after: Mapping[str, str]
    reason_substrings: Mapping[str, tuple[str, ...]]
    preempted_by: Mapping[str, str]
    cancelled_command_ids: tuple[str, ...]
    untouched_command_ids: tuple[str, ...]
    blockers_before: tuple[str, ...]
    blockers_after: tuple[str, ...]
    resolved_blocked: tuple[str, ...]
    rerun_contract: str
    rerun_refusal: str


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
        lease_owner=None,
        lease_expires_at=None,
        created_at=created_at,
        updated_at=created_at,
    )


def seed_compile_blocked(store: Any, *, now: datetime | None = None) -> ModeSeed:
    """A workflow the compiler refused, next to one that belongs to ``--mode restore``.

    Eligible: BLOCKED with ``no executable owner for efaDriverRemediation``, no
    step execution, no command, no source plan, incident settled -- the record
    observed live on 2026-09-04, which ``workflow_safety`` counts as active
    destructive work and which therefore blocks the release that would add the
    missing owner.

    Refused twin: also BLOCKED, but it *was* plan-driven, so it carries a
    ``source_plan_id`` and belongs to ``--mode restore``. Its incident already
    RECOVERED through a successor that completed ``RESTORE_SCHEDULING``, which is
    why ``workflow_safety`` reports it as ``resolved_blocked`` rather than a
    blocker: the twin must survive this mode untouched *and* leave the blocker
    list empty once the eligible record is closed.
    """

    stamp = now or datetime.now(timezone.utc)
    earlier = stamp - timedelta(hours=6)
    blocked_id = "p035cb-blocked"
    twin_id = "p035cb-restore-twin"
    successor_id = "p035cb-restored"
    blocked_incident = _incident(
        "p035cb-incident-efa",
        state=IncidentState.ESCALATED,
        workflow_request_id=blocked_id,
        fencing_token=3,
        created_at=earlier,
    )
    blocked = WorkflowRequest(
        request_id=blocked_id,
        incident_id=blocked_incident.incident_id,
        status=WorkflowStatus.BLOCKED,
        official_action="REMEDIATE_EFA_DRIVER",
        fencing_token=3,
        official_steps=_hardware_steps(WorkflowOperation.REMEDIATE_EFA_DRIVER),
        blocked_reasons=["no executable owner for efaDriverRemediation"],
        created_at=earlier,
        updated_at=earlier,
    )
    twin_incident = _incident(
        "p035cb-incident-restore",
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
        fencing_token=2,
        source_plan_id="p035cb-plan-1",
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
    return ModeSeed(
        mode=COMPILE_BLOCKED_MODE,
        plan_mode="compile-blocked-plan",
        apply_mode="compile-blocked-apply",
        workflow_ids=(blocked_id, twin_id),
        actionable_ids=(blocked_id,),
        refused_ids=(twin_id,),
        plan_flags={
            blocked_id: {
                "eligible": True,
                "already_closed": False,
            },
            twin_id: {
                "eligible": False,
                "already_closed": False,
            },
        },
        plan_reasons={
            blocked_id: (),
            twin_id: ("workflow has a source recovery plan; use --mode restore",),
        },
        refusal_substrings={
            twin_id: "workflow has a source recovery plan; use --mode restore",
        },
        approved_digest_field="settled_plan_sha256",
        settled_digest_differs=False,
        statuses_after={
            blocked_id: WorkflowStatus.SUPERSEDED.value,
            twin_id: WorkflowStatus.BLOCKED.value,
            successor_id: WorkflowStatus.SUCCEEDED.value,
        },
        reason_substrings={
            blocked_id: (
                f"operator reconciliation {REFERENCE}",
                "closed compile-time BLOCKED workflow",
                "no executable owner for efaDriverRemediation",
            ),
        },
        preempted_by={},
        cancelled_command_ids=(),
        untouched_command_ids=(),
        blockers_before=(blocked_id,),
        blockers_after=(),
        resolved_blocked=(twin_id,),
        rerun_contract=RERUN_NO_OP,
        rerun_refusal="",
    )


def seed_orphaned_commands(store: Any, *, now: datetime | None = None) -> ModeSeed:
    """A command a FAILED workflow left WAITING, next to a live one that must not move.

    Eligible: the 2026-09-05 shape -- ``CHECK_MECHANICALS`` still ``WAITING``
    thirteen hours after its workflow FAILED. The release upgrade refuses to
    start while any command is open, so the orphan blocks the release that stops
    orphans from forming.

    Refused twin: a RUNNING workflow with a ``WAITING`` ``RESET_GPU`` command.
    That command is live work only the executor holding its lease may settle, so
    the plan must name it refused and its command must come out of the apply
    byte-for-byte unchanged.
    """

    stamp = now or datetime.now(timezone.utc)
    earlier = stamp - timedelta(hours=13)
    failed_id = "p035oc-failed"
    running_id = "p035oc-running"
    orphan_command = "p035oc-command-orphan"
    live_command = "p035oc-command-live"
    failed_incident = _incident(
        "p035oc-incident-failed",
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
        "p035oc-incident-running",
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
        execution_owner_id="p035-executor-b",
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
    return ModeSeed(
        mode=ORPHANED_COMMANDS_MODE,
        plan_mode="orphaned-commands-plan",
        apply_mode="orphaned-commands-apply",
        workflow_ids=(failed_id, running_id),
        actionable_ids=(failed_id,),
        refused_ids=(running_id,),
        plan_flags={
            failed_id: {"eligible": True},
            running_id: {"eligible": False},
        },
        plan_reasons={
            failed_id: (),
            running_id: (
                "workflow is RUNNING; its open commands are live work",
                "workflow still has an execution owner",
            ),
        },
        refusal_substrings={
            running_id: "workflow is RUNNING; its open commands are live work",
        },
        approved_digest_field="settled_plan_sha256",
        settled_digest_differs=False,
        statuses_after={
            failed_id: WorkflowStatus.FAILED.value,
            running_id: WorkflowStatus.RUNNING.value,
        },
        reason_substrings={},
        preempted_by={},
        cancelled_command_ids=(orphan_command,),
        untouched_command_ids=(live_command,),
        blockers_before=(running_id,),
        blockers_after=(running_id,),
        resolved_blocked=(),
        rerun_contract=RERUN_REFUSED,
        rerun_refusal="workflow has no open remote commands",
    )


def seed_retired_generation(store: Any, *, now: datetime | None = None) -> ModeSeed:
    """A generation the incident re-planned away from, next to one that changed a node.

    Eligible: the 2026-09-04 shape -- an older generation still RUNNING with an
    execution owner, a renewing lease and a remediation budget claim, being
    dispatched on a loop while the successor starves. It holds one ``WAITING``
    remote command, so the plan reports it ``cancellable`` rather than
    ``eligible``: the apply cancels the command and then revokes, which is the
    two-pass hinge and the reason the case seeds a command here at all.

    Refused twin: an even older generation that already completed ``RESET_GPU``.
    Effect a later workflow would have to compensate for is the one blocker no
    apply may clear, so it must reach a human and must still be a release blocker
    afterwards.
    """

    stamp = now or datetime.now(timezone.utc)
    earlier = stamp - timedelta(hours=3)
    current_id = "p035rg-current"
    retired_id = "p035rg-retired"
    mutated_id = "p035rg-mutated"
    command_id = "p035rg-command-retired"
    incident = _incident(
        "p035rg-incident",
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
        execution_owner_id="p035-executor-a",
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
    return ModeSeed(
        mode=RETIRED_GENERATION_MODE,
        plan_mode="retired-generation-plan",
        apply_mode="retired-generation-apply",
        workflow_ids=(retired_id, mutated_id),
        actionable_ids=(retired_id,),
        refused_ids=(mutated_id,),
        plan_flags={
            retired_id: {
                "eligible": False,
                "cancellable": True,
                "already_revoked": False,
            },
            mutated_id: {
                "eligible": False,
                "cancellable": False,
                "already_revoked": False,
            },
        },
        plan_reasons={
            retired_id: (f"workflow has open remote commands: {command_id}",),
            mutated_id: (
                "workflow already completed destructive operations: RESET_GPU",
            ),
        },
        refusal_substrings={
            mutated_id: "workflow already completed destructive operations: RESET_GPU",
        },
        approved_digest_field="plan_sha256",
        settled_digest_differs=True,
        statuses_after={
            retired_id: WorkflowStatus.SUPERSEDED.value,
            mutated_id: WorkflowStatus.RUNNING.value,
            current_id: WorkflowStatus.RUNNING.value,
        },
        reason_substrings={
            retired_id: (
                f"operator reconciliation {REFERENCE}",
                f"revoked retired generation {retired_id}",
                f"in favour of {current_id}",
            ),
        },
        preempted_by={retired_id: current_id},
        cancelled_command_ids=(command_id,),
        untouched_command_ids=(),
        blockers_before=(current_id, mutated_id, retired_id),
        blockers_after=(current_id, mutated_id),
        resolved_blocked=(),
        rerun_contract=RERUN_NO_OP,
        rerun_refusal="",
    )


SEEDERS = {
    COMPILE_BLOCKED_MODE: seed_compile_blocked,
    ORPHANED_COMMANDS_MODE: seed_orphaned_commands,
    RETIRED_GENERATION_MODE: seed_retired_generation,
}


def seed_mode(mode: str, store: Any, *, now: datetime | None = None) -> ModeSeed:
    try:
        seeder = SEEDERS[mode]
    except KeyError:
        raise ValueError(f"unsupported workflow-reconcile mode: {mode}") from None
    return seeder(store, now=now)


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
            # The attributed audit trail (ARCH-I1): who wrote, under which
            # approval, from which status. Judged by ``event_errors``.
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


# --------------------------------------------------------------------------- #
# ARCH-I1: the operator write is attributed
# --------------------------------------------------------------------------- #
OPERATOR_RECONCILED_KIND = "OPERATOR_RECONCILED"
OPERATOR_RETIRED_GENERATION_KIND = "OPERATOR_RETIRED_GENERATION"
OPERATOR_EVENT_KINDS = frozenset(
    {OPERATOR_RECONCILED_KIND, OPERATOR_RETIRED_GENERATION_KIND}
)
# Which event kind each mode appends to the record it acts on, and whether the
# record's status changes under it (orphaned-commands leaves the FAILED
# workflow as it was and only records the cancel).
EVENT_KIND_BY_MODE = {
    COMPILE_BLOCKED_MODE: OPERATOR_RECONCILED_KIND,
    ORPHANED_COMMANDS_MODE: OPERATOR_RECONCILED_KIND,
    RETIRED_GENERATION_MODE: OPERATOR_RETIRED_GENERATION_KIND,
}
STATUS_BEFORE_BY_MODE = {
    COMPILE_BLOCKED_MODE: WorkflowStatus.BLOCKED.value,
    ORPHANED_COMMANDS_MODE: WorkflowStatus.FAILED.value,
    RETIRED_GENERATION_MODE: WorkflowStatus.RUNNING.value,
}
UNKNOWN_ACTOR = "unknown-identity"


def event_errors(
    snapshot: Mapping[str, Any],
    seed: ModeSeed,
    *,
    actor: str,
    admin_plan_sha256: str,
    approved_plan_sha256: str,
) -> list[str]:
    """Every record the mode wrote carries one attributed event; no other does.

    The event is the per-record projection of ``applied.json``: the operator
    (STS ARN or user@host, never ``unknown-identity`` when one was sent), the
    reference, the runtime digest the apply was bound to, the admin-side digest
    the operator approved, and the status the write moved the record from and
    to. Refused twins and untouched siblings must carry none.
    """

    errors: list[str] = []
    kind = EVENT_KIND_BY_MODE[seed.mode]
    for request_id in seed.actionable_ids:
        events = [
            item
            for item in (snapshot.get(request_id) or {}).get("events") or []
            if item.get("kind") == kind
        ]
        if len(events) != 1:
            errors.append(
                f"{request_id}: {len(events)} {kind} event(s), expected exactly 1"
            )
            continue
        event = events[0]
        details = event.get("details") or {}
        if event.get("actor") != actor or actor == UNKNOWN_ACTOR:
            errors.append(
                f"{request_id}: event actor is {event.get('actor')!r}, expected {actor!r}"
            )
        if details.get("reference") != REFERENCE:
            errors.append(
                f"{request_id}: event reference is {details.get('reference')!r}"
            )
        if details.get("admin_plan_sha256") != admin_plan_sha256:
            errors.append(
                f"{request_id}: event admin_plan_sha256 is "
                f"{details.get('admin_plan_sha256')!r}, expected the operator's digest"
            )
        runtime_digest = str(details.get("plan_sha256") or "")
        if _SHA256.fullmatch(runtime_digest) is None:
            errors.append(f"{request_id}: event carries no runtime plan_sha256")
        elif not seed.settled_digest_differs and runtime_digest != approved_plan_sha256:
            errors.append(
                f"{request_id}: event plan_sha256 {runtime_digest!r} is not the "
                f"approved {approved_plan_sha256!r}"
            )
        expected_before = STATUS_BEFORE_BY_MODE[seed.mode]
        if details.get("previous_status") != expected_before:
            errors.append(
                f"{request_id}: event previous_status is "
                f"{details.get('previous_status')!r}, expected {expected_before!r}"
            )
        expected_after = seed.statuses_after[request_id]
        if details.get("new_status") != expected_after or event.get("status") != (
            expected_after
        ):
            errors.append(
                f"{request_id}: event new_status is {details.get('new_status')!r}, "
                f"expected {expected_after!r}"
            )
        if REFERENCE not in str(event.get("reason") or ""):
            errors.append(f"{request_id}: event reason does not carry {REFERENCE!r}")
    for request_id in seed.refused_ids:
        stray = (snapshot.get(request_id) or {}).get("events") or []
        if stray:
            errors.append(
                f"{request_id}: a refused record carries {len(stray)} operator event(s)"
            )
    return errors


def rerun_event_errors(
    before: Mapping[str, Any], after: Mapping[str, Any], seed: ModeSeed
) -> list[str]:
    """A no-op or refused rerun appends nothing to the audit trail."""

    errors: list[str] = []
    for request_id in seed.workflow_ids:
        earlier = (before.get(request_id) or {}).get("events") or []
        later = (after.get(request_id) or {}).get("events") or []
        if len(later) != len(earlier):
            errors.append(
                f"{request_id}: the rerun changed the operator event count "
                f"{len(earlier)} -> {len(later)}"
            )
    return errors


def command_log_errors(
    log_path: Any, *, state_dir: Any, kind: str = "mutating"
) -> list[str]:
    """ARCH-I3: a mutating admin command's console log lands under logs/<kind>/."""

    if log_path is None:
        return ["the admin command log was not opened (state directory refused)"]
    path = str(log_path)
    expected_parent = str(state_dir / "logs" / kind)
    errors: list[str] = []
    if not path.startswith(expected_parent):
        errors.append(f"command log {path} is not under {expected_parent}")
    if not path.endswith(".log"):
        errors.append(f"command log {path} is not a .log file")
    return errors


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


def _plan_items(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["request_id"]): item
        for item in plan.get("items") or []
        if isinstance(item, Mapping)
    }


def plan_errors(plan: Mapping[str, Any], seed: ModeSeed) -> list[str]:
    """Whether the plan named exactly the actionable records and refused the twin."""

    errors: list[str] = []
    if plan.get("mode") != seed.plan_mode:
        errors.append(f"plan mode is {plan.get('mode')!r}, expected {seed.plan_mode!r}")
    digest = str(plan.get("plan_sha256") or "")
    if _SHA256.fullmatch(digest) is None:
        errors.append("plan does not carry a sha256 digest")
    items = _plan_items(plan)
    if sorted(items) != sorted(seed.workflow_ids):
        errors.append(
            f"plan covers {sorted(items)}, expected {sorted(seed.workflow_ids)}"
        )
    for request_id, expected in seed.plan_flags.items():
        item = items.get(request_id)
        if item is None:
            errors.append(f"{request_id}: plan has no item")
            continue
        for field, value in expected.items():
            if item.get(field) is not value:
                errors.append(
                    f"{request_id}: plan {field} is {item.get(field)!r}, "
                    f"expected {value!r}"
                )
    for request_id, substrings in seed.plan_reasons.items():
        item = items.get(request_id) or {}
        reasons = [str(value) for value in item.get("reasons") or []]
        if not substrings:
            if reasons:
                errors.append(
                    f"{request_id}: item was expected to carry no reasons, "
                    f"got {reasons!r}"
                )
            continue
        if len(reasons) != len(substrings):
            errors.append(
                f"{request_id}: item carries {len(reasons)} reasons, "
                f"expected {len(substrings)}: {reasons!r}"
            )
        joined = " | ".join(reasons)
        for substring in substrings:
            if substring not in joined:
                errors.append(
                    f"{request_id}: reasons do not name {substring!r}, got {joined!r}"
                )
    return errors


def refusal_errors(label: str, output: str, substring: str) -> list[str]:
    """Whether a refusal that must happen happened, and named the right reason."""

    if not substring:
        return [f"{label}: no refusal substring was configured"]
    if substring in output:
        return []
    return [f"{label}: refusal does not name {substring!r}, got {output.strip()!r}"]


def apply_errors(
    result: Mapping[str, Any],
    seed: ModeSeed,
    *,
    approved_plan_sha256: str,
) -> list[str]:
    """Whether the apply did exactly what the approved plan said, and nothing else."""

    errors: list[str] = []
    if result.get("mode") != seed.apply_mode:
        errors.append(
            f"apply mode is {result.get('mode')!r}, expected {seed.apply_mode!r}"
        )
    if result.get("reference") != REFERENCE:
        errors.append(f"apply reference is {result.get('reference')!r}")
    applied = sorted(str(value) for value in result.get("applied_workflow_ids") or [])
    if applied != sorted(seed.actionable_ids):
        errors.append(
            f"apply changed {applied}, expected {sorted(seed.actionable_ids)}"
        )
    if result.get("failed_workflow_ids"):
        errors.append(f"apply reported failures for {result['failed_workflow_ids']}")
    if result.get("failures"):
        errors.append(f"apply reported failures {result['failures']}")
    if result.get("records_deleted") != 0:
        errors.append(f"apply deleted {result.get('records_deleted')!r} records")
    bound = str(result.get(seed.approved_digest_field) or "")
    if bound != approved_plan_sha256:
        errors.append(
            f"apply {seed.approved_digest_field} is {bound!r}, "
            f"expected the approved {approved_plan_sha256!r}"
        )
    settled = str(result.get("settled_plan_sha256") or "")
    if seed.settled_digest_differs and settled == approved_plan_sha256:
        errors.append(
            "apply settled the approved digest unchanged, so the cancel pass "
            "left nothing to reconcile"
        )
    if not seed.settled_digest_differs and settled != approved_plan_sha256:
        errors.append(
            "apply settled a different plan than the approved digest: "
            f"{settled!r} != {approved_plan_sha256!r}"
        )
    cancelled = result.get("cancelled_remote_commands")
    if isinstance(cancelled, Mapping):
        counted = sum(
            int(counts.get("cancelled", 0))
            for counts in cancelled.values()
            if isinstance(counts, Mapping)
        )
        if counted != len(seed.cancelled_command_ids):
            errors.append(
                f"apply cancelled {counted} remote commands, "
                f"expected {len(seed.cancelled_command_ids)}"
            )
    return errors


def record_errors(snapshot: Mapping[str, Any], seed: ModeSeed) -> list[str]:
    """Whether every workflow ended in the state the mode promises, refused ones included."""

    errors: list[str] = []
    for request_id, status in seed.statuses_after.items():
        observed = (snapshot.get(request_id) or {}).get("status")
        if observed != status:
            errors.append(f"{request_id}: status is {observed!r}, expected {status!r}")
    for request_id, substrings in seed.reason_substrings.items():
        reason = str((snapshot.get(request_id) or {}).get("preemption_reason") or "")
        for substring in substrings:
            if substring not in reason:
                errors.append(
                    f"{request_id}: preemption_reason does not carry {substring!r}"
                )
    for request_id, successor in seed.preempted_by.items():
        observed = (snapshot.get(request_id) or {}).get("preempted_by_workflow_id")
        if observed != successor:
            errors.append(
                f"{request_id}: preempted_by_workflow_id is {observed!r}, "
                f"expected {successor!r}"
            )
        if (snapshot.get(request_id) or {}).get("execution_owner_id") is not None:
            errors.append(f"{request_id}: revoked record still holds an owner")
    return errors


def command_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ModeSeed,
) -> list[str]:
    """Whether the orphans were failed with the expected source and the live one untouched."""

    errors: list[str] = []
    for command_id in seed.cancelled_command_ids:
        observed = after.get(command_id)
        if observed is None:
            errors.append(f"{command_id}: command is missing after the apply")
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
        if REFERENCE not in str(observed.get("error") or ""):
            errors.append(f"{command_id}: error does not carry {REFERENCE!r}")
        if observed.get("lease_owner") is not None:
            errors.append(f"{command_id}: cancelled command still holds a lease")
    for command_id in seed.untouched_command_ids:
        if before.get(command_id) != after.get(command_id):
            errors.append(
                f"{command_id}: live command changed from {before.get(command_id)!r} "
                f"to {after.get(command_id)!r}"
            )
    return errors


def safety_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ModeSeed,
) -> list[str]:
    """Whether the release preflight stopped naming the records the mode closed.

    ``blockers_after`` is not always empty and must not be asserted as empty. A
    successor generation that is genuinely running, and a record that already
    changed a node and therefore needs a human, are real blockers both before and
    after -- reporting them as cleared would be the bug this case exists to
    prevent. What must hold is that every record the mode closed leaves the list
    and nothing else does.
    """

    errors: list[str] = []
    observed_before = sorted(str(value) for value in before.get("blockers") or [])
    observed_after = sorted(str(value) for value in after.get("blockers") or [])
    if observed_before != sorted(seed.blockers_before):
        errors.append(
            f"blockers before the apply are {observed_before}, "
            f"expected {sorted(seed.blockers_before)}"
        )
    if observed_after != sorted(seed.blockers_after):
        errors.append(
            f"blockers after the apply are {observed_after}, "
            f"expected {sorted(seed.blockers_after)}"
        )
    for name, values in (("before", before), ("after", after)):
        count = values.get("blocker_count")
        listed = len(values.get("blockers") or [])
        if count != listed:
            errors.append(f"blocker_count {name} is {count!r} for {listed} blockers")
    resolved = sorted(str(value) for value in before.get("resolved_blocked") or [])
    if resolved != sorted(seed.resolved_blocked):
        errors.append(
            f"resolved_blocked is {resolved}, expected {sorted(seed.resolved_blocked)}"
        )
    cleared = set(seed.blockers_before) - set(seed.blockers_after)
    if cleared and not cleared <= set(seed.actionable_ids):
        errors.append(
            f"the seed expects records outside the plan to be cleared: "
            f"{sorted(cleared - set(seed.actionable_ids))}"
        )
    return errors


def open_command_errors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: ModeSeed,
) -> list[str]:
    """Whether the Store's open-command counters, the release gate, dropped by the orphans."""

    errors: list[str] = []
    expected = len(seed.cancelled_command_ids)
    dropped = _open_commands(before) - _open_commands(after)
    if dropped != expected:
        errors.append(f"open remote commands dropped by {dropped}, expected {expected}")
    return errors


OPEN_COMMAND_STATUSES = (
    RemoteCommandStatus.PENDING.value,
    RemoteCommandStatus.LEASED.value,
    RemoteCommandStatus.WAITING.value,
)


def _open_commands(stats: Mapping[str, Any]) -> int:
    """The counter the release upgrade refuses to start against, from ``by_status``."""

    counters = stats.get("by_status")
    if not isinstance(counters, Mapping):
        raise ValueError("remote command stats carry no by_status counters")
    return sum(int(counters.get(name, 0)) for name in OPEN_COMMAND_STATUSES)


def rerun_errors(
    seed: ModeSeed,
    *,
    plan: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    output: str,
) -> list[str]:
    """Whether running the same command again is safe, in this mode's own terms."""

    if seed.rerun_contract == RERUN_REFUSED:
        return refusal_errors("rerun", output, seed.rerun_refusal)
    errors: list[str] = []
    if plan is None or result is None:
        return [f"rerun did not produce a plan and an apply: {output.strip()!r}"]
    items = _plan_items(plan)
    settled_key = (
        "already_closed" if seed.mode == COMPILE_BLOCKED_MODE else "already_revoked"
    )
    settled_field = (
        "already_closed_workflow_ids"
        if seed.mode == COMPILE_BLOCKED_MODE
        else "already_revoked_workflow_ids"
    )
    for request_id in seed.actionable_ids:
        item = items.get(request_id) or {}
        if item.get(settled_key) is not True:
            errors.append(f"{request_id}: rerun plan {settled_key} is not True")
    if result.get("applied_workflow_ids"):
        errors.append(f"rerun changed {result['applied_workflow_ids']} a second time")
    settled = sorted(str(value) for value in result.get(settled_field) or [])
    if settled != sorted(seed.actionable_ids):
        errors.append(
            f"rerun {settled_field} is {settled}, "
            f"expected {sorted(seed.actionable_ids)}"
        )
    return errors


def case_verdict(stages: Mapping[str, Sequence[str]]) -> str:
    return "PASS" if not any(errors for errors in stages.values()) else "FAIL"


def all_errors(stages: Mapping[str, Sequence[str]]) -> list[str]:
    return [f"{stage}: {error}" for stage in sorted(stages) for error in stages[stage]]
