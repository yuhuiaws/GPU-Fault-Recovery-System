from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
    bounded_reasons,
)
from gpu_fault.handoff_hold import held_nodes, record_held_nodes
from gpu_fault.notifications import HardwareEscalationEmailBuilder
from gpu_fault.health_recovery_contract import diagnostic_workflow_failures
from gpu_fault.operation_registry import (
    HARDWARE_ESCALATION_RELEVANT_OPERATIONS,
    NODE_ACTION_SCOPE_OPERATIONS,
)
from gpu_fault.orchestration.failure_outcome import (
    EXECUTOR_REJECTED_STATUS_SOURCE as EXECUTOR_REJECTED_STATUS_SOURCE,
    ISOLATION_NOT_ISOLATED,
    NO_SECOND_TICKET_OPERATIONS as _NO_SECOND_TICKET_OPERATIONS,
    UNKNOWN_OUTCOME_FLAGS as _UNKNOWN_OUTCOME_FLAGS,
    executor_rejected_failure as executor_rejected_failure,
    fabric_manager_unit_failed,
    held_chain_hand_off,
    lifecycle_isolation_refusals,
    node_failure_reasons,
    reported_failed_nodes,
    unknown_outcome_failure as unknown_outcome_failure,
)
from gpu_fault.orchestration.families.identity import note_stale_event_link
from gpu_fault.orchestration.gpu_placement import (
    EscalationGpuScope,
    escalation_gpu_scope,
    scope_validation_steps,
)
from gpu_fault.orchestration.validation_evidence import (
    VALIDATION_OPERATIONS,
    held_for_unsent_hand_off,
    requested_escalations,
    validation_unobserved_failure,
    withheld_escalation,
    withheld_escalation_reasons,
)
from gpu_fault.orchestration.validation_rungs import (
    EFA_RECOVERY_STAGE,
    RESET_OPERATIONS as _RESET_OPERATIONS,
    VALIDATION_JUDGED_OPERATIONS,
    validation_rung,
)
from gpu_fault.store.shared.errors import NotFoundError

RESTART_SAFETY_PARAMETERS = (
    "cluster_id",
    "job_id",
    "source_attempt_id",
    "source_gpu_count",
    "restart_budget",
)


_VALIDATION_OPERATIONS = VALIDATION_OPERATIONS
# The node actions a validation judges: the reset, reboot, replacement,
# runtime restart, remediation or diagnostic that ran before it
# (``validation_rungs.VALIDATION_RUNGS``). A failed validation escalates the
# nodes one of these acted on, never a sibling the step merely checked.
_RECOVERY_ACTION_OPERATIONS = VALIDATION_JUDGED_OPERATIONS
# Containment and release steps. Their failure is not a hardware rung but it
# is not nothing either: a node whose RESTORE_SCHEDULING failed stays
# cordoned until someone acts, so the failure goes to an operator (F-H1).
_CONTAINMENT_RELEASE_OPERATIONS = frozenset(
    {
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.RESTORE_SCHEDULING,
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.QUARANTINE,
    }
)
_CLASSIFIABLE_OPERATIONS = (
    HARDWARE_ESCALATION_RELEVANT_OPERATIONS
    | _RECOVERY_ACTION_OPERATIONS
    | _CONTAINMENT_RELEASE_OPERATIONS
    | _VALIDATION_OPERATIONS
)

# Stage names the classifier hands to ``emit``. A containment or release step
# that failed for an ordinary reason is re-cordoned by the support workflow
# (F-H1); one the adapter *refused* -- the node is absent, or owned by another
# generation -- cannot be isolated by planning the same step again, so the
# support workflow carries no isolation (ARCH-E2E-2A finding 1, DESTR-020).
CONTAINMENT_STAGE = "containment_or_release"
CONTAINMENT_REFUSED_STAGE = "containment_refused"
# The Node Agent refused the action at its safety boundary and attests that
# nothing was applied (``node_action_refused_before_acting``): the hand-off
# plans evidence and the support ticket only -- no quarantine, no reboot --
# because the chain's own cordon is already on a node nothing was done to.
# So does a failed Fabric Manager restart with no report that its unit failed
# (``_fabric_manager_restart_rung``): it never ran on a node nothing stopped.
NODE_ACTION_REFUSED_STAGE = "node_action_refused"
# A validation that never observed its node: every node it waited for was
# pending behind the product's own quiesce taint (``quiesce_taint_present`` in
# ``node_pending``, written by the validation adapter) and no node was seen
# failing. Nothing about the hardware was learned, so neither ladder climbs
# -- live 2026-10-07 (gpu-eks-cluster DESTR-025) the reboot rung's VALIDATE_GPU
# starved behind the taint the retired restore would have lifted, climbed to a
# REPLACE_NODE no spare could serve, and the exhausted branch quarantined the
# node. The hand-off plans evidence and the ticket only, like a refusal: the
# node keeps the chain's cordon and the operator's restore lifts the taint.
VALIDATION_UNOBSERVED_STAGE = "validation_unobserved"
# A validation a bound failed before it observed any node failure -- the
# per-step cap or the workflow deadline, the node's collectors dark or no
# fresh sample yet -- whose plan asked for an escalation on failure
# (``failure_escalation_action``). The request is for a validation that looked
# and found the fault still there; this one could not look, so it is withheld
# (``validation_evidence.withheld_escalation``) -- live 2026-10-25 (plain EKS,
# GF-GENERAL-LIVE-013 link_inactive) the EFA diagnostic chain's VALIDATE_FABRIC
# waited out the 600 s cap with ``missing_or_stale_collectors`` as its only
# word and the requested REBOOT_NODE rebooted a node nothing had been learned
# about. Same hand-off as the unobserved stage: evidence and the ticket, the
# incident reasons naming the missing evidence and the withheld request.
VALIDATION_EVIDENCE_MISSING_STAGE = "validation_evidence_missing"
# A failure whose outcome nobody knows. Three producers write it into the step
# details (``status_source`` never crosses into a step record, so details are
# the contract): the node-action fold, when the agent's attempt closed
# INTERRUPTED (``node_action_interrupted``); the regional executor, when it
# abandoned a mutating command at its execution cap (``outcome_unknown``); and
# the node-action transport, when the agent refused a body change under an id
# whose earlier attempt may already have run (COMMAND_ID_REUSED). All set
# ``manual_confirmation_required``. The reset may have run, the reboot may still
# be in flight -- climbing the hardware ladder from there reboots a node whose
# GPU was possibly just reset, or replaces one that is possibly still rebooting.
# Both ladders read this: ``_classify`` here and the DAG branch escalator
# (``execution/branch_escalation.py``), which exhausts such a branch at once.
MANUAL_CONFIRMATION_STAGE = "manual_confirmation_required"
# Written by the Kubernetes node adapter on a step it refused rather than
# failed: ``safety_rejection`` on every refusal, ``absent`` when the node
# could not be read at all.
SAFETY_REJECTION_DETAIL = "safety_rejection"
NODE_ABSENT_DETAIL = "absent"

ESCALATION_NAMES: dict[RecoveryAction, str] = {
    RecoveryAction.DRAIN: "drain",
    RecoveryAction.REBOOT_NODE: "reboot",
    RecoveryAction.REPLACE_NODE: "replace",
    RecoveryAction.ESCALATE_OPERATOR: "support",
}
_ESCALATION_EVENT_SEPARATOR = "-after-"
# Escalations that are the last rung. Their product is an operator hand-off;
# when that product fails, opening another one on the same node answers
# nothing and, driven by the failed-workflow reconcile, never stops.
CHAIN_TERMINAL_ESCALATIONS = frozenset({"support"})
ESCALATION_CHAIN_TERMINATED_REASON = "escalation chain terminated"


def escalation_origin(event_id: str) -> tuple[str, str] | None:
    """``(escalation_name, source_workflow_request_id)`` when ``event_id`` was
    minted by :meth:`HardwareEscalationService.escalate`, else ``None``.

    The event id is the field the escalation is deduplicated on
    (``create_incident_workflow_if_absent``), so it is the authoritative record
    of an incident being an escalation product; the workflow request id
    repeats it with a ``workflow-`` prefix.
    """

    name, separator, source_request_id = event_id.partition(_ESCALATION_EVENT_SEPARATOR)
    if not separator or not source_request_id:
        return None
    if name not in ESCALATION_NAMES.values():
        return None
    return name, source_request_id


def node_action_refused_before_acting(details: Mapping[str, Any] | None) -> bool:
    """Whether a FAILED node action was refused by the Node Agent's own safety
    boundary without anything being applied to the node.

    ``safety_rejection`` is the agent's refusal (a persistent device holder,
    an expired ownership permit, a moved fence); with it, either the agent
    attests that every physical call it made returned a verdict
    (``node_action_outcome_known``: the final ownership boundary attests its
    own refusals for every operation it guards -- a reset refused busy and
    then refused at the boundary before its retry, a fabric reset or a Fabric
    Manager restart refused before its spawn) or says nothing was started
    (``node_action_not_started``), nothing completed on any node and no
    outcome-unknown flag stands. Neither ladder climbs on it: a reboot or a drain would act
    on a node the refusal left untouched (live, gpu-eks-cluster: a reset
    refused for an expired permit opened a drain successor that quarantined
    and cordoned a healthy node). The incident is the operator's as
    ESCALATED, cordoned by the chain's own MARK_UNSCHEDULABLE, and the
    restore of the administrator's ``restore`` disposition puts it back.
    """

    if (
        details is None
        or details.get(SAFETY_REJECTION_DETAIL) is not True
        or details.get("reset_completed")
        or details.get("completed_nodes")
    ):
        return False
    # ``manual_confirmation_required`` rides on every agent refusal ("an
    # operator, never a reboot rung"); with the attestation it asks for a
    # decision, not for a confirmation of an outcome. The other flags say the
    # outcome is really unknown, and those keep the operator-confirmation stage.
    if any(
        details.get(flag) is True
        for flag in _UNKNOWN_OUTCOME_FLAGS
        if flag != "manual_confirmation_required"
    ):
        return False
    attested = (
        details.get("node_action_outcome_known") is True
        or details.get("node_action_not_started") is True
    )
    verdict = details.get("reset_verdict")
    return attested and verdict in (None, "not-attempted", "not-applied")


def _contract_ends_classification(
    workflow: WorkflowRequest, failed_executions: list[WorkflowStepExecution]
) -> bool:
    """Whether a diagnostic contract ends the classification at the held-chain
    rule: every failure is a contract step's, none is a containment or release
    step, and none of a chain that holds its node has an outcome nobody knows.
    A cordon, quarantine or restore that failed goes on to the hand-off every
    other chain gets (F-H1, the containment stages below), and an unknown
    outcome in a chain that cordoned its node (ENVIRONMENT_REVIEW, the older
    cooldown plan) to the manual-confirmation hand-off
    (``_no_verdict_classification``), which holds every node of the step -- a
    job-wide stop's whole allocation -- until an operator confirms. The
    contract keeps the hardware rungs away, and neither hand-off plans one; a
    read-only chain (a hung or diagnostic-only plan) holds nothing and keeps
    ending here, so its unknown evidence step never cordons a node."""

    return (
        diagnostic_workflow_failures(workflow, failed_executions)
        and not any(
            execution.operation in _CONTAINMENT_RELEASE_OPERATIONS
            for execution in failed_executions
        )
        and not (
            _unknown_outcome_failures(failed_executions) and record_held_nodes(workflow)
        )
    )


def _pre_action_classification(
    failed_executions: list[WorkflowStepExecution],
) -> tuple[bool, Any]:
    """``(handled, classification)`` for failures that never acted on the node.

    Every failed step refused by the cluster executor before dispatch (a stale
    generation, a scope mismatch): the node was never touched, so there is no
    hardware verdict to climb from and no ticket to open -- handled, no
    successor; the record's terminal reason names the refusal. Every failed
    step refused before acting with at least one refused by the Node Agent's
    safety boundary (``node_action_refused_before_acting``): handled, and an
    operator decides with the evidence and the ticket while the node keeps the
    chain's cordon (``NODE_ACTION_REFUSED_STAGE``: no quarantine, no reboot).
    Anything else is not handled here and climbs the ladder.
    """

    if not failed_executions:
        return False, None
    rejected = [
        executor_rejected_failure(execution.details) for execution in failed_executions
    ]
    refused = [
        node_action_refused_before_acting(execution.details)
        for execution in failed_executions
    ]
    if all(rejected):
        return True, None
    if all(left or right for left, right in zip(rejected, refused, strict=True)):
        return True, (
            NODE_ACTION_REFUSED_STAGE,
            RecoveryAction.ESCALATE_OPERATOR,
            WorkflowOperation.ESCALATE_SUPPORT,
            [
                execution
                for execution, flagged in zip(failed_executions, refused, strict=True)
                if flagged
            ],
        )
    return False, None


def escalation_descends_from(
    store: Any, workflow_request_id: str | None, incident_id: str, *, limit: int = 16
) -> bool:
    """Whether workflow ``workflow_request_id`` is, by escalation or queueing,
    a descendant of a workflow of ``incident_id``.

    Walks ``predecessor_workflow_id`` and the ``<rung>-after-<workflow>`` event
    ids of the incidents on the way (``escalation_origin``), at most ``limit``
    records. A node's isolation names the incident that placed it; the
    successor asking to take it over may be two rungs away (live: the reboot
    successor of a failed plugin restart was refused, its support successor
    then asked about the *original* incident, whose current workflow the one
    -hop check ``escalation_continues`` did not know). An unreadable record ends
    the walk as "no": the caller fails closed.
    """

    frontier = [workflow_request_id] if workflow_request_id else []
    seen: set[str] = set()
    while frontier and len(seen) < limit:
        request_id = frontier.pop()
        if request_id in seen:
            continue
        seen.add(request_id)
        try:
            workflow = store.get_workflow(request_id)
        except (NotFoundError, KeyError, TypeError):
            continue
        if workflow.incident_id == incident_id and request_id != workflow_request_id:
            return True
        if workflow.predecessor_workflow_id:
            frontier.append(workflow.predecessor_workflow_id)
        try:
            incident = store.get_incident(workflow.incident_id)
        except (NotFoundError, KeyError, TypeError):
            continue
        origin = escalation_origin(incident.event_id)
        if origin is not None:
            frontier.append(origin[1])
    return False


def _acted_nodes(workflow: WorkflowRequest, before_index: int) -> set[str]:
    """Nodes a recovery action ran on before step ``before_index``."""

    steps = workflow.official_steps
    return {
        node_id
        for execution in workflow.step_executions
        if execution.operation in _RECOVERY_ACTION_OPERATIONS
        and 0 <= execution.step_index < min(before_index, len(steps))
        for node_id in steps[execution.step_index].node_ids
    }


def _failed_step_nodes(
    workflow: WorkflowRequest,
    execution: WorkflowStepExecution,
    step_nodes: list[str],
) -> list[str]:
    """The nodes a failed step escalates.

    The step's ``failed_nodes`` when it reported them; the ``pending_nodes``
    of a step the per-step cap failed (its last WAITING details are what it
    keeps); else every node of the step. A failed *validation* is narrowed to
    the nodes the recovery action acted on: after a merge the validation
    names siblings the fault never touched, and the rung it fails into is a
    hardware action on the acted node, not on the sibling (live 2026-10-05: a
    RESET_GPU on one node escalated to a reboot that cordoned two).
    """

    selected = reported_failed_nodes(execution.details, step_nodes)
    if execution.operation in _VALIDATION_OPERATIONS:
        acted = _acted_nodes(workflow, execution.step_index) & set(step_nodes)
        if acted:
            selected = [node_id for node_id in selected if node_id in acted] or sorted(
                acted
            )
    return selected


def _unknown_outcome_failures(
    executions: list[WorkflowStepExecution],
) -> list[WorkflowStepExecution]:
    """FAILED executions that demand an operator, not a rung.

    Every operation qualifies -- a STOP_WORKLOADS or RESTART_VM the executor
    abandoned mid-flight is not a hardware rung, but classifying it as None
    made it a silent FAILED the dispatcher only stamps ``failure_handled_at``
    on -- except the three in ``failure_outcome.NO_SECOND_TICKET_OPERATIONS``: an abandoned
    ESCALATE_SUPPORT, FREEZE_EVIDENCE or CHECKPOINT_WORKLOADS carries the same
    flags, but a ``support-after-<wf>`` successor for a support escalation that
    timed out is a second vendor ticket. Those keep the previous behaviour -- no
    classification, the workflow stays FAILED with no successor.
    """

    return [
        execution
        for execution in executions
        if execution.status is WorkflowStepStatus.FAILED
        and execution.operation not in _NO_SECOND_TICKET_OPERATIONS
        and unknown_outcome_failure(execution.details)
    ]


def _latest_failed_validation(
    executions: list[WorkflowStepExecution],
) -> WorkflowStepExecution | None:
    """The record's latest FAILED step when it is a validation; else ``None``.

    The latest failure is the one the ladder stopped on: an earlier FAILED
    reset was answered by the reboot rung that ran after it, and classifying
    the record by that reset would plan a second reboot of a node whose first
    one succeeded.
    """

    latest = max(
        (
            execution
            for execution in executions
            if execution.status is WorkflowStepStatus.FAILED
        ),
        key=lambda execution: execution.step_index,
        default=None,
    )
    if latest is None or latest.operation not in _VALIDATION_OPERATIONS:
        return None
    return latest


def _no_verdict_classification(
    workflow: WorkflowRequest,
) -> tuple[str, RecoveryAction, WorkflowOperation, list[WorkflowStepExecution]] | None:
    """The operator hand-offs for a record whose failure carries no hardware
    verdict: an unknown outcome (the step may have run, or may still be
    running -- an operator confirms, no rung is climbed on a guess), a
    validation that could not look at the node because the product's own
    quiesce taint kept the exporter off it (no verdict, no rung, no
    quarantine -- the operator's restore lifts the taint), or a validation a
    bound failed before it observed any failure while its plan asked for an
    escalation (``withheld_escalation``: the request was for a verdict)."""

    unknown_failures = _unknown_outcome_failures(workflow.step_executions)
    if unknown_failures:
        return (
            MANUAL_CONFIRMATION_STAGE,
            RecoveryAction.ESCALATE_OPERATOR,
            WorkflowOperation.ESCALATE_SUPPORT,
            unknown_failures,
        )
    latest = _latest_failed_validation(workflow.step_executions)
    if latest is None:
        return None
    if validation_unobserved_failure(latest.details):
        stage = VALIDATION_UNOBSERVED_STAGE
    elif withheld_escalation(workflow, latest) is not None:
        stage = VALIDATION_EVIDENCE_MISSING_STAGE
    else:
        return None
    return (
        stage,
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
        [latest],
    )


def _fabric_manager_restart_rung(
    failed_executions: list[WorkflowStepExecution],
) -> tuple[str, RecoveryAction, WorkflowOperation]:
    """A failed RESTART_FABRIC_MANAGER climbs like a failed reset, through the
    whole record (the solo XID 45 plan holds no cordon; the reboot successor
    cordons first) -- only when the agent reports the unit failed
    (``failure_outcome.fabric_manager_unit_failed``). Any other failure never
    ran the restart, or says nothing about the unit, on a node nothing
    stopped: no reboot, the evidence and the ticket only, the reason naming
    why (``failure_outcome.fabric_manager_restart_reason``)."""

    restarts = [
        execution
        for execution in failed_executions
        if execution.operation is WorkflowOperation.RESTART_FABRIC_MANAGER
    ]
    if restarts and all(
        fabric_manager_unit_failed(execution.details) for execution in restarts
    ):
        return (
            "fabric_manager_restart",
            RecoveryAction.REBOOT_NODE,
            WorkflowOperation.RESTART_NODE,
        )
    return (
        NODE_ACTION_REFUSED_STAGE,
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
    )


def _refused_containment(
    executions: list[WorkflowStepExecution],
) -> bool:
    """Every failed containment/release execution was a safety refusal."""

    refusals = [
        execution
        for execution in executions
        if execution.status is WorkflowStepStatus.FAILED
        and execution.operation in _CONTAINMENT_RELEASE_OPERATIONS
    ]
    return bool(refusals) and all(
        execution.details.get(SAFETY_REJECTION_DETAIL) is True for execution in refusals
    )


def next_rung(
    failed_operation: WorkflowOperation,
    recovery_context: set[WorkflowOperation] | frozenset[WorkflowOperation],
    details: Mapping[str, Any] | None = None,
) -> WorkflowOperation | None:
    """The next hardware-recovery rung after ``failed_operation`` failed.

    Mirrors the ladder ``HardwareEscalationService._classify`` applies to a
    whole workflow, for one node branch (F-N1): reset -> reboot -> warm-spare
    replacement. ``None`` means the branch takes no rung in place and the
    outcome goes to the whole-workflow path once the workflow fails: an
    operator (support escalation / drain), or the reboot the whole record
    climbs to after a runtime restart -- a failed Fabric Manager restart or a
    validation behind one, whose successor cordons first -- or a plan's own
    request (``validation_rungs.VALIDATION_RUNGS``, ``EFA_RECOVERY_STAGE``).
    ``recovery_context`` is the set of recovery operations that already ran
    on the branch; it decides what a failed *validation* escalates from.
    """

    if failed_operation is WorkflowOperation.RECOVER_DRAIN_P2P:
        evidence = details or {}
        if evidence.get("gpu_recovery_reboot_required") is True or evidence.get(
            "reset_failed"
        ):
            return WorkflowOperation.RESTART_NODE
        return None
    if failed_operation in _RESET_OPERATIONS:
        return WorkflowOperation.RESTART_NODE
    if failed_operation is WorkflowOperation.RESTART_NODE:
        return WorkflowOperation.REPLACE_NODE
    if failed_operation in _VALIDATION_OPERATIONS:
        if WorkflowOperation.REPLACE_NODE in recovery_context:
            return None
        if WorkflowOperation.RESTART_NODE in recovery_context:
            return WorkflowOperation.REPLACE_NODE
        if recovery_context & _RESET_OPERATIONS:
            return WorkflowOperation.RESTART_NODE
        return None
    return None


def escalation_lineage_nodes(
    store: Any, incident: FaultIncident, *, limit: int = 8
) -> set[str]:
    """The nodes of the workflows and incidents ``incident``'s
    ``<rung>-after-<workflow>`` event ids lead back to, at most ``limit`` hops;
    an unreadable link ends the walk."""

    nodes: set[str] = set()
    current = incident
    for _ in range(limit):
        origin = escalation_origin(current.event_id)
        if origin is None:
            break
        try:
            workflow = store.get_workflow(origin[1])
            current = store.get_incident(workflow.incident_id)
        except (NotFoundError, KeyError, TypeError):
            break
        nodes.update(current.node_ids)
        nodes.update(node for step in workflow.official_steps for node in step.node_ids)
    return nodes


def _successor_gpu_scope(
    workflow: WorkflowRequest,
    source: FaultIncident,
    failed_nodes: list[str],
    store: Any,
) -> EscalationGpuScope:
    """``escalation_gpu_scope`` with the store and the source's own lineage."""

    return escalation_gpu_scope(
        workflow,
        source,
        failed_nodes,
        store,
        lineage_node_ids=(
            () if store is None else escalation_lineage_nodes(store, source)
        ),
    )


def _escalation_operations(
    failed_stage: str,
    next_action: RecoveryAction,
    next_operation: WorkflowOperation | None,
    workload_ids: list[str],
) -> list[WorkflowOperation]:
    """The operation sequence the escalation workflow plans (see ``emit``)."""

    if failed_stage in (
        CONTAINMENT_REFUSED_STAGE,
        NODE_ACTION_REFUSED_STAGE,
        VALIDATION_UNOBSERVED_STAGE,
        VALIDATION_EVIDENCE_MISSING_STAGE,
    ):
        # The adapter refused the isolation (absent node, foreign
        # generation): there is nothing this workflow can isolate, and
        # planning MARK_UNSCHEDULABLE / QUARANTINE again on the same node
        # only reproduces the refusal. Evidence and the hand-off remain. The
        # Node Agent's attested refusal takes the same shape: the chain's
        # cordon is on the node, nothing was applied to it, and a quarantine
        # would isolate a healthy node (live, gpu-eks-cluster). A validation
        # that could not look learned nothing to isolate on either: the node
        # keeps whatever cordon its own chain placed.
        operations = [WorkflowOperation.FREEZE_EVIDENCE]
    elif next_action is RecoveryAction.DRAIN:
        operations = [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
        ]
        if workload_ids:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations.extend(
            [
                WorkflowOperation.QUARANTINE,
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                WorkflowOperation.VALIDATE_GPU,
                # Same exit as the node-health DRAIN chain: the node stays
                # quarantined (RMA class, no RESTORE_SCHEDULING), and the
                # explicit hand-off is what moves the incident to ESCALATED
                # and tells an operator. Without it the drain parked in
                # QUARANTINED with nobody notified.
                WorkflowOperation.ESCALATE_SUPPORT,
            ]
        )
    else:
        operations = [
            WorkflowOperation.FREEZE_EVIDENCE,
            WorkflowOperation.MARK_UNSCHEDULABLE,
            WorkflowOperation.QUARANTINE,
        ]
    if next_operation is WorkflowOperation.ESCALATE_SUPPORT:
        operations.append(WorkflowOperation.ESCALATE_SUPPORT)
    elif next_action is not RecoveryAction.DRAIN:
        if workload_ids:
            operations.append(WorkflowOperation.STOP_WORKLOADS)
        operations.extend(
            [
                next_operation,
                WorkflowOperation.VALIDATE_GPU,
                WorkflowOperation.VALIDATE_HOST,
                WorkflowOperation.VALIDATE_FABRIC,
                WorkflowOperation.RESTORE_SCHEDULING,
            ]
        )
        if workload_ids:
            operations.append(WorkflowOperation.RESTART_WORKLOAD)
    return operations


def _successor_lifetime(
    workflow: WorkflowRequest,
    next_action: RecoveryAction,
    next_operation: WorkflowOperation | None,
    now: datetime,
) -> datetime | None:
    """The lifetime the escalation successor starts with.

    The hard lifetime (F-N1) bounds *automatic* remediation: a reboot or a
    replacement emitted after a failed reset shares the chain's single
    clock, so an escalation ladder cannot outlive the window by re-issuing
    itself. An operator hand-off -- the support ticket, a drain -- is the
    thing the window ends *in*: inheriting an already-expired deadline
    failed the support workflow at its first claim, before
    FREEZE_EVIDENCE ran, and the node never reached an operator
    (DESTR-018). Hand-offs therefore start their own clock; ``None`` lets
    the first claim stamp the node lifetime (and, for CHECK_MECHANICALS,
    the operator-acknowledgement floor). An automatic rung emitted while
    the chain's lifetime is still ahead keeps it; one emitted after the
    lifetime passed keeps the expired value on purpose, so it fails closed
    instead of running hardware actions past the window.
    """

    operator_hand_off = (
        next_operation is WorkflowOperation.ESCALATE_SUPPORT
        or next_operation is WorkflowOperation.CHECK_MECHANICALS
        or next_action in {RecoveryAction.ESCALATE_OPERATOR, RecoveryAction.DRAIN}
    )
    if operator_hand_off:
        return None
    return workflow.lifetime_deadline_at


def _chain_termination_reason(
    workflow: WorkflowRequest,
    source: FaultIncident,
    origin: tuple[str, str],
    failed_operations: list[str],
) -> str:
    """The incident reason ``_terminate_chain`` records; also its idempotency
    key, so the text must stay stable for a given failure."""

    escalation_name, source_request_id = origin
    return (
        f"{ESCALATION_CHAIN_TERMINATED_REASON}: {escalation_name} escalation "
        f"{source.event_id} (after {source_request_id}) failed again in "
        f"{workflow.request_id} at "
        f"{', '.join(failed_operations) or 'no recorded step'}; no further "
        f"{escalation_name} workflow is opened, operator attention required"
    )


class HardwareEscalationService:
    def __init__(self, store, builder) -> None:
        self.store = store
        self.builder = builder
        # Failed support escalations that were handed to an operator instead
        # of a further support escalation (the chain bound), and the Unix
        # time of the newest one (ARCH-E E4 convention).
        self.escalation_chain_terminated_total = 0
        self.escalation_chain_terminated_last_seen_timestamp_seconds = 0.0
        # Support escalations opened for a refused containment, without
        # re-planning the isolation the adapter refused.
        self.containment_refused_escalations_total = 0

    @staticmethod
    def _active_workload_dcgm_execution_review(
        workflow: WorkflowRequest,
    ) -> bool:
        if not any(step.workload_ids for step in workflow.official_steps):
            return False
        failed = [
            execution
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.FAILED
            and execution.operation is WorkflowOperation.RUN_DCGM_DIAGNOSTIC
        ]
        if not failed:
            return False
        for execution in failed:
            results = execution.details.get("node_results")
            if not isinstance(results, dict) or not results:
                return False
            # A step that gave up mid-batch reports only the nodes it folded.
            # The nodes with no verdict at all cannot be waved through on
            # their siblings' "rerun the diagnostic", so demand full coverage
            # of the step's targets, and escalate when the targets cannot be
            # resolved.
            if not 0 <= execution.step_index < len(workflow.official_steps):
                return False
            targeted = workflow.official_steps[execution.step_index].node_ids
            if any(node_id not in results for node_id in targeted):
                return False
            for result in results.values():
                if not isinstance(result, dict):
                    return False
                if any(
                    result.get(name)
                    for name in (
                        "diagnostic_findings",
                        "failed_checks",
                        "warning_checks",
                    )
                ):
                    return False
                recommendations = result.get("recommended_actions", [])
                if not isinstance(recommendations, list) or {
                    item.get("action_code")
                    for item in recommendations
                    if isinstance(item, dict)
                } != {"DCGM_EXECUTION_REVIEW"}:
                    return False
        return True

    @staticmethod
    def classify(workflow: WorkflowRequest):
        if HardwareEscalationService._active_workload_dcgm_execution_review(workflow):
            return None
        return HardwareEscalationService._classify(workflow)

    @staticmethod
    def _classify(workflow: WorkflowRequest):
        lifetime_failures = [
            execution
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.FAILED
            and execution.details.get("workflow_lifetime_exceeded") is True
        ]
        if lifetime_failures:
            # The remediation ran out of lifetime (F-N1): no further hardware
            # rung, the node belongs to an operator now.
            return (
                "lifetime_exceeded",
                RecoveryAction.ESCALATE_OPERATOR,
                WorkflowOperation.ESCALATE_SUPPORT,
                lifetime_failures,
            )
        failed_executions = [
            execution
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.FAILED
        ]
        if _contract_ends_classification(workflow, failed_executions):
            return held_chain_hand_off(workflow, failed_executions)
        handled, verdict = _pre_action_classification(failed_executions)
        if handled:
            return verdict
        hand_off = _no_verdict_classification(workflow)
        if hand_off is not None:
            return hand_off
        failed_operation_set = {execution.operation for execution in failed_executions}
        successful_operation_set = set(workflow.completed_operations) | {
            execution.operation
            for execution in workflow.step_executions
            if execution.status is WorkflowStepStatus.SUCCEEDED
        }
        # What actually ran before the failure, by execution record -- not the
        # steps that merely sit earlier in the list. In a DAG a sibling branch's
        # RESTART_NODE may precede the failed step positionally without ever
        # having run on this node (F-C6).
        executed_operation_set = {
            execution.operation for execution in workflow.step_executions
        }
        recovery_context_operations = successful_operation_set | executed_operation_set
        if (
            not (recovery_context_operations - failed_operation_set)
            & HARDWARE_ESCALATION_RELEVANT_OPERATIONS
        ):
            # A record with no execution history for its recovery steps (rows
            # written before completions were recorded): fall back to the
            # steps planned ahead of the failure.
            recovery_context_operations |= {
                step.operation
                for execution in workflow.step_executions
                if (
                    execution.status is WorkflowStepStatus.FAILED
                    and 0 <= execution.step_index < len(workflow.official_steps)
                )
                for step in workflow.official_steps[: execution.step_index]
            }
        # The plan's own requests, less a validation's that observed no
        # failure (``validation_evidence.requested_escalations``).
        requested = requested_escalations(workflow)
        if (
            RecoveryAction.REBOOT_NODE.value in requested
            and WorkflowOperation.RESTART_NODE not in recovery_context_operations
        ):
            failed_stage = EFA_RECOVERY_STAGE
            next_action = RecoveryAction.REBOOT_NODE
            next_operation = WorkflowOperation.RESTART_NODE
        elif WorkflowOperation.RECOVER_DRAIN_P2P in failed_operation_set:
            failed_stage = "drain_p2p_recovery"
            reboot = any(
                execution.operation is WorkflowOperation.RECOVER_DRAIN_P2P
                and execution.status is WorkflowStepStatus.FAILED
                and next_rung(execution.operation, set(), execution.details)
                is WorkflowOperation.RESTART_NODE
                for execution in workflow.step_executions
            )
            next_action = (
                RecoveryAction.REBOOT_NODE
                if reboot
                else RecoveryAction.ESCALATE_OPERATOR
            )
            next_operation = (
                WorkflowOperation.RESTART_NODE
                if reboot
                else WorkflowOperation.ESCALATE_SUPPORT
            )
        elif WorkflowOperation.RUN_DCGM_DIAGNOSTIC in (failed_operation_set):
            failed_stage = "temperature_diagnostic"
            next_action = RecoveryAction.DRAIN
            next_operation = None
        elif failed_operation_set.intersection(
            {
                WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
            }
        ):
            failed_stage = "nvidia_field_diagnostic"
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        elif failed_operation_set.intersection(
            {
                WorkflowOperation.REMEDIATE_DRIVER,
                WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
            }
        ):
            failed_stage = "software_or_firmware_remediation"
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        elif (refused := lifecycle_isolation_refusals(failed_executions)) is not None:
            # A reboot or replacement refused for want of observed isolation:
            # put it back and hand over, no rung (``failure_outcome``).
            failed_stage = (
                CONTAINMENT_STAGE
                if ISOLATION_NOT_ISOLATED in refused
                else CONTAINMENT_REFUSED_STAGE
            )
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        elif WorkflowOperation.REPLACE_NODE in failed_operation_set:
            failed_stage = "replacement"
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        elif WorkflowOperation.RESTART_NODE in failed_operation_set:
            failed_stage = "reboot"
            next_action = RecoveryAction.REPLACE_NODE
            next_operation = WorkflowOperation.REPLACE_NODE
        elif failed_operation_set.intersection(
            {
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            }
        ):
            failed_stage = "reset"
            next_action = RecoveryAction.REBOOT_NODE
            next_operation = WorkflowOperation.RESTART_NODE
        elif WorkflowOperation.RESTART_FABRIC_MANAGER in failed_operation_set:
            failed_stage, next_action, next_operation = _fabric_manager_restart_rung(
                failed_executions
            )
        elif failed_operation_set.intersection(_VALIDATION_OPERATIONS):
            rung = validation_rung(recovery_context_operations)
            if rung is not None:
                failed_stage, next_action, next_operation = rung
            elif held_for_unsent_hand_off(workflow):
                failed_stage = "drain_validation"
                next_action = RecoveryAction.ESCALATE_OPERATOR
                next_operation = WorkflowOperation.ESCALATE_SUPPORT
            else:
                # Validate-then-readmit: no recovery step ran before the
                # validation, so no rung answers it, but a node the record's
                # own cordon still holds goes to support; a record that holds
                # nothing (a read-only chain, the operator's validated restore)
                # opens nothing.
                return held_chain_hand_off(workflow, failed_executions)
        elif failed_operation_set.intersection(_CONTAINMENT_RELEASE_OPERATIONS):
            failed_stage = (
                CONTAINMENT_REFUSED_STAGE
                if _refused_containment(workflow.step_executions)
                else CONTAINMENT_STAGE
            )
            next_action = RecoveryAction.ESCALATE_OPERATOR
            next_operation = WorkflowOperation.ESCALATE_SUPPORT
        else:
            return held_chain_hand_off(workflow, failed_executions)

        failed_executions = [
            execution
            for execution in workflow.step_executions
            if (
                execution.status is WorkflowStepStatus.FAILED
                and execution.operation in _CLASSIFIABLE_OPERATIONS
            )
        ]
        if not failed_executions:
            return None

        return (
            failed_stage,
            next_action,
            next_operation,
            failed_executions,
        )

    @staticmethod
    def _exhausted_branch_executions(
        workflow: WorkflowRequest,
        failed_executions: list[WorkflowStepExecution],
    ) -> list[WorkflowStepExecution]:
        """Keep only the failures on the branches that actually exhausted.

        A job DAG's other branch may have failed a step, escalated in place and
        recovered (RESET_GPU failed, the reboot rung succeeded, RESTORE_SCHEDULING
        ran): that FAILED record is history, not hardware to hand to support.
        Read by node -- the exhausted rung's steps name the node, the branch's
        earlier steps may still carry ``branch:initial`` -- so every failure on
        the exhausted node stays in scope and nothing else does. Live 2026-09-09
        (DESTR-014): the support-after incident covered both nodes and
        re-quarantined the one that had just been restored.
        """

        exhausted = set(workflow.exhausted_branch_ids)
        if not exhausted:
            return failed_executions
        steps = workflow.official_steps
        exhausted_nodes = {
            node_id
            for step in steps
            if step.branch_id in exhausted
            for node_id in (step.branch_node_ids or step.node_ids)
        }
        if not exhausted_nodes:
            return failed_executions
        scoped = [
            execution
            for execution in failed_executions
            if 0 <= execution.step_index < len(steps)
            and set(steps[execution.step_index].node_ids) & exhausted_nodes
        ]
        return scoped or failed_executions

    @staticmethod
    def collect_scope(
        workflow: WorkflowRequest,
        source: FaultIncident,
        failed_executions,
        store: Any = None,
    ) -> dict:
        failed_nodes: set[str] = set()
        node_failures: dict[str, list[str]] = {}
        failed_operations = set()
        for execution in failed_executions:
            failed_operations.add(execution.operation)
            step_nodes = (
                workflow.official_steps[execution.step_index].node_ids
                if execution.step_index < len(workflow.official_steps)
                else source.node_ids
            )
            selected = _failed_step_nodes(workflow, execution, list(step_nodes))
            failed_nodes.update(selected)
            for node_id in selected:
                node_failures.setdefault(node_id, []).extend(
                    node_failure_reasons(execution, node_id)
                )
        if not failed_nodes:
            failed_nodes.update(source.node_ids)
        ordered_failed_nodes = sorted(failed_nodes)

        workload_ids = list(
            dict.fromkeys(
                workload_id
                for step in workflow.official_steps
                for workload_id in step.workload_ids
            )
        )
        # Placement read from ``store`` too; a sibling's GPU is never named on
        # a failed node's step (batch 149, ``orchestration/gpu_placement.py``).
        gpu_uuids, gpu_uuids_by_node, validation_gpus = _successor_gpu_scope(
            workflow, source, ordered_failed_nodes, store
        )
        failed = ", ".join(sorted(operation.value for operation in failed_operations))
        node_reason = "; ".join(
            (
                f"{node_id}: "
                + (
                    ",".join(sorted(set(node_failures[node_id])))
                    if node_failures.get(node_id)
                    else f"{failed or 'remediation'} failed"
                )
            )
            for node_id in ordered_failed_nodes
        )
        diagnostic_guidance: list[str] = []
        diagnostic_evidence: list[str] = []
        for execution in workflow.step_executions:
            if execution.operation is not WorkflowOperation.RUN_DCGM_DIAGNOSTIC:
                continue
            raw_node_results = execution.details.get("node_results", {})
            if not isinstance(raw_node_results, dict):
                continue
            for (
                node_id,
                node_result,
            ) in raw_node_results.items():
                if node_id not in ordered_failed_nodes or not isinstance(
                    node_result, dict
                ):
                    continue
                evidence_ref = node_result.get("evidence_ref")
                if isinstance(evidence_ref, str) and evidence_ref:
                    diagnostic_evidence.append(
                        f"DCGM evidence for {node_id}: {evidence_ref}"
                    )
                recommendations = node_result.get("recommended_actions", [])
                if not isinstance(recommendations, list):
                    continue
                for recommendation in recommendations:
                    if not isinstance(recommendation, dict):
                        continue
                    action_code = recommendation.get("action_code")
                    instruction = recommendation.get("instruction")
                    if not (
                        isinstance(action_code, str)
                        and action_code
                        and isinstance(instruction, str)
                        and instruction
                    ):
                        continue
                    trigger_tests = recommendation.get("trigger_tests", [])
                    tests = (
                        ", ".join(str(item) for item in trigger_tests)
                        if isinstance(trigger_tests, list)
                        else "unknown"
                    )
                    diagnostic_guidance.append(
                        f"DCGM guidance for {node_id} "
                        f"[{action_code}] ({tests}): "
                        f"{instruction}"
                    )
        return {
            "ordered_failed_nodes": ordered_failed_nodes,
            "workload_ids": workload_ids,
            "gpu_uuids": gpu_uuids,
            # Per-node GPU scope of the failed nodes: node-action steps of
            # the replacement plan carry it so a multi-node step is not
            # refused by the agent barrier (F-H1).
            "gpu_uuids_by_node": {
                node_id: list(dict.fromkeys(gpu_uuids_by_node[node_id]))
                for node_id in ordered_failed_nodes
                if gpu_uuids_by_node.get(node_id)
            },
            "validation_gpu_uuids_by_node": validation_gpus,
            "failed_operations": failed_operations,
            "failed": failed,
            "node_reason": node_reason,
            "diagnostic_guidance": diagnostic_guidance,
            "diagnostic_evidence": diagnostic_evidence,
        }

    def _compile_replacement_steps(
        self,
        workflow: WorkflowRequest,
        operations: list[WorkflowOperation],
        workload_scope: list[str],
        ordered_failed_nodes: list[str],
        gpu_uuids: list[str],
        workload_ids: list[str],
        gpu_uuids_by_node: dict[str, list[str]] | None = None,
        validation_gpu_uuids_by_node: dict[str, list[str]] | None = None,
    ) -> tuple[list, list[str]]:
        profile = None
        errors = []
        if workflow.runtime_profile_version:
            try:
                profile = self.store.get_profile(workflow.runtime_profile_version)
            except NotFoundError:
                errors.append(
                    "runtime profile does not exist: "
                    + workflow.runtime_profile_version
                )
        else:
            errors.append("runtime_profile_version is required for execution")
        replacement_steps = []
        inventory_parameters_by_operation: dict[WorkflowOperation, dict] = {}
        restart_source_steps = [
            source_step
            for source_step in workflow.official_steps
            if source_step.operation is WorkflowOperation.RESTART_WORKLOAD
        ]
        restart_contexts = [
            {name: source_step.parameters[name] for name in RESTART_SAFETY_PARAMETERS}
            for source_step in restart_source_steps
            if all(name in source_step.parameters for name in RESTART_SAFETY_PARAMETERS)
        ]
        restart_context = restart_contexts[0] if restart_contexts else None
        restart_context_error = None
        if len(restart_contexts) != len(restart_source_steps):
            restart_context_error = (
                "failed workflow has no complete restart safety context"
            )
            restart_context = None
        elif any(value != restart_context for value in restart_contexts[1:]):
            restart_context_error = (
                "failed workflow has inconsistent restart safety context"
            )
            restart_context = None
        if restart_context_error is not None:
            errors.append(restart_context_error)
        for source_step in workflow.official_steps:
            requirements = source_step.parameters.get("inventory_requirements_by_node")
            if (
                source_step.operation
                in {
                    WorkflowOperation.VALIDATE_GPU,
                    WorkflowOperation.VALIDATE_FABRIC,
                }
                and isinstance(requirements, dict)
                and requirements
            ):
                inventory_parameters_by_operation[source_step.operation] = {
                    "inventory_requirements_by_node": {
                        node_id: requirement
                        for node_id, requirement in requirements.items()
                        if node_id in ordered_failed_nodes
                    }
                }
        workload_operations = {
            WorkflowOperation.CHECKPOINT_WORKLOADS,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowOperation.RESTART_WORKLOAD,
        }
        for operation in operations:
            compiled, compile_errors = self.builder.compile_steps(
                [operation],
                profile,
                (
                    workload_scope
                    if operation in workload_operations
                    else ordered_failed_nodes
                ),
                gpu_uuids,
                workload_ids,
            )
            replacement_steps.extend(compiled)
            errors.extend(compile_errors)
        normalized_steps = []
        missing_restart_context_reported = restart_context_error is not None
        for step in replacement_steps:
            parameters = dict(step.parameters)
            if step.operation is WorkflowOperation.REPLACE_NODE:
                parameters["replacement_strategy"] = "HEALTHY_WARM_SPARE_ONLY"
            if step.operation is WorkflowOperation.RESTART_WORKLOAD:
                if restart_context is None:
                    if not missing_restart_context_reported:
                        errors.append(
                            "failed workflow has no complete restart safety context"
                        )
                        missing_restart_context_reported = True
                else:
                    parameters.update(restart_context)
            inventory_parameters = inventory_parameters_by_operation.get(step.operation)
            if inventory_parameters:
                parameters.update(inventory_parameters)
            if step.operation in NODE_ACTION_SCOPE_OPERATIONS and gpu_uuids_by_node:
                scoped = {
                    node_id: list(gpu_uuids_by_node[node_id])
                    for node_id in step.node_ids
                    if gpu_uuids_by_node.get(node_id)
                }
                if scoped:
                    parameters["gpu_uuids_by_node"] = scoped
            normalized_steps.append(step.model_copy(update={"parameters": parameters}))
        return scope_validation_steps(
            normalized_steps, validation_gpu_uuids_by_node or {}
        ), errors

    def emit(
        self,
        workflow: WorkflowRequest,
        source: FaultIncident,
        *,
        failed_stage: str,
        next_action: RecoveryAction,
        next_operation: WorkflowOperation | None,
        escalation_name: str,
        event_id: str,
        scope: dict,
    ) -> tuple[FaultIncident, WorkflowRequest]:
        ordered_failed_nodes = scope["ordered_failed_nodes"]
        workload_ids = scope["workload_ids"]
        gpu_uuids = scope["gpu_uuids"]
        failed = scope["failed"]
        node_reason = scope["node_reason"]
        diagnostic_guidance = scope["diagnostic_guidance"]
        diagnostic_evidence = scope["diagnostic_evidence"]
        now = datetime.now(timezone.utc)
        incident = FaultIncident(
            incident_id=f"inc-{event_id}",
            event_id=event_id,
            # The escalation is a continuation of the source incident: it
            # reports the policy, workload and drill that produced it (F-H1).
            event_type=source.event_type,
            event_source=source.event_source,
            source_boot_id=source.source_boot_id,
            cluster_id=source.cluster_id,
            node_ids=ordered_failed_nodes,
            gpu_uuids=gpu_uuids,
            job_id=source.job_id,
            attempt_id=source.attempt_id,
            workload_identity_source=source.workload_identity_source,
            policy_version=source.policy_version,
            policy_source=source.policy_source,
            policy_reference=source.policy_reference,
            drill_id=source.drill_id,
            effective_action=next_action,
            state=IncidentState.DETECTED,
            reasons=[
                f"{failed_stage} remediation failed; "
                f"failed operation or validation: {failed}; "
                f"nodes: {node_reason}",
                *withheld_escalation_reasons(workflow),
                *list(dict.fromkeys(diagnostic_evidence)),
                *list(dict.fromkeys(diagnostic_guidance)),
            ],
            created_at=now,
            updated_at=now,
        )
        operations = _escalation_operations(
            failed_stage, next_action, next_operation, workload_ids
        )

        workload_scope = list(
            dict.fromkeys(
                node_id
                for step in workflow.official_steps
                if step.operation
                in {
                    WorkflowOperation.CHECKPOINT_WORKLOADS,
                    WorkflowOperation.STOP_WORKLOADS,
                    WorkflowOperation.RESTART_WORKLOAD,
                }
                for node_id in step.node_ids
            )
        ) or list(source.node_ids)
        replacement_steps, errors = self._compile_replacement_steps(
            workflow,
            operations,
            workload_scope,
            ordered_failed_nodes,
            gpu_uuids,
            workload_ids,
            scope.get("gpu_uuids_by_node") or None,
            scope.get("validation_gpu_uuids_by_node"),
        )
        replacement = WorkflowRequest(
            request_id=(f"workflow-{escalation_name}-after-{workflow.request_id}"),
            incident_id=incident.incident_id,
            runtime_profile_version=(workflow.runtime_profile_version),
            status=(WorkflowStatus.PENDING if not errors else WorkflowStatus.BLOCKED),
            official_action=next_action.value,
            fencing_token=incident.fencing_token,
            official_steps=replacement_steps,
            blocked_reasons=errors,
            blocked_kind=(BlockedKind.NEEDS_OPERATOR if errors else None),
            lifetime_deadline_at=_successor_lifetime(
                workflow, next_action, next_operation, now
            ),
            created_at=now,
            updated_at=now,
        )
        incident = incident.model_copy(
            update={
                "state": (
                    IncidentState.ACTION_PENDING
                    if not errors
                    else IncidentState.ESCALATED
                ),
                "workflow_request_id": replacement.request_id,
            }
        )
        # One transaction, keyed by the deterministic event id: two workers
        # escalating the same failed workflow get the same pair, and a
        # workflow never exists ahead of its incident (F-H1 / F-B1).
        created_incident, created_workflow, created = (
            self.store.create_incident_workflow_if_absent(
                event_id, lambda: (incident, replacement)
            )
        )
        if created and failed_stage == CONTAINMENT_REFUSED_STAGE:
            self.containment_refused_escalations_total += 1
        return created_incident, created_workflow

    def _terminate_chain(
        self,
        workflow: WorkflowRequest,
        source: FaultIncident,
        origin: tuple[str, str],
    ) -> None:
        """Hand a failed operator hand-off to an operator without opening
        another one.

        ``source`` is itself the product of a terminal escalation (a support
        ticket) and the workflow that was to deliver it failed. Escalating that
        failure would mint one more FAILED incident/workflow pair on the same
        node every reconcile tick (ARCH-E2E-2A finding 1). Instead the incident
        stays ESCALATED with the reason recorded, the operator receives the
        hardware-escalation notification the ESCALATE_SUPPORT step would have
        sent (same builder, same deduplication key, so a step that did run
        never doubles it), and the event is counted. Idempotent per incident:
        a replay finds the reason and the notification already there.
        """

        failed_operations = sorted(
            {
                execution.operation.value
                for execution in workflow.step_executions
                if execution.status is WorkflowStepStatus.FAILED
            }
        )
        reason = _chain_termination_reason(workflow, source, origin, failed_operations)
        first_termination = reason not in source.reasons
        if first_termination or source.state is not IncidentState.ESCALATED:
            now = datetime.now(timezone.utc)
            self.store.save_incident(
                source.model_copy(
                    update={
                        "state": IncidentState.ESCALATED,
                        "reasons": bounded_reasons([*source.reasons, reason]),
                        "updated_at": now,
                    }
                ),
                expected=source,
            )
        workload_ids = list(
            dict.fromkeys(
                workload_id
                for step in workflow.official_steps
                for workload_id in step.workload_ids
            )
        )
        lineage = [workflow]
        with suppress(NotFoundError, KeyError, TypeError):
            lineage.append(self.store.get_workflow(origin[1]))
        notification = HardwareEscalationEmailBuilder().build(
            cluster_id=source.cluster_id,
            incident_id=source.incident_id,
            workflow_id=workflow.request_id,
            event_id=source.event_id,
            event_type=source.event_type,
            node_ids=list(source.node_ids),
            workload_ids=workload_ids,
            reasons=[*source.reasons, reason],
            failed_operations=failed_operations,
            policy_source=source.policy_source,
            official_action=source.official_action,
            ticket_id=f"vendor-ticket-{source.incident_id}",
            held_node_ids=held_nodes(lineage, source.node_ids),
        )
        self.store.save_notification_if_absent(notification)
        if first_termination:
            self.escalation_chain_terminated_total += 1
            self.escalation_chain_terminated_last_seen_timestamp_seconds = time.time()

    def escalate(
        self, workflow: WorkflowRequest
    ) -> tuple[FaultIncident, WorkflowRequest] | None:
        if workflow.status is not WorkflowStatus.FAILED:
            return None
        source = self.store.get_incident(workflow.incident_id)
        origin = escalation_origin(source.event_id)
        if origin is not None and origin[0] in CHAIN_TERMINAL_ESCALATIONS:
            # Bound on the escalation chain: a failed support escalation is
            # handed over, never escalated into another support escalation,
            # whatever its failed step would otherwise classify as.
            self._terminate_chain(workflow, source, origin)
            return None
        classification = self.classify(workflow)
        if classification is None:
            return None
        (
            failed_stage,
            next_action,
            next_operation,
            failed_executions,
        ) = classification
        escalation_name = ESCALATION_NAMES[next_action]
        event_id = (
            f"{escalation_name}{_ESCALATION_EVENT_SEPARATOR}{workflow.request_id}"
        )
        existing = self.store.get_incident_by_event(event_id)
        if existing is not None:
            if not existing.workflow_request_id:
                return None
            try:
                return (
                    existing,
                    self.store.get_workflow(existing.workflow_request_id),
                )
            except NotFoundError:
                # C-04: the escalation was recorded but its workflow row is
                # gone; ``emit`` rebuilds through
                # ``create_incident_workflow_if_absent``, which treats the
                # dirty link the same way and re-links this event id.
                note_stale_event_link(
                    self.store,
                    event_id=event_id,
                    incident_id=existing.incident_id,
                    pointer=existing.workflow_request_id,
                )
        scope = self.collect_scope(
            workflow,
            source,
            self._exhausted_branch_executions(workflow, failed_executions),
            self.store,
        )
        return self.emit(
            workflow,
            source,
            failed_stage=failed_stage,
            next_action=next_action,
            next_operation=next_operation,
            escalation_name=escalation_name,
            event_id=event_id,
            scope=scope,
        )
