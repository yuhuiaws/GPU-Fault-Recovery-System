"""What a FAILED step's record says about its outcome, and the hand-off it owes.

Both ladders read the outcome predicates -- ``orchestration/escalation.py`` for
the whole record (which re-exports them under their old names) and
``execution/branch_escalation.py`` for one branch -- and the node-action and
Kubernetes adapters stamp and read the unknown-outcome flags. They live here,
below the classifier, so the hand-off rule a held chain is owed
(:func:`held_chain_hand_off`) can read them too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gpu_fault.handoff_hold import record_held_nodes
from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
)
from gpu_fault.orchestration.validation_evidence import missing_evidence_reason

#: ``remote_status_source`` of a command the cluster executor refused before any
#: adapter ran it (``cluster_executor.dispatch._validate``: cluster mismatch, a
#: stale fencing token, no or an ambiguous adapter, a namespace outside the
#: executor's scope). Deterministic per command and never a node failure.
EXECUTOR_REJECTED_STATUS_SOURCE = "executor-rejected"
#: The step-detail flags of a failure whose outcome nobody knows; the three
#: producers are named at ``escalation.MANUAL_CONFIRMATION_STAGE``.
UNKNOWN_OUTCOME_FLAGS = (
    "manual_confirmation_required",
    "node_action_interrupted",
    "outcome_unknown",
)
# Unknown-outcome failures that must not open a successor: a second support
# case for a support escalation the executor abandoned, or a support case for
# an evidence freeze / checkpoint that may still be running inside the job.
NO_SECOND_TICKET_OPERATIONS = frozenset(
    {
        WorkflowOperation.ESCALATE_SUPPORT,
        WorkflowOperation.FREEZE_EVIDENCE,
        WorkflowOperation.CHECKPOINT_WORKLOADS,
    }
)
#: The classification stage of a held chain's step failure (see
#: :func:`held_chain_hand_off`).
HELD_CHAIN_STEP_STAGE = "held_chain_step"
#: The step detail :func:`held_scope` writes on each failure it hands over:
#: why each of its nodes is held, which the hand-off's reason gives for a node
#: with no finding of its own (``escalation.HardwareEscalationService
#: .collect_scope``).
HELD_NODE_REASONS_DETAIL = "held_node_reasons"


def executor_rejected_failure(details: Mapping[str, Any] | None) -> bool:
    """Whether a FAILED step never reached its node: the executor refused the
    command before dispatch.

    Neither ladder climbs on it. Live (gpu-eks-cluster, XID 46 reset): the
    reset chain's generation was fenced by a successor's plan rewrite, the
    RESTART_NODE rung was refused "command fencing token does not match
    workflow/incident", the branch ladder appended REPLACE_NODE, that was
    refused the same way, and the compensation restore too -- three rungs
    spent on a record whose incident had already moved on. The refusal names
    the stale generation; nothing about the hardware was learned.
    """

    return (
        details is not None
        and details.get("remote_status_source") == EXECUTOR_REJECTED_STATUS_SOURCE
    )


#: The step-detail key the Node Agent's GPU-client check writes when it
#: refuses (``node_agent/operations/clients.GpuClientsActive``: ``agent``,
#: ``compute`` or ``device`` holders), carried into the step record by the
#: node-action fold (``adapters/node_action/verification_continuation
#: .VERIFY_EVIDENCE_KEYS``).
GPU_CLIENT_KIND_DETAIL = "gpu_client_kind"


def gpu_clients_refused(details: Mapping[str, Any] | None) -> bool:
    """Whether a FAILED node action was refused by the agent's GPU-client
    check: it found GPU clients on its targets. The Fabric Manager restart
    runs that check before it touches the unit, so such a restart never ran
    -- and the GPUs are in use on a node nothing stopped."""

    return details is not None and bool(details.get(GPU_CLIENT_KIND_DETAIL))


#: The step detail the node-action fold writes on a FAILED RESTART_FABRIC_MANAGER
#: the Node Agent answered (``adapters/node_action/fabric_manager_outcome``):
#: what the agent's verdict says about the Fabric Manager unit, one of
#: :data:`FABRIC_MANAGER_RESTART_OUTCOMES`. A failure without it has no such
#: verdict: it was refused or lost before the agent's handler answered (the
#: fleet gate, an allowlist, the transport, a step cap or deadline), or it
#: carries flags of its own that are judged before the ladder (an attested
#: safety refusal, an unknown outcome).
FABRIC_MANAGER_RESTART_OUTCOME_DETAIL = "fabric_manager_restart_outcome"
#: A ``systemctl`` call on the unit exited non-zero once the agent's own checks
#: had passed -- the restart, the active check after it, or a read of the unit:
#: systemd could not restart the unit or answer for it.
FM_RESTART_UNIT_FAILED = "UNIT_FAILED"
#: The node's own configuration does not allow the restart: it never ran.
FM_RESTART_REFUSED_BY_NODE_CONFIGURATION = "REFUSED_BY_NODE_CONFIGURATION"
#: The agent's GPU-client check found clients on the GPUs: it never ran.
FM_RESTART_REFUSED_GPU_CLIENTS = "REFUSED_GPU_CLIENTS"
#: Any other answer: a read before the unit failed (the compute-apps query),
#: the re-submits ran out on a retryable error, an error that names no unit
#: call. Nothing in it says the unit failed.
FM_RESTART_NO_UNIT_VERDICT = "NO_UNIT_VERDICT"
FABRIC_MANAGER_RESTART_OUTCOMES = frozenset(
    {
        FM_RESTART_UNIT_FAILED,
        FM_RESTART_REFUSED_BY_NODE_CONFIGURATION,
        FM_RESTART_REFUSED_GPU_CLIENTS,
        FM_RESTART_NO_UNIT_VERDICT,
    }
)
#: How much of a failed step's error a no-verdict reason quotes.
REASON_ERROR_LIMIT = 200


def fabric_manager_unit_failed(details: Mapping[str, Any] | None) -> bool:
    """Whether a FAILED RESTART_FABRIC_MANAGER is the agent's report that the
    unit failed: the only failure of the restart that is evidence for the
    reboot rung. Every other one never ran the restart -- the operator's
    configuration or the GPU-client check refused it, a read before the unit
    failed, the command never reached the agent's handler -- or says nothing
    about the unit; on the solo plan's node, which nothing stopped, none of
    them is a reason to reboot."""

    return (
        details is not None
        and details.get(FABRIC_MANAGER_RESTART_OUTCOME_DETAIL) == FM_RESTART_UNIT_FAILED
        and not gpu_clients_refused(details)
    )


def fabric_manager_restart_reason(execution: WorkflowStepExecution) -> str | None:
    """What a hand-off says for a FAILED RESTART_FABRIC_MANAGER that is not
    rebooted (:func:`fabric_manager_unit_failed`): why it never ran, or that
    nothing says the unit failed; ``None`` for any other step and for a unit
    that failed (whose reboot reads ``RESTART_FABRIC_MANAGER failed``)."""

    details = execution.details
    if execution.operation is not WorkflowOperation.RESTART_FABRIC_MANAGER or (
        fabric_manager_unit_failed(details)
    ):
        return None
    outcome = details.get(FABRIC_MANAGER_RESTART_OUTCOME_DETAIL)
    if gpu_clients_refused(details):
        return (
            f"RESTART_FABRIC_MANAGER refused: GPU {details[GPU_CLIENT_KIND_DETAIL]} "
            "clients active on a node nothing stopped; the restart never ran"
        )
    if outcome == FM_RESTART_REFUSED_BY_NODE_CONFIGURATION:
        return (
            "RESTART_FABRIC_MANAGER refused by the node configuration (Fabric "
            "Manager restart not enabled); the restart never ran"
        )
    cause = (execution.error or "no error recorded")[:REASON_ERROR_LIMIT]
    return f"RESTART_FABRIC_MANAGER failed with no verdict on the unit: {cause}"


#: The step detail the node lifecycle adapter writes when it refuses a
#: RESTART_NODE / REPLACE_NODE before any provider call because the scheduler
#: does not show the node isolated by this incident right now
#: (``adapters/hyperpod/lifecycle._observe_isolation``, HyperPod and plain
#: EKS): :data:`ISOLATION_NOT_ISOLATED` -- the node was read and its cordon,
#: its quarantine taint, its incident or its fencing-token annotation is not
#: this workflow's; :data:`ISOLATION_UNOBSERVABLE` -- no Kubernetes node
#: answers to it, or the adapter has no Kubernetes client to read it with.
#: Nothing about the hardware was learned: the escalation puts the isolation
#: back and hands the node over (``containment_or_release``), or hands it
#: over alone when it cannot be observed at all (``containment_refused``),
#: instead of climbing from the reboot to the replacement.
LIFECYCLE_ISOLATION_DETAIL = "isolation_not_observed"
ISOLATION_NOT_ISOLATED = "NOT_ISOLATED"
ISOLATION_UNOBSERVABLE = "UNOBSERVABLE"
LIFECYCLE_OPERATIONS = frozenset(
    {WorkflowOperation.RESTART_NODE, WorkflowOperation.REPLACE_NODE}
)


def lifecycle_isolation_refused(details: Mapping[str, Any] | None) -> str | None:
    """The kind of a node lifecycle refusal for want of observed isolation
    (:data:`LIFECYCLE_ISOLATION_DETAIL`), or ``None``."""

    value = details.get(LIFECYCLE_ISOLATION_DETAIL) if details is not None else None
    return value if value in (ISOLATION_NOT_ISOLATED, ISOLATION_UNOBSERVABLE) else None


def lifecycle_isolation_refusals(
    failed: Sequence[WorkflowStepExecution],
) -> list[str] | None:
    """The refusal kinds when every FAILED RESTART_NODE / REPLACE_NODE of a
    record is such a refusal; ``None`` when none failed or one failed for
    another reason (a provider refusal, a preflight gate, a lost spare: the
    ladder's own rung)."""

    kinds = [
        lifecycle_isolation_refused(item.details)
        for item in failed
        if item.operation in LIFECYCLE_OPERATIONS
    ]
    if not kinds or any(kind is None for kind in kinds):
        return None
    return [kind for kind in kinds if kind is not None]


def unknown_outcome_failure(details: Mapping[str, Any] | None) -> bool:
    """Whether a FAILED step's details say its outcome is not actually known.

    Shared by both ladders: ``_classify`` (whole workflow) and the DAG branch
    escalator's guard, which must stop where this stops -- a rung appended in
    place is the same reboot over a possibly-reset GPU.
    """

    return details is not None and any(
        details.get(flag) is True for flag in UNKNOWN_OUTCOME_FLAGS
    )


def held_chain_hand_off(
    workflow: WorkflowRequest, failed: Sequence[WorkflowStepExecution]
) -> tuple[str, RecoveryAction, WorkflowOperation, list[WorkflowStepExecution]] | None:
    """The support hand-off a FAILED record that holds its node is owed, or
    ``None``.

    The classifier calls this where the ladder has no rung for a failure:
    its last branch, a validation with no recovery step before it, and the
    diagnostic contract's short-circuit. A record whose own cordon or
    quarantine still holds a node then needs an operator, whatever the step:
    the chains that hand the node over themselves (DRAIN, the
    ENVIRONMENT_REVIEW chain, an operator escalation) whose diagnostic
    bundle, health snapshot, job stop or checkpoint failed before their
    ESCALATE_SUPPORT; the plans whose end *is* the hold (the node-health
    QUARANTINE plan, a terminal quarantine folded into another chain, a
    containment prefix) whose job stop failed after the cordon; the release
    chains -- a reset, a reboot, the older cooldown plan -- whose job stop,
    GPU-client check, answered checkpoint or cooldown validation, snapshot or
    bundle failed after the cordon; and a validate-then-readmit whose
    validation failed after the cordon with no recovery step before it (no
    recovery's verdict, so no rung). Each used to end FAILED + QUARANTINED
    with no support record and no mail. The answer is
    ``(HELD_CHAIN_STEP_STAGE, ESCALATE_OPERATOR, ESCALATE_SUPPORT, failed)``:
    a support-after successor that keeps the hold and sends the mail.

    Not owed -- the endings that open no ticket on purpose: no hold left (a
    read-only chain, a node a RESTORE_SCHEDULING released); a hand-off of the
    record that already ran or failed (:func:`owes_hand_off`); every failure
    refused by the executor before dispatch (a stale generation's record: a
    newer one owns the incident); an abandoned step of
    :data:`NO_SECOND_TICKET_OPERATIONS` (the support escalation, the evidence
    freeze, the checkpoint that may still be running inside the job).

    The hand-off keeps the chain's hold and never extends it: each failure's
    ``failed_nodes`` is narrowed to the nodes the record holds
    (:func:`held_scope`), so a job-wide STOP or CHECKPOINT -- compiled for
    the whole allocation -- hands over the faulty node, not the job's healthy
    ones.
    """

    if (
        not failed
        or not owes_hand_off(workflow)
        or all(executor_rejected_failure(item.details) for item in failed)
        or any(
            item.operation in NO_SECOND_TICKET_OPERATIONS
            and unknown_outcome_failure(item.details)
            for item in failed
        )
    ):
        return None
    scoped = held_scope(workflow, failed)
    if not scoped:
        return None
    return (
        HELD_CHAIN_STEP_STAGE,
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
        scoped,
    )


def owes_hand_off(workflow: WorkflowRequest) -> bool:
    """Whether a FAILED record's hold is owed a support hand-off: no
    ESCALATE_SUPPORT of the record ran. A hand-off that ran or failed keeps
    its own ending -- a second ticket answers nothing. Whether the record
    still holds a node is :func:`held_scope`'s.
    """

    return not any(
        execution.operation is WorkflowOperation.ESCALATE_SUPPORT
        for execution in workflow.step_executions
    )


def reported_failed_nodes(
    details: Mapping[str, Any], step_nodes: Sequence[str]
) -> list[str]:
    """The nodes of ``step_nodes`` a FAILED step says it failed on: its
    ``failed_nodes``, else the ``pending_nodes`` of a step a bound failed while
    it waited, else -- none of them on the step -- every node of the step."""

    reported = details.get("failed_nodes")
    if not isinstance(reported, list):
        reported = details.get("pending_nodes")
    if not isinstance(reported, list):
        return list(step_nodes)
    return [node for node in reported if node in step_nodes] or list(step_nodes)


def node_failure_reasons(execution: WorkflowStepExecution, node_id: str) -> list[str]:
    """What a hand-off's reason says for ``node_id`` of FAILED ``execution``.

    Its findings (``node_failures``); else, for a validation still waiting on
    the node, what kept it from looking (``node_pending``: the adapter's
    reason and the operator's exit when it wrote one, else the missing
    evidence itself, ``validation_evidence.missing_evidence_reason``); else
    why the record holds the node (:data:`HELD_NODE_REASONS_DETAIL`); else why
    a Fabric Manager restart is not rebooted
    (:func:`fabric_manager_restart_reason`); else the failed step's own
    ``<OPERATION> failed`` -- never a validation the node may have passed.
    """

    details = execution.details
    reported = details.get("node_failures")
    reasons = reported.get(node_id, []) if isinstance(reported, Mapping) else []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    pending = details.get("node_pending")
    record = pending.get(node_id) if isinstance(pending, Mapping) else None
    if not reasons and isinstance(record, Mapping):
        reason = missing_evidence_reason(record)
        if reason:
            reasons = [reason]
    if not reasons:
        held = details.get(HELD_NODE_REASONS_DETAIL)
        held = held if isinstance(held, Mapping) else {}
        reasons = [
            held.get(node_id)
            or fabric_manager_restart_reason(execution)
            or f"{execution.operation.value} failed"
        ]
    return [str(reason) for reason in reasons]


def held_scope(
    workflow: WorkflowRequest, failed: Sequence[WorkflowStepExecution]
) -> list[WorkflowStepExecution]:
    """``failed`` with each failure's ``failed_nodes`` set to the nodes of its
    step the record still holds (``handoff_hold.record_held_nodes``); a
    failure on none of them is dropped.

    The escalation's scope reads ``failed_nodes`` within the step's nodes
    (``escalation._failed_step_nodes``), so the support successor cordons,
    quarantines and names exactly the held nodes. Each failure also says why
    each of those nodes is held (:data:`HELD_NODE_REASONS_DETAIL`): ``held
    after <OPERATION> failed`` for a node the step failed on, ``held;
    <OPERATION> failed on another node`` for one it did not fail on -- a held
    node that passed the validation of a validate-then-readmit -- so the
    hand-off never reports a node that passed as failing.
    """

    held = set(record_held_nodes(workflow))
    steps = workflow.official_steps
    scoped = []
    for item in failed:
        step_nodes = (
            steps[item.step_index].node_ids if 0 <= item.step_index < len(steps) else []
        )
        nodes = [node for node in step_nodes if node in held]
        if nodes:
            failed_on = set(reported_failed_nodes(item.details, step_nodes))
            operation = item.operation.value
            reasons = {
                node: (
                    f"held after {operation} failed"
                    if node in failed_on
                    else f"held; {operation} failed on another node"
                )
                for node in nodes
            }
            scoped.append(
                item.model_copy(
                    update={
                        "details": {
                            **item.details,
                            "failed_nodes": nodes,
                            HELD_NODE_REASONS_DETAIL: reasons,
                        }
                    }
                )
            )
    return scoped
