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


def held_scope(
    workflow: WorkflowRequest, failed: Sequence[WorkflowStepExecution]
) -> list[WorkflowStepExecution]:
    """``failed`` with each failure's ``failed_nodes`` set to the nodes of its
    step the record still holds (``handoff_hold.record_held_nodes``); a
    failure on none of them is dropped.

    The escalation's scope reads ``failed_nodes`` within the step's nodes
    (``escalation._failed_step_nodes``), so the support successor cordons,
    quarantines and names exactly the held nodes.
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
            scoped.append(
                item.model_copy(
                    update={"details": {**item.details, "failed_nodes": nodes}}
                )
            )
    return scoped
