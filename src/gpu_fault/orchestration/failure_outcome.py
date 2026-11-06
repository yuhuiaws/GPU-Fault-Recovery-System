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

from gpu_fault.models import (
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
)
from gpu_fault.orchestration.validation_evidence import held_for_unsent_hand_off

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
    """The support hand-off a FAILED held chain is owed, or ``None``.

    A chain that holds its node for an operator -- the DRAIN chain, the
    ENVIRONMENT_REVIEW chain of a confirmed thermal drain, an operator
    escalation -- plans its ESCALATE_SUPPORT last. A step before it that fails
    where the ladder has no rung -- the diagnostic bundle, the health snapshot,
    the job stop, a checkpoint that answered -- used to end the record
    QUARANTINED with no support record and no mail; the ENVIRONMENT_REVIEW
    contract keeps every hardware rung away and so ended it there for any
    step. The hand-off is the chain's own, so it is owed whatever the step:
    ``(HELD_CHAIN_STEP_STAGE, ESCALATE_OPERATOR, ESCALATE_SUPPORT, failed)``,
    a support-after successor that keeps the hold and sends the mail.

    Not owed -- the endings that open no ticket on purpose: no hold or no
    planned hand-off (``validation_evidence.held_for_unsent_hand_off``: a
    read-only chain, a cooldown plan, a hand-off that already ran or failed);
    every failure refused by the executor before dispatch (a stale
    generation's record: a newer one owns the incident); an abandoned step of
    :data:`NO_SECOND_TICKET_OPERATIONS` (the support escalation, the evidence
    freeze, the checkpoint that may still be running inside the job).
    """

    if (
        not failed
        or not held_for_unsent_hand_off(workflow)
        or all(executor_rejected_failure(item.details) for item in failed)
        or any(
            item.operation in NO_SECOND_TICKET_OPERATIONS
            and unknown_outcome_failure(item.details)
            for item in failed
        )
    ):
        return None
    return (
        HELD_CHAIN_STEP_STAGE,
        RecoveryAction.ESCALATE_OPERATOR,
        WorkflowOperation.ESCALATE_SUPPORT,
        list(failed),
    )
