"""What a client-verification WAITING record keeps under a pending poll.

A ``VERIFY_NO_GPU_CLIENTS`` attempt the agent accepted but has not finished
when the executor's poll window ends comes back as a WAITING with
``node_action_state: PENDING``. ``complete_remote_command`` replaces the
step's record with it, and a compound command replaces the step's
``batched_results`` entry the same way. Without the previous verdict's keys the
next cycle's ``_verify_attempt`` finds no attempt and restarts the 60-attempt
bound (live, eks-first site: the counter went 11 -> 3 under one pending poll),
and the record a preemption freezes mid-poll no longer says the step was
parked on live GPU clients -- the barrier contract of the preempting-reboot
case. The pending poll therefore carries the last verdict's continuation
forward; its own keys win.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import WorkflowStepStatus

__all__ = [
    "VERIFY_CONTINUATION_KEYS",
    "VERIFY_EVIDENCE_KEYS",
    "verification_continuation",
]

#: The agent's evidence on who held the GPU on an attempt.
VERIFY_EVIDENCE_KEYS = (
    "gpu_client_kind",
    "gpu_client_holders",
    "gpu_client_holders_total",
    "transient_device_clients",
)
#: Everything a client-verification WAITING carries that a later cycle reads:
#: the attempt the bound counts, the reason the step is parked on, the node it
#: waits for, the nodes already folded, and the holder evidence.
VERIFY_CONTINUATION_KEYS = (
    "gpu_client_quiesce_attempt",
    "reason",
    "waiting_node",
    "node_results",
    "completed_nodes",
    *VERIFY_EVIDENCE_KEYS,
)


def verification_continuation(context: WorkflowStepContext) -> dict[str, Any]:
    """The last client-verification verdict this step recorded, or ``{}``.

    Only a WAITING record of this very step (index and operation) that carries
    ``gpu_client_quiesce_attempt`` counts -- the record ``_verify_attempt``
    counts from -- and never for a spare eligibility probe, whose answer is
    definitive and is never written back (``barriers._maintenance_generations``).
    """

    if context.step.parameters.get("spare_health_check"):
        return {}
    previous = next(
        (
            item
            for item in reversed(context.workflow.step_executions)
            if item.step_index == context.step_index
            and item.operation is context.step.operation
            and item.status is WorkflowStepStatus.WAITING
            and "gpu_client_quiesce_attempt" in item.details
        ),
        None,
    )
    if previous is None:
        return {}
    return {
        key: previous.details[key]
        for key in VERIFY_CONTINUATION_KEYS
        if key in previous.details
    }
