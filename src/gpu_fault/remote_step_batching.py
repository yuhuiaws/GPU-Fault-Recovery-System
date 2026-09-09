"""One compound remote command for a contiguous run of node-side steps (性能 C).

A GPU reset chain is MARK_UNSCHEDULABLE → QUIESCE_GPU_SERVICES →
VERIFY_NO_GPU_CLIENTS → RESET_GPU → RESTORE_GPU_SERVICES → VALIDATE_GPU →
RESTORE_SCHEDULING. The four node-side steps in the middle each used to cost a
full control-plane ↔ executor round trip (mint, claim, execute, report, next
dispatcher tick) although they run on one node through one local adapter, and
the maintenance-window and agent-generation checks between them read the
QUIESCE step's details from the workflow copy the command carries -- nothing on
the control plane sits between them. So the control plane may hand the executor
the whole run as one command, and the executor replays exactly the per-step
contexts today's four commands would have carried
(``gpu_fault.cluster_executor_batching``).

Rejected alternative: a multi-action command in the node protocol. It would
save the same round trips but widen the signed agent API and the agent's
ledger; the node side stays untouched here, which is the point.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from gpu_fault.env import env_bool
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    resolved_step_indexes,
)
from gpu_fault.regional_compatibility import (
    REMOTE_STEP_BATCHING_PROTOCOL_VERSION,
    RegionalExecutorCompatibilityPolicy,
)
from gpu_fault.remote_command_models import (
    BATCHED_RESULTS_KEY,
    BatchedStep,
    RemoteCommandStatus,
)
from gpu_fault.store.shared.remote_helpers import OPEN_REMOTE_COMMAND_STATUSES

REMOTE_STEP_BATCHING_ENV = "GPU_FAULT_REMOTE_STEP_BATCHING"

# Single-node, node-action-adapter operations whose only inter-step
# dependency is the QUIESCE evidence the adapter reads from the workflow copy.
# Deliberately absent: the multi-node barrier reset (RESET_ALL_GPUS_NVSWITCHES
# needs the control-plane barrier coordinator), the long diagnostics and
# remediations (RUN_NVLINK74_WORKFLOW, RUN_FIELD_DIAGNOSTIC, REMEDIATE_*,
# UPDATE_SOFTWARE_FIRMWARE) whose WAITING/cancel semantics are their own, and
# everything another adapter owns.
BATCHABLE_OPERATIONS = frozenset(
    {
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
    }
)


@dataclass(frozen=True)
class RemoteStepBatchingPolicy:
    """Whether the control plane may mint compound commands right now.

    Two switches, both required: the operator flag, and the compatibility
    policy admitting no executor older than
    ``REMOTE_STEP_BATCHING_PROTOCOL_VERSION``. The claim response carries
    ``batched_steps`` only on a compound command, and an older executor's
    ``extra="forbid"`` model would refuse it, so the feature stays off while a
    rolling upgrade still accepts the previous version (deploy.sh pins
    ``required=<previous>, compatible=<current>`` for that window).
    """

    enabled: bool
    minimum_executor_protocol_version: int

    @property
    def active(self) -> bool:
        return (
            self.enabled
            and self.minimum_executor_protocol_version
            >= REMOTE_STEP_BATCHING_PROTOCOL_VERSION
        )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> RemoteStepBatchingPolicy:
        values = os.environ if environ is None else environ
        return cls(
            enabled=env_bool(REMOTE_STEP_BATCHING_ENV, True, environ=values),
            minimum_executor_protocol_version=(
                RegionalExecutorCompatibilityPolicy.from_mapping(
                    values
                ).minimum_accepted_version
            ),
        )

    @classmethod
    def disabled(cls) -> RemoteStepBatchingPolicy:
        return cls(enabled=False, minimum_executor_protocol_version=0)


def step_idempotency_key(
    workflow: WorkflowRequest, index: int, step: WorkflowStepSpec
) -> str:
    """The key ``ProductionWorkflowExecutor._dispatch_step`` gives a step.

    Restated here because a batched step is never dispatched by the executor
    loop, yet the node agent's ledger and the adapter's command ids are keyed
    by it. ``batchable_steps`` refuses to batch when the head's key disagrees,
    so a change to the executor's format cannot silently desynchronise them.
    """

    return f"{workflow.request_id}/{index}/{step.operation.value}"


def _batchable(step: WorkflowStepSpec, head: WorkflowStepSpec) -> bool:
    return (
        step.operation in BATCHABLE_OPERATIONS
        and len(step.node_ids) == 1
        and step.node_ids == head.node_ids
        and step.execution_owner == head.execution_owner
        and step.branch_id is None
        and not step.depends_on_step_indexes
    )


def batchable_steps(
    context: WorkflowStepContext,
    *,
    supports: Callable[[WorkflowStepSpec], bool],
) -> list[BatchedStep]:
    """The steps that ride along with ``context.step``, or ``[]``.

    The predicate: the workflow is not a DAG; the head and every following step
    (by index, contiguous, stopping at the first that fails) is in
    ``BATCHABLE_OPERATIONS``, targets exactly the head's one node under the
    head's ``execution_owner``, carries no ``branch_id`` and no
    ``depends_on_step_indexes``, is not yet resolved, and is one ``supports``
    (the calling adapter's own). A run of one is not a batch.
    """

    workflow = context.workflow
    if workflow.dag_enabled:
        return []
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    head = context.step
    if (
        not 0 <= context.step_index < len(steps)
        or steps[context.step_index] != head
        or not _batchable(head, head)
        or context.idempotency_key
        != step_idempotency_key(workflow, context.step_index, head)
    ):
        return []
    resolved = resolved_step_indexes(workflow)
    batched: list[BatchedStep] = []
    for index in range(context.step_index + 1, len(steps)):
        step = steps[index]
        if index in resolved or not _batchable(step, head) or not supports(step):
            break
        batched.append(
            BatchedStep(
                step_index=index,
                step=step,
                idempotency_key=step_idempotency_key(workflow, index, step),
            )
        )
    return batched


def batched_result_entry(command: Any, step_index: int) -> dict[str, Any] | None:
    results = command.result_details.get(BATCHED_RESULTS_KEY)
    if not isinstance(results, dict):
        return None
    entry = results.get(str(step_index))
    return entry if isinstance(entry, dict) else None


def covered_step_outcome(command: Any, step_index: int) -> WorkflowStepOutcome | None:
    """Map step ``step_index``'s share of compound ``command`` to an outcome.

    ``None`` means the command does not settle this step: it is terminal and
    never reached the step (it failed, or was cancelled, at an earlier one).
    The caller then mints afresh -- the step never started on the node, so a
    new command is safe, and the RESTORE_GPU_SERVICES compensation the chain
    dispatches after a failed RESET still runs on the node as it does today.
    Answering "earlier batched step failed" instead would have turned that
    compensation into a recorded failure and left the node quiesced until the
    agent's fail-safe timer.
    """

    operation_id = f"remote/{command.command_id}"
    entry = batched_result_entry(command, step_index)
    if entry is not None:
        status = entry.get("status")
        details = entry.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        source = entry.get("status_source")
        if status == RemoteCommandStatus.SUCCEEDED.value:
            return WorkflowStepOutcome.succeeded(
                operation_id=operation_id, details=details
            )
        if status == RemoteCommandStatus.FAILED.value:
            return WorkflowStepOutcome.failed(
                str(entry.get("error") or "remote cluster action failed"),
                details={
                    **details,
                    **({"remote_status_source": source} if source else {}),
                },
            )
    if command.status in OPEN_REMOTE_COMMAND_STATUSES:
        return WorkflowStepOutcome.waiting(
            operation_id=operation_id,
            details={
                "remote_command_id": command.command_id,
                "remote_cluster_id": command.cluster_id,
                "remote_status": command.status.value,
                "batched_step_index": step_index,
                "batched_step_indexes": list(command.covered_step_indexes),
                "mutation_submitted_by_control_plane": False,
            },
        )
    if step_index != command.step_index:
        return None
    # The head maps like a single command: the command's verdict is its own.
    if command.status is RemoteCommandStatus.SUCCEEDED:
        return WorkflowStepOutcome.succeeded(
            operation_id=operation_id, details=command.result_details
        )
    return WorkflowStepOutcome.failed(
        command.error or "remote cluster action failed",
        details={
            **command.result_details,
            **(
                {"remote_status_source": command.status_source}
                if command.status_source is not None
                else {}
            ),
        },
    )
