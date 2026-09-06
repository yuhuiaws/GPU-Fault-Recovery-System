"""Where a running workflow may still be superseded (F-C1).

Two callers ask this and used to answer it with different rules. The
executor's ``_supersede_if_safe`` asks at a step boundary, before it hands the
next step to an adapter: may this workflow give way to its preempting successor
right now? The merge service's ``preempt_parallel_branch`` asks at planning
time: which steps of a node branch does a stronger candidate replace, and behind
which step does the new branch chain? The executor judged every WAITING record
by whether its wait could be stopped -- a local wait only for the read-only
validations, a remote command only while the node has not started it or the
operation is a collection. The merge judged only the branch's records, and only
by a registry flag that ignored the command's remote state, the operator
acknowledgements and whether the record still described the step at its index.
A step the executor refused to abandon was one the merge marked superseded.

One pure function now answers both. It reports the remote commands that would
have to be cancelled first; the caller that can cancel them (the executor) does
so and proceeds, the caller that cannot (planning) treats them as a boundary.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS,
    SAFE_WAITING_PREEMPT_OPERATIONS,
)

REMOTE_OPERATION_PREFIX = "remote/"


class WaitingVerdict(Enum):
    """What a WAITING record means for preemption."""

    # A local wait that can simply be abandoned (the read-only validations).
    STOPPABLE = "stoppable"
    # A remote command the node has not started, or a collection it has: safe
    # to cancel, and cancelling it is the price of superseding the step.
    CANCELLABLE = "cancellable"
    # In-flight effect nobody can stop: a destructive remote command past
    # PENDING, an operator acknowledgement, any other local wait.
    BLOCKING = "blocking"
    # The record's operation no longer matches the step at its index: a DAG
    # rewrite replaced the step, and the record is history.
    STALE = "stale"
    # The record points past the end of the step set; nothing can say what,
    # if anything, is running for it.
    MALFORMED = "malformed"


@dataclass(frozen=True)
class WaitingRecord:
    step_index: int
    operation: WorkflowOperation
    verdict: WaitingVerdict
    reason: str
    remote_command_id: str | None = None


@dataclass(frozen=True)
class PreemptionBoundary:
    """The verdict on one workflow (or one branch of it).

    ``indexes`` are the steps under judgement, ``completed`` the ones among
    them that already ran, ``waiting`` every WAITING record in scope with its
    verdict. ``remote_cancellation_available`` says whether the asker can
    cancel a remote command; without that a CANCELLABLE record blocks.
    """

    indexes: frozenset[int]
    completed: frozenset[int]
    waiting: tuple[WaitingRecord, ...]
    remote_cancellation_available: bool

    @property
    def cancellations(self) -> tuple[WaitingRecord, ...]:
        return tuple(
            record
            for record in self.waiting
            if record.verdict is WaitingVerdict.CANCELLABLE
        )

    @property
    def blocking(self) -> frozenset[int]:
        """Steps whose wait holds the boundary closed."""

        held = {WaitingVerdict.BLOCKING}
        if not self.remote_cancellation_available:
            held.add(WaitingVerdict.CANCELLABLE)
        return frozenset(
            record.step_index for record in self.waiting if record.verdict in held
        )

    @property
    def malformed(self) -> frozenset[int]:
        return frozenset(
            record.step_index
            for record in self.waiting
            if record.verdict is WaitingVerdict.MALFORMED
        )

    @property
    def stale(self) -> frozenset[int]:
        return frozenset(
            record.step_index
            for record in self.waiting
            if record.verdict is WaitingVerdict.STALE
        )

    @property
    def open(self) -> bool:
        """May the steps under judgement be superseded now?"""

        return not self.blocking and not self.malformed

    @property
    def protected(self) -> frozenset[int]:
        """Steps a successor must leave alone: they ran, or cannot be stopped."""

        return self.completed | self.blocking

    @property
    def replaceable(self) -> frozenset[int]:
        return self.indexes - self.protected

    @property
    def predecessor(self) -> int | None:
        """The step a replacing branch chains behind: the last one that holds
        the boundary, else the last one that completed, else nothing."""

        if self.blocking:
            return max(self.blocking)
        if self.completed:
            return max(self.completed)
        return None

    @property
    def first_supersedable(self) -> int | None:
        """The first step that may still be superseded, or ``None`` while the
        boundary is closed."""

        if not self.open or not self.replaceable:
            return None
        return min(self.replaceable)

    @property
    def reason(self) -> str:
        for record in self.waiting:
            if record.verdict is WaitingVerdict.MALFORMED or (
                record.step_index in self.blocking
            ):
                return record.reason
        return "every waiting step can be stopped"


def preemption_boundary(
    workflow: WorkflowRequest,
    *,
    indexes: Collection[int] | None = None,
    remote_cancellation_available: bool = True,
) -> PreemptionBoundary:
    """Judge ``workflow``'s WAITING records against the steps it executes.

    ``indexes`` narrows the judgement to a branch: records outside it are not
    read at all. Without it the whole step set is judged, and a record that
    points outside the step set closes the boundary -- the executor's reading,
    since it cannot tell what such a record has running.
    """

    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    scoped = frozenset(indexes) if indexes is not None else frozenset(range(len(steps)))
    completed = frozenset(workflow.completed_step_indexes) & scoped
    waiting: list[WaitingRecord] = []
    for record in workflow.step_executions:
        if record.status is not WorkflowStepStatus.WAITING:
            continue
        if not 0 <= record.step_index < len(steps):
            if indexes is None:
                waiting.append(
                    WaitingRecord(
                        record.step_index,
                        record.operation,
                        WaitingVerdict.MALFORMED,
                        f"step {record.step_index} record lies outside the "
                        f"{len(steps)} steps this workflow executes",
                    )
                )
            continue
        if record.step_index not in scoped:
            continue
        waiting.append(_judge(record, steps[record.step_index]))
    return PreemptionBoundary(
        indexes=scoped,
        completed=completed,
        waiting=tuple(waiting),
        remote_cancellation_available=remote_cancellation_available,
    )


def _judge(record: WorkflowStepExecution, step: WorkflowStepSpec) -> WaitingRecord:
    index = record.step_index
    if record.operation is not step.operation:
        return WaitingRecord(
            index,
            record.operation,
            WaitingVerdict.STALE,
            f"step {index} record for {record.operation.value} no longer matches "
            f"the step's {step.operation.value}",
        )
    operation_id = record.adapter_operation_id or ""
    if operation_id.startswith(REMOTE_OPERATION_PREFIX):
        return _judge_remote(record, operation_id)
    if step.operation in SAFE_WAITING_PREEMPT_OPERATIONS:
        return WaitingRecord(
            index,
            step.operation,
            WaitingVerdict.STOPPABLE,
            f"step {index} {step.operation.value} wait can be abandoned",
        )
    return WaitingRecord(
        index,
        step.operation,
        WaitingVerdict.BLOCKING,
        f"step {index} {step.operation.value} is waiting on a local adapter "
        "and cannot be stopped",
    )


def _judge_remote(record: WorkflowStepExecution, operation_id: str) -> WaitingRecord:
    index = record.step_index
    operation = record.operation
    remote_status = str(record.details.get("remote_status") or "")
    can_cancel = remote_status == "PENDING" or (
        remote_status == "WAITING"
        and operation in SAFE_REMOTE_WAITING_PREEMPT_OPERATIONS
    )
    if not can_cancel:
        return WaitingRecord(
            index,
            operation,
            WaitingVerdict.BLOCKING,
            f"step {index} {operation.value} has a remote command in "
            f"{remote_status or 'an unknown'} state that cannot be cancelled",
        )
    command_id = record.details.get("remote_command_id")
    if not isinstance(command_id, str) or not command_id:
        command_id = operation_id.removeprefix(REMOTE_OPERATION_PREFIX)
    if not command_id:
        return WaitingRecord(
            index,
            operation,
            WaitingVerdict.BLOCKING,
            f"step {index} {operation.value} names no remote command to cancel",
        )
    return WaitingRecord(
        index,
        operation,
        WaitingVerdict.CANCELLABLE,
        f"step {index} {operation.value} has a cancellable remote command {command_id}",
        remote_command_id=command_id,
    )
