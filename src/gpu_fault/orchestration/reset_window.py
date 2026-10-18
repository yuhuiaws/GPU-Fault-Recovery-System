"""A node's GPU reset window keeps its scope once its QUIESCE is dispatched.

The merge rule widened every pending step of a node to a second fault's GPUs,
and treated a run that had only reached QUIESCE_GPU_SERVICES as a reset not yet
submitted (GF-REGIONAL-PREEMPT-021). The window was committed by then, though,
in two places the plan does not show as records of the later steps:

- On the Node Agent: the QUIESCE released persistence (``nvidia-smi -i <bus id>
  -pm 0``) and swept clients for the GPUs it named only. The
  ``nvidia-persistenced`` it kept holds every other GPU's device node, so the
  widened VERIFY_NO_GPU_CLIENTS met it on the new GPU and failed closed into the
  reboot rung, and the Agent refuses a second QUIESCE with another target set
  for the same incident ("quiesce target GPU scope changed after quiesce").
- With remote step batching the QUIESCE is the head of one compound command
  that carries the window's VERIFY, RESET and RESTORE as passengers, each with
  the scope it had when the command was minted, and the data plane runs the
  whole window under one lease. A passenger has no record of its own until its
  result is read back, by step index: the widened RESET_GPU was recorded
  SUCCEEDED for a GPU the node never reset.

So a QUIESCE that completed or has a record of any status commits its window:
the QUIESCE and every later step of its branch on one of its nodes keep the
node and GPU scope they were dispatched with (``fixed_scope_step_indexes``, the
set every widening pass leaves alone), and a fault naming a GPU that window did
not quiesce is not a widening (``widens_dispatched_window``). It gets a window
of its own: a node successor branch before the job's single join, or a
successor workflow where there is no join.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
    execution_matches_step,
    resolved_step_indexes,
)

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES


def quiesce_dispatched(workflow: WorkflowRequest, index: int) -> bool:
    """Whether the official QUIESCE at ``index`` completed or has a record."""

    return index in workflow.completed_step_indexes or any(
        execution_matches_step(item, index, QUIESCE, "official")
        for item in workflow.step_executions
    )


def _dispatched_quiesces(
    workflow: WorkflowRequest, indexes: Iterable[int] | None = None
) -> list[int]:
    if workflow.executes_safety_steps:
        # The record runs its safety steps; nothing of the official plan ran.
        return []
    selected = None if indexes is None else set(indexes)
    return [
        index
        for index, step in enumerate(workflow.official_steps)
        if step.operation is QUIESCE
        and (selected is None or index in selected)
        and quiesce_dispatched(workflow, index)
    ]


def committed_step_indexes(workflow: WorkflowRequest) -> frozenset[int]:
    """Each dispatched QUIESCE and every later step of its branch on its nodes.

    ``branch_id`` is None throughout a flat plan; in a DAG another branch on
    the same node -- a successor branch with a window of its own -- is not part
    of this one.
    """

    steps = workflow.official_steps
    committed: set[int] = set()
    for index in _dispatched_quiesces(workflow):
        quiesce = steps[index]
        nodes = set(quiesce.node_ids)
        committed.add(index)
        committed.update(
            later
            for later in range(index + 1, len(steps))
            if steps[later].branch_id == quiesce.branch_id
            and nodes.intersection(steps[later].node_ids)
        )
    return frozenset(committed)


def fixed_scope_step_indexes(workflow: WorkflowRequest) -> set[int]:
    """The steps whose node and GPU scope a merge no longer changes (F-B6).

    Finished or superseded steps are history and a step with a record is a
    command an agent holds; a dispatched window's steps are committed with it
    (``committed_step_indexes``). Every widening pass leaves these alone.
    """

    return (
        set(resolved_step_indexes(workflow))
        | {execution.step_index for execution in workflow.step_executions}
        | committed_step_indexes(workflow)
    )


def _quiesced_gpus(step: WorkflowStepSpec, node_id: str) -> frozenset[str] | None:
    """The GPUs a QUIESCE named on ``node_id``; None when it named every GPU.

    The node-action adapter reads ``gpu_uuids_by_node`` when the step carries
    it (a node missing from it is refused, so nothing was quiesced there) and
    ``gpu_uuids`` otherwise, where an empty list is the whole node.
    """

    raw = step.parameters.get("gpu_uuids_by_node")
    if isinstance(raw, Mapping):
        values = raw.get(node_id)
        return frozenset(map(str, values)) if isinstance(values, list) else frozenset()
    return frozenset(step.gpu_uuids) or None


def widens_dispatched_window(
    workflow: WorkflowRequest,
    node_id: str,
    gpu_uuids: Collection[str],
    indexes: Iterable[int] | None = None,
) -> bool:
    """Whether merging ``gpu_uuids`` of ``node_id`` would widen a committed window.

    True when a QUIESCE among ``indexes`` (every step when None) was dispatched
    on ``node_id`` and left one of ``gpu_uuids`` out. An event that names no GPU
    adds nothing to widen.
    """

    wanted = set(gpu_uuids)
    if not wanted:
        return False
    steps = workflow.official_steps
    for index in _dispatched_quiesces(workflow, indexes):
        if node_id not in steps[index].node_ids:
            continue
        quiesced = _quiesced_gpus(steps[index], node_id)
        if quiesced is not None and not wanted <= quiesced:
            return True
    return False
