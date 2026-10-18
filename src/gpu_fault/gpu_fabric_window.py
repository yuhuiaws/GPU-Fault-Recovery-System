"""Fabric Manager across one GPU quiesce window (batch 128).

The quiesce stopped ``nvidia-fabricmanager`` in every window and the restore
started it again. NVIDIA's Fabric Manager user guide ("Stopping Fabric
Manager"): stopping the service loses the fabric state, and once it runs again
every GPU in the NVLink fabric needs a reset. A window that reset one GPU
therefore left the new Fabric Manager instance knowing that GPU alone. Live
(plain EKS p5en, H200 HGX, driver 580.178.04, NVLink Inband + ALI), after the
product's single-GPU RESET_GPU windows the Fabric Manager log said "failed to
find the member GPU handle <n> in partition -1 in the multicast team setup
request" and "All GPUs in the partition need to be reset to recover": every
multicast team with one of the other seven GPUs failed, so the whole node lost
NVLink SHARP (NVLS), while unicast P2P and DCGM field 170 still looked healthy.

From Hopper on, an NVSwitch system resets one GPU while Fabric Manager runs (the
guide: "a GPU can be individually reset"; nvidia-smi: "the dependency on Fabric
Manager to facilitate reset is removed"), and with NVLink Inband Fabric Manager
does not open the GPUs at all (live: it holds only the NVSwitch, NVLink and caps
device nodes). Hence two rules here:

- :func:`fabric_manager_policy`: a window whose only GPU-state change is a
  RESET_GPU keeps Fabric Manager running. Its QUIESCE command carries
  ``fabric_manager=keep``; the Agent then neither stops it nor starts it again
  in any restore. Every other window -- the full GPU + NVSwitch reset, a driver
  or firmware remediation, the NVLink field diagnostic, a reboot, an operation
  this module does not know -- keeps the stop and the start. The policy is read
  off the persisted plan alone (the window's steps on its own branch, whatever
  their status), so every send of one QUIESCE carries the same body: an Agent
  answers a reused command_id with another body 409 COMMAND_ID_REUSED.
- :func:`fabric_state_loss`: VALIDATE_GPU's guard for a window that did stop
  Fabric Manager around per-GPU resets (an Agent that predates the parameter,
  or any later path), with no full reset, reboot or replacement after it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gpu_fault.cluster_scope import FABRIC_MANAGER_UNIT
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.node_agent.protocol import (
    QUIESCE_FABRIC_MANAGER_KEEP,
    QUIESCE_FABRIC_MANAGER_PARAMETER,
)

#: What a window may hold beside its per-GPU resets and still keep Fabric
#: Manager running: the client check only reads.
FABRIC_NEUTRAL_OPERATIONS = frozenset({WorkflowOperation.VERIFY_NO_GPU_CLIENTS})
#: The node actions that bring every GPU back into a fresh fabric: the full GPU
#: + NVSwitch reset (inside a window that stops Fabric Manager), a reboot, a
#: replacement.
FABRIC_REREGISTRATION_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
    }
)
FABRIC_STATE_LOST = "nvlink_fabric_state_lost"


def window_operations(
    steps: Sequence[WorkflowStepSpec], quiesce_index: int
) -> list[WorkflowOperation] | None:
    """The operations between the QUIESCE at ``quiesce_index`` and its RESTORE.

    A step is in the window when it is on the QUIESCE's branch (``branch_id``,
    None throughout a flat plan) and acts on one of its nodes; steps of other
    branches and job-level steps are not. None when no RESTORE closes the window
    on that branch (a window handed to a successor) or another QUIESCE comes
    first: a window this module cannot read.
    """

    quiesce = steps[quiesce_index]
    nodes = set(quiesce.node_ids)
    operations: list[WorkflowOperation] = []
    for step in steps[quiesce_index + 1 :]:
        if step.branch_id != quiesce.branch_id or not nodes & set(step.node_ids):
            continue
        if step.operation is WorkflowOperation.RESTORE_GPU_SERVICES:
            return operations
        if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
            return None
        operations.append(step.operation)
    return None


def fabric_manager_policy(
    steps: Sequence[WorkflowStepSpec], quiesce_index: int
) -> str | None:
    """``keep`` when the window's only GPU-state change is RESET_GPU, else None.

    None is today's window: the quiesce stops Fabric Manager and the restore
    starts it. Provider- and product-agnostic on purpose: the persisted plan
    names no GPU product, a node without NVSwitches runs no Fabric Manager to
    keep, every supported NVSwitch type is Hopper or later
    (``gpu_instance_inventory``: p5/p5e/p5en Hopper, p6-b200/p6-b300 NVL5, whose
    unit keeps NVLSM running with Fabric Manager), and VERIFY_NO_GPU_CLIENTS
    still fails closed
    should Fabric Manager ever hold a target GPU's device node.
    """

    operations = window_operations(steps, quiesce_index)
    if not operations or WorkflowOperation.RESET_GPU not in operations:
        return None
    allowed = FABRIC_NEUTRAL_OPERATIONS | {WorkflowOperation.RESET_GPU}
    return QUIESCE_FABRIC_MANAGER_KEEP if set(operations) <= allowed else None


def quiesce_command_parameters(
    workflow: WorkflowRequest, step_index: int, step: WorkflowStepSpec
) -> dict[str, Any]:
    """The parameters the QUIESCE command for ``step`` carries.

    The step's own parameters, plus ``fabric_manager`` when the window keeps
    Fabric Manager. A value the plan already carries is sent as it is; any other
    operation's parameters are returned unchanged.
    """

    parameters = dict(step.parameters)
    if (
        step.operation is not WorkflowOperation.QUIESCE_GPU_SERVICES
        or QUIESCE_FABRIC_MANAGER_PARAMETER in parameters
    ):
        return parameters
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    if not 0 <= step_index < len(steps) or steps[step_index].operation is not (
        step.operation
    ):
        return parameters
    policy = fabric_manager_policy(steps, step_index)
    if policy is not None:
        parameters[QUIESCE_FABRIC_MANAGER_PARAMETER] = policy
    return parameters


def _per_node(value: Any, node_id: str) -> Any:
    return value.get(node_id) if isinstance(value, Mapping) else None


def fabric_manager_stopped(details: Mapping[str, Any], node_id: str) -> bool:
    """Whether a QUIESCE step's result says it stopped Fabric Manager on the node.

    Read from the Agent's receipt in ``node_results`` (``active_services`` is
    what the quiesce stopped), else from the step's lifted ``stopped_services``.
    A window that kept it running names it in ``kept_services``. A result that
    names no services says nothing, which is not "stopped".
    """

    receipt = _per_node(details.get("node_results"), node_id)
    if isinstance(receipt, Mapping) and "active_services" in receipt:
        stopped = receipt.get("active_services")
        kept = receipt.get("kept_services")
    else:
        stopped = _per_node(details.get("stopped_services"), node_id)
        kept = _per_node(details.get("kept_services"), node_id)
    if not isinstance(stopped, list) or FABRIC_MANAGER_UNIT not in stopped:
        return False
    return not any(
        isinstance(item, Mapping) and item.get("service") == FABRIC_MANAGER_UNIT
        for item in (kept if isinstance(kept, list) else ())
    )


def _acted_on(
    execution: WorkflowStepExecution,
    steps: Sequence[WorkflowStepSpec],
    node_id: str,
) -> bool:
    if (
        execution.status is not WorkflowStepStatus.SUCCEEDED
        or execution.phase == "safety"
        or execution.operation
        not in FABRIC_REREGISTRATION_OPERATIONS
        | {WorkflowOperation.QUIESCE_GPU_SERVICES, WorkflowOperation.RESET_GPU}
        or not 0 <= execution.step_index < len(steps)
    ):
        return False
    step = steps[execution.step_index]
    return step.operation is execution.operation and node_id in step.node_ids


def fabric_state_loss(workflow: WorkflowRequest, node_id: str) -> str | None:
    """VALIDATE_GPU's failure reason when this workflow lost the node's fabric state.

    A QUIESCE that stopped Fabric Manager followed by a RESET_GPU: the restore's
    new Fabric Manager instance knows only the GPUs reset in the window, so the
    multicast teams of every other GPU fail until all GPUs and NVSwitches are
    reset or the node reboots. The workflow's successful steps are walked in the
    order they finished; a full GPU + NVSwitch reset, a reboot or a replacement
    of the node after the reset clears it (the reboot rung the ladder climbs to
    from this very failure). None: no such window, or one that kept Fabric
    Manager running.
    """

    steps = workflow.official_steps
    executions = sorted(
        (item for item in workflow.step_executions if _acted_on(item, steps, node_id)),
        key=lambda item: (item.updated_at, item.step_index),
    )
    stopped_at: int | None = None
    lost: str | None = None
    for execution in executions:
        if execution.operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
            stopped = fabric_manager_stopped(execution.details, node_id)
            stopped_at = execution.step_index if stopped else None
        elif execution.operation is WorkflowOperation.RESET_GPU:
            if stopped_at is not None:
                lost = (
                    f"{FABRIC_STATE_LOST}(step {stopped_at} stopped "
                    f"{FABRIC_MANAGER_UNIT}, step {execution.step_index} reset "
                    "GPUs individually; no full GPU + NVSwitch reset or reboot since)"
                )
        else:
            stopped_at, lost = None, None
    return lost
