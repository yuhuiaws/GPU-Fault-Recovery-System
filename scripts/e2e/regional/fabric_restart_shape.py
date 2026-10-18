"""Which workflow a solo XID 45 compiles to on the case's node (batches 128, 131).

NVIDIA's catalog finalizes a solo XID 45 as RESTART_FM. On a node whose NVLink
fabric the product grades -- a Hopper SXM part or an HGX B200/B300 on
NVSwitches whose DCGM exporter reports field 170 -- a Fabric Manager restart
alone loses the fabric state: the restarted service needs every GPU of the
fabric reset (NVIDIA Fabric Manager user guide). The product therefore plans
NVIDIA's H100-and-later procedure there: cordon, quiesce (Fabric Manager
stopped), one reset of every GPU and NVSwitch, restore (Fabric Manager
started), VALIDATE_GPU, VALIDATE_FABRIC and scheduling back. Every other node
keeps FREEZE_EVIDENCE -> RESTART_FABRIC_MANAGER.

From batch 131 that chain is the site's opt-in: by default a solo XID 45 on a
graded node is observe-only (``gpu_fault.orchestration.xid45_policy``: NVIDIA's
GPU debug guidelines read it as informational except on A100) -- MONITOR_ONLY,
the incident RECOVERED with its reasons, no workflow, no node command, no
cordon. ``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`` on the control-worker
(the case opens it as a ``control_plane_env_window``, ``xid45_opt_in``) brings
the chain back.

DESTR-010 and DESTR-019 drive the same injection, so both read the shape here:
from the installed control plane's own grading of the node's latest samples
(``gpu_fault.gpu_fabric_manager.graded_fabric_gpus``) at preflight, recorded in
the plan so that a node whose grading moved between plan and execute is refused.
A control plane without that function cannot plan the fabric chain, so its
shape is the Fabric Manager restart; one without ``xid45_policy`` plans the
chain without an opt-in.
"""

from __future__ import annotations

from typing import Any

FABRIC_MANAGER_RESTART = "fabric-manager-restart"
FABRIC_RESET = "fabric-reset"
OBSERVE_ONLY = "observe-only"
SHAPES = (FABRIC_MANAGER_RESTART, FABRIC_RESET, OBSERVE_ONLY)

FABRIC_MANAGER_RESTART_STEPS = ("FREEZE_EVIDENCE", "RESTART_FABRIC_MANAGER")
# On an idle node: no STOP_WORKLOADS, no RESTART_WORKLOAD.
FABRIC_RESET_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)
NODE_AGENT = "gpu-fault-node-agent"
CONTROL_PLANE = "gpu-fault-control-plane"
FABRIC_RESET_NODE_STEPS = (
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
)
# The steps that each run exactly once: the window, the one reset, the restore.
FABRIC_RESET_SINGLE_COMMAND_STEPS = (
    "QUIESCE_GPU_SERVICES",
    "RESET_ALL_GPUS_NVSWITCHES",
    "RESTORE_GPU_SERVICES",
)
FABRIC_RESET_FORBIDDEN = frozenset(
    {
        "QUARANTINE",
        "RESTART_FABRIC_MANAGER",
        "RESET_GPU",
        "RESTART_NODE",
        "REPLACE_NODE",
        "STOP_WORKLOADS",
    }
)
# ``dcgmFabricManagerStatus_t`` SUCCESS: the GPU registered with the fabric.
FABRIC_STATE_SUCCESS = 3
# The subject each shape's completion notification carries
# (``notifications/restart_guard.py``); the observe-only shape completes
# nothing and sends the XID investigatory notification alone.
NOTIFICATION_SUBJECTS = {
    FABRIC_MANAGER_RESTART: "Fabric Manager",
    FABRIC_RESET: "GPU/NVSwitch reset",
    OBSERVE_ONLY: "[XID 45 investigatory action]",
}
# The observe-only decision (``gpu_fault.orchestration.xid45_policy``).
XID45_SWITCH = "GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM"
XID45_EVIDENCE_REFERENCE = (
    "https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html"
    "#understanding-xid-messages"
)
# The node actions each shape's capability check needs.
REQUIRED_AGENT_OPERATIONS = {
    FABRIC_MANAGER_RESTART: frozenset({"RESTART_FABRIC_MANAGER"}),
    FABRIC_RESET: frozenset(FABRIC_RESET_NODE_STEPS),
    OBSERVE_ONLY: frozenset(),
}
REQUIRED_CAPABILITIES = {
    FABRIC_MANAGER_RESTART: ("fabricManagerRestart",),
    FABRIC_RESET: ("gpuReset", "fabricReset"),
    OBSERVE_ONLY: (),
}

# A function both CPU probes define: the installed control plane's grading of
# the node, never a copy of it. ``product`` stands in for samples that carry no
# ``modelName``, as the XID event's product does when the builder grades.
GRADING_SOURCE = r"""
def fabric_grading(store, cluster_id, node_id):
    try:
        from gpu_fault.gpu_fabric_manager import (
            FABRIC_MANAGER_STATUS,
            graded_fabric_gpus,
        )
    except ImportError:
        return {
            "available": False,
            "graded_gpus": [],
            "states": {},
            "product": None,
            "observe_only_default": False,
        }
    try:
        from gpu_fault.orchestration.xid45_policy import XID45_SOLO_RESTART_FM_ENV
    except ImportError:
        observe_only_default = False
    else:
        observe_only_default = bool(XID45_SOLO_RESTART_FM_ENV)
    inventory = store.get_gpu_inventory_snapshot(cluster_id, node_id)
    products = (
        sorted({item.product for item in inventory.devices if item.product})
        if inventory is not None
        else []
    )
    product = products[0] if len(products) == 1 else None
    latest = store.list_gpu_metrics_latest(cluster_id, node_id)
    graded = graded_fabric_gpus(
        ((item.sample, item.observed_at) for item in latest), product=product
    )
    states = {
        item.sample.gpu_uuid: item.sample.value
        for item in latest
        if item.sample.canonical_name == FABRIC_MANAGER_STATUS
        and item.sample.gpu_uuid
    }
    return {
        "available": True,
        "graded_gpus": list(graded),
        "states": states,
        "product": product,
        "observe_only_default": observe_only_default,
    }
"""

FABRIC_GRADING_PROBE = (
    r"""
import json
import sys

from gpu_fault.app import ApplicationContext
"""
    + GRADING_SOURCE
    + r"""
cluster_id, node_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
print(json.dumps(fabric_grading(store, cluster_id, node_id), sort_keys=True, default=str))
"""
)


def graded(grading: Any) -> bool:
    """Whether the installed control plane grades the node's fabric."""

    return bool(
        isinstance(grading, dict)
        and grading.get("available") is True
        and grading.get("graded_gpus")
    )


def observe_only_default(grading: Any) -> bool:
    """Whether a solo XID 45 there is observe-only unless the site opts in."""

    return graded(grading) and grading.get("observe_only_default") is True


def node_action_shape(grading: Any) -> str:
    """The node action a case drives on this node: the full chain on a graded
    node (under the opt-in where the default is observe-only), the Fabric
    Manager restart elsewhere."""

    return FABRIC_RESET if graded(grading) else FABRIC_MANAGER_RESTART


def shape_of(grading: Any, *, opted_in: bool = False) -> str:
    """The shape the installed control plane plans for this node's solo XID 45.

    ``opted_in``: every ready control-worker replica reads
    ``GPU_FAULT_XID45_SOLO_NVSWITCH_RESTART_FM=true`` (the case's window, or a
    site that opted in for good).
    """

    if not graded(grading):
        return FABRIC_MANAGER_RESTART
    if opted_in or not observe_only_default(grading):
        return FABRIC_RESET
    return OBSERVE_ONLY


def graded_gpus(grading: Any) -> list[str]:
    values = grading.get("graded_gpus") if isinstance(grading, dict) else None
    return sorted(str(item) for item in values or [])


def covered_operations(command: dict[str, Any]) -> list[str]:
    """The head step's operation and every step a compound command carries."""

    head = str((command.get("step") or {}).get("operation") or "")
    batched = [
        str((item.get("step") or {}).get("operation") or "")
        for item in command.get("batched_steps") or []
        if isinstance(item, dict)
    ]
    return [operation for operation in (head, *batched) if operation]


def command_for(commands: list[dict[str, Any]], operation: str) -> dict[str, Any]:
    """The SUCCEEDED remote command whose head step is ``operation``, else {}."""

    return next(
        (
            item
            for item in commands
            if (item.get("step") or {}).get("operation") == operation
            and item.get("status") == "SUCCEEDED"
        ),
        {},
    )


def node_action_command_id(
    command: dict[str, Any], *, node: str, generation: int
) -> str:
    """What ``_send_action`` names a single-node command pinned to a generation."""

    return f"{command.get('idempotency_key')}/{node}/agent-{generation}"


def fabric_reset_workflow_errors(
    workflow: dict[str, Any],
    commands: list[dict[str, Any]],
    expected_gpus: list[str],
) -> list[str]:
    """The full GPU + NVSwitch reset chain, each node action exactly once."""

    errors: list[str] = []
    steps = workflow.get("official_steps") or []
    operations = [item.get("operation") for item in steps]
    if operations != list(FABRIC_RESET_STEPS):
        errors.append(
            f"workflow steps {operations} are not the full GPU + NVSwitch reset "
            f"chain {list(FABRIC_RESET_STEPS)}"
        )
    for item in steps:
        operation = item.get("operation")
        owner = item.get("execution_owner")
        if operation == "FREEZE_EVIDENCE" and owner != CONTROL_PLANE:
            errors.append(f"FREEZE_EVIDENCE is owned by {owner}, not {CONTROL_PLANE}")
        if operation in FABRIC_RESET_NODE_STEPS and owner != NODE_AGENT:
            errors.append(f"{operation} is owned by {owner}, not {NODE_AGENT}")
    reset = next(
        (
            item
            for item in steps
            if item.get("operation") == "RESET_ALL_GPUS_NVSWITCHES"
        ),
        {},
    )
    if sorted(reset.get("gpu_uuids") or []) != sorted(expected_gpus):
        errors.append("the full reset does not name exactly the node's graded GPUs")
    if (reset.get("parameters") or {}).get("xid") != 45:
        errors.append("the full reset does not carry the XID 45 it answers")
    completed = set(workflow.get("completed_operations") or [])
    missing = sorted(set(FABRIC_RESET_STEPS) - completed)
    if missing:
        errors.append(f"workflow did not complete {missing}")
    reached = sorted(completed & FABRIC_RESET_FORBIDDEN)
    if reached:
        errors.append(f"workflow reached forbidden operations {reached}")
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"workflow is {workflow.get('status')}, not SUCCEEDED")
    if any(item.get("status") == "FAILED" for item in commands):
        errors.append("a remote command of the chain FAILED")
    for operation in FABRIC_RESET_SINGLE_COMMAND_STEPS:
        matches = [
            item
            for item in commands
            if operation in covered_operations(item)
            and item.get("status") == "SUCCEEDED"
        ]
        if len(matches) != 1:
            errors.append(
                f"{operation} has {len(matches)} SUCCEEDED remote commands, not one"
            )
    return errors


def fabric_state_errors(grading: Any, expected_gpus: list[str]) -> list[str]:
    """Every graded GPU reports Success after the reset (the store's latest)."""

    states = (grading or {}).get("states") if isinstance(grading, dict) else None
    states = states if isinstance(states, dict) else {}
    pending = []
    for gpu in expected_gpus:
        value = states.get(gpu)
        try:
            settled = value is not None and float(value) == FABRIC_STATE_SUCCESS
        except (TypeError, ValueError):
            settled = False
        if not settled:
            pending.append(gpu)
    if pending:
        return [
            "GPUs whose NVLink fabric state is not Success after the reset: "
            + ", ".join(sorted(pending))
        ]
    return []


def shape_notifications(
    notifications: list[dict[str, Any]], shape: str
) -> list[dict[str, Any]]:
    """The completion notification(s) of the shape's node action.

    The Fabric Manager restart's is ``ACTION_COMPLETED``; the fabric reset's is
    the product's GPU/NVSwitch reset notice, whatever its category; the
    observe-only shape completes nothing, so its one notification is the XID
    investigatory notice (``FAULT_DETECTED``).
    """

    marker = NOTIFICATION_SUBJECTS[shape]
    category = {
        FABRIC_MANAGER_RESTART: "ACTION_COMPLETED",
        OBSERVE_ONLY: "FAULT_DETECTED",
    }.get(shape)
    return [
        item
        for item in notifications
        if marker in str(item.get("subject") or "")
        and (category is None or item.get("category") == category)
    ]


def observe_only_errors(state: dict[str, Any], expected_gpus: list[str]) -> list[str]:
    """A solo XID 45 on a graded node by default: recorded and notified, nothing run.

    The decision keeps RESTART_FM as the official action but is MONITOR_ONLY /
    NO_ACTION; the incident is RECOVERED without a workflow, cites NVIDIA's GPU
    debug guidelines as its policy reference and says why and how to opt in.
    """

    errors: list[str] = []
    decision = state.get("decision") or {}
    incident = state.get("incident") or {}
    if decision.get("official_action") != "RESTART_FM":
        errors.append("the decision does not record the catalog's RESTART_FM")
    if decision.get("disposition") != "MONITOR_ONLY":
        errors.append(
            f"the decision is {decision.get('disposition')}, not MONITOR_ONLY"
        )
    if decision.get("action") != "NO_ACTION":
        errors.append(
            f"the effective action is {decision.get('action')}, not NO_ACTION"
        )
    if (decision.get("marker") or {}).get("active"):
        errors.append("the decision's marker is active")
    if incident.get("state") != "RECOVERED":
        errors.append(f"the incident is {incident.get('state')}, not RECOVERED")
    if incident.get("workflow_request_id") or state.get("workflow"):
        errors.append("a workflow was planned for the observe-only XID 45")
    if state.get("commands"):
        errors.append("remote commands were created for the observe-only XID 45")
    if incident.get("policy_reference") != XID45_EVIDENCE_REFERENCE:
        errors.append("the incident does not cite NVIDIA's GPU debug guidelines")
    reasons = [str(item) for item in incident.get("reasons") or []]
    why = [item for item in reasons if item.startswith("observe-only")]
    if len(why) != 1 or f"{len(expected_gpus)} GPUs report" not in why[0]:
        errors.append("the incident does not say why no action was taken")
    if not any(XID45_SWITCH in item for item in reasons):
        errors.append("the incident does not say how to opt in")
    return errors


def observe_only_host_errors(
    baseline: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    """Nothing reached the node: Fabric Manager, the Agent ledger and the
    gpu-fault timers are exactly as they were."""

    errors: list[str] = []
    before_service = baseline.get("fabric_manager") or {}
    after_service = after.get("fabric_manager") or {}
    if after_service.get("ActiveState") != "active":
        errors.append("Fabric Manager is not active after the observe-only XID 45")
    for field in ("MainPID", "InvocationID"):
        if before_service.get(field) != after_service.get(field):
            errors.append(f"Fabric Manager {field} changed")
    journal = after.get("journal") or {}
    if journal.get("started_count") or journal.get("stopped_count"):
        errors.append("the journal shows a Fabric Manager stop or start")
    before_ids = {item.get("command_id") for item in baseline.get("ledger") or []}
    after_ids = {item.get("command_id") for item in after.get("ledger") or []}
    if after_ids != before_ids:
        errors.append("the Node Agent ledger changed: a node action ran")
    if baseline.get("gpu_fault_timers") != after.get("gpu_fault_timers"):
        errors.append("gpu-fault timer inventory changed")
    return errors
