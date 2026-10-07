"""The device-plugin health gate a workload restart waits behind.

Live 2026-10-07 (gpu-eks-cluster DESTR-015, attempt 9): both branches reset
their GPU, restored the services, validated and released scheduling, and the
join's RESTART_WORKLOAD created the new PyTorchJob attempt one second later.
kubelet admitted its Pods -- pinned by ``nodeName`` to the two nodes -- onto
devices the upstream NVIDIA device plugin still reported unhealthy across the
reset (``UnexpectedAdmissionError: Allocate failed due to no healthy devices
present; cannot allocate unhealthy devices nvidia.com/gpu``), and the attempt
ended Failed. Node allocatable showed 8/8 again a little later. HyperPod's
plugin timing never exposed the window; a plain EKS node does.

The gate reads what kubelet publishes about the node and the plugin: the
``nvidia.com/gpu`` capacity and allocatable (allocatable is capacity minus the
devices the plugin reports unhealthy), the plugin's Pod on the node and when it
became Ready, and the node's Ready-condition heartbeat (kubelet reports node
status when allocatable changes). A node is ready for GPU Pods when allocatable
equals capacity, the plugin Pod is Ready, and the node status was reported
after the plugin became Ready -- a status that predates the plugin's return
may still describe the devices as they were. The restart waits (bounded, a
hold that owes no restart budget) and the step that proceeds carries the
receipt per node. A node that reports no GPU capacity is outside the gate; a
node the restart is not pinned to and this workflow did not repair is not
judged at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from gpu_fault.adapters.kubernetes.node_operations import pod_terminating
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus

GPU_RESOURCE = "nvidia.com/gpu"
PLUGIN_NAMESPACE = "kube-system"
PLUGIN_SELECTOR = "app.kubernetes.io/name=nvidia-device-plugin"
#: Step ``parameters`` key bounding the wait; the default is the device plugin
#: restart's own restore timeout.
TIMEOUT_PARAMETER = "device_plugin_health_timeout_seconds"
DEFAULT_TIMEOUT_SECONDS = 180
RECEIPT_KEY = "device_plugin_health"
PENDING_KEY = "device_plugin_health_pending"
UNHEALTHY_REASON = "DEVICE_PLUGIN_UNHEALTHY"
#: A SUCCEEDED step of one of these says this workflow worked on the node and
#: the restarted job comes back to it.
RECOVERY_OPERATIONS = frozenset(
    {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.REPLACE_NODE,
        WorkflowOperation.RESTART_VM,
        WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
        WorkflowOperation.RESTORE_GPU_SERVICES,
        WorkflowOperation.RESTORE_SCHEDULING,
    }
)


@dataclass(frozen=True)
class DevicePluginGate:
    #: The hold (WAITING) or refusal (FAILED) when a node is not ready; None
    #: when the restart may proceed.
    outcome: WorkflowStepOutcome | None
    #: Per-node receipt of what was read; empty when no node was judged.
    receipt: dict[str, dict[str, Any]]


def _field(value: Any, key: str, attribute: str | None = None) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, attribute or key, None)


def _status(resource: Any) -> Any:
    return _field(resource, "status") or {}


def _quantity(node: Any, section: str) -> int:
    values = _field(_status(node), section) or {}
    raw = values.get(GPU_RESOURCE) if isinstance(values, dict) else None
    try:
        return int(str(raw)) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _condition_time(resource: Any, kind: str, field: str, attribute: str) -> Any:
    conditions = _field(_status(resource), "conditions") or []
    for condition in conditions:
        if str(_field(condition, "type")) == kind:
            return _timestamp(_field(condition, field, attribute))
    return None


def _pod_ready(pod: Any) -> bool:
    conditions = _field(_status(pod), "conditions") or []
    return any(
        str(_field(condition, "type")) == "Ready"
        and str(_field(condition, "status")).lower() == "true"
        for condition in conditions
    )


def _pod_name(pod: Any) -> str:
    metadata = _field(pod, "metadata") or {}
    return str(_field(metadata, "name") or "")


def gpu_resource_health(
    node: Any, pods: list[Any] | None, *, now: datetime, pods_error: str | None = None
) -> dict[str, Any]:
    """One node's receipt: what kubelet publishes and whether GPU Pods may land.

    ``pods`` is the device plugin's Pods on the node (``None`` when they could
    not be listed: the allocatable comparison then stands alone and the
    receipt says so).
    """

    capacity = _quantity(node, "capacity")
    allocatable = _quantity(node, "allocatable")
    receipt: dict[str, Any] = {
        "resource": GPU_RESOURCE,
        "capacity": capacity,
        "allocatable": allocatable,
        "read_at": now.isoformat(),
    }
    if capacity <= 0:
        return {**receipt, "not_applicable": True, "healthy": True}
    heartbeat = _condition_time(
        node, "Ready", "lastHeartbeatTime", "last_heartbeat_time"
    )
    receipt["node_status_heartbeat_at"] = heartbeat.isoformat() if heartbeat else None
    reasons: list[str] = []
    if allocatable != capacity:
        reasons.append(
            f"{GPU_RESOURCE} allocatable {allocatable} != capacity {capacity}: "
            "kubelet still reports unhealthy devices"
        )
    if pods is None:
        receipt["plugin_pods"] = None
        receipt["plugin_pods_error"] = pods_error or "device plugin Pods not read"
    else:
        live = [pod for pod in pods if not pod_terminating(pod)]
        ready = [pod for pod in live if _pod_ready(pod)]
        receipt["plugin_pods"] = [_pod_name(pod) for pod in live]
        receipt["plugin_pods_ready"] = [_pod_name(pod) for pod in ready]
        ready_since = max(
            (
                stamp
                for pod in ready
                if (
                    stamp := _condition_time(
                        pod, "Ready", "lastTransitionTime", "last_transition_time"
                    )
                )
                is not None
            ),
            default=None,
        )
        receipt["plugin_ready_since"] = ready_since.isoformat() if ready_since else None
        if live and not ready:
            reasons.append(
                "the device plugin Pod on the node is not Ready: "
                + ", ".join(receipt["plugin_pods"])
            )
        elif ready and ready_since is not None and heartbeat is not None:
            if heartbeat < ready_since:
                reasons.append(
                    f"the node status (heartbeat {heartbeat.isoformat()}) predates "
                    f"the device plugin's readiness ({ready_since.isoformat()}); "
                    "kubelet has not reported the re-registered devices yet"
                )
    receipt["healthy"] = not reasons
    if reasons:
        receipt["reasons"] = reasons
    return receipt


def restart_target_nodes(
    adapter: Any,
    context: WorkflowStepContext,
    workloads: Sequence[tuple[str, str, str, str, Any]],
) -> list[str]:
    """The nodes the restarted Pods come back to: pinned by the templates of
    ``workloads`` (as the mutation read them), or repaired by this workflow; a
    node the restart avoids is neither."""

    rebindings = adapter._node_rebindings(context)
    avoided = set(adapter._avoid_node_ids(context))
    targets: set[str] = set()
    for _namespace, kind, _name, _workload_id, workload in workloads:
        targets.update(adapter._pinned_nodes(kind, workload, rebindings))
    steps = context.workflow.official_steps
    for execution in context.workflow.step_executions:
        if (
            execution.status is WorkflowStepStatus.SUCCEEDED
            and execution.operation in RECOVERY_OPERATIONS
            and 0 <= execution.step_index < len(steps)
        ):
            targets.update(
                rebindings.get(node_id, node_id)
                for node_id in steps[execution.step_index].node_ids
            )
    return sorted(targets - avoided)


def _wait_started(context: WorkflowStepContext, now: datetime) -> datetime:
    """When this step first held for the gate (its earliest pending record)."""

    return min(
        (
            execution.started_at
            for execution in context.workflow.step_executions
            if execution.step_index == context.step_index
            and execution.operation is context.step.operation
            and execution.status is WorkflowStepStatus.WAITING
            and isinstance(execution.details, dict)
            and PENDING_KEY in execution.details
        ),
        default=now,
    )


def device_plugin_health_gate(
    adapter: Any,
    context: WorkflowStepContext,
    workloads: Sequence[tuple[str, str, str, str, Any]],
) -> DevicePluginGate:
    """Hold a RESTART_WORKLOAD until every target node's devices are healthy.

    Runs after the restart guard, on the workloads the mutation read (no read
    of its own). The hold is WAITING with ``device_plugin_health_pending`` and
    ``restart_submitted: False`` (a hold owes no restart budget); past the
    bound it is a FAILED refusal (``DEVICE_PLUGIN_UNHEALTHY``), never a
    submission onto unhealthy devices.
    """

    targets = restart_target_nodes(adapter, context, workloads)
    if not targets:
        return DevicePluginGate(None, {})
    now = datetime.now(timezone.utc)
    receipt: dict[str, dict[str, Any]] = {}
    for node_id in targets:
        try:
            node = adapter.core.read_node(node_id)
        except Exception as exc:
            if isinstance(exc, KeyError) or getattr(exc, "status", None) == 404:
                receipt[node_id] = {"absent": True, "healthy": True}
                continue
            raise
        pods: list[Any] | None = None
        pods_error: str | None = None
        if _quantity(node, "capacity") > 0:
            try:
                pods = list(
                    adapter._efa_plugin_pods(
                        node_id,
                        namespace=PLUGIN_NAMESPACE,
                        label_selector=PLUGIN_SELECTOR,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - the allocatable check stands alone
                pods_error = f"{type(exc).__name__}: {exc}"
        receipt[node_id] = gpu_resource_health(
            node, pods, now=now, pods_error=pods_error
        )
    unhealthy = {
        node_id: item for node_id, item in receipt.items() if not item.get("healthy")
    }
    if not unhealthy:
        return DevicePluginGate(None, receipt)
    started = _wait_started(context, now)
    waited = max(0, int((now - started).total_seconds()))
    raw_timeout = context.step.parameters.get(
        TIMEOUT_PARAMETER, DEFAULT_TIMEOUT_SECONDS
    )
    try:
        timeout = max(0, int(raw_timeout))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS
    summary = "; ".join(
        f"{node_id}: " + "; ".join(item.get("reasons") or ["unhealthy"])
        for node_id, item in sorted(unhealthy.items())
    )
    if waited >= timeout:
        return DevicePluginGate(
            WorkflowStepOutcome.failed(
                "device plugin health did not converge before the workload restart "
                f"({waited}s of {timeout}s): {summary}",
                details={
                    "reason": UNHEALTHY_REASON,
                    RECEIPT_KEY: receipt,
                    "waited_seconds": waited,
                    "timeout_seconds": timeout,
                    "restart_submitted": False,
                },
            ),
            receipt,
        )
    return DevicePluginGate(
        WorkflowStepOutcome.waiting(
            operation_id=context.idempotency_key,
            details={
                PENDING_KEY: {
                    "nodes": receipt,
                    "started_at": started.isoformat(),
                    "waited_seconds": waited,
                    "timeout_seconds": timeout,
                    "reason": summary,
                },
                "restart_submitted": False,
            },
        ),
        receipt,
    )


def with_device_plugin_receipt(
    outcome: WorkflowStepOutcome, receipt: dict[str, dict[str, Any]]
) -> WorkflowStepOutcome:
    """The restart's outcome with the gate's receipt on it, when a node was judged."""

    if not receipt or outcome.status is not WorkflowStepStatus.SUCCEEDED:
        return outcome
    return replace(outcome, details={**(outcome.details or {}), RECEIPT_KEY: receipt})
