"""The quiesce taint an eks-managed node carries while its GPU services are down.

On HyperPod the quiesce stops kubelet, so the GPU client containers it kills
stay down until RESTORE_GPU_SERVICES starts kubelet again. On a plain EKS node
kubelet and containerd are protected (``node_agent.config
EKS_MANAGED_PROTECTED_QUIESCE_SERVICES``) and every container the agent stops
is restarted by kubelet within seconds -- the product's own DCGM exporter
(embedded hostengine), the NVIDIA device plugin, GPU feature discovery -- so
``VERIFY_NO_GPU_CLIENTS`` never saw the GPU free (live 2026-10-05,
gpu-eks-cluster DESTR-001: seven minutes of retries, then "quiesce maintenance
window expired"). The executor therefore taints the node
``gpu-fault.io/quiesce=<incident value>:NoExecute`` before it dispatches the
quiesce and removes the taint before it dispatches the restore: DaemonSet Pods
that do not tolerate the key are evicted for the window and recreated once it
is gone. Applied only on eks-managed (the adapter is wired a Kubernetes core
client there and nowhere else); fail closed: a quiesce whose taint cannot be
applied is not sent to the agent.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.adapters.common import QUIESCE_TAINT, quarantine_taint_value
from gpu_fault.adapters.kubernetes.primitives import (
    KubernetesPrimitivesMixin,
    patch_node_with_retry,
)


def quiesce_taint(incident_id: str) -> dict[str, str]:
    return {
        "key": QUIESCE_TAINT,
        "value": quarantine_taint_value(incident_id),
        "effect": "NoExecute",
    }


def _body(node: Any, taints: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metadata": {
            "resourceVersion": KubernetesPrimitivesMixin._resource_version(node),
            "annotations": {},
        },
        "spec": {"taints": taints},
    }


def quiesce_taint_patch(node: Any, incident_id: str) -> dict[str, Any] | None:
    """The patch that adds the quiesce taint, ``None`` when the node has it."""

    taints = KubernetesPrimitivesMixin._taints(node)
    wanted = quiesce_taint(incident_id)
    if any(
        item.get("key") == QUIESCE_TAINT and item.get("value") == wanted["value"]
        for item in taints
    ):
        return None
    kept = [item for item in taints if item.get("key") != QUIESCE_TAINT]
    return _body(node, [*kept, wanted])


def quiesce_untaint_patch(node: Any) -> dict[str, Any] | None:
    """The patch that removes every quiesce taint, ``None`` when there is none."""

    taints = KubernetesPrimitivesMixin._taints(node)
    kept = [item for item in taints if item.get("key") != QUIESCE_TAINT]
    if len(kept) == len(taints):
        return None
    return _body(node, kept)


def apply_quiesce_taint(core: Any, node_ids: list[str], incident_id: str) -> list[str]:
    """Taint ``node_ids`` for ``incident_id``; returns the nodes now tainted."""

    for node_id in node_ids:
        patch_node_with_retry(
            core, node_id, lambda node: quiesce_taint_patch(node, incident_id)
        )
    return list(node_ids)


def remove_quiesce_taint(core: Any, node_ids: list[str]) -> list[str]:
    """Remove the quiesce taint from ``node_ids``; returns the nodes handled."""

    for node_id in node_ids:
        patch_node_with_retry(core, node_id, quiesce_untaint_patch)
    return list(node_ids)
