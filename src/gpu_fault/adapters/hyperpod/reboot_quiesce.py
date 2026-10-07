"""The quiesce taint a confirmed reboot lifts from its node.

The eks-managed quiesce window taints the node ``gpu-fault.io/quiesce``
(NoExecute) and the RESTORE_GPU_SERVICES that lifts it is exactly the step a
RESET_GPU -> RESTART_NODE escalation retires. Taints survive a reboot, so the
exporter DaemonSet could not come back and the rung's VALIDATE_GPU waited out
its cap on a node nothing had observed (live 2026-10-07, gpu-eks-cluster
DESTR-025: ``quiesce_taint_present``, then a REPLACE_NODE rung no spare could
serve, then quarantine). A reboot is the restore of a quiesce -- every host
service and container restarts with the node -- so the confirmed reboot
carries the restore's taint half, for both providers: a HyperPod node carries
no such taint and gets no patch.

``quiesce_taint_removed_nodes`` is recorded on the reboot step as on a restore,
so the record's accounting (``quiesce_taint_nodes_held``) balances; a node that
cannot be read is skipped, and a patch that fails is named
(``quiesce_taint_error``) without failing the reboot it follows -- the
validation's ``quiesce_taint_present`` then says what is left.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.adapters.kubernetes.primitives import patch_node_with_retry
from gpu_fault.adapters.node_action.quiesce_taint import quiesce_untaint_patch
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.models import WorkflowOperation
from gpu_fault.workflow_quarantine import quiesce_taint_nodes_held


def release_quiesce_taint(
    kubernetes_adapter: Any, context: WorkflowStepContext, details: dict[str, Any]
) -> dict[str, Any]:
    """Strip every quiesce taint of a RESTART_NODE's nodes; the step details to add."""

    if (
        kubernetes_adapter is None
        or context.step.operation is not WorkflowOperation.RESTART_NODE
    ):
        return {}
    core = kubernetes_adapter.core
    observed = details.get("observed_isolation")
    observed = observed if isinstance(observed, dict) else {}
    held = quiesce_taint_nodes_held(context.workflow)
    removed: list[str] = []
    errors: list[str] = []
    for node_id in context.step.node_ids:
        record = observed.get(node_id)
        kubernetes_node = str(
            (record.get("kubernetes_node") if isinstance(record, dict) else None)
            or node_id
        )
        stripped: list[bool] = []

        def body(node: Any) -> dict[str, Any] | None:
            patch = quiesce_untaint_patch(node)
            if patch is not None:
                stripped.append(True)
            return patch

        try:
            patch_node_with_retry(core, kubernetes_node, body)
        except Exception as exc:  # noqa: BLE001 - the error is the evidence
            if isinstance(exc, KeyError) or getattr(exc, "status", None) == 404:
                continue
            errors.append(f"{node_id}: {type(exc).__name__}: {exc}")
            continue
        if stripped or node_id in held:
            removed.append(node_id)
    result: dict[str, Any] = {}
    if removed:
        result["quiesce_taint_removed_nodes"] = removed
    if errors:
        result["quiesce_taint_error"] = "; ".join(errors)
    return result
