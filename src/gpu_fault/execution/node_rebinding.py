"""Point a workflow's later steps at replacement nodes, GPU scope included.

A node replacement (``REPLACE_NODE`` with ``node_rebindings`` in its outcome)
rewrites every later step and the incident from the fault node to the spare.
The GPU scope has to move with them: a replaced node's GPU UUIDs do not exist
on the spare, and the per-GPU validation rule -- every GPU a step names must
report for itself -- waited on the fault node's eight UUIDs on the spare until
the step cap failed the workflow and the escalation quarantined the healthy
spare (GF-REGIONAL-DESTR-003, 2026-09-08). A step whose nodes changed names the
new nodes' GPUs from their inventory snapshot; with no snapshot yet the scope
goes node-wide, which the validation adapter judges by the node's whole metric
set.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from gpu_fault.models import FaultIncident, WorkflowRequest

LOGGER = logging.getLogger(__name__)


def inventory_gpu_uuids(store: Any, cluster_id: str, node_id: str) -> list[str]:
    """The GPUs a node currently enumerates, or nothing when unknown."""

    try:
        snapshot = store.get_gpu_inventory_snapshot(cluster_id, node_id)
    except Exception:  # noqa: BLE001 - an unknown inventory is not an error here
        LOGGER.exception(
            "GPU inventory unavailable while rebinding node scope: node=%s", node_id
        )
        return []
    if snapshot is None:
        return []
    return [device.gpu_uuid for device in snapshot.devices]


def rebind_nodes(
    store: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    rebindings: dict[str, str],
    *,
    is_safety: bool,
    after_index: int,
) -> tuple[WorkflowRequest, FaultIncident]:
    def replace(node_ids: list[str]) -> list[str]:
        return list(
            dict.fromkeys(rebindings.get(node_id, node_id) for node_id in node_ids)
        )

    def rebound_gpu_uuids(node_ids: list[str]) -> list[str]:
        uuids: list[str] = []
        for node_id in node_ids:
            uuids.extend(inventory_gpu_uuids(store, incident.cluster_id, node_id))
        return list(dict.fromkeys(uuids))

    field = "safety_steps" if is_safety else "official_steps"
    steps = list(getattr(workflow, field))
    for index in range(after_index + 1, len(steps)):
        step = steps[index]
        node_ids = replace(step.node_ids)
        update: dict[str, Any] = {"node_ids": node_ids}
        if set(node_ids) != set(step.node_ids):
            if step.gpu_uuids:
                update["gpu_uuids"] = rebound_gpu_uuids(node_ids)
            by_node = step.parameters.get("gpu_uuids_by_node")
            if isinstance(by_node, dict):
                update["parameters"] = {
                    **step.parameters,
                    "gpu_uuids_by_node": {
                        rebindings.get(node_id, node_id): (
                            list(values)
                            if node_id not in rebindings
                            else rebound_gpu_uuids([rebindings[node_id]])
                        )
                        for node_id, values in by_node.items()
                    },
                }
        steps[index] = step.model_copy(update=update)
    now = datetime.now(timezone.utc)
    incident_nodes = replace(incident.node_ids)
    incident_update: dict[str, Any] = {"node_ids": incident_nodes, "updated_at": now}
    if set(incident_nodes) != set(incident.node_ids) and incident.gpu_uuids:
        incident_update["gpu_uuids"] = rebound_gpu_uuids(incident_nodes)
    return (
        workflow.model_copy(update={field: steps, "updated_at": now}),
        incident.model_copy(update=incident_update),
    )
