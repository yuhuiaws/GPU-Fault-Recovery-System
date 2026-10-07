"""The identity a node's Agent records as ``node_instance_id``, per provider.

HyperPod installs record the Kubernetes Node UID; a plain-EKS (``eks-managed``)
install records the EC2 instance id behind the Node's ``providerID`` -- the join
binds the Agent to the instance the eks adapter reboots. Every runner check
that compares an Agent (its Store record or its ``/etc/gpu-fault/node-agent.env``
``GPU_FAULT_NODE_INSTANCE_ID``) against "the node" must compare against this,
not the Node UID: DESTR-014's preflight, its host recovery binding and the
DESTR-008 service window each stopped an eks run on exactly that assumption.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gpu_fault.node_provider import parse_provider_id
from scripts.e2e.regional.regional_workload_inventory import node_provider


def agent_node_identity(node: Mapping[str, Any]) -> str | None:
    """The identity the node's Agent records as ``node_instance_id``.

    ``node`` is the warm-spare node snapshot (``labels``, ``provider_id``,
    ``uid``); the plain regional projection carries neither labels nor
    ``providerID`` and must not be used here. ``None`` when the view cannot
    tell its provider or lacks the fact that provider needs -- fail-closed: a
    label-less view once fell back to the Node UID for an eks node whose Agent
    correctly records the instance id (live DESTR-014 attempt 12).
    """

    labels = node.get("labels") or {}
    provider = node_provider(labels)
    if provider == "unknown" and any(
        key.startswith("sagemaker.amazonaws.com/") and value
        for key, value in labels.items()
    ):
        # The warm-spare view projects a fixed label set; any HyperPod label
        # (health status, instance group) names the provider as surely as the
        # cluster label does.
        provider = "hyperpod"
    if provider == "eks-managed":
        parsed = parse_provider_id(node.get("provider_id"))
        return parsed[1] if parsed else None
    if provider != "hyperpod":
        return None
    uid = node.get("uid")
    return str(uid) if uid else None
