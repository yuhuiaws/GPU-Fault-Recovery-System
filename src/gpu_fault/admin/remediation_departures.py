"""The departure proof ``submit-remediation`` reads for a node the cluster lost.

``--disposition restore`` skips an incident node whose departure is proven and
validates and releases the nodes the cluster still has; ``--disposition
confirm-node-action`` settles a departed node's parked node actions
``node_departed`` (batch 139). Both judge the proof the evidence close reads
(``admin/incident_close.py`` ``departure_evidence``): the ``kubectl get
nodes`` through the site's GPU kubeconfig that no longer lists the node, and
the provider read -- HyperPod ``list-cluster-nodes`` for the instance the node
name carries, or EC2 ``DescribeInstances`` for the instance the fleet recorded
for the node (the inspection's Agent record ``node_instance_id``). A lookup
that fails is recorded as a blocker on the proof, never as a departure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.workflow_reconcile import _departed_node_evidence, cluster_target
from gpu_fault.departed_nodes import (
    node_evidence,
    provider_lookup_blocker,
    recorded_instance_id,
)


def node_departures(
    site: RenderedSite,
    cluster_id: str,
    node_ids: Iterable[str],
    *,
    agents: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """The departure evidence of each of ``node_ids`` (absent from Kubernetes).

    One provider read per cluster, shared by the nodes; ``read_at`` records
    when Kubernetes and the provider were read.
    """

    target = cluster_target(site, cluster_id)
    provider_inventories: dict[str, Any] = {}
    read_at = {"kubernetes": datetime.now(timezone.utc).isoformat()}
    proofs: dict[str, dict[str, Any]] = {}
    for node_id in sorted(set(node_ids)):
        raw = agents.get(node_id) if isinstance(agents, Mapping) else None
        recorded = (
            recorded_instance_id(raw.get("node_instance_id"))
            if isinstance(raw, Mapping)
            else None
        )
        try:
            departed = _departed_node_evidence(
                site,
                cluster_id,
                node_id,
                provider_inventories,
                read_at,
                instance_id=recorded,
            )
        except BootstrapError as exc:
            departed = {
                "node_id": node_id,
                "exists": False,
                "restored": False,
                "blockers": [provider_lookup_blocker(exc)],
            }
        record = node_evidence(
            node_id,
            departed=departed,
            target=target,
            aws_region=site.release_config.get("aws_region"),
        )
        record["read_at"] = dict(read_at)
        proofs[node_id] = record
    return proofs
