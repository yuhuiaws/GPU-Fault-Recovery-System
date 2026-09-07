"""Node failure domains, derived from Kubernetes node labels.

One GPU cluster in this architecture is one availability zone, so the zone
label that ordinary Kubernetes topology tooling treats as *the* failure domain
cannot separate any two nodes of the same cluster. The domains that do matter
at 500 nodes in one zone are smaller: a HyperPod instance group (the unit
capacity is bought and rolled in), a rack or PDU, a network leaf. The label
priority below is the single place that ordering lives; the remediation
budget's ``domain:`` scope and the fleet rollout's per-domain cap both read it
so they cannot disagree about what a "domain" is.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Finest known domain first. Operators whose fleet carries EC2 instance
# topology labels (``topology.k8s.aws/network-node-layer-N``) can pass those
# keys explicitly; they are not defaulted because which layer is a rack and
# which is a spine depends on the instance family.
FAILURE_DOMAIN_LABELS: tuple[str, ...] = (
    "sagemaker.amazonaws.com/instance-group-name",
    "topology.kubernetes.io/zone",
    "failure-domain.beta.kubernetes.io/zone",
)


def node_failure_domain(
    labels: Mapping[str, Any],
    *,
    label_keys: Sequence[str] = FAILURE_DOMAIN_LABELS,
) -> str | None:
    """The first non-blank label value in priority order, or ``None``."""

    for key in label_keys:
        value = str(labels.get(key) or "").strip()
        if value:
            return value
    return None


def failure_domain_map(
    cluster_id: str,
    node_items: Sequence[Mapping[str, Any]],
    *,
    label_keys: Sequence[str] = FAILURE_DOMAIN_LABELS,
) -> dict[str, dict[str, str]]:
    """``{cluster_id: {node_id: domain}}`` for the nodes that carry a domain.

    Nodes without any of the labels are left out rather than mapped to an
    ``UNKNOWN`` bucket: one shared bucket would make every unlabelled node
    contend for a single ``domain:`` slot, which is a stricter limit than the
    operator asked for and would look like a stuck budget.
    """

    nodes: dict[str, str] = {}
    for item in node_items:
        metadata = item.get("metadata") or {}
        node_id = str(metadata.get("name") or "")
        if not node_id:
            continue
        domain = node_failure_domain(
            metadata.get("labels") or {}, label_keys=label_keys
        )
        if domain is not None:
            nodes[node_id] = domain
    return {cluster_id: nodes}
