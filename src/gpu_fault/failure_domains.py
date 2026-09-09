"""Node failure domains, derived from Kubernetes node labels.

One GPU cluster in this architecture is one availability zone, so the zone
label that ordinary Kubernetes topology tooling treats as *the* failure domain
cannot separate any two nodes of the same cluster. The domains that do matter
at 500 nodes in one zone are smaller: a HyperPod instance group (the unit
capacity is bought and rolled in), a rack or PDU, a network leaf. The label
priority below is the single place that ordering lives; the remediation
budget's ``domain:`` scope and the fleet rollout's per-domain cap both read it
so they cannot disagree about what a "domain" is.

Two consumers, two shapes of the same reading. The remediation map
(``failure_domain_map``) *omits* nodes that carry none of the labels: an
unlabelled node consumes node, cluster and region quota only, never a
``domain:`` slot, because one shared bucket would make every unlabelled node
contend for a single slot and look like a stuck budget. The fleet rollout's
request model needs a value for every node it rolls, so that path substitutes
``UNKNOWN_FAILURE_DOMAIN`` for the same nodes, and its wave policy reads the
sentinel as "no known blast radius, one node per wave". The labelled nodes are
identical in both shapes; the unlabelled set is the same set.

The ConfigMap that ships the remediation map to the control-worker is also
rendered here so the release engine and the administrator tooling produce one
byte-identical document and one digest for it.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# The fleet rollout's stand-in for a node without any failure-domain label. It
# never appears in the remediation map; see the module docstring.
UNKNOWN_FAILURE_DOMAIN = "UNKNOWN"

FAILURE_DOMAIN_CONFIGMAP = "gpu-fault-failure-domain-map"
FAILURE_DOMAIN_FILE = "failure-domains.json"
FAILURE_DOMAIN_MOUNT_DIR = "/etc/gpu-fault/failure-domains"
FAILURE_DOMAIN_MAP_PATH = f"{FAILURE_DOMAIN_MOUNT_DIR}/{FAILURE_DOMAIN_FILE}"
FAILURE_DOMAIN_MAP_ANNOTATION = "gpu-fault.io/failure-domain-map-sha256"

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
    operator asked for and would look like a stuck budget. The fleet rollout
    reads the same labels through ``node_failure_domain`` and fills the gap
    with ``UNKNOWN_FAILURE_DOMAIN`` instead; see the module docstring.
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


def failure_domain_document(mapping: Mapping[str, Mapping[str, str]]) -> str:
    """The JSON file the executor loads, in one canonical byte order."""

    return (
        json.dumps(
            {
                cluster: dict(sorted(nodes.items()))
                for cluster, nodes in sorted(mapping.items())
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def failure_domain_configmap(
    mapping: Mapping[str, Mapping[str, str]],
    *,
    namespace: str = "gpu-fault-system",
) -> dict[str, Any]:
    """The ConfigMap the control-worker mounts; ``map-path`` feeds the env var."""

    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": FAILURE_DOMAIN_CONFIGMAP, "namespace": namespace},
        "data": {
            FAILURE_DOMAIN_FILE: failure_domain_document(mapping),
            "map-path": FAILURE_DOMAIN_MAP_PATH,
        },
    }


def failure_domain_map_sha256(manifest: Mapping[str, Any]) -> str:
    """Digest of the map document alone.

    Stamped on the control-worker pod template so the worker rolls exactly
    when the document it mounts changed -- not when the namespace, the
    manifest key order or an unrelated release field did.
    """

    data = manifest.get("data") or {}
    return hashlib.sha256(
        str(data.get(FAILURE_DOMAIN_FILE) or "").encode("utf-8")
    ).hexdigest()


def validate_failure_domain_manifest(manifest: Mapping[str, Any]) -> None:
    """Run the rendered document through the executor's own loader.

    The control-worker refuses to start on a malformed map, so the render
    refuses to ship one: whatever ``load_failure_domain_map`` would reject at
    worker start-up is rejected here, before ``kubectl apply``, with the same
    message. Raises ``ValueError``.
    """

    # Imported here so this module stays free of the execution layer for its
    # other readers (the fleet rollout imports it for the label priority only).
    from gpu_fault.execution.remediation_budget import load_failure_domain_map

    document = str((manifest.get("data") or {}).get(FAILURE_DOMAIN_FILE) or "")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-failure-domains-") as directory:
        path = Path(directory) / FAILURE_DOMAIN_FILE
        path.write_text(document, encoding="utf-8")
        load_failure_domain_map(path)
