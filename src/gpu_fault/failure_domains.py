"""Node failure domains, derived from Kubernetes node labels.

One GPU cluster in this architecture is one availability zone, so the zone
label that ordinary Kubernetes topology tooling treats as *the* failure domain
cannot separate any two nodes of the same cluster. The domains that do matter
at 500 nodes in one zone are smaller: a HyperPod instance group (the unit
capacity is bought and rolled in), a rack or PDU, a network leaf. The label
priority below is the single place that ordering lives; the fleet rollout's
per-domain wave cap reads it (``node_failure_domain``).

The remediation budget does not. A training job, or a GPU node that runs
none, is one workflow -- its faults merge into it and its nodes and GPUs are
repaired in parallel inside it -- and the per-node scope (with a job token for
the steps that stop or restart a whole job) serializes a second workflow; the
only ``domain:`` scope left is an SXID's NVSwitch fabric partition. Read from these labels, every GPU node of an
instance group or EKS node group shared one slot, so two jobs, or two idle
nodes, of one group were repaired one at a time.

The node map (``failure_domain_map``) is still rendered and published as the
ConfigMap ``gpu-fault-failure-domain-map`` -- by every ``deploy`` and by
``join-cluster``/``remove-cluster``, which the release engine and the
administrator tooling produce as one byte-identical document with one digest
-- and the control-worker still mounts it, but nothing at runtime reads it:
it is the published label reading of the fleet, a read-only topology view, not
a limit. It *omits* nodes that carry none of the labels, while the fleet
rollout's request model needs a value for every node it rolls and substitutes
``UNKNOWN_FAILURE_DOMAIN``, which its wave policy reads as "no known blast
radius, one node per wave". The labelled nodes are identical in both shapes;
the unlabelled set is the same set.
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
# The control-worker manifest points this variable at the mounted map (the
# ConfigMap's ``map-path``); no runtime code reads it (module docstring).
FAILURE_DOMAIN_MAP_ENV = "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP"

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
    """The JSON file the ConfigMap ships, in one canonical byte order."""

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
    """The ConfigMap the control-worker mounts; ``map-path`` sets the env var."""

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


def load_failure_domain_map(path: str | Path) -> dict[str, dict[str, str]]:
    """Read ``{cluster_id: {node_id: failure_domain}}`` or raise ``ValueError``.

    Malformed input is refused rather than skipped: a map that quietly drops
    half its nodes is not a reading of the fleet.
    """

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"failure domain map {path} is unreadable: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"failure domain map {path} must be a JSON object")
    mapping: dict[str, dict[str, str]] = {}
    for cluster_id, nodes in document.items():
        if not isinstance(nodes, dict):
            raise ValueError(
                f"failure domain map {path}: cluster {cluster_id!r} must map "
                "node ids to failure domains"
            )
        for node_id, domain in nodes.items():
            if not isinstance(domain, str) or not domain:
                raise ValueError(
                    f"failure domain map {path}: {cluster_id}/{node_id} must "
                    "name a non-empty failure domain"
                )
        mapping[str(cluster_id)] = {
            str(node_id): domain for node_id, domain in nodes.items()
        }
    return mapping


def validate_failure_domain_manifest(manifest: Mapping[str, Any]) -> None:
    """Run the rendered document through ``load_failure_domain_map``.

    The render refuses to ship a map its own reader rejects: the check runs
    before ``kubectl apply``, with the reader's message. Raises ``ValueError``.
    """

    document = str((manifest.get("data") or {}).get(FAILURE_DOMAIN_FILE) or "")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-failure-domains-") as directory:
        path = Path(directory) / FAILURE_DOMAIN_FILE
        path.write_text(document, encoding="utf-8")
        load_failure_domain_map(path)
