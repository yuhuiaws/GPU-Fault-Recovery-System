"""Freshly bound CPU credential shape checks, never password currency proofs.

The ``cpu_secrets`` check (``cpu_secrets_check``) reads three CPU Secrets. The
node-key Secret mirrors one Node Action key per GPU node, so it must hold a key
while any GPU cluster of the site has a node. It may be empty -- there, with no
key -- when the site has no GPU cluster, and when no GPU cluster has a node:
every cluster's selector selects none, parked or unavailable, by the release
engine's zero-node proof (``regional_parked_cluster.without_nodes``). That is a
site whose clusters have no keyed node -- its last keyed cluster removed, or a
first cluster joined without nodes. Only an empty Secret asks for the proof,
and the proof's reads join the report snapshot the caller opened (``status``,
``preflight``, ``verify``), never opening one. A node list that cannot be read
proves nothing, so the empty Secret fails as before; so does one that is
missing or cannot be read (its read fails) or that the read did not return as
itself (``_present``): "empty but present" is the only relaxed case.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.execution import PROOFS, ProofKey, ProofSubject
from gpu_fault_release import regional_parked_cluster as parked_cluster
from gpu_fault_release.regional_release_config import ReleaseError

CPU_SECRET_NAMES = (
    "gpu-fault-aurora",
    "gpu-fault-control-plane-active",
    "gpu-fault-node-action-keys",
)
NODE_KEYS_SECRET = CPU_SECRET_NAMES[-1]
#: The summary of a passing ``cpu_secrets`` check.
SUMMARY = "required CPU secrets exist without exposing their values"
#: The ``cpu_secrets`` detail of an empty node-key Secret no GPU cluster needs:
#: one line naming every GPU cluster and its zero-node state.
NODE_KEYS_NOT_NEEDED_KEY = "node_action_keys_not_needed"


def _validate_shapes(
    data: list[dict[str, Any]], *, requires_node_keys: bool
) -> dict[str, object]:
    aurora, active, node_keys = data
    required_aurora = {"postgres-url", "master-secret-arn"}
    required_active = {
        "execution-token",
        "processor-replay-secret",
        "node-action-secret",
    }
    if missing := sorted(required_aurora - set(aurora)):
        raise ReleaseError("gpu-fault-aurora is missing: " + ", ".join(missing))
    if missing := sorted(required_active - set(active)):
        raise ReleaseError(
            "gpu-fault-control-plane-active is missing: " + ", ".join(missing)
        )
    decoded = {
        name: base64.b64decode(active[name].encode(), validate=True)
        for name in required_active
    }
    if any(len(value) < 32 for value in decoded.values()):
        raise ReleaseError("control-plane active secrets must be at least 32 bytes")
    if len({hashlib.sha256(value).hexdigest() for value in decoded.values()}) != len(
        required_active
    ):
        raise ReleaseError("control-plane active secrets must be pairwise distinct")
    if requires_node_keys and not node_keys:
        raise ReleaseError("gpu-fault-node-action-keys is empty")
    return {
        "aurora_keys": sorted(aurora),
        "active_keys": sorted(active),
        "node_action_key_count": len(node_keys),
    }


def cpu_secret_shapes(
    documents: Sequence[dict[str, Any]],
    *,
    context: list[str],
    namespace: str,
    requires_node_keys: bool,
) -> dict[str, object]:
    if len(documents) != len(CPU_SECRET_NAMES):
        raise ReleaseError("CPU credential document set is incomplete")
    versions = [
        (
            str((item.get("metadata") or {}).get("uid") or ""),
            str((item.get("metadata") or {}).get("resourceVersion") or ""),
        )
        for item in documents
    ]
    data = [item.get("data") or {} for item in documents]
    if any(not isinstance(item, dict) for item in data):
        raise ReleaseError("CPU credential data is malformed")

    def verify() -> dict[str, object]:
        return _validate_shapes(data, requires_node_keys=requires_node_keys)

    if any(not uid or not version for uid, version in versions):
        return verify()
    return PROOFS.verify(
        ProofKey(
            ProofSubject.CREDENTIAL_SHAPE,
            json.dumps([context, namespace, versions]),
            hashlib.sha256(
                json.dumps([requires_node_keys, data], sort_keys=True).encode()
            ).hexdigest(),
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            json.dumps(versions),
        ),
        verify,
        max_age=30,
    )


def _present(document: Mapping[str, Any], namespace: str) -> bool:
    """Whether the node-key read returned that Secret, there to stay: its name
    in ``namespace``, a UID, no deletion under way and, when the answer names
    one, kind ``Secret``. Nothing else stands in for an empty Secret."""

    metadata = document.get("metadata")
    return (
        isinstance(metadata, Mapping)
        and document.get("kind", "Secret") == "Secret"
        and metadata.get("name") == NODE_KEYS_SECRET
        and metadata.get("namespace") == namespace
        and bool(metadata.get("uid"))
        and not metadata.get("deletionTimestamp")
    )


def _zero_node_states(release: Any) -> dict[str, str]:
    """Cluster id -> ``parked`` or ``unavailable`` for every GPU cluster of the
    site, each one's selector proven to select no node; the empty node-key
    Secret is refused, naming the cluster, at the first one that selects a
    node or whose node list cannot be read."""

    states: dict[str, str] = {}
    for target in release.config.clusters:
        proof = parked_cluster.without_nodes(release, target)
        if proof is None:
            raise ReleaseError(
                f"{NODE_KEYS_SECRET} is empty, but GPU cluster {target.cluster_id} "
                "selects a node (or its node list cannot be read)"
            )
        states[str(target.cluster_id)] = parked_cluster.state_of(proof)
    return states


def cpu_secrets_check(
    release: Any, documents: Sequence[dict[str, Any]], *, context: list[str]
) -> tuple[str, dict[str, object]]:
    """The ``cpu_secrets`` check of ``documents`` -- the CPU Secrets as read, in
    ``CPU_SECRET_NAMES`` order; a Secret that is missing or cannot be read
    failed its read before this: its summary and details.

    An empty node-key Secret passes only while no GPU cluster has a node (the
    module docstring); its summary then ends with the one line the details
    keep under ``NODE_KEYS_NOT_NEEDED_KEY``, naming every cluster's state.
    """

    namespace = release.config.namespace
    details = cpu_secret_shapes(
        documents, context=context, namespace=namespace, requires_node_keys=False
    )
    if details["node_action_key_count"] or not release.config.clusters:
        return SUMMARY, details
    if not _present(documents[-1], namespace):
        raise ReleaseError(
            f"{NODE_KEYS_SECRET} is empty and was not read back as the live "
            f"Secret in {namespace}"
        )
    states = _zero_node_states(release)
    line = (
        f"{NODE_KEYS_SECRET} is empty and needs no key: no GPU cluster has a node ("
        + ", ".join(f"{name}: {state}" for name, state in sorted(states.items()))
        + ")"
    )
    return f"{SUMMARY}; {line}", {**details, NODE_KEYS_NOT_NEEDED_KEY: line}
