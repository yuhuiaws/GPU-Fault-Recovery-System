"""Freshly bound CPU credential shape checks, never password currency proofs."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.execution import PROOFS, ProofKey, ProofSubject
from gpu_fault_release.regional_release_config import ReleaseError

CPU_SECRET_NAMES = (
    "gpu-fault-aurora",
    "gpu-fault-control-plane-active",
    "gpu-fault-node-action-keys",
)


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
