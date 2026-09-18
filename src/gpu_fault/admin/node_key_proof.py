"""Private, identity-checked reads of node-key material; only digests escape."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.node_key_custody import ProvisionCustody

NODE_KEY_ROTATION_ANNOTATION = "gpu-fault.io/node-action-key-rotation"


def load_node_key_custody_request(
    path: Path, trust_sha256: str, *, expected_input_sha256: str | None = None
) -> ProvisionCustody:
    """Keep the provisioning receipt implementation in the deploy-host closure."""
    if expected_input_sha256 is None:
        return ProvisionCustody.load(path, trust_sha256)
    return ProvisionCustody.load(
        path, trust_sha256, expected_input_sha256=expected_input_sha256
    )


@dataclass(frozen=True)
class NodeKeyProof:
    uid: str
    version: str
    digests: dict[str, str]
    rotation_pending: bool


def node_key_digest(encoded: object) -> str:
    try:
        if not isinstance(encoded, str):
            raise ValueError
        value = base64.b64decode(encoded, validate=True)
        if (
            base64.b64encode(value).decode("ascii") != encoded
            or len(value.decode("utf-8").strip()) < 32
        ):
            raise ValueError
    except (ValueError, UnicodeError):
        raise BootstrapError("node-key Secret contains invalid key material") from None
    return hashlib.sha256(value).hexdigest()


def read_node_key_proof(
    runner: CommandRunner,
    kubectl: Sequence[str],
    namespace: str,
    *,
    secret_name: str = "gpu-fault-node-action-keys",
) -> NodeKeyProof | None:
    output = runner.run(
        [
            *kubectl,
            "-n",
            namespace,
            "get",
            "secret",
            secret_name,
            "--ignore-not-found",
            "-o",
            "json",
        ],
        capture=True,
        sensitive=True,
        timeout_seconds=30,
    )
    if not output.strip():
        return None
    try:
        document = json.loads(output)
        metadata = document["metadata"]
        uid, version = metadata["uid"], metadata["resourceVersion"]
        data = document.get("data")
        data = {} if data is None else data
        annotations = metadata.get("annotations")
        annotations = {} if annotations is None else annotations
        if (
            document.get("apiVersion") != "v1"
            or document.get("kind") != "Secret"
            or document.get("type") != "Opaque"
            or metadata.get("name") != secret_name
            or metadata.get("namespace") != namespace
            or not isinstance(uid, str)
            or not uid
            or not isinstance(version, str)
            or not version
            or metadata.get("deletionTimestamp")
            or not isinstance(data, dict)
            or any(not isinstance(node, str) or not node for node in data)
            or not isinstance(annotations, dict)
        ):
            raise ValueError
        digests = {node: node_key_digest(value) for node, value in data.items()}
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BootstrapError("node-key Secret identity or data is invalid") from None
    return NodeKeyProof(
        uid, version, digests, NODE_KEY_ROTATION_ANNOTATION in annotations
    )
