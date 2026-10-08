from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path


class NodeActionKeyUnavailableError(ValueError):
    """No action key for ``node_id`` is available to this process.

    A ``ValueError`` so every caller that fails closed on one still does; the
    type lets the node-action adapter tell this from any other refusal and hold
    a freshly onboarded node's step while its per-node key is still propagating
    (live 2026-10-08: the first command for a replacement node landed the second
    its Agent registered, before the key Secret had reached the executor).
    """

    def __init__(self, node_id: str, *, shared: bool = False) -> None:
        self.node_id = node_id
        super().__init__(
            f"shared node action secret is unavailable for {node_id}"
            if shared
            else f"node action secret is unavailable for {node_id}"
        )


def derive_node_action_secret(fleet_secret: str, cluster_id: str, node_id: str) -> str:
    """Derive a node-scoped HMAC key from the fleet master secret."""
    if len(fleet_secret) < 32:
        raise ValueError("node action fleet secret must be at least 32 characters")
    if not cluster_id or not node_id:
        raise ValueError("node action key derivation requires cluster_id and node_id")
    context = (f"gpu-fault/node-action/v1\0{cluster_id}\0{node_id}").encode()
    return hmac.new(fleet_secret.encode(), context, hashlib.sha256).hexdigest()


def node_action_secrets_from_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    raw = os.getenv(
        "GPU_FAULT_NODE_ACTION_KEYS_JSON",
        os.getenv("GPU_FAULT_NODE_ACTION_SECRETS_JSON", ""),
    ).strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("GPU_FAULT_NODE_ACTION_KEYS_JSON must be JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "GPU_FAULT_NODE_ACTION_KEYS_JSON must map node IDs to secrets"
            )
        values.update(parsed)
    directory_name = os.getenv(
        "GPU_FAULT_NODE_ACTION_KEYS_DIR",
        os.getenv("GPU_FAULT_NODE_ACTION_SECRETS_DIR", ""),
    ).strip()
    if directory_name:
        directory = Path(directory_name)
        if not directory.is_dir():
            raise ValueError("GPU_FAULT_NODE_ACTION_KEYS_DIR must be a directory")
        values.update(
            {
                path.name: path.read_text(encoding="utf-8").strip()
                for path in directory.iterdir()
                if path.is_file()
            }
        )
    if any(
        not node_id or not isinstance(secret, str) or len(secret) < 32
        for node_id, secret in values.items()
    ):
        raise ValueError(
            "node action node secrets must map node IDs to "
            "secrets of at least 32 characters"
        )
    return values


def resolve_node_action_secret(
    fleet_secret: str,
    node_secrets: dict[str, str],
    cluster_id: str,
    node_id: str,
    key_version: int,
) -> str:
    if key_version == 1:
        if not fleet_secret:
            raise NodeActionKeyUnavailableError(node_id, shared=True)
        return fleet_secret
    if key_version == 2:
        node_secret = node_secrets.get(node_id) or node_secrets.get(
            f"{cluster_id}/{node_id}"
        )
        if node_secret is not None:
            return node_secret
        if not fleet_secret:
            raise NodeActionKeyUnavailableError(node_id)
        return derive_node_action_secret(fleet_secret, cluster_id, node_id)
    raise ValueError(f"unsupported node action key version {key_version}")
