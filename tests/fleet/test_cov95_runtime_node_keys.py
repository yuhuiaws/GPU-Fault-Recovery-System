from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpu_fault.node_action_keys import (
    derive_node_action_secret,
    node_action_secrets_from_environment,
    resolve_node_action_secret,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    ("master", "cluster", "node"),
    [("short", "a", "node"), ("m" * 32, "", "node"), ("m" * 32, "a", "")],
)
def test_node_key_derivation_requires_master_and_both_identity_fields(
    master: str, cluster: str, node: str
) -> None:
    with pytest.raises(ValueError, match="32 characters|cluster_id and node_id"):
        derive_node_action_secret(master, cluster, node)


def test_derived_key_is_bound_to_cluster_and_node() -> None:
    keys = [
        derive_node_action_secret("m" * 32, cluster, node)
        for cluster, node in (("a", "node-a"), ("a", "node-b"), ("b", "node-a"))
    ]
    assert all(len(key) == 64 for key in keys), (
        "derived keys must have the protocol SHA256 length"
    )
    assert len(set(keys)) == 3
    assert keys[0] == derive_node_action_secret("m" * 32, "a", "node-a")


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[]",
        '{"node":"short"}',
        '{"": "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn"}',
        '{"node": 123}',
    ],
)
def test_node_key_environment_rejects_malformed_or_short_entries(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEYS_JSON", raw)
    with pytest.raises(ValueError, match="JSON|map node IDs"):
        node_action_secrets_from_environment()


@pytest.mark.parametrize("legacy", [False, True])
def test_node_key_directory_overrides_inline_values_without_reading_subdirectories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, legacy: bool
) -> None:
    prefix = "GPU_FAULT_NODE_ACTION_SECRETS" if legacy else "GPU_FAULT_NODE_ACTION_KEYS"
    monkeypatch.setenv(
        prefix + "_JSON", json.dumps({"node-a": "a" * 32, "node-b": "b" * 32})
    )
    directory = tmp_path / "keys"
    directory.mkdir()
    (directory / "node-a").write_text("c" * 32 + "\n", encoding="ascii")
    (directory / "skip-directory").mkdir()
    monkeypatch.setenv(prefix + "_DIR", str(directory))
    values = node_action_secrets_from_environment()
    assert set(values) == {"node-a", "node-b"}
    assert (
        hashlib.sha256(values["node-a"].encode()).hexdigest()
        == hashlib.sha256(b"c" * 32).hexdigest()
    )
    assert len(values["node-b"]) == 32


def test_node_key_directory_must_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_KEYS_DIR", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="must be a directory"):
        node_action_secrets_from_environment()


@pytest.mark.parametrize("version", [0, 3, -1])
def test_unknown_key_version_never_falls_back_to_shared_master(version: int) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        resolve_node_action_secret("m" * 32, {}, "a", "node-a", version)


@pytest.mark.parametrize("version", [1, 2])
def test_missing_credential_is_not_an_empty_valid_key(version: int) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        resolve_node_action_secret("", {}, "a", "node-a", version)


def test_key_resolution_keeps_explicit_node_rotation_and_derived_fallback() -> None:
    master, rotated = "m" * 32, "r" * 32
    assert resolve_node_action_secret(master, {}, "a", "node-a", 1) == master
    assert (
        resolve_node_action_secret(master, {"node-a": rotated}, "a", "node-a", 2)
        == rotated
    )
    assert (
        resolve_node_action_secret("", {"a/node-a": rotated}, "a", "node-a", 2)
        == rotated
    )
    assert resolve_node_action_secret(
        master, {}, "a", "node-a", 2
    ) == derive_node_action_secret(master, "a", "node-a")
