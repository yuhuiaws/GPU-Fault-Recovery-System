from __future__ import annotations

import base64
import copy
from types import SimpleNamespace

import pytest

from gpu_fault.admin.execution import ProofCache
from gpu_fault_release import regional_admin_checks
from gpu_fault_release import regional_secret_checks as checks
from gpu_fault_release.regional_release_config import ReleaseError


def documents():
    return [
        {
            "metadata": {"uid": "database", "resourceVersion": "1"},
            "data": {"postgres-url": "reference", "master-secret-arn": "reference"},
        },
        {
            "metadata": {"uid": "active", "resourceVersion": "2"},
            "data": {
                name: base64.b64encode(character.encode() * 32).decode()
                for name, character in (
                    ("execution-token", "a"),
                    ("processor-replay-secret", "b"),
                    ("node-action-secret", "c"),
                )
            },
        },
        {
            "metadata": {"uid": "keys", "resourceVersion": "3"},
            "data": {"node-a": "reference"},
        },
    ]


def test_shape_proof_reuse_is_bound_to_uid_version_content_and_scope(monkeypatch):
    monkeypatch.setattr(checks, "PROOFS", ProofCache())
    decodes = []
    decode = base64.b64decode

    def record(*args, **kwargs):
        decodes.append(True)
        return decode(*args, **kwargs)

    monkeypatch.setattr(checks.base64, "b64decode", record)
    source = documents()
    arguments = {
        "context": ["kubectl", "--kubeconfig", "example-cpu"],
        "namespace": "gpu-fault-system",
        "requires_node_keys": True,
    }
    first = checks.cpu_secret_shapes(source, **arguments)
    checks.cpu_secret_shapes(source, **arguments)
    assert len(decodes) == 3
    first["active_keys"].append("not-a-real-key")
    assert (
        "not-a-real-key"
        not in checks.cpu_secret_shapes(source, **arguments)["active_keys"]
    )
    source[1]["metadata"]["resourceVersion"] = "4"
    checks.cpu_secret_shapes(source, **arguments)
    assert len(decodes) == 6
    checks.cpu_secret_shapes(source, **{**arguments, "namespace": "another"})
    assert len(decodes) == 9
    source[1]["data"]["execution-token"] = base64.b64encode(b"too short").decode()
    with pytest.raises(ReleaseError, match="at least 32"):
        checks.cpu_secret_shapes(source, **arguments)
    assert len(decodes) == 12


def test_missing_metadata_is_revalidated_and_no_secret_values_are_returned(monkeypatch):
    monkeypatch.setattr(checks, "PROOFS", ProofCache())
    source = documents()
    source[1].pop("metadata")
    arguments = {
        "context": ["kubectl"],
        "namespace": "gpu-fault-system",
        "requires_node_keys": True,
    }
    report = checks.cpu_secret_shapes(source, **arguments)
    assert report == {
        "aurora_keys": ["master-secret-arn", "postgres-url"],
        "active_keys": [
            "execution-token",
            "node-action-secret",
            "processor-replay-secret",
        ],
        "node_action_key_count": 1,
    }
    changed = copy.deepcopy(source)
    changed[1]["data"]["node-action-secret"] = changed[1]["data"]["execution-token"]
    with pytest.raises(ReleaseError, match="pairwise distinct"):
        checks.cpu_secret_shapes(changed, **arguments)


def test_empty_node_action_keys_are_valid_for_an_empty_gpu_registry():
    source = documents()
    source[2]["data"] = None
    values = dict(zip(checks.CPU_SECRET_NAMES, source, strict=True))
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=[], namespace="gpu-fault-system"),
        _cpu=lambda *args: ["kubectl", "--kubeconfig", "cpu", *args],
        _get_json=lambda arguments: values[arguments[-1]],
    )
    result = regional_admin_checks.check_cpu_secrets(release)
    assert result.details["node_action_key_count"] == 0
