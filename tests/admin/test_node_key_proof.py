from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pytest

from gpu_fault.admin import node_key_proof
from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.node_key_proof import read_node_key_proof


class Runner(CommandRunner):
    def __init__(self, output: str) -> None:
        super().__init__()
        self.output = output
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(self, arguments: Sequence[str], **kwargs: Any) -> str:
        self.calls.append((list(arguments), kwargs))
        assert kwargs["capture"] is True and kwargs["sensitive"] is True
        assert kwargs["timeout_seconds"] == 30
        return self.output


def secret() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": "gpu-fault-node-action-keys",
            "namespace": "fixture",
            "uid": "fixture-uid",
            "resourceVersion": "1",
        },
        "data": {"node-a": base64.b64encode(b"fixture-key-material" * 3).decode()},
    }


def test_private_key_read_returns_digests_without_material() -> None:
    document = secret()
    runner = Runner(json.dumps(document))
    proof = read_node_key_proof(runner, ["kubectl", "--context", "fixture"], "fixture")
    assert proof is not None
    assert proof.digests == {
        "node-a": hashlib.sha256(b"fixture-key-material" * 3).hexdigest()
    }
    assert proof.uid == "fixture-uid" and proof.version == "1"
    assert "fixture-key-material" not in repr(proof)
    assert document["data"]["node-a"] not in repr(runner.calls)


@pytest.mark.parametrize(
    ("field", "value"),
    [("name", "other"), ("namespace", "other"), ("uid", ""), ("resourceVersion", None)],
)
def test_secret_identity_is_required(field: str, value: object) -> None:
    document = secret()
    document["metadata"][field] = value
    with pytest.raises(BootstrapError, match="identity"):
        read_node_key_proof(Runner(json.dumps(document)), ["kubectl"], "fixture")


@pytest.mark.parametrize(
    "encoded", ["not base64", 123, base64.b64encode(b"short").decode(), "/w==" * 12]
)
def test_invalid_material_is_refused_without_echoing_it(encoded: object) -> None:
    document = secret()
    document["data"]["node-a"] = encoded
    with pytest.raises(BootstrapError, match="invalid") as rejected:
        read_node_key_proof(Runner(json.dumps(document)), ["kubectl"], "fixture")
    assert str(encoded) not in str(rejected.value)


@pytest.mark.parametrize("output", ["[]", "{bad", "null"])
def test_malformed_secret_response_is_not_absence(output: str) -> None:
    with pytest.raises(BootstrapError):
        read_node_key_proof(Runner(output), ["kubectl"], "fixture")


def test_only_a_successful_empty_read_proves_absence() -> None:
    assert read_node_key_proof(Runner(""), ["kubectl"], "fixture") is None


def test_any_rotation_marker_requires_reconciliation() -> None:
    document = secret()
    document["metadata"]["annotations"] = {
        "gpu-fault.io/node-action-key-rotation": "not-a-valid-marker"
    }
    proof = read_node_key_proof(Runner(json.dumps(document)), ["kubectl"], "fixture")
    assert proof is not None and proof.rotation_pending


def test_custody_loader_is_part_of_the_deploy_host_import_closure(
    tmp_path, monkeypatch
) -> None:
    from scripts.component_wheels import component_modules

    expected = {
        "gpu_fault.admin.node_key_custody",
        "gpu_fault.admin.node_key_custody_models",
        "gpu_fault.admin.node_key_custody_crypto",
        "gpu_fault.admin.node_key_custody_chain",
        "gpu_fault.admin.node_key_custody_admin",
        "gpu_fault.admin.node_key_custody_admin_config",
        "gpu_fault.admin.node_key_custody_admin_probe",
        "gpu_fault.admin.node_key_custody_admin_entry",
    }
    assert expected <= component_modules("deploy_host")
    for component in ("node_runtime", "executor", "control_plane"):
        assert not expected & component_modules(component)
    calls = []
    sentinel = object()

    def load(path, pin):
        calls.append((path, pin))
        return sentinel

    monkeypatch.setattr(node_key_proof.ProvisionCustody, "load", load)
    path = tmp_path / "request.json"
    assert node_key_proof.load_node_key_custody_request(path, "a" * 64) is sentinel
    assert calls == [(path, "a" * 64)]
