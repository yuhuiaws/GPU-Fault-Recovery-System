from __future__ import annotations

import base64
import hashlib
import json

import pytest

from scripts.e2e.regional.credential_value_scan import (
    CredentialScanError,
    credential_value_digests,
)
from scripts.e2e.regional.identity_auth_checks import execution_token_hits
from tests.regional.test_blast_acceptance_review import make_runner


@pytest.mark.parametrize("container", ["list", "nested", "encoded", "key"])
def test_json_credential_leaf_is_detected_without_exposing_it(container: str) -> None:
    secret = "private-fixture-value-" + "x" * 40
    values = {
        "list": [{"token": secret}],
        "nested": {"clusters": {"peer": {"retiring_token": secret}}},
        "encoded": {"value": json.dumps({"execution-token": secret})},
        "key": {secret: "value"},
    }
    digests = credential_value_digests(json.dumps(values[container]).encode())
    assert hashlib.sha256(secret.encode()).hexdigest() in digests
    assert secret not in json.dumps(sorted(digests))


@pytest.mark.parametrize(
    "raw", [b'{"x":1,"x":2}', b'{"unfinished"', b"[" * 100, b'"' * 12]
)
def test_unreadable_json_container_cannot_prove_absence(raw: bytes) -> None:
    with pytest.raises(CredentialScanError):
        credential_value_digests(raw, require_json=True)


def test_opaque_binary_and_literal_values_keep_exact_and_normalized_digests() -> None:
    raw = b"\xffliteral \n"
    assert credential_value_digests(raw) == {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(raw.strip()).hexdigest(),
    }
    with pytest.raises(CredentialScanError):
        credential_value_digests(b"x" * (2 * 1024 * 1024 + 1))


def test_blast_finds_registry_secret_copied_under_an_innocent_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, "GF-REGIONAL-BLAST-004", count=2)
    local, peer = "local-fixture-" + "a" * 40, "peer-fixture-" + "b" * 40
    objects = [
        {
            "kind": "Secret",
            "metadata": {
                "namespace": runner.namespace,
                "name": "gpu-fault-regional-connection",
            },
            "data": {"cluster-token": base64.b64encode(local.encode()).decode()},
        },
        {
            "kind": "Secret",
            "metadata": {"namespace": "foreign", "name": "copied-registry"},
            "data": {
                "clusters.json": base64.b64encode(
                    json.dumps([{"cluster_id": "peer", "token": peer}]).encode()
                ).decode()
            },
        },
    ]
    monkeypatch.setattr(runner, "gpu_json", lambda *_args: {"items": objects})
    scan, _ = runner.scan_gpu_objects(
        runner.targets[0],
        execution_hash="0" * 64,
        foreign_cluster_hashes={hashlib.sha256(peer.encode()).hexdigest()},
    )
    assert len(scan["foreign_cluster_token_hash_hits"]) == 1
    assert peer not in json.dumps(scan)
    assert local not in json.dumps(scan)


def test_auth_scan_checks_json_and_ephemeral_containers() -> None:
    secret = "execution-fixture-" + "z" * 40
    found = execution_token_hits(
        {
            "items": [
                {
                    "metadata": {"name": "copied"},
                    "data": {
                        "config.json": base64.b64encode(
                            json.dumps({"credentials": [secret]}).encode()
                        ).decode()
                    },
                }
            ]
        },
        {
            "items": [
                {
                    "metadata": {"name": "pod"},
                    "spec": {
                        "ephemeralContainers": [
                            {
                                "name": "debug",
                                "env": [
                                    {"name": "CONFIG", "value": json.dumps([secret])}
                                ],
                            }
                        ]
                    },
                }
            ]
        },
        digests={hashlib.sha256(secret.encode()).hexdigest()},
    )
    assert {item["kind"] for item in found} == {"Secret", "Pod"}
    assert secret not in json.dumps(found)
