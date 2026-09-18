from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import blast_acceptance_cases_2 as cases
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_blast_acceptance_review import make_runner


def canonical(namespace: str) -> dict[str, Any]:
    return {
        "kind": "Secret",
        "metadata": {"name": "gpu-fault-regional-connection", "namespace": namespace},
        "data": {"cluster-token": base64.b64encode(b"a" * 40).decode()},
    }


@pytest.mark.parametrize(
    "location",
    [
        "secret",
        "data",
        "binary",
        "literal",
        "secret-ref",
        "config-ref",
        "secret-from",
        "config-from",
    ],
)
@pytest.mark.parametrize("matching", [True, False])
def test_gpu_scan_inspects_all_credential_storage_and_reference_forms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, location: str, matching: bool
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[3])
    raw = b"example-value-for-test"
    key = "execution-token" if matching else "ordinary"
    name = "execution-token-copy" if matching else "ordinary"
    extra: dict[str, Any] = {"metadata": {"name": name, "namespace": "other"}}
    if location in {"secret", "data", "binary"}:
        extra["kind"] = "Secret" if location == "secret" else "ConfigMap"
        extra["binaryData" if location == "binary" else "data"] = {
            key: raw.decode() if location == "data" else base64.b64encode(raw).decode()
        }
    else:
        extra["kind"] = "Pod"
        container: dict[str, Any] = {"name": "tool"}
        if location == "literal":
            container["env"] = [{"name": key, "value": raw.decode()}]
        elif location.endswith("-ref"):
            kind = "secretKeyRef" if location == "secret-ref" else "configMapKeyRef"
            container["env"] = [
                {"name": "ordinary", "valueFrom": {kind: {"name": name, "key": key}}}
            ]
        else:
            kind = "secretRef" if location == "secret-from" else "configMapRef"
            container["envFrom"] = [{kind: {"name": name}}]
        extra["spec"] = {
            field: [container]
            for field in ("containers", "initContainers", "ephemeralContainers")
        }
    monkeypatch.setattr(
        runner,
        "gpu_json",
        lambda *args: {
            "items": [canonical(runner.namespace), extra, {"kind": "Other"}]
        },
    )
    digest = base.sha256_bytes(raw)
    result, own_hash = runner.scan_gpu_objects(
        runner.targets[0],
        execution_hash=digest if matching else "f" * 64,
        foreign_cluster_hashes={digest} if matching else set(),
    )
    value_form = location in {"secret", "data", "binary", "literal"}
    assert bool(result["execution_token_value_hash_hits"]) is (matching and value_form)
    assert bool(result["foreign_cluster_token_hash_hits"]) is (matching and value_form)
    assert bool(result["execution_token_name_or_reference_hits"]) is matching
    assert own_hash == base.sha256_bytes(b"a" * 40)
    assert raw.decode() not in json.dumps(result), (
        "scan evidence must not contain credential bytes"
    )


@pytest.mark.parametrize(
    "defect", ["inventory", "absent", "duplicate", "short", "base64"]
)
def test_gpu_scan_cannot_derive_a_pass_from_ambiguous_canonical_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[3])
    item = canonical(runner.namespace)
    document: dict[str, Any] = {"items": [item]}
    if defect == "inventory":
        document = {}
    elif defect == "absent":
        document["items"] = []
    elif defect == "duplicate":
        document["items"].append(item)
    else:
        item["data"]["cluster-token"] = (
            "!" if defect == "base64" else base64.b64encode(b"short").decode()
        )
    monkeypatch.setattr(runner, "gpu_json", lambda *args: document)
    with pytest.raises(base.CheckError):
        runner.scan_gpu_objects(runner.targets[0], execution_hash="e" * 64)


@pytest.mark.parametrize("length", [0, 31, 32])
def test_execution_token_probe_requires_the_minimum_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, length: int
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[3])
    monkeypatch.setattr(
        runner,
        "cpu_text",
        lambda *args: json.dumps({"sha256": "e" * 64, "length": length}),
    )
    if length < 32:
        with pytest.raises(base.CheckError, match="missing or too short"):
            runner.execution_token_digest("api")
    else:
        assert runner.execution_token_digest("api") == ("e" * 64, 32)


@pytest.mark.parametrize("field", ["resourceNames", "nonResourceURLs"])
def test_executor_role_rejects_unmodeled_rule_scopes(field: str) -> None:
    with pytest.raises(base.CheckError, match="unsupported rule scope"):
        cases.BlastCasesTwo.normalized_role_rules(
            {"rules": [{field: ["scoped"], "resources": ["pods"], "verbs": ["get"]}]}
        )


def test_executor_iam_does_not_guess_a_missing_hyperpod_arn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[2])
    monkeypatch.setattr(runner, "gpu_json", lambda *args: {})
    monkeypatch.setattr(runner, "iam_role_policies", lambda *args: ([], {}))
    monkeypatch.setattr(runner, "aws", lambda *args: {})
    with pytest.raises(base.CheckError, match="actual HyperPod ARN"):
        runner.executor_iam_scope(runner.targets[0], {})


def test_blast003_refuses_an_unexpected_manifest_before_audit_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[2])
    monkeypatch.setattr(
        cases, "expected_executor_role", lambda: {"core:nodes": ["delete"]}
    )
    calls = []
    monkeypatch.setattr(runner, "gpu_json", lambda *args: calls.append(args))
    with pytest.raises(base.CheckError, match="ClusterRole grants"):
        runner.blast_003()
    assert calls == []


@pytest.mark.parametrize(
    "runtime",
    [
        None,
        {},
        {"cluster_id": "foreign", "digests": {"x": "a"}},
        {"cluster_id": "cluster-0", "digests": []},
    ],
)
def test_blast004_requires_runtime_binding_not_just_secret_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime: Any
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[3])
    monkeypatch.setattr(runner, "ready_cpu_pod", lambda: "api")
    monkeypatch.setattr(runner, "execution_token_digest", lambda *args: ("e" * 64, 40))
    monkeypatch.setattr(runner, "cluster_token_digest", lambda *args: "a" * 64)
    monkeypatch.setattr(
        runner, "scan_gpu_objects", lambda *args, **kwargs: ({}, "a" * 64)
    )
    monkeypatch.setattr(runner, "ready_executor_pod", lambda *args: "executor")
    monkeypatch.setattr(runner, "gpu_text", lambda *args: json.dumps(runtime))
    with pytest.raises(base.CheckError, match="incomplete or foreign"):
        runner.blast_004()


def test_one_cluster_cannot_pass_cross_cluster_token_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[3])
    monkeypatch.setattr(runner, "ready_cpu_pod", lambda: "api")
    monkeypatch.setattr(runner, "execution_token_digest", lambda *args: ("e" * 64, 40))
    monkeypatch.setattr(runner, "cluster_token_digest", lambda *args: "a" * 64)
    clean = {
        "execution_token_value_hash_hits": [],
        "execution_token_name_or_reference_hits": [],
        "foreign_cluster_token_hash_hits": [],
    }
    monkeypatch.setattr(
        runner, "scan_gpu_objects", lambda *args, **kwargs: (clean, "a" * 64)
    )
    monkeypatch.setattr(runner, "ready_executor_pod", lambda *args: "executor")
    monkeypatch.setattr(
        runner,
        "gpu_text",
        lambda *args: json.dumps(
            {
                "cluster_id": "cluster-0",
                "digests": {"GPU_FAULT_CONTROL_PLANE_TOKEN": "a" * 64},
            }
        ),
    )
    assert runner.run() == 1
    result = json.loads((runner.run_dir / f"{runner.case_id}.json").read_text())
    assert result["verdict"] == "FAIL"
    assert result["checks"]["cluster_token_count"] == 1
    assert any(
        "one registered physical GPU cluster" in message
        for message in result["limitations"]
    ), "single-cluster evidence omitted its cross-cluster validation limit"
