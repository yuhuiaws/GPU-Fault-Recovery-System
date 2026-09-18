from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import audit_destr013_replacement_invariant as audit
from scripts.e2e.regional import destr013_audit_evidence as binding
from scripts.e2e.regional import regional_commands
from tests.regional.test_destr013_audit_integration import EKS, LocalAudit, write_json


@pytest.mark.parametrize("value", [None, 1, "[]", "false", "{broken"])
def test_binding_requires_a_json_object_without_echoing_raw_probe_data(
    value: Any,
) -> None:
    with pytest.raises(
        binding.AuditError, match="probe is not (valid JSON|a JSON object)"
    ):
        binding.json_object(value, "probe")


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-an-arn",
        "arn:aws:eks:us-west-2:123456789012:cluster/",
        "arn:aws:iam::123456789012:role/executor",
    ],
)
def test_binding_rejects_missing_wrong_service_or_incomplete_eks_arn(
    value: Any,
) -> None:
    with pytest.raises(
        binding.AuditError, match="target eks ARN is (missing|malformed)"
    ):
        binding.arn_parts(value, "eks")


@pytest.mark.parametrize(
    "defect", ["contexts", "clusters", "tls", "server", "missing-file", "yaml"]
)
def test_kubeconfig_binding_requires_one_verified_tls_target(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)
    cluster, digest = binding.kubeconfig_cluster(h.gpu_config, "selected")
    assert cluster["server"] == "https://gpu.example.invalid", cluster
    assert digest == hashlib.sha256(h.gpu_config.read_bytes()).hexdigest(), digest
    if defect == "contexts":
        h.kubeconfig["contexts"] *= 2
    elif defect == "clusters":
        h.kubeconfig["clusters"] *= 2
    elif defect == "tls":
        h.kubeconfig["clusters"][0]["cluster"]["insecure-skip-tls-verify"] = True
    elif defect == "server":
        h.kubeconfig["clusters"][0]["cluster"]["server"] = "http://gpu.example.invalid"
    write_json(h.gpu_config, h.kubeconfig)
    if defect == "missing-file":
        h.gpu_config.unlink()
    elif defect == "yaml":
        h.gpu_config.write_text("contexts: [", encoding="utf-8")
    with pytest.raises(binding.AuditError, match="verified TLS identity is invalid"):
        binding.kubeconfig_cluster(h.gpu_config, "selected")
    assert h.calls == [], "kubeconfig validation must be a local parse"


def test_eks_region_mismatch_refuses_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(binding, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(binding.AuditError, match="Region differs"):
        binding.eks_binding(tmp_path / "unread", "selected", "unit", "us-east-1", EKS)
    assert calls == [], calls


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("registry-generation", "registry generation is missing"),
        ("component-pin", "component pins are missing or malformed"),
        ("template-container", "complete stable Ready Pod population"),
        ("pod-container-id", "complete stable Ready Pod population"),
        ("namespace-anchor", "namespace UID is missing"),
    ],
)
def test_final_audit_main_rejects_unbound_release_and_replica_proof(
    defect: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)
    if defect == "registry-generation":
        h.registry["generation"] = 0
    elif defect == "component-pin":
        h.state["executor_wheel_sha256"] = "not-a-digest"
    elif defect == "template-container":
        containers = h.deployments[binding.API_APP]["spec"]["template"]["spec"][
            "containers"
        ]
        containers.append(dict(containers[1]))
    elif defect == "pod-container-id":
        h.pods[binding.API_APP]["items"][0]["status"]["containerStatuses"][1][
            "containerID"
        ] = ""
    else:
        original = h.command

        def transport(
            argv: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            result = original(argv, **kwargs)
            if "namespace" in argv:
                result.stdout = json.dumps({"metadata": {}})
            return result

        monkeypatch.setattr(regional_commands, "run_command", transport)
    code, report = h.run()
    assert code == 1 and report["verdict"] == "FAIL", report
    assert expected in str(report), report
    assert report["formal_sequence_satisfied"] is False, report
    assert h.iam_calls() == [], "invalid target evidence must not reach IAM simulation"


def test_final_audit_accepts_matching_relative_public_ca_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = LocalAudit(tmp_path, monkeypatch)
    cluster = h.kubeconfig["clusters"][0]["cluster"]
    cluster.pop("certificate-authority-data")
    cluster["certificate-authority"] = "public-ca.pem"
    (tmp_path / "public-ca.pem").write_bytes(b"public-test-certificate")
    write_json(h.gpu_config, h.kubeconfig)
    code, report = h.run()
    assert code == 0 and report["verdict"] == "PASS", report
    assert report["executor_replica_count"] == 3, report
    assert len(h.iam_calls()) == 1, h.calls


@pytest.mark.parametrize("defect", ["replace", "legacy", "missing-switch", "malformed"])
def test_manifest_invariant_audits_parsed_local_configuration(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    executor = root / "deploy" / "dataplane" / "cluster-action-executor.yaml"
    executor.parent.mkdir(parents=True)
    value: Any = {
        "env": [{"name": "GPU_FAULT_ALLOW_HYPERPOD_REPLACE", "value": "false"}]
    }
    expected = ""
    if defect == "replace":
        value["env"][0]["value"] = "true"
        expected = "enables provider replace"
    elif defect == "legacy":
        value["env"].append(
            {"name": "GPU_FAULT_ALLOW_HYPERPOD_MUTATION", "value": "false"}
        )
        expected = "contains legacy mutation switch"
    elif defect == "missing-switch":
        value = {"env": []}
        expected = "has no replace invariant"
    else:
        expected = "cannot be parsed"
    executor.write_text(
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE: ["
        if defect == "malformed"
        else json.dumps(value),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "ROOT", root)
    result = audit.manifest_invariants()
    assert any(expected in error for error in result["violations"]), result
    assert (
        result["source_manifest"] == "deploy/dataplane/cluster-action-executor.yaml"
    ), result


@pytest.mark.parametrize("defect", ["case-file", "nested-symlink"])
def test_timestamp_inventory_refuses_non_directory_or_redirected_case(
    defect: str, tmp_path: Path
) -> None:
    path = tmp_path / "cases" / "GF-REGIONAL-DESTR-002"
    path.parent.mkdir()
    if defect == "case-file":
        path.write_text("not a case directory", encoding="utf-8")
        expected = "case directory is invalid"
    else:
        path.mkdir()
        target = tmp_path / "elsewhere"
        target.mkdir()
        (path / "nested").symlink_to(target, target_is_directory=True)
        expected = "directory must not be a symlink"
    with pytest.raises(audit.AuditError, match=expected):
        audit.run_timestamps(tmp_path)
