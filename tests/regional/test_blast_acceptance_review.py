from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import blast_acceptance_cases_2 as cases
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne

SOURCE_DIGEST = hashlib.sha256(b"unit-blast-source").hexdigest()


def make_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_id: str, *, count: int = 1
) -> cases.BlastCasesTwo:
    site = tmp_path / "site.yaml"
    site.write_text("test site", encoding="ascii")
    for plane in ("cpu", "gpu"):
        (tmp_path / f"{plane}.kubeconfig").write_text(
            f"{plane} connection fixture", encoding="ascii"
        )
    monkeypatch.setattr(base, "source_digest", lambda: SOURCE_DIGEST)
    config = {
        "namespace": "gpu-system",
        "aws_region": "us-west-2",
        "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
        "gpu_kubeconfig": str(tmp_path / "gpu.kubeconfig"),
        "cpu_eks_arn": "arn:aws:eks:us-west-2:000000000000:cluster/cpu",
        "notifications": {"email_sender": "sender@example.com"},
        "clusters": [
            {
                "cluster_id": f"cluster-{index}",
                "context": f"context-{index}",
                "hyperpod_cluster_name": f"hyperpod-{index}",
                "eks_cluster_arn": f"arn:aws:eks:us-west-2:000000000000:cluster/gpu-{index}",
                "executor_irsa_role_arn": f"arn:aws:iam::000000000000:role/executor-{index}",
                "allowed_namespaces": ["training"],
            }
            for index in range(count)
        ],
    }
    monkeypatch.setattr(
        base,
        "load_site",
        lambda path: SimpleNamespace(
            release_config=config,
            environment={},
            source_sha256=base.sha256_bytes(site.read_bytes()),
        ),
    )
    runner = cases.BlastCasesTwo(
        site_path=site,
        run_dir=tmp_path / "run",
        case_id=case_id,
        e2e_dir=tmp_path / "e2e",
        trusted_cpu_baseline=tmp_path / "baseline.json",
        predecessor={"valid": True},
    )
    monkeypatch.setattr(
        runner,
        "cpu_json",
        lambda *args: {
            "items": [
                {"metadata": {"name": "gpu-system"}},
                {"metadata": {"name": "foreign"}},
            ]
        }
        if "namespaces" in args
        else {"items": []}
        if cases.RBAC_INVENTORY in args
        else {"data": {"state.json": json.dumps({"release_id": "release-test"})}},
    )
    monkeypatch.setattr(runner, "preflight", lambda: None)
    return runner


def evidence(runner: cases.BlastCasesTwo) -> dict[str, Any]:
    return json.loads((runner.run_dir / f"{runner.case_id}.json").read_text())


@pytest.mark.parametrize("drift", ["site", "context", "namespace", "release"])
def test_blast_cache_requires_the_complete_current_scope(
    drift: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, "GF-REGIONAL-BLAST-002")
    cache = {
        "captured_at": base.utc_now(),
        "site": str(runner.site_path),
        "gpu_clusters": [{"cluster_id": "cluster-0"}],
        "binding": runner.preflight_binding(),
    }
    base.write_json(runner.root_run_dir / base.PREFLIGHT_CACHE_NAME, cache)
    assert runner.reusable_preflight() is not None
    if drift == "site":
        runner.site_path.write_text("changed site", encoding="ascii")
    elif drift == "context":
        first = runner.targets[0]
        runner.targets = [
            base.ClusterTarget(
                first.cluster_id,
                "other-context",
                first.hyperpod_cluster_name,
                first.eks_cluster_arn,
                first.executor_role_arn,
            )
        ]
    elif drift == "namespace":
        runner.namespace = "other"
    else:
        monkeypatch.setattr(
            runner,
            "cpu_json",
            lambda *args: {
                "data": {"state.json": json.dumps({"release_id": "changed"})}
            },
        )
        with pytest.raises(base.CheckError, match="identity changed"):
            runner.reusable_preflight()
        return
    assert runner.reusable_preflight() is None


@pytest.mark.parametrize(
    "defect", ["missing-ca", "wrong-ca", "insecure", "arn", "context"]
)
def test_kubeconfig_binding_cannot_pass_missing_or_conflicting_tls_identity(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, "GF-REGIONAL-BLAST-002")
    ca = base64.b64encode(b"test-public-ca").decode()
    cluster_data: dict[str, Any] = {"server": "https://api.example"}
    if defect == "insecure":
        cluster_data["insecure-skip-tls-verify"] = True
    view = {
        "contexts": [
            {
                "name": "context",
                "context": {"cluster": "other" if defect == "context" else "eks"},
            }
        ],
        "clusters": [{"name": "eks", "cluster": cluster_data}],
    }
    description = {
        "cluster": {
            "arn": "wrong" if defect == "arn" else runner.cpu_eks_arn,
            "endpoint": "https://api.example",
            "certificateAuthority": {"data": ca},
        }
    }
    actual_ca = (
        ""
        if defect == "missing-ca"
        else base64.b64encode(b"other").decode()
        if defect == "wrong-ca"
        else ca
    )
    monkeypatch.setattr(base, "json_command", lambda *args: view)
    monkeypatch.setattr(
        base,
        "command",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, actual_ca, ""),
    )
    with pytest.raises(base.CheckError):
        runner.kubeconfig_binding(
            kube_args=("--context", "context"),
            expected_arn=runner.cpu_eks_arn,
            eks_description=description,
        )


def test_record_case_cannot_keep_pass_when_release_identity_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, "GF-REGIONAL-BLAST-002")
    monkeypatch.setattr(
        runner, "cpu_json", lambda *args: {"data": {"state.json": "{}"}}
    )
    with pytest.raises(base.CheckError, match="lost its deployment identity"):
        runner.record_case(runner.case_id, "PASS", checks={"permission": True})
    assert evidence(runner)["verdict"] == "FAIL"


@pytest.mark.parametrize(
    ("channel", "defect"),
    [
        ("ses", "none"),
        ("ses", "sender-wildcard"),
        ("ses", "unrelated-condition"),
        ("ses", "wrong-sa"),
        ("ses", "service-wildcard"),
        ("ses", "extra-channel"),
        ("sns", "none"),
        ("sns", "topic-wildcard"),
        ("sns", "missing-topic"),
        ("sns", "other-topic"),
        ("sns", "extra-channel"),
        ("disabled", "none"),
        ("disabled", "extra-channel"),
    ],
)
def test_cpu_permission_audit_checks_actual_identity_and_sender_scope(
    channel: str, defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, "GF-REGIONAL-BLAST-002")
    topic = "arn:aws:sns:us-west-2:000000000000:site-alerts"
    monkeypatch.setattr(
        runner,
        "notification_config",
        lambda: {
            "GPU_FAULT_NOTIFICATION_CHANNEL": channel,
            "GPU_FAULT_SNS_TOPIC_ARN": "" if defect == "missing-topic" else topic,
        },
    )
    monkeypatch.setattr(runner, "ready_cpu_pod", lambda: "cpu")
    release_read = runner.cpu_json
    monkeypatch.setattr(
        runner,
        "cpu_json",
        lambda *args: {
            "spec": {
                "serviceAccountName": "other"
                if defect == "wrong-sa"
                else "gpu-fault-control-plane"
            }
        }
        if "pod" in args
        else release_read(*args),
    )
    monkeypatch.setattr(
        runner, "cpu_text", lambda *args: "home_kube=absent\nkubeconfig_env=unset\n"
    )
    monkeypatch.setattr(
        runner,
        "auth_can_i",
        lambda **kwargs: {
            verb: {resource: False for resource in kwargs["resources"]}
            for verb in kwargs["verbs"]
        },
    )
    monkeypatch.setattr(runner, "cpu_control_plane_role_arn", lambda: ("role-test", {}))
    statement: dict[str, Any] = {
        "Effect": "Allow",
        "Action": "ses:SendEmail",
        "Resource": "arn:aws:ses:us-west-2:000000000000:identity/sender@example.com",
        "Condition": {"StringEquals": {"ses:FromAddress": "sender@example.com"}},
    }
    if defect == "sender-wildcard":
        statement["Resource"] = "arn:aws:ses:us-west-2:000000000000:identity/*"
    elif defect == "unrelated-condition":
        statement["Condition"] = {"Bool": {"aws:SecureTransport": "true"}}
    sns_statement = {"Effect": "Allow", "Action": "sns:Publish", "Resource": topic}
    if defect == "topic-wildcard":
        sns_statement["Resource"] = "*"
    elif defect == "other-topic":
        sns_statement["Resource"] = topic + "-other"
    statements = (
        [statement] if channel == "ses" else [sns_statement] if channel == "sns" else []
    )
    if defect == "extra-channel":
        statements.append(sns_statement if channel == "ses" else statement)
    if defect == "service-wildcard":
        statements.append(
            {"Effect": "Allow", "Action": "sage*:UpdateCluster", "Resource": "*"}
        )
    monkeypatch.setattr(runner, "iam_role_policies", lambda role: (statements, {}))
    assert runner.run() == (0 if defect == "none" else 1)
    assert evidence(runner)["verdict"] == ("PASS" if defect == "none" else "FAIL")


@pytest.mark.parametrize(
    ("config", "channel"),
    [
        ({}, "disabled"),
        ({"GPU_FAULT_SNS_TOPIC_ARN": "unit-topic"}, "sns"),
        ({"GPU_FAULT_EMAIL_SENDER": "sender@example.com"}, "ses"),
        ({"GPU_FAULT_EMAIL_RECIPIENTS": "recipient@example.com"}, "ses"),
        (
            {
                "GPU_FAULT_NOTIFICATION_CHANNEL": "disabled",
                "GPU_FAULT_SNS_TOPIC_ARN": "unit-topic",
            },
            "disabled",
        ),
    ],
)
def test_notification_channel_uses_the_declared_or_inferred_delivery_mode(
    config: dict[str, str], channel: str
) -> None:
    assert base.notification_channel(config) == channel


def test_unknown_notification_channel_cannot_be_inferred_as_a_known_mode() -> None:
    with pytest.raises(base.CheckError, match="channel is unknown"):
        base.notification_channel(
            {
                "GPU_FAULT_NOTIFICATION_CHANNEL": "unknown",
                "GPU_FAULT_SNS_TOPIC_ARN": "unit-topic",
            }
        )


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "jobs",
        "pytorchjobs.kubeflow.org",
        "jobsets.jobset.x-k8s.io",
        "namespace-jobset",
        "pod-sa",
        "iam",
        "named-secret-drift",
        "unnamed-secret",
    ],
)
def test_executor_audit_judges_every_workload_permission_column(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, "GF-REGIONAL-BLAST-003")
    manifest = [
        item
        for item in yaml.safe_load_all(cases.EXECUTOR_MANIFEST.read_text())
        if isinstance(item, dict)
    ]
    role = next(item for item in manifest if item["kind"] == "ClusterRole")
    # The live inventory as deploy renders it: the manifest's namespaced
    # node-key Role + RoleBinding in the site's system namespace.
    inventory = []
    for item in manifest:
        if item["kind"] not in {"Role", "RoleBinding"}:
            continue
        item = json.loads(json.dumps(item))
        item["metadata"]["namespace"] = runner.namespace
        for subject in item.get("subjects", []):
            subject["namespace"] = runner.namespace
        if item["kind"] == "Role":
            rule = item["rules"][0]
            if defect == "named-secret-drift":
                rule["resourceNames"] = ["gpu-fault-cluster-token"]
            elif defect == "unnamed-secret":
                rule.pop("resourceNames")
        inventory.append(item)
    expected = cases.expected_executor_role()
    keys = {
        "nodes": "core:nodes",
        "pods": "core:pods",
        "jobs": "batch:jobs",
        "pytorchjobs.kubeflow.org": "kubeflow.org:pytorchjobs",
        "jobsets.jobset.x-k8s.io": "jobset.x-k8s.io:jobsets",
    }

    def matrix(**kwargs: Any) -> dict[str, dict[str, bool]]:
        namespace_policy = runner.expected_namespace_roles(runner.targets[0]).get(
            kwargs.get("namespace"), {}
        )
        result = {
            verb: {
                resource: verb
                in (
                    expected.get(keys.get(resource, ""), [])
                    + namespace_policy.get(keys.get(resource, ""), [])
                )
                for resource in kwargs["resources"]
            }
            for verb in kwargs["verbs"]
        }
        if defect in keys and defect in kwargs["resources"]:
            result["create"][defect] = True
        if (
            defect == "namespace-jobset"
            and kwargs.get("namespace") == "training"
            and "jobsets.jobset.x-k8s.io" in kwargs["resources"]
        ):
            result["create"]["jobsets.jobset.x-k8s.io"] = False
        return result

    def gpu_json(target: base.ClusterTarget, *args: str) -> dict[str, Any]:
        if cases.RBAC_INVENTORY in args:
            return {"items": inventory}
        if "clusterrole" in args:
            return role
        if "pod" in args:
            return {
                "items": [
                    {
                        "metadata": {"name": "executor", "uid": "executor-uid"},
                        "spec": {
                            "serviceAccountName": "other"
                            if defect == "pod-sa"
                            else "gpu-fault-cluster-executor",
                            "containers": [{"name": "executor"}],
                        },
                        "status": {
                            "phase": "Running",
                            "conditions": [{"type": "Ready", "status": "True"}],
                            "containerStatuses": [{"name": "executor", "ready": True}],
                        },
                    }
                ]
            }
        if "namespaces" in args:
            return {
                "items": [
                    {"metadata": {"name": runner.namespace}},
                    {"metadata": {"name": "foreign"}},
                    {"metadata": {"name": "training"}},
                    {"metadata": {"name": "kube-system"}},
                ]
            }
        return {
            "metadata": {
                "annotations": {"eks.amazonaws.com/role-arn": target.executor_role_arn}
            }
        }

    statements = [
        {
            "Effect": "Allow",
            "Action": "iam:PassRole"
            if defect == "iam"
            else "sagemaker:DescribeCluster",
            "Resource": "scoped-resource",
        }
    ]
    monkeypatch.setattr(runner, "auth_can_i", matrix)
    monkeypatch.setattr(runner, "gpu_json", gpu_json)
    monkeypatch.setattr(runner, "iam_role_policies", lambda role: (statements, {}))
    monkeypatch.setattr(runner, "aws", lambda *args: {"ClusterArn": "scoped-resource"})
    assert runner.run() == (0 if defect == "none" else 1)
    result = evidence(runner)
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL")
    if defect == "namespace-jobset":
        checks = result["checks"]["clusters"][0]
        assert checks["sensitive_resources_all_denied"] is True
        assert checks["workloads_exact_permissions"] is False
    role_evidence = json.loads(
        (runner.run_dir / "BLAST-003-role-cluster-0.json").read_text()
    )
    assert role_evidence["expected_named_namespace_grants"] == {
        runner.namespace: {"core:secrets@gpu-fault-node-action-keys": ["get", "patch"]}
    }
    grants = role_evidence["unexpected_bound_grants"]
    if defect in {"named-secret-drift", "unnamed-secret"}:
        assert [(item["verb"], item["resource_names"]) for item in grants] == [
            (
                verb,
                ["gpu-fault-cluster-token"] if defect == "named-secret-drift" else [],
            )
            for verb in ("get", "patch")
        ]
        assert all(
            item["binding"] == "RoleBinding/gpu-fault-cluster-executor-node-keys"
            and item["namespace"] == runner.namespace
            for item in grants
        ), grants
    else:
        assert grants == [], "the manifest's named node-key grant is not a finding"


@pytest.mark.parametrize(
    "defect", ["none", "peer-copy", "execution-copy", "whitespace-reuse", "short"]
)
def test_token_sweep_detects_peer_copies_and_normalized_credential_reuse(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, "GF-REGIONAL-BLAST-004", count=2)
    execution = b"e" * 40
    tokens = {"cluster-0": b"a" * 40, "cluster-1": b"b" * 40}
    if defect == "whitespace-reuse":
        tokens["cluster-1"] = b"\n" + tokens["cluster-0"] + b" "
    elif defect == "short":
        tokens["cluster-1"] = b""

    def encoded(raw: bytes) -> str:
        return base64.b64encode(raw).decode()

    def gpu_json(target: base.ClusterTarget, *args: str) -> dict[str, Any]:
        canonical = {
            "kind": "Secret",
            "metadata": {
                "name": "gpu-fault-regional-connection",
                "namespace": runner.namespace,
            },
            "data": {"cluster-token": encoded(tokens[target.cluster_id])},
        }
        if "secret,configmap,pod" not in args:
            return canonical
        items = [canonical]
        if target.cluster_id == "cluster-0" and defect in {
            "peer-copy",
            "execution-copy",
        }:
            value = tokens["cluster-1"] if defect == "peer-copy" else execution + b"\n"
            items.append(
                {
                    "kind": "Secret",
                    "metadata": {"name": "innocent", "namespace": "other"},
                    "data": {"config": encoded(value)},
                }
            )
        return {"items": items}

    monkeypatch.setattr(runner, "gpu_json", gpu_json)
    monkeypatch.setattr(
        runner,
        "gpu_text",
        lambda target, *args: json.dumps(
            {
                "cluster_id": target.cluster_id,
                "digests": {
                    "GPU_FAULT_CONTROL_PLANE_TOKEN": hashlib.sha256(
                        tokens[target.cluster_id].strip()
                    ).hexdigest()
                },
            }
        ),
    )
    monkeypatch.setattr(runner, "ready_executor_pod", lambda target: "executor")
    monkeypatch.setattr(runner, "ready_cpu_pod", lambda: "cpu")
    monkeypatch.setattr(
        runner,
        "cpu_text",
        lambda *args: json.dumps(
            {"sha256": hashlib.sha256(execution).hexdigest(), "length": len(execution)}
        ),
    )
    assert runner.run() == (0 if defect == "none" else 1)
    result = evidence(runner)
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL")
    assert execution.decode() not in json.dumps(result), "execution credential leaked"
    assert tokens["cluster-0"].decode() not in json.dumps(result), (
        "cluster credential leaked"
    )


def test_sagemaker_service_wildcards_cannot_hide_mutation_permissions() -> None:
    patterns = BlastCasesOne.sagemaker_patterns(
        ["sage*:UpdateCluster", "ses:SendEmail"]
    )
    assert patterns == ["sage*:UpdateCluster"]
    assert BlastCasesOne.sagemaker_read_only(patterns) is False
