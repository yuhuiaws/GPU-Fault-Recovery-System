from __future__ import annotations

import hashlib
import json
import lzma
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import bootstrap_services as services
from gpu_fault.admin.bootstrap_common import (
    SITE_TAG_KEY,
    BootstrapError,
    ReadOnlyProbeRunner,
)
from tests.admin._cov95_join_support import target
from tests.admin.test_admin_bootstrap_iam import SITE, Account
from tests.admin.test_monitoring_policy import MonitoringAWS


@pytest.fixture
def cpu():
    return replace(target("a"), role="cpu")


@pytest.mark.parametrize(
    "operation,payload",
    [
        ("list-workspaces", {"workspaces": [{}]}),
        ("list-workspaces", {"workspaces": [None]}),
        ("list-tags-for-resource", {"tags": []}),
        ("list-tags-for-resource", {"tags": {"example": None}}),
        ("describe-workspace", {"workspace": None}),
    ],
)
def test_monitoring_requires_observable_workspace_identity_and_tags(
    cpu, operation, payload
):
    runner = MonitoringAWS(cpu)
    runner.overrides["amp", operation] = payload
    with pytest.raises(BootstrapError, match="identity|tags|workspace"):
        services.ensure_monitoring_resources(
            runner, cpu=cpu, site_id=runner.site_id, alert_email=None
        )
    assert runner.mutations() == []


def test_monitoring_rejects_foreign_identity_returned_by_create(cpu):
    runner = MonitoringAWS(cpu, initial=True)
    runner.overrides["amp", "create-workspace"] = {
        "workspaceId": "ws-example",
        "arn": "arn:aws:aps:us-west-2:123456789012:workspace/ws-example",
    }
    with pytest.raises(BootstrapError, match="created AMP workspace identity differs"):
        services.ensure_monitoring_resources(
            runner, cpu=cpu, site_id=runner.site_id, alert_email=None
        )
    assert runner.mutations() == [("amp", "create-workspace")]


@pytest.mark.parametrize("status", [None, "DELETING", "FAILED"])
def test_monitoring_refuses_unavailable_workspace_states(cpu, status):
    runner = MonitoringAWS(cpu)
    runner.workspace["status"] = None if status is None else {"statusCode": status}
    with pytest.raises(BootstrapError, match="not in an admissible state"):
        services.ensure_monitoring_resources(
            runner, cpu=cpu, site_id=runner.site_id, alert_email=None
        )
    assert runner.mutations() == []


@pytest.mark.parametrize("finishes", [False, True])
def test_monitoring_waits_with_a_bounded_clock_before_publishing(
    cpu, monkeypatch, finishes
):
    runner = MonitoringAWS(cpu)
    runner.workspace["status"] = {"statusCode": "CREATING"}
    clock = SimpleNamespace(value=0.0)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.value += 600
        if finishes:
            clock.value = 5
            runner.workspace["status"] = {"statusCode": "ACTIVE"}

    monkeypatch.setattr(
        services, "time", SimpleNamespace(monotonic=lambda: clock.value, sleep=sleep)
    )
    if finishes:
        result = services.ensure_monitoring_resources(
            runner, cpu=cpu, site_id=runner.site_id, alert_email=None
        )
        assert result["workspace_id"] == runner.workspace["workspaceId"]
    else:
        with pytest.raises(BootstrapError, match="did not become ACTIVE"):
            services.ensure_monitoring_resources(
                runner, cpu=cpu, site_id=runner.site_id, alert_email=None
            )
        assert runner.mutations() == []
    assert sleeps == [5]


def test_control_plane_service_account_read_failure_is_not_absence(tmp_path):
    class Unavailable(Account):
        def run(self, arguments, **options):
            if arguments[0] == "kubectl":
                self.calls.append(list(arguments))
                raise BootstrapError("example transport unavailable")
            return super().run(arguments, **options)

    runner = Unavailable()
    runner.addon_installed = True
    with pytest.raises(BootstrapError, match="transport unavailable"):
        runner.control_plane(tmp_path)
    kubectl_calls = [call for call in runner.calls if call[0] == "kubectl"]
    assert len(kubectl_calls) == 1
    assert "get" in kubectl_calls[0]
    assert runner.mutations("create") == []
    assert runner.mutations("put-role-policy") == []


def test_role_with_missing_inline_policy_reconciles_only_after_absence_proof():
    runner = Account()
    runner.role_exists = True
    runner.role_tags = [{"Key": SITE_TAG_KEY, "Value": SITE}]
    runner.policy_lookup = (254, "NoSuchEntity")
    result = runner.executor()
    assert result["inline_policy_name"] == "GPUFaultRegionalExecutor"
    assert len(runner.mutations("put-role-policy")) == 1
    assert runner.mutations("create-role") == []


def test_public_role_probe_can_verify_a_role_without_an_inline_policy(cpu):
    runner = Account()
    runner.role_exists = True
    runner.role_tags = [{"Key": SITE_TAG_KEY, "Value": SITE}]
    runner.role_trust = services.pod_identity_trust(cpu)
    result = services.probe_iam_role(
        runner,
        account_id=cpu.account_id,
        role_name="example-role",
        trust=runner.role_trust,
        policy_name=None,
        policy=None,
        site_id=SITE,
    )
    assert result["inline_policy_name"] == ""
    assert len(runner.calls) == 1


def test_executor_rejects_oidc_issuer_without_a_host_before_certificate_command():
    runner = Account()
    runner.issuer = "https:///"
    with pytest.raises(BootstrapError, match="invalid EKS OIDC issuer"):
        runner.executor()
    assert not runner.mutations("openssl"), "invalid issuer reached certificate I/O"


class RefreshAccount(Account):
    def __init__(self, root, *, exists=True, read_error=None):
        super().__init__()
        self.root = root
        self.exists = exists
        self.read_error = read_error
        self.uploads = []
        self.projections = {}
        self.wheel = root / "example.whl"
        self.wheel.write_bytes(b"owned-fake-wheel-payload")
        digest = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.configmap = "gpu-fault-control-plane-wheel-0100-" + digest[:12]

    def run(self, arguments, **options):
        if list(arguments[:3]) == ["aws", "eks", "list-pod-identity-associations"]:
            self.calls.append(list(arguments))
            return json.dumps({"associations": self.associations})
        if list(arguments[:3]) == ["aws", "eks", "describe-pod-identity-association"]:
            self.calls.append(list(arguments))
            return json.dumps({"association": {"roleArn": self.association_role}})
        if arguments[0] == "kubectl" and "configmap" in arguments:
            self.calls.append(list(arguments))
            if "get" in arguments:
                if self.read_error:
                    raise BootstrapError(self.read_error)
                if not self.exists:
                    raise BootstrapError("Error from server (NotFound)")
                return self.configmap
            assert "create" in arguments, "unexpected artifact ConfigMap operation"
            source = next(item for item in arguments if item.startswith("--from-file="))
            stored_key, name = source.removeprefix("--from-file=").split("=", 1)
            self.uploads.append((stored_key, lzma.decompress(Path(name).read_bytes())))
            self.exists = True
            return ""
        if arguments[0] == "kubectl" and "cronjob" in arguments:
            self.calls.append(list(arguments))
            projection = next(
                item for item in arguments if item.startswith("jsonpath=")
            )
            if ".image}" in projection:
                return self.projections["image"]
            if ".configMap.name}" in projection:
                return self.configmap
            return self.projections["master"]
        return super().run(arguments, **options)


@pytest.fixture
def refresh(tmp_path, cpu):
    template = Path("deploy/control-plane/regional/aurora-credential-refresh.yaml")
    destination = tmp_path / template
    destination.parent.mkdir(parents=True)
    destination.write_text(template.read_text())
    runner = RefreshAccount(tmp_path)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"wheel": "example.whl"}))
    arguments = {
        "repository_root": tmp_path,
        "cpu": cpu,
        "cpu_kubeconfig": tmp_path / "cpu.kubeconfig",
        "namespace": "gpu-fault-system",
        "site_id": SITE,
        "release_manifest": manifest,
        "runtime_image": "example.invalid/runtime@sha256:" + "a" * 64,
        "aurora": {
            "master_secret_arn": "arn:aws:secretsmanager:us-east-1:123456789012:secret:example"
        },
    }
    runner.projections = {
        "image": arguments["runtime_image"],
        "master": arguments["aurora"]["master_secret_arn"],
    }
    return runner, arguments


@pytest.mark.parametrize(
    "exists,absolute,kms", [(False, False, False), (True, True, True)]
)
def test_refresh_installer_uploads_only_missing_artifact_and_orders_verification(
    refresh, exists, absolute, kms
):
    runner, arguments = refresh
    runner.exists = exists
    if absolute:
        arguments["release_manifest"].write_text(
            json.dumps({"wheel": str(runner.wheel)})
        )
    if kms:
        arguments["aurora"]["master_secret_kms_key_arn"] = (
            "arn:aws:kms:us-east-1:123456789012:key/example"
        )
    result = services.install_aurora_refresh(runner, **arguments)
    assert result["wheel_configmap"] == runner.configmap
    assert runner.uploads == (
        [] if exists else [("example.whl.xz", runner.wheel.read_bytes())]
    )
    job_calls = [call for call in runner.calls if "job" in call or "420" in call]
    assert len(job_calls) == 3
    assert "delete" in job_calls[0]
    assert "create" in job_calls[1]
    assert "420" in job_calls[2]
    policy_call = runner.mutations("put-role-policy")[0]
    policy = json.loads(policy_call[policy_call.index("--policy-document") + 1])
    assert len(policy["Statement"]) == (2 if kms else 1)


def test_refresh_artifact_read_failure_prevents_all_installer_mutation(refresh):
    runner, arguments = refresh
    runner.read_error = "example API authentication failure"
    with pytest.raises(BootstrapError, match="authentication failure"):
        services.install_aurora_refresh(runner, **arguments)
    assert len(runner.calls) == 1
    assert runner.uploads == []


def test_refresh_probe_reuses_matching_artifact_role_and_association(refresh, cpu):
    runner, arguments = refresh
    role = f"arn:aws:iam::{cpu.account_id}:role/gpu-fault-{SITE}-aurora-refresh"
    runner.role_exists = True
    runner.role_tags = [{"Key": SITE_TAG_KEY, "Value": SITE}]
    runner.role_trust = services.pod_identity_trust(cpu)
    runner.role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue"],
                "Resource": arguments["aurora"]["master_secret_arn"],
            }
        ],
    }
    runner.associations = [
        {
            "namespace": arguments["namespace"],
            "serviceAccount": "gpu-fault-aurora-credential-refresh",
            "associationId": "assoc-example",
        }
    ]
    runner.association_role = role
    result = services.install_aurora_refresh(
        ReadOnlyProbeRunner(runner), **arguments, probe_only=True
    )
    assert result["association_id"] == "assoc-example"
    assert runner.uploads == []
    assert runner.mutations("create") == []
    assert runner.mutations("put-role-policy") == []


def test_monitoring_installer_calls_fake_script_with_scoped_identity(tmp_path, cpu):
    runner = Account()
    result = services.install_monitoring(
        runner,
        repository_root=tmp_path,
        cpu=cpu,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id=SITE,
        monitoring={
            "workspace_id": "ws-example",
            "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:example",
        },
        adot_image="example.invalid/adot:example",
        alert_email=None,
    )
    assert len(runner.mutations("install-amp-monitoring.sh")) == 1
    assert result["runtime_managed_by_release"] is False
    assert result["service_account"] == "gpu-fault-adot"
