from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile
from gpu_fault.training_submit_cli import render_workload
from tests._script_loader import lazy_script_module
from tests.regional._release_orchestrator_support import (
    RuntimeProfileRunner,
    config_file,
    manifest_config_file,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
MODULE = lazy_script_module("rollout_regional_release", MODULE_PATH)
RUNTIME_PROFILE_MODULE = lazy_script_module(
    "regional_runtime_profile",
    ROOT / "deploy/control-plane/regional/regional_runtime_profile.py",
)
RENDERING_MODULE = lazy_script_module(
    "regional_release_rendering",
    ROOT / "deploy/control-plane/regional/regional_release_rendering.py",
)
ADMIN_COMMANDS_MODULE = lazy_script_module(
    "regional_admin_commands",
    ROOT / "deploy/control-plane/regional/regional_admin_commands.py",
)
DNS_MODULE = lazy_script_module(
    "regional_dns", ROOT / "deploy/control-plane/regional/regional_dns.py"
)
DIFF_MODULE = lazy_script_module(
    "regional_release_diff",
    ROOT / "deploy/control-plane/regional/regional_release_diff.py",
)
ORCHESTRATION_MODULE = lazy_script_module(
    "regional_release_orchestration_tests",
    ROOT / "deploy/control-plane/regional/regional_release_orchestration.py",
)
REGION = "us-east-1"
CPU_EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-fault-control-plane"
GPU_EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"


def test_upgrade_ensures_schema_before_rolling_cpu() -> None:
    source = inspect.getsource(ORCHESTRATION_MODULE.run_upgrade_phases)

    assert source.index("self._upload_release(diff)") < source.index(
        "self._ensure_schema()"
    )
    assert source.index("self._ensure_schema()") < source.index("self._apply_cpu(")


def test_control_plane_only_upgrade_skips_schema_and_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    calls = []
    monkeypatch.setattr(release, "_ensure_contexts", lambda: None)
    monkeypatch.setattr(release, "_require_cpu_secrets", lambda: None)
    monkeypatch.setattr(release, "_remote_commands_are_idle", lambda: True)
    monkeypatch.setattr(release, "_capture_previous", lambda: {"metadata": {}})
    monkeypatch.setattr(release, "_backup_release_secrets", lambda: {})
    monkeypatch.setattr(
        release, "_save_state", lambda phase, **_updates: calls.append(phase)
    )
    monkeypatch.setattr(
        release,
        "_upload_release",
        lambda diff: calls.append(("upload", diff.kind.value)),
    )
    monkeypatch.setattr(
        release,
        "_ensure_schema",
        lambda: pytest.fail("control-only upgrade ran schema"),
    )
    monkeypatch.setattr(
        release,
        "_apply_cpu",
        lambda *, finalize, **_kwargs: calls.append(("cpu", finalize)),
    )
    monkeypatch.setattr(release, "_stage_registry", lambda: False)
    monkeypatch.setattr(
        release,
        "_upgrade_gpu_target",
        lambda *_args: pytest.fail("control-only upgrade touched GPU"),
    )
    monkeypatch.setattr(
        release, "_validate_release_quick", lambda _plan: calls.append("verify")
    )
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    release.upgrade(diff=diff)

    assert ("cpu", True) in calls
    assert ("cpu", False) not in calls
    assert "schema-ready" in calls
    assert "verify" in calls


def test_new_upgrade_discards_stale_rollback_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    release.state = {
        "release_id": "previous-release",
        "rollback_completed_phases": [
            "rollback-controller-staged",
            "rollback-data-restored",
            "rollback-cpu-restored",
            "rollback-verified",
        ],
        "rollback_completed_cluster_ids": ["gpu-a"],
        "rollback_result": {"status": "PASSED"},
    }
    saves: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(release, "_ensure_contexts", lambda: None)
    monkeypatch.setattr(release, "_require_cpu_secrets", lambda: None)
    monkeypatch.setattr(release, "_remote_commands_are_idle", lambda: True)
    monkeypatch.setattr(release, "_capture_previous", lambda: {"metadata": {}})
    monkeypatch.setattr(release, "_backup_release_secrets", lambda: {})
    monkeypatch.setattr(
        release,
        "_save_state",
        lambda phase, **_updates: saves.append((phase, dict(release.state))),
    )
    monkeypatch.setattr(release, "_upload_release", lambda _diff: None)
    monkeypatch.setattr(release, "_ensure_schema", lambda: None)
    monkeypatch.setattr(release, "_apply_cpu", lambda **_kwargs: None)
    monkeypatch.setattr(release, "_stage_registry", lambda: False)
    monkeypatch.setattr(release, "_validate_release_quick", lambda _plan: None)
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    release.upgrade(diff=diff)

    assert saves[0] == ("preflight", {})


def test_endpoint_only_upgrade_skips_executor_and_node_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    calls = []
    monkeypatch.setattr(
        release, "_ensure_connection_secret", lambda _target: calls.append("secret")
    )
    monkeypatch.setattr(
        release,
        "_verify_gpu_control_plane_endpoint",
        lambda _target: calls.append("endpoint"),
    )
    monkeypatch.setattr(
        release, "_apply_gpu_dcgm_exporter", lambda _target: calls.append("dcgm")
    )
    monkeypatch.setattr(
        release, "_apply_gpu_deployments", lambda *_args: calls.append("executor")
    )
    monkeypatch.setattr(
        release, "_deploy_reconciler", lambda *_args, **_kwargs: calls.append("node")
    )
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"endpoint"}),
    )

    MODULE.upgrade_gpu_target(release, target, diff)

    assert calls == ["secret", "endpoint"]


def test_bootstrap_gates_executor_on_gpu_dns_and_tls() -> None:
    source = inspect.getsource(MODULE.RegionalRelease.bootstrap)

    assert source.index("self._ensure_connection_secret(target)") < source.index(
        "self._quiesce_gpu_executor(target)"
    )
    assert source.index("self._quiesce_gpu_executor(target)") < source.index(
        "self._verify_gpu_control_plane_endpoint(target)"
    )
    assert source.index(
        "self._verify_gpu_control_plane_endpoint(target)"
    ) < source.index("self._apply_gpu_dcgm_exporter(target)")
    assert source.index("self._apply_gpu_dcgm_exporter(target)") < source.index(
        "self._apply_gpu_deployments(target, self.executor_wheel_cm)"
    )


def test_dcgm_exporter_is_ready_before_node_installer(tmp_path: Path) -> None:
    runtime_image = "registry.example/dcgm@sha256:" + "b" * 64
    path = config_file(tmp_path)
    config = MODULE.ReleaseConfig.load(path)

    class RecordingRunner:
        dry_run = False

        def __init__(self) -> None:
            self.calls = []

        def run(self, arguments, **kwargs):
            self.calls.append((arguments, kwargs))
            if "create" in arguments and "configmap" in arguments:
                return (
                    "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: test\n"
                    "data:\n  gpu-fault-counters.csv: test\n"
                )
            return ""

    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    release.dcgm_exporter_image = runtime_image
    release._apply_gpu_dcgm_exporter(config.clusters[0])

    rendered = [kwargs.get("input_text", "") for _arguments, kwargs in runner.calls]
    assert any("gpu-fault-counters.csv" in value for value in rendered), (
        "DCGM counters ConfigMap was not rendered"
    )
    assert any(runtime_image in value for value in rendered), (
        "configured DCGM image was not rendered"
    )
    assert any(
        "daemonset/gpu-fault-dcgm-exporter" in arguments and "rollout" in arguments
        for arguments, _kwargs in runner.calls
    ), "DCGM DaemonSet readiness was not awaited"


def test_node_installer_starts_after_dcgm_rollout() -> None:
    source = inspect.getsource(MODULE.RegionalRelease.bootstrap)

    assert source.index("self._apply_gpu_dcgm_exporter(target)") < source.index(
        "self._roll_node_runtime("
    )


def test_explicit_deploy_retries_only_failed_installer_jobs(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class RetryRunner:
        dry_run = False

        def __init__(self) -> None:
            self.calls = []

        def run(self, arguments, **_kwargs):
            self.calls.append(arguments)
            if "get" in arguments and "jobs" in arguments:
                return json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "failed-job"},
                                "status": {
                                    "conditions": [{"type": "Failed", "status": "True"}]
                                },
                            },
                            {
                                "metadata": {"name": "running-job"},
                                "status": {"conditions": []},
                            },
                            {
                                "metadata": {"name": "complete-job"},
                                "status": {
                                    "conditions": [
                                        {"type": "Complete", "status": "True"}
                                    ]
                                },
                            },
                        ]
                    }
                )
            return ""

    runner = RetryRunner()
    release = MODULE.RegionalRelease(config, runner)
    release._retry_failed_installer_jobs(config.clusters[0])

    deletes = [
        arguments
        for arguments in runner.calls
        if "delete" in arguments and "job" in arguments
    ]
    assert len(deletes) == 1
    assert "failed-job" in deletes[0]
    assert "running-job" not in deletes[0]
    assert "complete-job" not in deletes[0]


def test_dns_gate_waits_for_nlb_before_route53_change() -> None:
    source = inspect.getsource(DNS_MODULE.ensure_control_plane_dns)
    upsert_source = inspect.getsource(DNS_MODULE._upsert_cname_and_wait)

    assert source.index("_wait_service_hostname") < source.index("_wait_nlb_active")
    assert source.index("_wait_nlb_active") < source.index("_wait_raw_nlb_dns")
    assert source.index("_wait_raw_nlb_dns") < source.index("_wait_targets_healthy")
    assert source.index("_wait_targets_healthy") < source.index(
        "_upsert_cname_and_wait"
    )
    assert upsert_source.index("change-resource-record-sets") < upsert_source.index(
        "resource-record-sets-changed"
    )
    assert upsert_source.index("resource-record-sets-changed") < upsert_source.index(
        "get-change"
    )
    assert upsert_source.index("get-change") < upsert_source.index(
        'change.get("Status") != "INSYNC"'
    )


def test_nlb_service_is_created_after_dns_and_certificate_gate() -> None:
    source = inspect.getsource(DNS_MODULE.apply_control_plane_nlb)

    assert source.index("verify_control_plane_dns_prerequisites") < source.index(
        'release.runner.run(release._cpu("apply", "-f", "-")'
    )
    assert source.index(
        'release.runner.run(release._cpu("apply", "-f", "-")'
    ) < source.index("ensure_control_plane_dns")


def test_dns_gate_executes_the_strict_runtime_sequence(monkeypatch) -> None:
    dns_module = DNS_MODULE._load()
    events: list[str] = []
    certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/control-plane"
    raw_hostname = "internal-nlb.elb.us-east-1.amazonaws.com"

    class DnsRunner:
        dry_run = False

        def run(self, arguments, **_kwargs):
            command = " ".join(arguments)
            if "route53 get-hosted-zone" in command:
                events.append("hosted-zone")
                return json.dumps(
                    {
                        "HostedZone": {
                            "Id": "/hostedzone/Z123",
                            "Name": "site.gpu-fault.internal.",
                            "Config": {"PrivateZone": True},
                        }
                    }
                )
            if "acm describe-certificate" in command:
                events.append("certificate")
                return json.dumps(
                    {
                        "Certificate": {
                            "Status": "ISSUED",
                            "NotAfter": "2100-01-01T00:00:00+00:00",
                            "SubjectAlternativeNames": ["api.site.gpu-fault.internal"],
                        }
                    }
                )
            if "get service gpu-fault-api-nlb" in command:
                events.append("service-hostname")
                return raw_hostname
            if "elbv2 describe-load-balancers" in command:
                events.append("nlb-active")
                return json.dumps(
                    {
                        "LoadBalancers": [
                            {
                                "LoadBalancerArn": "arn:aws:elasticloadbalancing:"
                                "us-east-1:123456789012:loadbalancer/net/test/1",
                                "DNSName": raw_hostname,
                                "State": {"Code": "active"},
                            }
                        ]
                    }
                )
            if "elbv2 describe-listeners" in command:
                events.append("tls-listener")
                return json.dumps(
                    {
                        "Listeners": [
                            {
                                "Protocol": "TLS",
                                "Port": 443,
                                "Certificates": [{"CertificateArn": certificate_arn}],
                            }
                        ]
                    }
                )
            if "get deployment gpu-fault-api-ha" in command:
                events.append("expected-targets")
                return "3"
            if "elbv2 describe-target-groups" in command:
                events.append("target-group")
                return json.dumps(
                    {"TargetGroups": [{"TargetGroupArn": "target-group-arn"}]}
                )
            if "elbv2 describe-target-health" in command:
                events.append("targets-healthy")
                return json.dumps(
                    {
                        "TargetHealthDescriptions": [
                            {"TargetHealth": {"State": "healthy"}},
                            {"TargetHealth": {"State": "healthy"}},
                            {"TargetHealth": {"State": "healthy"}},
                        ]
                    }
                )
            if "route53 change-resource-record-sets" in command:
                events.append("cname-upsert")
                return json.dumps({"ChangeInfo": {"Id": "/change/C123"}})
            if "route53 wait resource-record-sets-changed" in command:
                events.append("route53-wait")
                return ""
            if "route53 get-change" in command:
                events.append("route53-insync")
                return json.dumps({"ChangeInfo": {"Status": "INSYNC"}})
            raise AssertionError(f"unexpected command: {command}")

    release = SimpleNamespace(
        runner=DnsRunner(),
        config=SimpleNamespace(
            aws_region=REGION,
            namespace="gpu-fault-system",
            nlb={"name": "gpu-fault-regional-test", "certificate_arn": certificate_arn},
            dns=SimpleNamespace(
                hosted_zone_id="Z123", hostname="api.site.gpu-fault.internal"
            ),
        ),
        _cpu=lambda *arguments: ["kubectl", *arguments],
    )

    def resolve(hostname, port):
        assert hostname == raw_hostname
        assert port == 443
        events.append("raw-nlb-dns")
        return [(2, 1, 6, "", ("192.0.2.10", port))]

    monkeypatch.setattr(dns_module.socket, "getaddrinfo", resolve)

    dns_module.verify_control_plane_dns_prerequisites(release)
    dns_module.ensure_control_plane_dns(release)

    assert events == [
        "hosted-zone",
        "certificate",
        "service-hostname",
        "nlb-active",
        "tls-listener",
        "raw-nlb-dns",
        "expected-targets",
        "target-group",
        "targets-healthy",
        "cname-upsert",
        "route53-wait",
        "route53-insync",
    ]


@pytest.mark.parametrize("method_name", ["upgrade", "rollback", "join_cluster"])
def test_executor_rollout_paths_require_gpu_dns_and_tls(method_name: str) -> None:
    inspected = (
        MODULE.upgrade_gpu_target
        if method_name == "upgrade"
        else (
            ORCHESTRATION_MODULE.rollback_target
            if method_name == "rollback"
            else getattr(MODULE.RegionalRelease, method_name)
        )
    )
    owner = "release" if method_name == "upgrade" else "self"
    source = inspect.getsource(inspected)

    assert source.index(
        f"{owner}._verify_gpu_control_plane_endpoint(target)"
    ) < source.index(f"{owner}._apply_gpu_dcgm_exporter(target)")
    assert source.index(f"{owner}._apply_gpu_dcgm_exporter(target)") < source.index(
        f"{owner}._apply_gpu_deployments"
    )


def test_gpu_endpoint_probe_checks_dns_ca_tls_and_health(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["dns"] = {"hosted_zone_id": "Z123", "hostname": "api.site.gpu-fault.internal"}
    path.write_text(json.dumps(value))
    config = MODULE.ReleaseConfig.load(path)

    class ProbeRunner:
        dry_run = False

        def __init__(self) -> None:
            self.manifest = ""

        def run(self, arguments, **kwargs):
            command = " ".join(arguments)
            if "apply -f -" in command:
                self.manifest = kwargs["input_text"]
            if "jsonpath={.status.phase}" in command:
                return "Succeeded"
            if " logs " in f" {command} ":
                return '{"tls": "verified", "status": 200}'
            return ""

    runner = ProbeRunner()
    release = MODULE.RegionalRelease(config, runner)
    release._verify_gpu_control_plane_endpoint(config.clusters[0])
    pod = yaml.safe_load(runner.manifest)
    script = pod["spec"]["containers"][0]["command"][2]

    assert "socket.getaddrinfo" in script
    assert "ssl.create_default_context" in script
    assert "urllib.request.urlopen" in script
    assert "'/healthz'" in script
    assert {
        item["name"]: item.get("value") for item in pod["spec"]["containers"][0]["env"]
    }["EXPECTED_HOSTNAME"] == "api.site.gpu-fault.internal"
    assert {
        (item["key"], item["operator"], item["effect"])
        for item in pod["spec"]["tolerations"]
    } == {
        ("gpu-fault.io/quarantined", "Exists", "NoSchedule"),
        ("node.kubernetes.io/unschedulable", "Exists", "NoSchedule"),
    }


def test_agent_convergence_uses_hyperpod_cluster_name(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    items = [
        {
            "metadata": {
                "name": "node-a",
                "uid": "uid-a",
                "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"},
                "annotations": {
                    "gpu-fault.io/installer-state": "Succeeded",
                    "gpu-fault.io/installer-artifact-sha256": hashlib.sha256(
                        b"wheel"
                    ).hexdigest(),
                    "gpu-fault.io/installer-config-digest": "config-a",
                    "gpu-fault.io/installer-node-uid": "uid-a",
                },
            }
        },
        {
            "metadata": {
                "labels": {"sagemaker.amazonaws.com/cluster-name": "another-hyperpod"},
                "annotations": {},
            }
        },
    ]

    assert MODULE.agents_converged(
        items,
        config.clusters[0],
        hashlib.sha256(b"wheel").hexdigest(),
        config_digest="config-a",
        require_node_uid=True,
    ), "agent convergence ignored the configured HyperPod cluster name"
    items[0]["metadata"]["annotations"]["gpu-fault.io/installer-node-uid"] = "stale"
    assert not MODULE.agents_converged(
        items,
        config.clusters[0],
        hashlib.sha256(b"wheel").hexdigest(),
        config_digest="config-a",
        require_node_uid=True,
    ), "agent convergence accepted a stale node UID annotation"


def test_release_config_requires_unique_clusters(tmp_path) -> None:
    cluster = {
        "cluster_id": "gpu-a",
        "context": "gpu-a-context",
        "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
        "region": REGION,
        "hyperpod_cluster_name": "hp-gpu-a",
        "eks_cluster_arn": GPU_EKS_ARN,
    }

    with pytest.raises(MODULE.ReleaseError, match="unique"):
        MODULE.ReleaseConfig.load(config_file(tmp_path, clusters=[cluster, cluster]))


def test_release_config_requires_runtime_profile_inputs(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value.pop("runtime_profile")
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="runtime_profile.source"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_allows_a_stable_site_profile_anchor(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["registration_cluster_id"] = "missing"
    path.write_text(json.dumps(value))

    config = MODULE.ReleaseConfig.load(path)

    assert config.runtime_profile_registration_cluster_id == "missing"


def test_release_config_allows_an_empty_managed_gpu_set(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["clusters"] = []
    path.write_text(json.dumps(value))

    config = MODULE.ReleaseConfig.load(path)

    assert config.clusters == ()


def test_release_config_requires_existing_profile_source(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["source"] = str(tmp_path / "missing-profile.yaml")
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="existing file"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_unsafe_profile_version(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["version"] = "hyperpod-v2&unexpected"
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="Runtime Profile version"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_requires_explicit_matching_region(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["aws_region"] = "REPLACE_WITH_AWS_REGION"
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="aws_region"):
        MODULE.ReleaseConfig.load(path)

    value["aws_region"] = REGION
    value["clusters"][0]["region"] = "us-west-2"
    path.write_text(json.dumps(value))
    with pytest.raises(MODULE.ReleaseError, match="does not match aws_region"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_cross_region_eks_arns(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["clusters"][0]["eks_cluster_arn"] = (
        "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
    )
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="eks_cluster_arn Region"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_cross_region_nlb_certificate(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["nlb"] = {
        "name": "gpu-fault-regional",
        "public_subnets": "subnet-a,subnet-b",
        "security_group": "sg-0123456789abcdef0",
        "certificate_arn": ("arn:aws:acm:us-west-2:123456789012:certificate/example"),
    }
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="certificate_arn Region"):
        MODULE.ReleaseConfig.load(path)


def test_nlb_manifest_uses_explicit_name_and_region_certificate(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["nlb"] = {
        "name": "gpu-fault-regional-test",
        "public_subnets": "subnet-a,subnet-b",
        "security_group": "sg-0123456789abcdef0",
        "certificate_arn": ("arn:aws:acm:us-east-1:123456789012:certificate/example"),
    }
    path.write_text(json.dumps(value))
    config = MODULE.ReleaseConfig.load(path)

    source = (
        ROOT / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
    ).read_text(encoding="utf-8")
    rendered = DNS_MODULE.render_nlb_manifest(config, source)

    assert "gpu-fault-regional-test" in rendered
    assert "arn:aws:acm:us-east-1:" in rendered
    assert "REPLACE_WITH" not in rendered


def test_plan_covers_first_deploy_and_rollback(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    deploy = release.plan("deploy")
    rollback = release.plan("rollback")
    join = release.plan("join-cluster")

    assert any("prerequisites" in step for step in deploy)
    assert any("PostgreSQL schema" in step for step in deploy)
    assert any("Runtime Profile" in step for step in deploy), (
        "deploy plan must include Runtime Profile registration"
    )
    assert any("previous required pins" in step for step in rollback)
    assert any("installer bundle" in step for step in rollback)
    assert any("Runtime Profile" in step for step in join), (
        "join plan must verify the shared Runtime Profile"
    )


def test_regional_parser_exposes_admin_health_commands() -> None:
    mode = next(action for action in MODULE.parser()._actions if action.dest == "mode")

    assert {"preflight", "verify", "release-summary", "status"} <= set(mode.choices)


def test_status_keeps_health_report_when_release_summary_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    module = MODULE._load()
    config = module.ReleaseConfig.load(config_file(tmp_path))
    release = module.RegionalRelease(config, module.Runner(dry_run=True))
    monkeypatch.setattr(
        module,
        "build_full_status",
        lambda _release: {
            "healthy": False,
            "release_status_error": "deployment is missing",
            "health": {"mode": "status"},
        },
    )

    report = release.status()

    assert report["healthy"] is False
    assert report["release_status_error"] == "deployment is missing"
    assert report["health"]["mode"] == "status"


def test_runtime_profile_payload_uses_declared_identity(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    payload = RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)

    assert payload["cluster_id"] == "gpu-a"
    assert payload["profile_version"] == "hyperpod-v1"
    assert payload["cluster_id"] != "hp-gpu-a"


def test_runtime_profile_policy_digest_ignores_identity_and_yaml_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        yaml.safe_dump(
            {
                "cluster_id": "placeholder",
                "environment": "hyperpod-eks",
                "profile_version": "profile-a",
                "claims": [
                    {
                        "capability": "gpuReset",
                        "mode": "OBSERVE",
                        "owner": "gpu-fault-node-agent",
                    },
                    {
                        "capability": "evidenceCapture",
                        "mode": "OWN",
                        "owner": "gpu-fault-control-plane",
                        "adapter": "control-plane-evidence",
                    },
                ],
                "observed": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    second.write_text(
        yaml.safe_dump(
            {
                "profile_version": "profile-b",
                "cluster_id": "gpu-a",
                "observed": [],
                "claims": list(reversed(yaml.safe_load(first.read_text())["claims"])),
                "environment": "hyperpod-eks",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert RUNTIME_PROFILE_MODULE.runtime_profile_policy_digest(
        first
    ) == RUNTIME_PROFILE_MODULE.runtime_profile_policy_digest(second)


def test_runtime_profile_is_registered_when_missing(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = RuntimeProfileRunner()
    release = MODULE.RegionalRelease(config, runner)

    MODULE.ensure_runtime_profile(release)

    assert len(runner.posted) == 1
    assert runner.posted[0]["cluster_id"] == "gpu-a"
    assert runner.posted[0]["profile_version"] == "hyperpod-v1"


def test_runtime_profile_registration_is_idempotent(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    desired = compile_runtime_profile(
        RuntimeProfile.model_validate(
            RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)
        )
    ).model_dump(mode="json")
    runner = RuntimeProfileRunner(existing=desired)
    release = MODULE.RegionalRelease(config, runner)

    MODULE.ensure_runtime_profile(release)

    assert runner.posted == []


def test_runtime_profile_drift_requires_a_new_version(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    existing = compile_runtime_profile(
        RuntimeProfile.model_validate(
            RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)
        )
    ).model_dump(mode="json")
    existing["capabilities"][0]["mode"] = "OBSERVE"
    runner = RuntimeProfileRunner(existing=existing)
    release = MODULE.RegionalRelease(config, runner)

    with pytest.raises(MODULE.ReleaseError, match="new profile version"):
        MODULE.ensure_runtime_profile(release)

    assert runner.posted == []


def test_runtime_profile_verify_is_read_only_when_missing(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = RuntimeProfileRunner()
    release = MODULE.RegionalRelease(config, runner)

    with pytest.raises(MODULE.ReleaseError, match="is not registered"):
        RUNTIME_PROFILE_MODULE.verify_runtime_profile(release)

    assert runner.posted == []


def test_release_config_loads_health_targets(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["site_name"] = "production"
    value["health"] = {
        "aurora_cluster_id": "gpu-fault-aurora",
        "amp_workspace_id": "ws-test",
        "amp_rule_namespace": "gpu-fault-rules",
        "sns_topic_arn": ("arn:aws:sns:us-east-1:123456789012:gpu-fault"),
        "certificate_min_validity_days": 45,
        "remote_command_max_unclaimed_seconds": 240,
        "require_confirmed_sns_subscription": True,
    }
    path.write_text(json.dumps(value))

    config = MODULE.ReleaseConfig.load(path)

    assert config.site_name == "production"
    assert config.health.aurora_cluster_id == "gpu-fault-aurora"
    assert config.health.amp_workspace_id == "ws-test"
    assert config.health.certificate_min_validity_days == 45
    assert config.health.remote_command_max_unclaimed_seconds == 240


class PreflightRunner:
    dry_run = False

    def __init__(self, *, gpu_eks_arn=GPU_EKS_ARN, gpu_node_recovery="None") -> None:
        self.gpu_eks_arn = gpu_eks_arn
        self.gpu_node_recovery = gpu_node_recovery

    def run(self, args, **kwargs):
        del kwargs
        if "config" in args and "view" in args:
            return CPU_EKS_ARN if "--kubeconfig" in args else self.gpu_eks_arn
        if args[:3] == ["aws", "sagemaker", "describe-cluster"]:
            cluster_name = args[args.index("--cluster-name") + 1]
            return json.dumps(
                {
                    "EksClusterArn": (
                        GPU_EKS_ARN if cluster_name == "hp-gpu-a" else CPU_EKS_ARN
                    ),
                    "NodeRecovery": (
                        self.gpu_node_recovery
                        if cluster_name == "hp-gpu-a"
                        else "Automatic"
                    ),
                }
            )
        if "--raw=/readyz" in args:
            return "ok"
        if "get" in args and "nodes" in args:
            return (
                '{"items":[{"metadata":{"name":"gpu-node-a"},"status":'
                '{"addresses":[{"type":"InternalIP","address":"10.0.1.10"}]}}]}'
            )
        raise AssertionError(f"unexpected preflight command: {args}")


def test_preflight_binds_contexts_and_hyperpod_to_config(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, PreflightRunner())
    monkeypatch.setattr(release, "_validate_executor_iam_role", lambda _target: None)

    MODULE.ensure_region_contexts(release)


def test_preflight_rejects_wrong_context_and_managed_gpu_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    wrong_context = MODULE.RegionalRelease(
        config,
        PreflightRunner(
            gpu_eks_arn=("arn:aws:eks:us-east-1:123456789012:cluster/unexpected-gpu")
        ),
    )
    monkeypatch.setattr(
        wrong_context, "_validate_executor_iam_role", lambda _target: None
    )

    with pytest.raises(MODULE.ReleaseError, match="does not match"):
        MODULE.ensure_region_contexts(wrong_context)

    managed_recovery = MODULE.RegionalRelease(
        config, PreflightRunner(gpu_node_recovery="Automatic")
    )
    monkeypatch.setattr(
        managed_recovery, "_validate_executor_iam_role", lambda _target: None
    )
    with pytest.raises(MODULE.ReleaseError, match="NodeRecovery=None"):
        MODULE.ensure_region_contexts(managed_recovery)


def test_join_cluster_requires_current_release_artifact(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    assert release.wheel_cm.startswith("gpu-fault-control-plane-wheel-0100-")
    assert release.bundle_cm.startswith("gpu-fault-node-installer-0100-")
    assert MODULE.STATE_CONFIG_MAP == ("gpu-fault-regional-release-state")


def test_sync_release_state_records_a_noop_topology() -> None:
    calls = []
    release = SimpleNamespace(
        state={
            "rollback_completed_phases": ["rollback-verified"],
            "rollback_result": {"status": "PASSED"},
        },
        _capture_previous=lambda: {"live_runtime_image": "legacy-runtime:stable"},
        _save_state=lambda phase, **updates: calls.append((phase, updates)),
    )

    MODULE.sync_release_state(release)

    assert release.state == {}
    assert calls == [
        (
            "complete",
            {
                "previous": None,
                "release_diff": {"kind": "NOOP", "changed": []},
                "adopted_live_runtime_image": "legacy-runtime:stable",
            },
        )
    ]


def test_join_cluster_requires_an_idle_remote_command_queue() -> None:
    source = inspect.getsource(MODULE.join_target)

    assert source.index("release._target(cluster_id)") < source.index(
        "release._remote_commands_are_idle()"
    )


def test_rollback_detects_legacy_component_pins_and_preserves_alerting() -> None:
    source = inspect.getsource(MODULE.build_rollback_environment)

    assert "legacy_component_pins = not any" in source
    assert '"GPU_FAULT_LEGACY_COMPONENT_PINS"' in source
    assert '"GPU_FAULT_ALLOW_EMAIL"' in source
    assert '"GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL"' in source


def test_last_cluster_can_be_removed_from_the_cpu_registry(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class RegistryRunner:
        dry_run = False

        def __init__(self) -> None:
            self.registrations = None

        def run(self, arguments, **kwargs):
            if "create" in arguments and "secret" in arguments:
                source = next(
                    item for item in arguments if item.startswith("--from-file=")
                )
                self.registrations = json.loads(
                    Path(source.split("=", 2)[2]).read_text()
                )
                return "apiVersion: v1\nkind: Secret\nmetadata:\n  name: test\n"
            return ""

    runner = RegistryRunner()
    release = MODULE.RegionalRelease(config, runner)
    monkeypatch.setattr(
        release, "_registry", lambda: [{"cluster_id": config.clusters[0].cluster_id}]
    )

    release._update_registry(config.clusters[0], remove=True)

    assert runner.registrations == []


@pytest.mark.parametrize(
    ("state_exists", "phase", "expected"),
    [
        (False, None, "bootstrap"),
        (True, "bootstrap-cleaned", "bootstrap"),
        (True, "bootstrap-cpu-ready", "bootstrap"),
        (True, "bootstrap-endpoint-ready", "bootstrap"),
        (True, "bootstrap-data-plane-progress", "bootstrap"),
        (True, "failed", "upgrade"),
        (True, "rolled-back", "upgrade"),
        (True, "complete", "upgrade"),
    ],
)
def test_deploy_selects_initial_or_upgrade_path(
    tmp_path: Path, monkeypatch, state_exists: bool, phase: str | None, expected: str
) -> None:
    module = MODULE._load()
    admin = ADMIN_COMMANDS_MODULE._load()
    release = module.RegionalRelease(
        module.ReleaseConfig.load(config_file(tmp_path)), module.Runner(dry_run=False)
    )
    calls: list[str] = []
    monkeypatch.setattr(
        admin.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0 if state_exists else 1
        ),
    )
    monkeypatch.setattr(release, "bootstrap", lambda: calls.append("bootstrap"))
    monkeypatch.setattr(release, "upgrade", lambda **_kwargs: calls.append("upgrade"))
    monkeypatch.setattr(release, "noop", lambda _diff: calls.append("noop"))
    if state_exists:
        monkeypatch.setattr(
            release,
            "_load_state",
            lambda: {
                "phase": phase,
                "release_diff": {
                    "kind": "CONTROL_PLANE_ONLY",
                    "changed": ["control_plane_wheel"],
                },
            },
        )

    admin.run_deploy(release)

    assert calls == [expected]


def test_bootstrap_live_checkpoint_recognizes_current_cpu_release(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    monkeypatch.setattr(
        release,
        "_config_map_data",
        lambda _name: {
            "required-agent-artifact-sha256": release.node_wheel_sha,
            "required-agent-compatibility-digest": (
                config.component_digests.get("node_runtime") or release.node_wheel_sha
            ),
            "required-regional-executor-artifact-sha256": (release.executor_wheel_sha),
            "required-regional-executor-compatibility-digest": (
                config.component_digests.get("executor") or release.executor_wheel_sha
            ),
            "required-agent-config-digest": config.agent_config_digest,
            "required-runtime-profile-version": config.runtime_profile_version,
        },
    )
    monkeypatch.setattr(
        release, "_deployment_wheel", lambda _kubectl, _deployment: release.wheel_cm
    )
    monkeypatch.setattr(
        release,
        "_get_json",
        lambda _arguments: {
            "metadata": {"generation": 7},
            "spec": {"replicas": 3},
            "status": {
                "observedGeneration": 7,
                "readyReplicas": 3,
                "updatedReplicas": 3,
                "availableReplicas": 3,
            },
        },
    )

    assert release._bootstrap_cpu_is_current() is True


def test_bootstrap_live_checkpoint_rejects_partial_cpu_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    monkeypatch.setattr(
        release,
        "_config_map_data",
        lambda _name: {
            "required-agent-artifact-sha256": release.wheel_sha,
            "required-agent-config-digest": config.agent_config_digest,
            "required-runtime-profile-version": config.runtime_profile_version,
        },
    )
    monkeypatch.setattr(
        release, "_deployment_wheel", lambda _kubectl, _deployment: release.wheel_cm
    )
    monkeypatch.setattr(
        release,
        "_get_json",
        lambda _arguments: {
            "metadata": {"generation": 7},
            "spec": {"replicas": 3},
            "status": {
                "observedGeneration": 7,
                "readyReplicas": 2,
                "updatedReplicas": 3,
                "availableReplicas": 2,
            },
        },
    )

    assert release._bootstrap_cpu_is_current() is False


def test_regional_release_shell_has_valid_syntax() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(ROOT / "deploy/control-plane/regional/rollout-regional-release.sh"),
        ],
        check=True,
    )


def test_regional_release_config_imports_with_runtime_pythonpath() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import regional_release_config"],
        cwd=ROOT / "deploy/control-plane/regional",
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_gpu_deployment_manifest_is_stamped_with_release_sha(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    document = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "executor"},
        "spec": {
            "template": {
                "metadata": {"annotations": {"gpu-fault.io/artifact-sha256": "old"}}
            }
        },
    }
    rendered = release._stamp_gpu_deployments(json.dumps(document))
    stamped = next(yaml.safe_load_all(rendered))
    annotations = stamped["spec"]["template"]["metadata"]["annotations"]

    assert annotations["gpu-fault.io/artifact-sha256"] == release.executor_wheel_sha
    assert (
        annotations["gpu-fault.io/executor-wheel-sha256"] == release.executor_wheel_sha
    )
    assert annotations["gpu-fault.io/executor-compatibility-digest"] == (
        config.component_digests.get("executor") or release.executor_wheel_sha
    )
    assert "gpu-fault.io/control-plane-wheel-sha256" not in annotations
    assert annotations["gpu-fault.io/release-rollout"] == release.release_id
    assert annotations["gpu-fault.io/runtime-image"] == MODULE.DEFAULT_RUNTIME_IMAGE


def test_executor_iam_boundary_accepts_minimal_role() -> None:
    MODULE.validate_executor_iam_documents(
        "arn:aws:iam::1:role/executor",
        [
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeCluster",
                            "sagemaker:ListClusterNodes",
                            "sagemaker:DescribeClusterNode",
                            "sagemaker:BatchRebootClusterNodes",
                            "s3:PutObject",
                        ],
                    }
                ]
            }
        ],
    )


@pytest.mark.parametrize(
    "action",
    [
        "ses:SendEmail",
        "sagemaker:BatchReplaceClusterNodes",
        "sagemaker:BatchDeleteClusterNodes",
        "sagemaker:*",
    ],
)
def test_executor_iam_boundary_rejects_excess_privilege(action: str) -> None:
    with pytest.raises(
        MODULE.ReleaseError, match="exceeds the regional data-plane boundary"
    ):
        MODULE.validate_executor_iam_documents(
            "arn:aws:iam::1:role/executor",
            [{"Statement": [{"Effect": "Allow", "Action": action}]}],
        )


def test_release_config_loads_content_addressed_manifest(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(manifest_config_file(tmp_path))

    assert config.wheel.name == "release.whl"
    assert config.bundle.name == "bundle.tar.gz"


def test_executor_iam_boundary_rejects_allow_not_action() -> None:
    with pytest.raises(MODULE.ReleaseError, match="Allow/NotAction"):
        MODULE.validate_executor_iam_documents(
            "arn:aws:iam::1:role/executor",
            [{"Statement": [{"Effect": "Allow", "NotAction": "iam:*"}]}],
        )


def test_release_renders_one_runtime_image_across_gpu_roles(
    tmp_path, monkeypatch
) -> None:
    runtime_image = "registry.example/gpu-fault/python@sha256:" + "a" * 64
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", runtime_image)
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class RecordingRunner:
        dry_run = True

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return ""

    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    target = config.clusters[0]

    release._apply_gpu_deployments(target, release.wheel_cm)
    rendered = [
        kwargs["input_text"]
        for _args, kwargs in runner.calls
        if kwargs.get("input_text")
    ]

    assert len(rendered) == 3
    assert all(runtime_image in item for item in rendered)
    assert all(MODULE.DEFAULT_RUNTIME_IMAGE not in item for item in rendered)
    assert any(f"value: {REGION}" in item for item in rendered), (
        "GPU manifests did not receive the configured Region"
    )
    assert all("REPLACE_WITH_AWS_REGION" not in item for item in rendered), (
        "GPU manifests retained an unresolved Region placeholder"
    )
    release._deploy_reconciler(
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
    )
    assert runner.calls[-1][1]["env"]["GPU_FAULT_RUNTIME_IMAGE"] == runtime_image
    assert runner.calls[-1][1]["env"]["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v1"
    assert runner.calls[-1][1]["env"]["GPU_FAULT_CLUSTER_ID"] == "gpu-a"
    assert runner.calls[-1][1]["env"]["GPU_FAULT_HYPERPOD_CLUSTER"] == "hp-gpu-a"


def test_release_rejects_invalid_runtime_image(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", "registry.example/bad image")
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    with pytest.raises(MODULE.ReleaseError, match="OCI image"):
        MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))


def test_non_default_runtime_profile_reaches_every_plane(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="hyperpod-v2")
    )

    class RecordingRunner:
        dry_run = True

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return ""

    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    target = config.clusters[0]

    cpu_environment = RENDERING_MODULE.build_cpu_apply_environment(
        release, finalize=False
    )
    assert (
        cpu_environment["GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"] == "hyperpod-v2"
    )
    assert cpu_environment["GPU_FAULT_ADMIN_CONFIG_SHA256"] == (
        release.admin_config_digest
    )

    rendered_manifests = [
        text
        for _deployment, text in RENDERING_MODULE.render_gpu_rollout_manifests(
            release, target, release.wheel_cm
        )
    ]
    resources = [
        document
        for text in rendered_manifests
        for document in yaml.safe_load_all(text)
        if isinstance(document, dict)
    ]
    collector = next(
        item
        for item in resources
        if item.get("kind") == "Deployment"
        and item["metadata"]["name"] == "gpu-fault-kubernetes-node-resource-collector"
    )
    collector_env = {
        item["name"]: item.get("value")
        for item in collector["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert collector_env["GPU_FAULT_RUNTIME_PROFILE_VERSION"] == "hyperpod-v2"
    assert all(
        "REPLACE_WITH_RUNTIME_PROFILE_VERSION" not in item
        for item in rendered_manifests
    ), "rendered GPU manifests retained the Runtime Profile placeholder"

    reconciler_environment = RENDERING_MODULE.build_reconciler_environment(
        release,
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
    )
    assert reconciler_environment["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v2"

    workload = render_workload(
        ROOT / "examples/hyperpod/three-node-pytorchjob.yaml",
        job_id="profile-v2-job",
        attempt_id=None,
        attempt_number=1,
        runtime_profile_version="hyperpod-v2",
        expected_critical_ranks=None,
        training_container="pytorch",
        restart_budget=1,
        namespace="training",
    )
    document = yaml.safe_load(workload.manifest)
    for replica in document["spec"]["pytorchReplicaSpecs"].values():
        annotations = replica["template"]["metadata"]["annotations"]
        assert annotations["gpu-fault.io/runtime-profile-version"] == "hyperpod-v2"


def test_runtime_profile_override_restores_rollback_version(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="hyperpod-v2")
    )

    class RecordingRunner:
        dry_run = True

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return ""

    release = MODULE.RegionalRelease(config, RecordingRunner())
    target = config.clusters[0]
    rendered = RENDERING_MODULE.render_gpu_rollout_manifests(
        release, target, release.wheel_cm, runtime_profile_version="hyperpod-v1"
    )
    resources = [
        document
        for _deployment, text in rendered
        for document in yaml.safe_load_all(text)
        if isinstance(document, dict)
    ]
    collector = next(
        item
        for item in resources
        if item.get("kind") == "Deployment"
        and item["metadata"]["name"] == "gpu-fault-kubernetes-node-resource-collector"
    )
    collector_env = {
        item["name"]: item.get("value")
        for item in collector["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert collector_env["GPU_FAULT_RUNTIME_PROFILE_VERSION"] == "hyperpod-v1"

    environment = RENDERING_MODULE.build_reconciler_environment(
        release,
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
        runtime_profile_version="hyperpod-v1",
    )
    assert environment["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v1"
