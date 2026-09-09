from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin.bootstrap_common import BootstrapResult
from gpu_fault.admin.site import (
    RegionalSite,
    SiteConfigError,
    effective_environment,
    load_site,
    materialized_release_config,
)
from gpu_fault_release import regional_release_config as RELEASE_CONFIG

REGION = "us-east-1"
ROOT = Path(__file__).resolve().parents[2]


def site_file(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    rollout.parent.mkdir(parents=True)
    rollout.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    rollout.chmod(0o755)
    (root / "dist").mkdir()
    wheel = root / "dist/release.whl"
    bundle = root / "dist/bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    manifest = json.dumps(
        {
            "release_id": "release-a",
            "wheel": str(wheel),
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "bundle": str(bundle),
            "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        }
    )
    (root / "dist/current-release.json").write_text(manifest, encoding="utf-8")
    immutable = root / "dist/release-a/release.json"
    immutable.parent.mkdir()
    immutable.write_text(manifest, encoding="utf-8")
    (root / "config").mkdir()
    (root / "config/profile.yaml").write_text(
        "cluster_id: placeholder\n"
        "environment: hyperpod-eks\n"
        "profile_version: hyperpod-v1\n"
        "claims: []\n"
        "observed: []\n",
        encoding="utf-8",
    )
    secure = tmp_path / "secure"
    secure.mkdir()
    for name, value in (
        ("cpu.kubeconfig", "kubeconfig"),
        ("token", "t" * 64),
        ("ca.crt", "certificate"),
        ("fleet-master", "f" * 64),
    ):
        path = secure / name
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    document = {
        "apiVersion": "gpu-fault.aws/v1alpha1",
        "kind": "RegionalSite",
        "metadata": {"name": "test-site"},
        "spec": {
            "repositoryRoot": str(root),
            "awsRegion": REGION,
            "cpu": {
                "kubeconfig": str(secure / "cpu.kubeconfig"),
                "eksArn": ("arn:aws:eks:us-east-1:123456789012:cluster/control"),
                "hyperpodClusterName": "control",
            },
            "release": {
                "manifest": "dist/current-release.json",
                "agentConfigDigest": "a" * 64,
            },
            "runtimeProfile": {
                "source": "config/profile.yaml",
                "version": "hyperpod-v1",
                "registrationClusterId": "gpu-a",
            },
            "nlb": {
                "name": "gpu-fault-regional",
                "publicSubnets": ["subnet-a", "subnet-b"],
                "securityGroup": "sg-123",
                "certificateArn": (
                    "arn:aws:acm:us-east-1:123456789012:certificate/test"
                ),
            },
            "images": {
                "runtime": "registry.example/runtime@sha256:" + "b" * 64,
                "nodeInstaller": "registry.example/installer:v1",
            },
            "health": {
                "auroraClusterId": "gpu-fault-aurora",
                "ampWorkspaceId": "ws-test",
                "snsTopicArn": ("arn:aws:sns:us-east-1:123456789012:gpu-fault"),
            },
            "clusters": [
                {
                    "clusterId": "gpu-a",
                    "context": "gpu-a",
                    "hyperpodClusterName": "hp-gpu-a",
                    "eksClusterArn": (
                        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"
                    ),
                    "executorIrsaRoleArn": ("arn:aws:iam::123456789012:role/executor"),
                    "allowedNamespaces": ["training", "gpu-fault-system"],
                    "controlPlaneUrl": "https://control.example",
                    "tokenFile": str(secure / "token"),
                    "caFile": str(secure / "ca.crt"),
                    "fleetMasterFile": str(secure / "fleet-master"),
                }
            ],
        },
    }
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)
    return path


def mock_live_release(
    monkeypatch: pytest.MonkeyPatch,
    *,
    release_id: str = "release-a",
    committed: bool = True,
    phase: str = "complete",
    admin_config_sha256: str | None = None,
    release_diff_kind: str | None = None,
    rollback_result: object = None,
) -> None:
    monkeypatch.setattr(
        admin_cli, "verify_prebuilt_release", lambda *_args, **_kwargs: None
    )

    def live_state(_site):
        state = {
            "release_id": release_id,
            "transaction_committed": committed,
            "phase": phase,
        }
        if admin_config_sha256 is not None:
            state["admin_config_sha256"] = admin_config_sha256
        if release_diff_kind is not None:
            state["release_diff"] = {"kind": release_diff_kind, "changed": []}
        if rollback_result is not None:
            state["rollback_result"] = rollback_result
        return state

    monkeypatch.setattr(admin_cli, "_live_release_state", live_state)


def reconcile_aurora_stub(**_kwargs):
    # A 32/50 preset's Aurora floor (82/128 ACU) makes applying it an RDS change.
    return {"modified": True, "before": {}, "after": {}}


def test_effective_environment_pins_repository_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/stale/source")
    site = load_site(site_file(tmp_path))

    environment = effective_environment(site)

    assert environment["PYTHONPATH"] == str(site.repository_root / "src")
    assert environment["GPU_FAULT_REPO_ROOT"] == str(site.repository_root)


def test_site_yaml_renders_the_existing_release_contract(tmp_path: Path) -> None:
    rendered = load_site(site_file(tmp_path))

    assert rendered.release_config["site_name"] == "test-site"
    assert rendered.release_config["aws_region"] == REGION
    assert rendered.release_config["runtime_profile"]["version"] == "hyperpod-v1"
    assert rendered.release_config["nlb"]["public_subnets"] == "subnet-a,subnet-b"
    assert rendered.release_config["health"]["amp_workspace_id"] == "ws-test"
    assert rendered.release_config["clusters"][0]["region"] == REGION
    # 0 is "auto": the release derives the wave size from the node count
    # instead of the site pinning a 1 that defeats the built-in cap.
    assert rendered.release_config["release"]["upgrade_max_unavailable"] == 0
    assert rendered.release_config["release"]["rollback_max_unavailable"] == 2
    assert rendered.release_config["release"]["upgrade_max_parallel_clusters"] == 1
    assert rendered.environment["GPU_FAULT_RUNTIME_IMAGE"].startswith(
        "registry.example/runtime"
    )

    with materialized_release_config(rendered) as path:
        assert path.stat().st_mode & 0o777 == 0o600
        assert json.loads(path.read_text()) == rendered.release_config
    assert not path.exists()


def test_runtime_profile_template_defaults_to_active_source(tmp_path: Path) -> None:
    document = yaml.safe_load(site_file(tmp_path).read_text(encoding="utf-8"))
    profile = RegionalSite.from_value(document).spec.runtime_profile

    assert profile.template_source == profile.source
    document["spec"]["runtimeProfile"]["templateSource"] = (
        "config/profile-template.yaml"
    )
    profile = RegionalSite.from_value(document).spec.runtime_profile
    assert profile.template_source == "config/profile-template.yaml"


def test_site_release_rollout_limits_are_bounded(tmp_path: Path) -> None:
    document = yaml.safe_load(site_file(tmp_path).read_text(encoding="utf-8"))
    document["spec"]["release"]["rollbackMaxUnavailable"] = 5

    with pytest.raises(SiteConfigError, match="rollbackMaxUnavailable"):
        RegionalSite.from_value(document)


def test_site_email_notification_contract(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["notifications"] = {
        "allowEmail": True,
        "acknowledgeExternalAlertChannel": False,
        "adminEmail": "ops@example.com",
        "emailSender": "sender@example.com",
        "emailRecipients": ["ops@example.com", "oncall@example.com"],
        "emailSubjectPrefix": "[PROD]",
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)

    site = load_site(path)

    assert site.release_config["notifications"] == {
        "allow_email": True,
        "acknowledge_external_alert_channel": False,
        "admin_email": "ops@example.com",
        "email_sender": "sender@example.com",
        "email_recipients": ["ops@example.com", "oncall@example.com"],
        "email_subject_prefix": "[PROD]",
    }


def test_admin_cli_exposes_single_cluster_removal(tmp_path, monkeypatch) -> None:
    site_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        admin_cli,
        "remove_cluster",
        lambda request: calls.append(request) or {"phase": "COMPLETED"},
    )
    arguments = admin_cli.parser().parse_args(
        [
            "remove-cluster",
            "--state-dir",
            str(tmp_path),
            "--cluster-id",
            "gpu-a",
            "--confirm",
            "REMOVE_GPU_CLUSTER",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[0].cluster_id == "gpu-a"
    assert calls[0].confirmation == "REMOVE_GPU_CLUSTER"


def test_admin_cli_exposes_arn_only_cluster_join(tmp_path, monkeypatch) -> None:
    site_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        admin_cli,
        "join_cluster",
        lambda request: calls.append(request) or {"phase": "COMPLETED"},
    )
    arguments = admin_cli.parser().parse_args(
        [
            "join-cluster",
            "--state-dir",
            str(tmp_path),
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[0].gpu_cluster_arn.endswith("cluster/gpu-b"), (
        "join-cluster did not preserve the requested GPU ARN"
    )
    assert calls[0].cluster_id is None
    assert calls[0].allowed_namespaces == ("gpu-fault-system", "training")
    assert calls[0].state_dir is None


def test_admin_cli_exposes_bounded_batch_cluster_join(tmp_path, monkeypatch) -> None:
    site_file(tmp_path)
    batches = []
    monkeypatch.setattr(
        admin_cli,
        "join_clusters",
        lambda requests: batches.append(requests)
        or {"phase": "COMPLETED", "joined": []},
    )
    arguments = admin_cli.parser().parse_args(
        [
            "join-cluster",
            "--state-dir",
            str(tmp_path),
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-c",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert [request.gpu_cluster_arn for request in batches[0]] == [
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-c",
    ]
    assert all(request.state_dir is None for request in batches[0]), (
        "public batch join leaked the site root as a transaction state directory"
    )


def test_generated_release_json_loads_through_the_existing_state_machine(
    tmp_path: Path,
) -> None:
    rendered = load_site(site_file(tmp_path))

    with materialized_release_config(rendered) as path:
        config = RELEASE_CONFIG.ReleaseConfig.load(path)

    assert config.site_name == "test-site"
    assert config.aws_region == REGION
    assert config.health.aurora_cluster_id == "gpu-fault-aurora"
    assert config.clusters[0].cluster_id == "gpu-a"
    assert config.admin_config.capacity.control_worker_replicas == 6


def test_site_yaml_must_be_private(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    path.chmod(0o644)

    with pytest.raises(SiteConfigError, match="group/other"):
        load_site(path)


def test_site_yaml_rejects_cross_region_monitoring(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["spec"]["health"]["snsTopicArn"] = (
        "arn:aws:sns:us-west-2:123456789012:gpu-fault"
    )
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(SiteConfigError, match="snsTopicArn"):
        load_site(path)


def test_admin_deploy_runs_preflight_before_bootstrap(
    tmp_path: Path, monkeypatch
) -> None:
    path = site_file(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append([str(item) for item in command])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(admin_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        admin_cli, "_configure_site_notifications", lambda site, **_kwargs: site
    )
    monkeypatch.setattr(
        admin_cli,
        "sync_installation_resource_registry",
        lambda _site: tmp_path / "installation-resources.json",
    )
    arguments = argparse.Namespace(
        command="deploy", file=path, repo_root=None, show_effective_config=False
    )

    assert admin_cli.run(arguments) == 0
    assert [call[1] for call in calls] == ["preflight", "deploy"]


def test_admin_deploy_stops_when_preflight_fails(tmp_path: Path, monkeypatch) -> None:
    path = site_file(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append([str(item) for item in command])
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(admin_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        admin_cli, "_configure_site_notifications", lambda site, **_kwargs: site
    )
    monkeypatch.setattr(
        admin_cli,
        "sync_installation_resource_registry",
        lambda _site: tmp_path / "installation-resources.json",
    )
    arguments = argparse.Namespace(
        command="deploy", file=path, repo_root=None, show_effective_config=False
    )

    assert admin_cli.run(arguments) == 1
    assert [call[1] for call in calls] == ["preflight"]


@pytest.mark.parametrize("command", ["preflight", "verify", "status"])
def test_admin_read_only_commands_map_to_regional_modes(
    tmp_path: Path, monkeypatch, command: str
) -> None:
    site_file(tmp_path)
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        del kwargs
        calls.append([str(item) for item in arguments])
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(admin_cli.subprocess, "run", fake_run)
    if command == "status":
        monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    monkeypatch.setattr(
        admin_cli, "_configure_site_notifications", lambda site, **_kwargs: site
    )
    arguments = argparse.Namespace(
        command=command,
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
    )

    assert admin_cli.run(arguments) == 0
    assert [call[1] for call in calls] == [command]


def test_status_uses_previous_management_baseline_after_verified_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = site_file(tmp_path)
    used = []

    monkeypatch.setattr(
        admin_cli,
        "_live_release_state",
        lambda _site: {
            "phase": "rolled-back",
            "rollback_result": {"status": "PASSED"},
            "previous": {"release_delivery_sha256": "a" * 64},
        },
    )

    @contextmanager
    def materialized(site, state, **_kwargs):
        used.append((site, state["phase"]))
        yield path

    monkeypatch.setattr(admin_cli, "materialized_rollback_status_site", materialized)
    monkeypatch.setattr(
        admin_cli.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0),
    )
    arguments = argparse.Namespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
    )

    assert admin_cli.run(arguments) == 0
    assert used == [(path, "rolled-back")]


def test_console_script_is_published() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    )

    assert project["project"]["scripts"]["gpu-fault-admin"] == (
        "gpu_fault.admin.cli:main"
    )


def test_deploy_host_binding_rejects_another_state_before_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "canonical"
    wrong = tmp_path / "wrong"
    monkeypatch.setattr(
        admin_cli, "_bound_deploy_host_state_dir", lambda: canonical.resolve()
    )
    arguments = argparse.Namespace(
        command="status",
        state_dir=wrong,
        file=None,
        repo_root=None,
        show_effective_config=False,
    )

    with pytest.raises(SiteConfigError, match="installed deploy-host is bound"):
        admin_cli.enforce_deploy_host_state_dir(arguments)


def test_deploy_host_binding_allows_canonical_state(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(
        admin_cli, "_bound_deploy_host_state_dir", lambda: canonical.resolve()
    )
    arguments = argparse.Namespace(
        command="status",
        state_dir=canonical,
        file=None,
        repo_root=None,
        show_effective_config=False,
    )

    admin_cli.enforce_deploy_host_state_dir(arguments)


def test_admin_main_enforces_binding_before_dispatch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class Parser:
        @staticmethod
        def parse_args():
            return arguments

    canonical = tmp_path / "canonical"
    wrong = tmp_path / "wrong"
    arguments = argparse.Namespace(command="status", state_dir=wrong, file=None)
    monkeypatch.setattr(admin_cli, "parser", Parser)
    monkeypatch.setattr(
        admin_cli, "_bound_deploy_host_state_dir", lambda: canonical.resolve()
    )
    monkeypatch.setattr(
        admin_cli,
        "run",
        lambda _arguments: pytest.fail("dispatch ran before state binding"),
    )

    assert admin_cli.main() == 2
    assert "installed deploy-host is bound" in capsys.readouterr().err


def test_legacy_uninstall_accepts_cluster_arns_without_site_file() -> None:
    arguments = admin_cli.parser().parse_args(
        [
            "uninstall",
            "--cpu-cluster-arn",
            "arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:sagemaker:us-west-2:123456789012:cluster/gpu",
            "--cpu-cluster",
            "keep",
            "--confirm",
            "UNINSTALL_GPU_FAULT",
        ]
    )

    assert arguments.file is None, "ARN uninstall unexpectedly requires a site file"
    assert arguments.cpu_cluster_arn.endswith("cluster/cpu"), "CPU ARN was not parsed"
    assert arguments.gpu_cluster_arn == [
        "arn:aws:sagemaker:us-west-2:123456789012:cluster/gpu"
    ], "GPU ARN was not parsed"


def test_uninstall_resolves_site_from_state_dir(tmp_path: Path, monkeypatch) -> None:
    site_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        admin_cli,
        "uninstall",
        lambda request: calls.append(request) or {"phase": "COMPLETED"},
    )
    arguments = admin_cli.parser().parse_args(
        [
            "uninstall",
            "--state-dir",
            str(tmp_path),
            "--cpu-cluster",
            "keep",
            "--confirm",
            "UNINSTALL_GPU_FAULT",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[0].cpu_disposition == "keep"
    assert calls[0].confirmation == "UNINSTALL_GPU_FAULT"


def test_arn_only_deploy_bootstraps_site_then_deploys_and_verifies(
    tmp_path: Path, monkeypatch
) -> None:
    path = site_file(tmp_path)
    calls: list[tuple[list[str], dict]] = []
    requests = []
    joined = []
    monkeypatch.setattr(
        admin_cli,
        "bootstrap_from_arns",
        lambda request: (
            requests.append(request)
            or BootstrapResult(
                site_file=path,
                state_file=request.state_dir / "bootstrap-state.json",
                pending_gpu_cluster_arns=(
                    "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-b",
                ),
            )
        ),
    )

    def fake_run(arguments, **kwargs):
        calls.append(([str(item) for item in arguments], kwargs))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(admin_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        admin_cli, "_configure_site_notifications", lambda site, **_kwargs: site
    )
    monkeypatch.setattr(
        admin_cli,
        "sync_installation_resource_registry",
        lambda _site: tmp_path / "installation-resources.json",
    )
    monkeypatch.setattr(
        admin_cli,
        "join_clusters",
        lambda requests: joined.extend(request.gpu_cluster_arn for request in requests)
        or {},
    )
    arguments = argparse.Namespace(
        command="deploy",
        file=None,
        cpu_cluster_arn=("arn:aws:sagemaker:us-east-1:123456789012:cluster/cpu"),
        gpu_cluster_arn=[
            "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
            "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-b",
        ],
        repo_root=tmp_path / "repo",
        state_dir=tmp_path / "state",
        alert_email="operations@example.com",
        staging_only_release=True,
        impact_base="origin/release",
        prepared_source_release=True,
        show_effective_config=False,
    )

    assert admin_cli.run(arguments) == 0
    assert len(calls) == 1
    command, options = calls[0]
    assert command[1].endswith("scripts/release_deploy.py"), (
        "ARN deploy did not delegate to the signed release state machine"
    )
    assert "--prebuilt-attestation" in command
    assert "--prebuilt-bundle" in command
    assert "--cosign-key" in command
    assert "--profile-approval" not in command
    assert "--allow-staging-release" in command
    assert options["env"]["PYTHONPATH"] == str((tmp_path / "repo") / "src")
    assert requests[0].gpu_cluster_arns == (
        "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
        "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-b",
    )
    assert requests[0].staging_only_release is True
    assert requests[0].impact_base == "origin/release"
    assert joined == ["arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-b"]


def test_public_arn_deploy_delegates_to_internal_source_preparation(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.chdir(tmp_path)
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
            "--state-dir",
            str(tmp_path / "state"),
            "--admin-email",
            "operations@example.com",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[0]["state_dir"] == tmp_path / "state"
    assert calls[0]["admin_email"] == "operations@example.com"
    assert calls[0]["gpu_cluster_arns"] == (
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
    )


def test_public_deploy_help_exposes_optional_admin_config(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args(["deploy", "--help"])

    help_text = capsys.readouterr().out
    for value in (
        "--cpu-cluster-arn",
        "--gpu-cluster-arn",
        "--state-dir",
        "--admin-email",
        "--config",
    ):
        assert value in help_text
    for value in (
        "--file",
        "--repo-root",
        "--release-ref",
        "--profile-approval",
        "--staging-only-release",
        "--impact-base",
        "--prepared-source-release",
        "--email-sender",
    ):
        assert value not in help_text


def test_approve_profile_command_records_pending_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[Path, str, str]] = []

    def approve(state_dir: Path, *, reference: str, expected_plan_sha256: str):
        calls.append((state_dir, reference, expected_plan_sha256))
        return {
            "site_identity": {
                "site_name": "test-site",
                "aws_region": REGION,
                "cpu_eks_arn": ("arn:aws:eks:us-east-1:123456789012:cluster/control"),
            },
            "site_identity_sha256": "b" * 64,
            "desired_version": "profile-v2",
            "change_kind": "EXPANSIVE",
            "plan_sha256": "a" * 64,
            "reference": reference,
            "approved_at": "2026-08-30T12:00:00+00:00",
            "approver_identity": "arn:aws:sts::123456789012:assumed-role/Admin/alice",
        }

    monkeypatch.setattr(admin_cli, "approve_profile", approve)
    state_dir = tmp_path / "state"
    arguments = admin_cli.parser().parse_args(
        [
            "approve-profile",
            "--state-dir",
            str(state_dir),
            "--plan-sha256",
            "a" * 64,
            "--reference",
            "CHG-12345",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert calls == [(state_dir, "CHG-12345", "a" * 64)]
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "APPROVED"
    assert output["plan_sha256"] == "a" * 64
    assert output["site_identity"]["cpu_eks_arn"].endswith("/control"), (
        "approve-profile output omitted the readable CPU control-plane identity"
    )
    assert output["approver_identity"].endswith("/Admin/alice"), (
        "approve-profile output must name who approved (I2)"
    )


def test_approve_profile_help_exposes_reviewed_plan_sha(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args(["approve-profile", "--help"])

    help_text = capsys.readouterr().out
    assert "--state-dir" in help_text
    assert "--plan-sha256" in help_text
    assert "--reference" in help_text
    assert "--file" not in help_text
    assert "--repo-root" not in help_text


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("preflight", ()),
        ("verify", ()),
        ("status", ()),
        ("join-cluster", ("--gpu-cluster-arn",)),
        ("remove-cluster", ("--cluster-id", "--confirm")),
        ("uninstall", ("--cpu-cluster", "--confirm")),
    ),
)
def test_managed_admin_help_uses_state_dir_not_site_file(
    capsys, command: str, expected: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args([command, "--help"])

    help_text = capsys.readouterr().out
    assert "--state-dir" in help_text
    assert "--file" not in help_text
    assert "--repo-root" not in help_text
    for value in expected:
        assert value in help_text


def test_managed_admin_command_requires_existing_state_site(tmp_path: Path) -> None:
    arguments = admin_cli.parser().parse_args(["status", "--state-dir", str(tmp_path)])

    with pytest.raises(SiteConfigError, match="no managed site"):
        admin_cli.run(arguments)


def test_arn_only_deploy_requires_private_state_dir(tmp_path: Path) -> None:
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:sagemaker:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
            "--repo-root",
            str(tmp_path),
        ]
    )

    with pytest.raises(SiteConfigError, match="requires --state-dir"):
        admin_cli.run(arguments)


def test_arn_only_deploy_requires_admin_email(tmp_path: Path) -> None:
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:sagemaker:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )

    with pytest.raises(SiteConfigError, match="requires --admin-email"):
        admin_cli.run(arguments)


def test_admin_email_option_keeps_alert_email_compatibility() -> None:
    preferred = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
            "--admin-email",
            "ops@example.com",
        ]
    )
    legacy = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
            "--alert-email",
            "ops@example.com",
        ]
    )

    assert preferred.alert_email == "ops@example.com"
    assert legacy.alert_email == preferred.alert_email


def test_admin_cli_accepts_independent_email_routing() -> None:
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
            "--admin-email",
            "owner@example.com",
            "--email-sender",
            "sender@example.com",
            "--email-recipient",
            "ops@example.com",
            "--email-recipient",
            "oncall@example.com",
            "--email-subject-prefix",
            "[PROD]",
        ]
    )

    assert arguments.email_sender == "sender@example.com"
    assert arguments.email_recipient == ["ops@example.com", "oncall@example.com"]
    assert arguments.email_subject_prefix == "[PROD]"


def test_accept_schema_change_travels_to_the_source_preparer_as_environment(
    tmp_path: Path, monkeypatch
) -> None:
    """One flag on the public command; the release engine reads one variable.

    The deploy is a chain of processes that each inherit their environment, so
    the CLI sets the variable once and never has to thread a new argument
    through the source preparer, the inner CLI or the release driver.
    """

    calls = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.delenv(admin_cli.ACCEPT_SCHEMA_CHANGE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    base = [
        "deploy",
        "--cpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
        "--gpu-cluster-arn",
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
        "--state-dir",
        str(tmp_path / "state"),
        "--admin-email",
        "operations@example.com",
    ]

    assert admin_cli.run(admin_cli.parser().parse_args(base)) == 0
    assert calls[-1]["extra_environment"] == {}, "no flag, no variable"
    assert (
        admin_cli.run(admin_cli.parser().parse_args([*base, "--accept-schema-change"]))
        == 0
    )
    assert calls[-1]["extra_environment"] == {
        admin_cli.ACCEPT_SCHEMA_CHANGE_ENV: "snapshot"
    }
    assert (
        admin_cli.run(
            admin_cli.parser().parse_args(
                [*base, "--accept-schema-change-without-snapshot"]
            )
        )
        == 0
    )
    assert calls[-1]["extra_environment"] == {
        admin_cli.ACCEPT_SCHEMA_CHANGE_ENV: "no-snapshot"
    }


def test_an_explicit_schema_change_variable_wins_over_the_flag(monkeypatch) -> None:
    monkeypatch.setenv(admin_cli.ACCEPT_SCHEMA_CHANGE_ENV, "no-snapshot")
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--accept-schema-change", "--state-dir", "/tmp/x"]
    )

    assert admin_cli.schema_change_environment(arguments) == {}


ADOT_ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-adot-writer"


def test_site_cluster_adot_irsa_role_reaches_the_release_config(tmp_path: Path) -> None:
    """``adotIrsaRoleArn`` is the product path to the data-plane collector.

    The regional release applies the per-cluster ADOT collector only for a
    target that carries ``adot_irsa_role_arn`` (F7); a site.yaml field that the
    generated release config dropped would leave every site on the skip path
    with no way through the sanctioned CLI.
    """
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["clusters"][0]["adotIrsaRoleArn"] = ADOT_ROLE_ARN
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    rendered = load_site(path)

    cluster = rendered.release_config["clusters"][0]
    assert cluster["adot_irsa_role_arn"] == ADOT_ROLE_ARN
    with materialized_release_config(rendered) as materialized:
        config = RELEASE_CONFIG.ReleaseConfig.load(materialized)
    assert config.clusters[0].adot_irsa_role_arn == ADOT_ROLE_ARN


def test_site_cluster_without_an_adot_role_stays_on_the_skip_path(
    tmp_path: Path,
) -> None:
    """Optional: a brownfield site deploys everything else and adds the role later."""
    rendered = load_site(site_file(tmp_path))

    cluster = rendered.release_config["clusters"][0]
    assert "adot_irsa_role_arn" not in cluster
    with materialized_release_config(rendered) as materialized:
        config = RELEASE_CONFIG.ReleaseConfig.load(materialized)
    assert config.clusters[0].adot_irsa_role_arn is None


def test_site_cluster_adot_role_must_be_text_when_present(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["spec"]["clusters"][0]["adotIrsaRoleArn"] = ["not", "text"]

    with pytest.raises(SiteConfigError, match=r"clusters\[0\]\.adotIrsaRoleArn"):
        RegionalSite.from_value(document)
