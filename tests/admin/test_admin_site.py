from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from gpu_fault import admin_cli
from gpu_fault.admin_bootstrap_common import BootstrapResult
from gpu_fault.admin_config import (
    AdminConfigError,
    admin_config_plan_path,
    load_desired_admin_config,
)
from gpu_fault.admin_config_file import (
    admin_config_file_path,
    initialize_desired_admin_config,
)
from gpu_fault.admin_site import (
    RegionalSite,
    SiteConfigError,
    effective_environment,
    load_site,
    materialized_release_config,
)
from tests._script_loader import lazy_script_module

REGION = "us-east-1"
ROOT = Path(__file__).resolve().parents[2]
RELEASE_CONFIG = lazy_script_module(
    "admin_site_release_config",
    ROOT / "deploy/control-plane/regional/regional_release_config.py",
)


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
    (root / "dist/current-release.json").write_text(
        json.dumps(
            {
                "release_id": "release-a",
                "wheel": str(wheel),
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "bundle": str(bundle),
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
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
    assert calls[0].allowed_namespaces == ()


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


def test_console_script_is_published() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    )

    assert project["project"]["scripts"]["gpu-fault-admin"] == (
        "gpu_fault.admin_cli:main"
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
    monkeypatch.setattr(
        admin_cli,
        "bootstrap_from_arns",
        lambda request: (
            requests.append(request)
            or BootstrapResult(
                site_file=path, state_file=request.state_dir / "bootstrap-state.json"
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


def test_config_help_is_single_level(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args(["config", "--help"])

    help_text = capsys.readouterr().out
    for value in ("--state-dir", "--file", "--preset", "--reference", "--dry-run"):
        assert value in help_text
    for value in ("config plan", "config apply", "--plan-sha256"):
        assert value not in help_text


def test_config_dry_run_from_private_yaml_is_role_scoped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)
    config = tmp_path / "capacity.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {
                    "capacity": {
                        "remediation": {
                            "maxActiveRegion": 128,
                            "maxActivePerCluster": 4,
                            "maxActivePerResourceClass": 4,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--file",
            str(config),
            "--reference",
            "CHG-12345",
            "--dry-run",
        ]
    )

    assert admin_cli.run(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "DRY_RUN"
    assert output["source"] == f"file:{config.resolve()}"
    assert output["affected_roles"] == ["worker"]
    assert output["desired_config"]["capacity"]["telemetry_spool"] == {
        "enabled": False,
        "replicas": 0,
    }
    assert not admin_config_plan_path(tmp_path).exists(), (
        "config --dry-run persisted an active internal plan"
    )


def test_config_preset_uses_existing_signed_release_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        admin_cli, "_run_automatic_release", lambda **kwargs: calls.append(kwargs) or 0
    )
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--preset",
            "32-disabled",
            "--reference",
            "CHG-12345",
        ]
    )

    assert admin_cli.run(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "APPLIED"
    assert output["release_id"] == "release-a"
    assert output["affected_roles"] == ["worker"]
    assert calls[0]["site_file"] == tmp_path / "site.yaml"
    assert admin_config_file_path(tmp_path).is_file(), (
        "config preset did not materialize the canonical editable file"
    )


def test_config_without_input_creates_canonical_file_and_stops(tmp_path: Path) -> None:
    site_file(tmp_path)
    arguments = admin_cli.parser().parse_args(
        ["config", "--state-dir", str(tmp_path), "--reference", "CHG-12345"]
    )

    with pytest.raises(AdminConfigError, match="edit it and rerun"):
        admin_cli.run(arguments)

    path = admin_config_file_path(tmp_path)
    assert path.is_file(), "config without input did not create admin-config.yaml"
    assert path.stat().st_mode & 0o777 == 0o600


def test_first_deploy_can_import_private_admin_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "admin-config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {"capacity": {"preset": "32-disabled"}},
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    calls = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
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
            "--config",
            str(config),
        ]
    )

    assert admin_cli.run(arguments) == 0
    desired = load_desired_admin_config(tmp_path / "state")
    assert desired.capacity.remediation.max_active_region == 128
    canonical = admin_config_file_path(tmp_path / "state")
    assert load_desired_admin_config(tmp_path / "state") == (
        admin_cli.load_admin_config_file(canonical)
    )
    assert calls, "first deploy did not continue into source preparation"


def test_existing_site_rejects_direct_config_change_on_deploy(tmp_path: Path) -> None:
    site_file(tmp_path)
    config = tmp_path / "admin-config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {"capacity": {"preset": "50-disabled"}},
            }
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
            "--state-dir",
            str(tmp_path),
            "--admin-email",
            "operations@example.com",
            "--config",
            str(config),
        ]
    )

    with pytest.raises(AdminConfigError, match="gpu-fault-admin config"):
        admin_cli.run(arguments)


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
