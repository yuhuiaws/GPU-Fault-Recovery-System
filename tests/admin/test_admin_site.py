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
        "channel": "ses",
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
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
            "--confirm",
            "REMOVE_GPU_CLUSTER",
        ]
    )

    assert admin_cli.run(arguments) == 0
    # The administrator names the cluster by ARN, as for deploy and
    # join-cluster; the derived cluster_id stays internal.
    assert calls[0].cluster_id == "gpu-a"
    assert calls[0].confirmation == "REMOVE_GPU_CLUSTER"


def test_admin_cli_names_the_managed_clusters_for_an_unknown_arn(
    tmp_path, monkeypatch
) -> None:
    site_file(tmp_path)
    monkeypatch.setattr(
        admin_cli, "remove_cluster", lambda request: pytest.fail("must not run")
    )
    arguments = admin_cli.parser().parse_args(
        [
            "remove-cluster",
            "--state-dir",
            str(tmp_path),
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-z",
            "--confirm",
            "REMOVE_GPU_CLUSTER",
        ]
    )

    with pytest.raises(admin_cli.BootstrapError, match="gpu-z") as failure:
        admin_cli.run(arguments)
    assert "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a" in str(failure.value)


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
    arguments = argparse.Namespace(
        command="deploy", file=path, repo_root=None, show_effective_config=False
    )

    assert admin_cli.run(arguments) == 1
    assert [call[1] for call in calls] == ["preflight"]


# `status` is the public verb; `preflight` and `verify` are the driver-only
# passthroughs `scripts/staging_deploy.py` and `scripts/release_deploy.py` use.
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


def test_uninstall_no_longer_discovers_a_site_from_cluster_arns(
    tmp_path: Path, capsys
) -> None:
    """Every production state directory carries ``site.yaml``; the ARN discovery
    branch (and ``legacy_site``) is gone, so the flags are unknown."""

    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(
            [
                "uninstall",
                "--cpu-cluster-arn",
                "arn:aws:sagemaker:us-west-2:123456789012:cluster/cpu",
                "--gpu-cluster-arn",
                "arn:aws:sagemaker:us-west-2:123456789012:cluster/gpu",
                "--confirm",
                "UNINSTALL_GPU_FAULT",
            ]
        )
    assert "--cpu-cluster-arn" in capsys.readouterr().err, (
        "argparse must name the unknown flag"
    )

    arguments = admin_cli.parser().parse_args(
        ["uninstall", "--state-dir", str(tmp_path), "--confirm", "UNINSTALL_GPU_FAULT"]
    )
    with pytest.raises(SiteConfigError, match="found no managed site"):
        admin_cli.run(arguments)


def test_uninstall_keeps_aurora_unless_the_database_reset_is_explicit(
    tmp_path: Path, monkeypatch
) -> None:
    site_file(tmp_path)
    calls = []
    monkeypatch.setattr(
        admin_cli,
        "uninstall",
        lambda request: calls.append(request) or {"phase": "COMPLETED"},
    )
    common = ["uninstall", "--state-dir", str(tmp_path), "--cpu-cluster", "keep"]

    admin_cli.run(
        admin_cli.parser().parse_args([*common, "--confirm", "UNINSTALL_GPU_FAULT"])
    )
    admin_cli.run(
        admin_cli.parser().parse_args(
            [*common, "--reset-database", "--confirm", "UNINSTALL_GPU_FAULT"]
        )
    )

    assert [request.reset_database for request in calls] == [False, True], (
        "only the explicit flag asks for the database wipe"
    )
    assert {request.final_snapshot_policy for request in calls} == {"retain"}, (
        "keep mode always retains the final snapshot"
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (["--cpu-cluster", "keep", "--aurora-final-snapshot", "skip"], "delete"),
        (["--cpu-cluster", "delete", "--reset-database"], "keep"),
    ),
)
def test_uninstall_refuses_contradictory_aurora_flags(
    tmp_path: Path, monkeypatch, extra: list[str], message: str
) -> None:
    site_file(tmp_path)
    monkeypatch.setattr(
        admin_cli, "uninstall", lambda request: pytest.fail("must not run")
    )
    arguments = admin_cli.parser().parse_args(
        ["uninstall", "--state-dir", str(tmp_path), *extra, "--confirm", "x"]
    )

    with pytest.raises(SiteConfigError, match=f"--cpu-cluster {message}"):
        admin_cli.run(arguments)


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
        "--rollback",
        "--approve-profile-plan",
        "--reference",
        "--wait-for-email-confirmation",
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


def test_public_verbs_are_the_six_plus_two_and_approve_profile_is_gone() -> None:
    """The verb table: no ``approve-profile`` (folded into ``deploy``), and the
    two wave-1 verbs ``rotate-token``/``submit-remediation`` registered."""

    choices = sorted(admin_cli.parser()._subparsers._group_actions[0].choices)

    assert "approve-profile" not in choices
    assert "rotate-token" in choices
    assert "submit-remediation" in choices
    assert "preflight" not in choices and "verify" not in choices, (
        "the driver passthroughs stay off the public table"
    )
    assert not hasattr(admin_cli, "_run_profile_approval"), (
        "the verb's runner went with it"
    )
    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(
            ["approve-profile", "--state-dir", "/tmp/x", "--plan-sha256", "a" * 64]
        )


def _managed_state(tmp_path: Path, *, gpu_names: tuple[str, ...] = ("gpu-a",)) -> Path:
    """A state directory holding a loadable ``site.yaml`` with an admin email."""

    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    first = document["spec"]["clusters"][0]
    document["spec"]["clusters"] = [
        {
            **first,
            "clusterId": name,
            "context": name,
            "hyperpodClusterName": f"hp-{name}",
            "eksClusterArn": f"arn:aws:eks:us-east-1:123456789012:cluster/{name}",
        }
        for name in gpu_names
    ]
    document["spec"]["notifications"] = {"adminEmail": "operations@example.com"}
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return tmp_path


def test_deploy_with_state_dir_alone_upgrades_the_managed_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deploy --state-dir X`` is the whole upgrade command on a managed site."""

    state_dir = _managed_state(tmp_path)
    calls: list[dict] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    arguments = admin_cli.parser().parse_args(["deploy", "--state-dir", str(state_dir)])

    assert admin_cli.run(arguments) == 0
    assert calls[0]["cpu_cluster_arn"] == (
        "arn:aws:eks:us-east-1:123456789012:cluster/control"
    )
    assert calls[0]["gpu_cluster_arns"] == (
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
    )
    assert calls[0]["admin_email"] == "operations@example.com"


def test_deploy_superset_joins_the_delta_without_a_rollout_when_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The managed ARNs plus a new one: the new one is joined; when the source
    deploy applied no release (the site would NOOP) the join is the whole cost."""

    state_dir = _managed_state(tmp_path)
    new_arn = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"
    calls: list[dict] = []
    joined: list[str] = []

    def source_deploy(**kwargs):
        calls.append(kwargs)
        (state_dir / "source-deploy-success.json").write_text(
            json.dumps({"schema_version": 1, "status": "PASSED", "mode": "UNCHANGED"}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(admin_cli, "run_source_deploy", source_deploy)
    monkeypatch.setattr(admin_cli, "load_site", lambda *_a, **_k: "site")
    monkeypatch.setattr(
        admin_cli,
        "join_clusters",
        lambda requests: joined.extend(r.gpu_cluster_arn for r in requests) or {},
    )
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(state_dir),
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
            "--gpu-cluster-arn",
            new_arn,
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[0]["gpu_cluster_arns"] == (
        "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        new_arn,
    )
    assert joined == [new_arn]


def test_deploy_superset_after_an_application_release_leaves_the_join_to_the_inner_hop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An application release joins the delta itself (``pending_gpu_cluster_arns``
    right after the release); the outer hop must not join it a second time."""

    state_dir = _managed_state(tmp_path)

    def source_deploy(**_kwargs):
        (state_dir / "source-deploy-success.json").write_text(
            json.dumps({"schema_version": 1, "mode": "APPLICATION_RELEASE"}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(admin_cli, "run_source_deploy", source_deploy)
    monkeypatch.setattr(
        admin_cli,
        "join_clusters",
        lambda requests: pytest.fail("the outer hop joined after a release"),
    )
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(state_dir),
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        ]
    )

    assert admin_cli.run(arguments) == 0


def test_deploy_subset_is_refused_naming_remove_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _managed_state(tmp_path, gpu_names=("gpu-a", "gpu-b"))
    monkeypatch.setattr(
        admin_cli,
        "run_source_deploy",
        lambda **_kwargs: pytest.fail("a subset reached the source preparer"),
    )
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(state_dir),
            "--gpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        ]
    )

    with pytest.raises(SiteConfigError, match="remove-cluster") as failure:
        admin_cli.run(arguments)
    assert "cluster/gpu-b" in str(failure.value), "the omitted cluster is named"


def test_deploy_with_a_different_cpu_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _managed_state(tmp_path)
    monkeypatch.setattr(
        admin_cli,
        "run_source_deploy",
        lambda **_kwargs: pytest.fail("a foreign CPU reached the source preparer"),
    )
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(state_dir),
            "--cpu-cluster-arn",
            "arn:aws:eks:us-east-1:123456789012:cluster/other",
        ]
    )

    with pytest.raises(SiteConfigError, match="CPU cluster identity differs"):
        admin_cli.run(arguments)


def test_deploy_rollback_dispatches_to_the_rollback_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _managed_state(tmp_path)
    seen: list[tuple[object, Path, dict[str, str]]] = []
    monkeypatch.setattr(admin_cli, "load_site", lambda *_a, **_k: "site")
    monkeypatch.setattr(
        admin_cli,
        "run_rollback",
        lambda site, *, state_dir, environment: (
            seen.append((site, state_dir, dict(environment))) or 2
        ),
    )
    monkeypatch.setattr(
        admin_cli,
        "run_source_deploy",
        lambda **_kwargs: pytest.fail("--rollback ran a source deploy"),
    )
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(state_dir), "--rollback"]
    )

    assert admin_cli.run(arguments) == 2, "the rollback command's exit code is returned"
    assert seen == [("site", state_dir.resolve(), {})], (
        "without a consent flag the rollback command gets an empty consent mapping"
    )


def test_deploy_rollback_carries_the_inflight_consent_to_the_rollback_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deploy --rollback --allow-inflight-installs`` reaches the engine as
    ``GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS=1`` through the same consent
    builder the upgrade path uses; the gate runs on the rollback too."""

    state_dir = _managed_state(tmp_path)
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(admin_cli, "load_site", lambda *_a, **_k: "site")
    monkeypatch.setattr(
        admin_cli,
        "run_rollback",
        lambda site, *, state_dir, environment: seen.append(dict(environment)) or 0,
    )
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(state_dir),
            "--rollback",
            "--allow-inflight-installs",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert seen == [{admin_cli.ALLOW_INFLIGHT_INSTALLS_ENV: "1"}], (
        "the flag is the only consent a rollback carries, as the engine's variable"
    )


def test_the_inflight_consent_without_rollback_still_reaches_the_source_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _managed_state(tmp_path)
    calls: list[dict] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.setattr(
        admin_cli, "run_rollback", lambda *_a, **_k: pytest.fail("rollback ran")
    )
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(state_dir), "--allow-inflight-installs"]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[-1]["extra_environment"] == {
        admin_cli.ALLOW_INFLIGHT_INSTALLS_ENV: "1"
    }, "the upgrade path is unchanged: the consent travels as before"


@pytest.mark.parametrize(
    "extra",
    (
        ["--gpu-cluster-arn", "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"],
        ["--approve-profile-plan", "a" * 64, "--reference", "CHG-1"],
        ["--accept-schema-change"],
        ["--accept-schema-change-without-snapshot"],
        ["--supersede-failed-transaction"],
    ),
)
def test_deploy_rollback_takes_no_other_deploy_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    state_dir = _managed_state(tmp_path)
    monkeypatch.setattr(
        admin_cli, "run_rollback", lambda *_a, **_k: pytest.fail("rollback ran")
    )
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(state_dir), "--rollback", *extra]
    )

    with pytest.raises(SiteConfigError, match="takes no other deploy option"):
        admin_cli.run(arguments)


def test_deploy_rollback_requires_a_managed_site(tmp_path: Path) -> None:
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(tmp_path / "empty"), "--rollback"]
    )

    with pytest.raises(SiteConfigError, match="requires a managed site"):
        admin_cli.run(arguments)


def test_deploy_approve_profile_plan_approves_inline_before_the_source_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_dir = _managed_state(tmp_path)
    events: list[str] = []

    def approve(path: Path, *, plan_sha256: str, reference: str):
        events.append(f"approve:{path}:{plan_sha256}:{reference}")
        return {
            "plan_sha256": plan_sha256,
            "reference": reference,
            "approved_at": "2026-09-08T12:00:00+00:00",
            "approver_identity": "arn:aws:sts::123456789012:assumed-role/Admin/alice",
        }

    monkeypatch.setattr(admin_cli, "approve_profile_plan_inline", approve)
    monkeypatch.setattr(
        admin_cli,
        "run_source_deploy",
        lambda **kwargs: events.append("source-deploy") or 0,
    )
    arguments = admin_cli.parser().parse_args(
        [
            "deploy",
            "--state-dir",
            str(state_dir),
            "--approve-profile-plan",
            "a" * 64,
            "--reference",
            "CHG-12345",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert events == [
        f"approve:{state_dir.resolve()}:{'a' * 64}:CHG-12345",
        "source-deploy",
    ], "the approval happens first, then the same deploy continues"
    err = capsys.readouterr().err
    assert "2026-09-08T12:00:00+00:00" in err and "/Admin/alice" in err, (
        "the approval record's time and approver are printed (I2)"
    )


def test_deploy_approve_profile_plan_requires_a_reference(tmp_path: Path) -> None:
    state_dir = _managed_state(tmp_path)
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(state_dir), "--approve-profile-plan", "a" * 64]
    )

    with pytest.raises(SiteConfigError, match="requires --reference"):
        admin_cli.run(arguments)


def test_wait_for_email_confirmation_travels_to_the_source_preparer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _managed_state(tmp_path)
    calls: list[dict] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(state_dir), "--wait-for-email-confirmation", "15"]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[0]["wait_for_email_confirmation"] == 15


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("status", ("--full",)),
        ("join-cluster", ("--gpu-cluster-arn",)),
        ("remove-cluster", ("--gpu-cluster-arn", "--confirm")),
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


def test_admin_email_option_is_the_alert_email() -> None:
    arguments = admin_cli.parser().parse_args(
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

    assert arguments.alert_email == "ops@example.com", (
        "--admin-email is the only notification address input"
    )


@pytest.mark.parametrize(
    "flag",
    (
        "--email-sender=sender@example.com",
        "--email-recipient=ops@example.com",
        "--email-subject-prefix=[PROD]",
        "--alert-email=ops@example.com",
        "--allow-legacy-python-foundation",
        "--show-effective-config",
    ),
)
def test_deploy_dropped_its_hidden_notification_and_debug_flags(
    flag: str, capsys
) -> None:
    """The routing overrides were never forwarded by the source preparer and the
    other two had no reader; sender = recipient = ``--admin-email``."""

    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(
            [
                "deploy",
                "--cpu-cluster-arn",
                "arn:aws:eks:us-east-1:123456789012:cluster/cpu",
                "--gpu-cluster-arn",
                "arn:aws:eks:us-east-1:123456789012:cluster/gpu",
                "--admin-email",
                "owner@example.com",
                flag,
            ]
        )
    assert flag.split("=", 1)[0] in capsys.readouterr().err, (
        "argparse must name the unknown flag"
    )
    assert not hasattr(admin_cli, "_configure_site_notifications"), (
        "the zero-caller notification rewriter went with its flags"
    )
    assert not hasattr(admin_cli, "_redacted_config"), (
        "nothing prints the effective config any more"
    )


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


def test_the_flag_wins_over_a_stale_schema_change_variable(monkeypatch) -> None:
    """A leftover ``export`` from an earlier session must not override the
    consent the operator typed; without a flag nothing is added and the
    inherited environment travels unchanged."""

    monkeypatch.setenv(admin_cli.ACCEPT_SCHEMA_CHANGE_ENV, "no-snapshot")
    with_flag = admin_cli.parser().parse_args(
        ["deploy", "--accept-schema-change", "--state-dir", "/tmp/x"]
    )
    without = admin_cli.parser().parse_args(["deploy", "--state-dir", "/tmp/x"])

    assert admin_cli.schema_change_environment(with_flag) == {
        admin_cli.ACCEPT_SCHEMA_CHANGE_ENV: "snapshot"
    }
    assert admin_cli.schema_change_environment(without) == {}


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
