"""The ``gpu-fault-admin config`` command: plan, apply, and live-state gates.

Split out of ``test_admin_site.py`` so neither file needs an architecture size
exception; the site-file and live-release helpers stay in the original module.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin.config import (
    AdminConfigError,
    admin_config_plan_path,
    create_admin_config_plan,
    load_desired_admin_config,
    prepare_admin_config_apply,
)
from gpu_fault.admin.config_file import (
    admin_config_file_path,
    initialize_desired_admin_config,
)
from gpu_fault.admin.config_patch import preset_admin_config
from gpu_fault.admin.site import SiteConfigError
from tests.admin.test_admin_site import (
    REGION,
    mock_live_release,
    reconcile_aurora_stub,
    site_file,
)


def test_config_help_is_single_level(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args(["config", "--help"])

    help_text = capsys.readouterr().out
    for value in ("--state-dir", "--file", "--preset", "--reference", "--dry-run"):
        assert value in help_text
    for value in ("config plan", "config apply", "--plan-sha256"):
        assert value not in help_text


def test_config_dry_run_from_private_yaml_is_role_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
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
    mock_live_release(monkeypatch)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        admin_cli, "_run_automatic_release", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.setattr(admin_cli, "reconcile_aurora_capacity", reconcile_aurora_stub)
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
    assert output["affected_roles"] == ["ingress", "spool", "worker"]
    assert calls[0]["site_file"] == tmp_path / "site.yaml"
    assert admin_config_file_path(tmp_path).is_file(), (
        "config preset did not materialize the canonical editable file"
    )


def test_config_applies_post_deploy_aurora_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    config = tmp_path / "aurora.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {"aurora": {"minAcu": 16, "maxAcu": 64}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    aurora_calls = []
    release_calls = []
    monkeypatch.setattr(
        admin_cli,
        "reconcile_aurora_capacity",
        lambda **kwargs: aurora_calls.append(kwargs)
        or {"modified": True, "before": {}, "after": {}},
    )
    monkeypatch.setattr(
        admin_cli,
        "_run_automatic_release",
        lambda **kwargs: release_calls.append(kwargs) or 0,
    )
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--file",
            str(config),
            "--reference",
            "CHG-AURORA-1",
        ]
    )

    assert admin_cli.run(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["affected_roles"] == []
    assert output["affected_resources"] == ["aurora"]
    assert aurora_calls[0]["expected"].min_acu == 8.0
    assert aurora_calls[0]["desired"].min_acu == 16.0
    assert release_calls, "Aurora-only config did not refresh live release metadata"
    assert load_desired_admin_config(tmp_path).aurora.min_acu == 16.0


def test_config_rolls_back_aurora_when_release_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    config = tmp_path / "aurora.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": {"aurora": {"minAcu": 16, "maxAcu": 64}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    aurora_calls = []

    def reconcile(**kwargs):
        aurora_calls.append(kwargs)
        return {"modified": True, "before": {}, "after": {}}

    monkeypatch.setattr(admin_cli, "reconcile_aurora_capacity", reconcile)
    monkeypatch.setattr(admin_cli, "_run_automatic_release", lambda **_kwargs: 7)
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--file",
            str(config),
            "--reference",
            "CHG-AURORA-2",
        ]
    )

    assert admin_cli.run(arguments) == 7
    assert len(aurora_calls) == 2
    assert aurora_calls[0]["desired"].min_acu == 16.0
    assert aurora_calls[1]["desired"].min_acu == 8.0
    plan = json.loads(admin_config_plan_path(tmp_path).read_text())
    failed = (
        tmp_path / "admin-config/history" / str(plan["plan_sha256"]) / "failed.json"
    )
    result = json.loads(failed.read_text())
    assert result["status"] == "FAILED"
    assert result["details"]["aurora_rollback"]["modified"] is True


def test_config_rejects_local_release_drift_from_live_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)
    mock_live_release(monkeypatch, release_id="release-b")
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--preset",
            "32-disabled",
            "--reference",
            "CHG-12345",
            "--dry-run",
        ]
    )

    with pytest.raises(SiteConfigError, match="differs from the live regional release"):
        admin_cli.run(arguments)

    assert not admin_config_plan_path(tmp_path).exists(), (
        "release drift persisted an admin config plan"
    )


def test_config_rejects_uncommitted_live_state_without_active_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)
    desired = preset_admin_config("32-enabled")
    mock_live_release(
        monkeypatch,
        committed=False,
        phase="cpu-staged",
        admin_config_sha256=desired.sha256(),
        release_diff_kind="CONTROL_PLANE_ONLY",
    )
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--preset",
            "32-enabled",
            "--reference",
            "CHG-12345",
        ]
    )

    with pytest.raises(SiteConfigError, match="live regional release is not committed"):
        admin_cli.run(arguments)


@pytest.mark.parametrize(
    "phase",
    # Every phase here is one `regional_admin_commands.RESUMABLE_PHASES` accepts,
    # so the admin CLI has to accept it too: a phase the release engine can
    # resume from but this list rejects turns an approved config apply into a
    # dead end.
    ("cpu-staged", "candidate-preflight-ready"),
)
def test_config_resumes_matching_approved_uncommitted_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    path = site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)
    root = Path(yaml.safe_load(path.read_text())["spec"]["repositoryRoot"])
    raw = (root / "dist/current-release.json").read_bytes()
    release_identity = {
        "release_id": "release-a",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "staging_only": False,
    }
    desired = preset_admin_config("32-enabled")
    plan = create_admin_config_plan(
        tmp_path,
        site_identity={
            "site_name": "test-site",
            "aws_region": REGION,
            "cpu_eks_arn": ("arn:aws:eks:us-east-1:123456789012:cluster/control"),
        },
        release_identity=release_identity,
        desired=desired,
        source="preset:32-enabled",
    )
    prepare_admin_config_apply(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        reference="CHG-12345",
        current_release_identity=release_identity,
    )
    mock_live_release(
        monkeypatch,
        committed=False,
        phase=phase,
        admin_config_sha256=desired.sha256(),
        release_diff_kind="CONTROL_PLANE_ONLY",
    )
    calls = []
    monkeypatch.setattr(
        admin_cli, "_run_automatic_release", lambda **kwargs: calls.append(kwargs) or 0
    )
    monkeypatch.setattr(admin_cli, "reconcile_aurora_capacity", reconcile_aurora_stub)
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--preset",
            "32-enabled",
            "--reference",
            "CHG-12345",
        ]
    )

    assert admin_cli.run(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "APPLIED"
    assert output["release_id"] == "release-a"
    assert calls, "matching approved plan did not resume release deployment"
    assert not admin_config_plan_path(tmp_path).exists(), (
        "successful resume left the active admin config plan"
    )


def test_config_reads_live_release_state_from_the_cpu_configmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)
    expected = {
        "release_id": "release-a",
        "transaction_committed": True,
        "phase": "complete",
    }
    calls: list[list[str]] = []
    monkeypatch.setattr(
        admin_cli, "verify_prebuilt_release", lambda *_args, **_kwargs: None
    )

    def run(arguments, **_kwargs):
        calls.append(list(arguments))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps({"data": {"state.json": json.dumps(expected)}}),
            stderr="",
        )

    monkeypatch.setattr(admin_cli.subprocess, "run", run)
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--preset",
            "32-disabled",
            "--reference",
            "CHG-12345",
            "--dry-run",
        ]
    )

    assert admin_cli.run(arguments) == 0
    assert calls[0][-5:] == [
        "get",
        "configmap",
        "gpu-fault-regional-release-state",
        "-o",
        "json",
    ]


def test_config_requires_content_addressed_release_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = site_file(tmp_path)
    root = Path(yaml.safe_load(site.read_text())["spec"]["repositoryRoot"])
    (root / "dist/release-a/release.json").unlink()
    initialize_desired_admin_config(tmp_path)
    mock_live_release(monkeypatch)
    arguments = admin_cli.parser().parse_args(
        [
            "config",
            "--state-dir",
            str(tmp_path),
            "--preset",
            "32-disabled",
            "--reference",
            "CHG-12345",
            "--dry-run",
        ]
    )

    with pytest.raises(
        SiteConfigError, match="content-addressed release manifest is missing"
    ):
        admin_cli.run(arguments)


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
