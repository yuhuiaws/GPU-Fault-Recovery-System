"""The ``gpu-fault-admin config`` command: dry run, apply, resume, live-state gates.

Split out of ``test_admin_site.py`` so neither file needs an architecture size
exception; the site-file and live-release helpers stay in the original module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import operator_identity
from gpu_fault.admin.config import (
    AdminConfigError,
    AuroraCapacityConfig,
    admin_config_desired_path,
    admin_config_history_path,
    admin_config_pending_path,
    begin_admin_config_apply,
    load_desired_admin_config,
    load_pending_admin_config_apply,
)
from gpu_fault.admin.config_file import (
    admin_config_file_path,
    initialize_desired_admin_config,
    load_admin_config_file,
)
from gpu_fault.admin.config_patch import preset_admin_config
from gpu_fault.admin.operation_lock import SiteOperationBusy, site_operation_lock
from gpu_fault.admin.site import SiteConfigError
from tests.admin.test_admin_config import legacy_capacity_record
from tests.admin.test_admin_site import REGION, mock_live_release, site_file

APPROVER = "arn:aws:sts::123456789012:assumed-role/Admin/alice"
SITE_IDENTITY = {
    "site_name": "test-site",
    "aws_region": REGION,
    "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
}
AURORA_16_64 = {"aurora": {"minAcu": 16, "maxAcu": 64}}


@pytest.fixture(autouse=True)
def fixed_approver(monkeypatch: pytest.MonkeyPatch) -> None:
    """The STS caller every apply in this module is attributed to."""

    monkeypatch.setattr(
        operator_identity, "caller_identity_arn", lambda **_kwargs: APPROVER
    )


def _yaml(path: Path, spec: dict[str, object]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "AdminConfig",
                "spec": spec,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _arguments(tmp_path: Path, *extra: str) -> argparse.Namespace:
    return admin_cli.parser().parse_args(
        ["config", "--state-dir", str(tmp_path), *extra]
    )


class Stubs:
    """The external effects of an apply -- Aurora request, role rollout, Aurora
    wait, Aurora rollback -- recorded in the order the command took them."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        release_returncode: int = 0,
        await_error: Exception | None = None,
    ) -> None:
        self.events: list[str] = []
        self.calls: dict[str, list[dict[str, object]]] = {
            "request": [],
            "release": [],
            "await": [],
            "rollback": [],
        }

        def request(**kwargs):
            self._record("request", kwargs)
            return {
                "before": {},
                "after": {"min_acu": kwargs["desired"].min_acu},
                "modified": True,
                "scale_up": kwargs["desired"].min_acu > kwargs["expected"].min_acu,
            }

        def release(**kwargs):
            self._record("release", kwargs)
            return release_returncode

        def wait(**kwargs):
            self._record("await", kwargs)
            if await_error is not None:
                raise await_error
            return {"instances": [], "min_acu": kwargs["desired"].min_acu}

        def rollback(**kwargs):
            self._record("rollback", kwargs)
            return {"modified": True, "before": {}, "after": {}}

        monkeypatch.setattr(admin_cli, "request_aurora_capacity", request)
        monkeypatch.setattr(admin_cli, "_run_automatic_release", release)
        monkeypatch.setattr(admin_cli, "await_aurora_capacity", wait)
        monkeypatch.setattr(admin_cli, "reconcile_aurora_capacity", rollback)

    def _record(self, event: str, kwargs: dict[str, object]) -> None:
        self.events.append(event)
        self.calls[event].append(kwargs)


def test_config_help_is_single_level_without_capacity_flags(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args(["config", "--help"])

    help_text = capsys.readouterr().out
    for value in ("--state-dir", "--file", "--reference", "--dry-run"):
        assert value in help_text
    for value in (
        "--preset",
        "--control-worker-replicas",
        "--spool",
        "--max-active",
        "--largest-cluster-node-count",
        "--managed-node-count",
        "--plan-sha256",
        "config plan",
        "config apply",
    ):
        assert value not in help_text, f"{value} is still advertised"
    assert (
        admin_cli.parser().parse_args(["config", "--state-dir", "s"]).reference is None
    )


def test_config_dry_run_is_local_and_prints_the_change_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No cosign, no kubectl, no STS, and a legacy desired.json is not migrated."""

    site_file(tmp_path)
    legacy = replace(
        preset_admin_config("32-disabled"),
        aurora=AuroraCapacityConfig(min_acu=0.5, max_acu=8.0),
    )
    desired_path = admin_config_desired_path(tmp_path)
    desired_path.parent.mkdir(parents=True)
    desired_path.write_text(
        json.dumps(legacy_capacity_record(legacy)), encoding="utf-8"
    )
    persisted = desired_path.read_bytes()

    def forbidden(*arguments, **_kwargs):
        raise AssertionError(f"dry-run ran a subprocess: {arguments[0]}")

    monkeypatch.setattr(admin_cli.subprocess, "run", forbidden)
    monkeypatch.setattr(admin_cli, "verify_prebuilt_release", forbidden)
    monkeypatch.setattr(admin_cli, "_live_release_state", forbidden)
    config = _yaml(
        tmp_path / "capacity.yaml",
        {
            "aurora": {"minAcu": 82, "maxAcu": 128},
            "capacity": {"remediation": {"maxActiveRegion": 200}},
        },
    )

    assert admin_cli.run(_arguments(tmp_path, "--file", str(config), "--dry-run")) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "DRY_RUN"
    assert output["source"] == f"file:{config.resolve()}"
    assert output["affected_roles"] == ["worker"]
    assert output["aurora_changed"] is True
    assert {change["field"] for change in output["changes"]} == {
        "aurora.max_acu",
        "aurora.min_acu",
        "capacity.remediation.max_active_region",
    }
    assert output["before_config_sha256"] == legacy.sha256()
    assert desired_path.read_bytes() == persisted, "dry-run migrated desired.json"
    assert not admin_config_pending_path(tmp_path).exists(), (
        "dry-run wrote a pending apply"
    )
    assert not (tmp_path / "admin-config/history").exists(), "dry-run wrote history"


def test_config_applies_with_a_default_reference_and_records_the_approver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    config = _yaml(tmp_path / "aurora.yaml", AURORA_16_64)
    stubs = Stubs(monkeypatch)

    assert admin_cli.run(_arguments(tmp_path, "--file", str(config))) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "APPLIED"
    assert output["release_id"] == "release-a"
    assert output["approver_identity"] == APPROVER
    assert output["reference"].startswith(f"{APPROVER}:2"), (
        "default reference is not approver:started"
    )
    assert output["affected_roles"] == []
    assert output["affected_resources"] == ["aurora"]
    # The window is requested first, the roles roll, then the ramp is awaited.
    assert stubs.events == ["request", "release", "await"]
    assert stubs.calls["request"][0]["expected"].min_acu == 8.0
    assert stubs.calls["request"][0]["desired"].min_acu == 16.0
    assert stubs.calls["release"][0]["site_file"] == tmp_path / "site.yaml"
    assert stubs.calls["await"][0]["modified"] is True
    history = admin_config_history_path(tmp_path, output["history"])
    result = json.loads((history / "result.json").read_text())
    assert result["status"] == "APPLIED"
    assert result["approver_identity"] == APPROVER
    assert result["reference"] == output["reference"]
    assert result["details"]["aurora"]["after"]["min_acu"] == 16.0
    persisted = json.loads(admin_config_desired_path(tmp_path).read_text())
    assert persisted["approver_identity"] == APPROVER
    assert load_desired_admin_config(tmp_path).aurora.min_acu == 16.0
    # The canonical file is normalised only once the apply succeeded.
    assert (
        load_admin_config_file(admin_config_file_path(tmp_path)).aurora.min_acu == 16.0
    )
    assert not admin_config_pending_path(tmp_path).exists(), (
        "successful apply left pending.json behind"
    )


def test_config_rolls_back_aurora_and_desired_when_the_release_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    config = _yaml(tmp_path / "aurora.yaml", AURORA_16_64)
    stubs = Stubs(monkeypatch, release_returncode=7)

    assert admin_cli.run(_arguments(tmp_path, "--file", str(config))) == 7

    assert stubs.events == ["request", "release", "rollback"]
    assert stubs.calls["rollback"][0]["expected"].min_acu == 16.0
    assert stubs.calls["rollback"][0]["desired"].min_acu == 8.0
    pending = load_pending_admin_config_apply(tmp_path)
    assert pending is not None, "failed apply did not keep pending.json"
    history = admin_config_history_path(tmp_path, str(pending["history"]))
    result = json.loads((history / "result.json").read_text())
    assert result["status"] == "FAILED"
    assert result["error"] == "release-deploy exited with status 7"
    assert result["details"]["aurora_rollback"]["modified"] is True
    assert load_desired_admin_config(tmp_path).aurora.min_acu == 8.0
    # A failed apply leaves the administrator's canonical file alone.
    assert (
        load_admin_config_file(admin_config_file_path(tmp_path)).aurora.min_acu == 8.0
    )


def test_config_keeps_the_target_when_only_the_aurora_wait_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The roles already run the new config; the rerun only finishes the wait."""

    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    config = _yaml(tmp_path / "aurora.yaml", AURORA_16_64)
    arguments = _arguments(tmp_path, "--file", str(config))
    stubs = Stubs(
        monkeypatch, await_error=AdminConfigError("Aurora capacity did not converge")
    )

    with pytest.raises(AdminConfigError, match="finish waiting for Aurora"):
        admin_cli.run(arguments)

    assert stubs.events == ["request", "release", "await"]
    assert load_desired_admin_config(tmp_path).aurora.min_acu == 16.0
    pending = load_pending_admin_config_apply(tmp_path)
    assert pending is not None, "the unfinished wait did not keep pending.json"
    first = admin_config_history_path(tmp_path, str(pending["history"]))
    result = json.loads((first / "result.json").read_text())
    assert result["status"] == "FAILED"
    assert f"gpu-fault-admin config --state-dir {tmp_path}" in result["error"]

    retry = Stubs(monkeypatch)
    assert admin_cli.run(arguments) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "APPLIED"
    assert output["history"] != pending["history"]
    assert retry.events == ["request", "release", "await"]
    assert (first / "result.json").is_file(), "the retry erased the failed attempt"
    assert not admin_config_pending_path(tmp_path).exists(), (
        "successful retry left pending.json behind"
    )


def test_config_noop_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    stubs = Stubs(monkeypatch)

    assert admin_cli.run(_arguments(tmp_path)) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "NOOP"
    assert output["approver_identity"] == APPROVER
    assert stubs.events == []
    assert not admin_config_pending_path(tmp_path).exists(), (
        "a no-op wrote pending.json"
    )
    assert not (tmp_path / "admin-config/history").exists(), "a no-op wrote history"


def test_config_holds_the_site_lock_across_the_whole_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    Stubs(monkeypatch)
    observed: list[str] = []

    def release(**_kwargs):
        try:
            with site_operation_lock(tmp_path, wait=False):
                observed.append("free")
        except SiteOperationBusy:
            observed.append("held")
        return 0

    monkeypatch.setattr(admin_cli, "_run_automatic_release", release)
    config = _yaml(
        tmp_path / "c.yaml", {"capacity": {"remediation": {"maxActiveRegion": 40}}}
    )

    assert admin_cli.run(_arguments(tmp_path, "--file", str(config))) == 0

    assert observed == ["held"]


def test_config_rejects_local_release_drift_from_live_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_file(tmp_path)
    initialize_desired_admin_config(tmp_path)
    mock_live_release(monkeypatch, release_id="release-b")
    config = _yaml(tmp_path / "aurora.yaml", AURORA_16_64)

    with pytest.raises(SiteConfigError, match="differs from the live regional release"):
        admin_cli.run(_arguments(tmp_path, "--file", str(config)))

    assert not admin_config_pending_path(tmp_path).exists(), (
        "release drift recorded a pending apply"
    )
    assert load_desired_admin_config(tmp_path).aurora.min_acu == 8.0


def test_config_rejects_uncommitted_live_state_without_a_pending_apply(
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
    config = _yaml(tmp_path / "c.yaml", {"capacity": {"preset": "32-enabled"}})

    with pytest.raises(
        SiteConfigError, match="live regional release is not committed"
    ) as e:
        admin_cli.run(_arguments(tmp_path, "--file", str(config)))

    assert f"gpu-fault-admin deploy --state-dir {tmp_path}" in str(e.value)


@pytest.mark.parametrize(
    "phase",
    # Every phase here is one `regional_admin_commands.RESUMABLE_PHASES` accepts,
    # so the admin CLI has to accept it too: a phase the release engine can
    # resume from but this list rejects turns a pending config apply into a
    # dead end.
    ("cpu-staged", "candidate-preflight-ready"),
)
def test_config_resumes_a_matching_pending_apply_on_uncommitted_live_state(
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
    pending = begin_admin_config_apply(
        tmp_path,
        site_identity=SITE_IDENTITY,
        release_identity=release_identity,
        desired=desired,
        source="file:/secure/admin-config.yaml",
        approver_identity=APPROVER,
        reference="CHG-12345",
    )
    mock_live_release(
        monkeypatch,
        committed=False,
        phase=phase,
        admin_config_sha256=desired.sha256(),
        release_diff_kind="CONTROL_PLANE_ONLY",
    )
    stubs = Stubs(monkeypatch)
    config = _yaml(tmp_path / "c.yaml", {"capacity": {"preset": "32-enabled"}})

    returncode = admin_cli.run(
        _arguments(tmp_path, "--file", str(config), "--reference", "CHG-12345")
    )

    assert returncode == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "APPLIED"
    assert output["history"] == pending.history
    assert output["reference"] == "CHG-12345"
    assert stubs.events == ["request", "release", "await"]
    assert not admin_config_pending_path(tmp_path).exists(), (
        "successful resume left pending.json behind"
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
    Stubs(monkeypatch)
    config = _yaml(
        tmp_path / "c.yaml",
        {
            "capacity": {
                "remediation": {"maxActiveRegion": 128, "maxActivePerCluster": 4}
            }
        },
    )

    assert admin_cli.run(_arguments(tmp_path, "--file", str(config))) == 0

    assert [call[-5:] for call in calls] == [
        ["get", "configmap", "gpu-fault-regional-release-state", "-o", "json"]
    ]


def test_config_requires_content_addressed_release_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = site_file(tmp_path)
    root = Path(yaml.safe_load(site.read_text())["spec"]["repositoryRoot"])
    (root / "dist/release-a/release.json").unlink()
    initialize_desired_admin_config(tmp_path)
    mock_live_release(monkeypatch)
    config = _yaml(tmp_path / "aurora.yaml", AURORA_16_64)

    with pytest.raises(
        SiteConfigError, match="content-addressed release manifest is missing"
    ):
        admin_cli.run(_arguments(tmp_path, "--file", str(config)))


def test_config_without_input_creates_canonical_file_and_stops(tmp_path: Path) -> None:
    site_file(tmp_path)

    with pytest.raises(AdminConfigError) as error:
        admin_cli.run(_arguments(tmp_path))

    assert f"edit it and rerun gpu-fault-admin config --state-dir {tmp_path}" in str(
        error.value
    )
    path = admin_config_file_path(tmp_path)
    assert path.is_file(), "config without input did not create admin-config.yaml"
    assert path.stat().st_mode & 0o777 == 0o600


def test_first_deploy_can_import_private_admin_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _yaml(
        tmp_path / "admin-config.yaml", {"capacity": {"preset": "32-disabled"}}
    )
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
    config = _yaml(
        tmp_path / "admin-config.yaml", {"capacity": {"preset": "50-disabled"}}
    )
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

    with pytest.raises(AdminConfigError, match="gpu-fault-admin config --state-dir"):
        admin_cli.run(arguments)


def test_config_hands_its_site_lock_to_release_deploy(
    tmp_path: Path, monkeypatch
) -> None:
    """Live 2026-09-13: `config` held the site lock and spawned release-deploy,
    which takes the same lock and -- without the inherited descriptor -- refused
    its own parent as "another administrator mutation". The release is now given
    the held descriptor, which must be the site lock file itself."""

    import os

    from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK

    site_file(tmp_path)
    mock_live_release(monkeypatch)
    initialize_desired_admin_config(tmp_path)
    Stubs(monkeypatch)
    handed: list[object] = []

    def release(**kwargs):
        lock_fd = kwargs.get("lock_fd")
        handed.append(lock_fd)
        if isinstance(lock_fd, int):
            held = os.fstat(lock_fd)
            expected = os.stat(tmp_path / SITE_OPERATION_LOCK)
            handed.append(
                (held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino)
            )
        return 0

    monkeypatch.setattr(admin_cli, "_run_automatic_release", release)
    config = _yaml(
        tmp_path / "c.yaml", {"capacity": {"remediation": {"maxActiveRegion": 40}}}
    )

    assert admin_cli.run(_arguments(tmp_path, "--file", str(config))) == 0

    assert len(handed) == 2 and isinstance(handed[0], int), (
        "the release is handed an open descriptor"
    )
    assert handed[1] is True, "the descriptor is the site operation lock"
