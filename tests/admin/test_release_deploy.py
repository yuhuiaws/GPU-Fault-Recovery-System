from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.admin import profile_approval as admin_profile_approval
from scripts import release_deploy, release_failure_recovery

REGION = "us-east-1"
READ_LIVE_RELEASE_STATE = release_deploy.read_live_release_state


def _verification_report() -> dict[str, object]:
    return {
        "mode": "verify",
        "site_name": "test-site",
        "healthy": True,
        "summary": {"PASS": 12, "WARN": 0, "FAIL": 0, "SKIP": 0},
        "checks": [{"name": "runtime_profile", "status": "PASS"}],
    }


def _release_summary(
    kind: str = "NOOP", changed: list[str] | None = None
) -> dict[str, object]:
    return {
        "mode": "release-summary",
        "site_name": "test-site",
        "next_deploy": {"kind": kind, "changed": changed or []},
    }


def _release_diff(
    kind: str = "NOOP", changed: list[str] | None = None
) -> dict[str, object]:
    return {
        "mode": "release-diff",
        "state_sha256": "a" * 64,
        "next_deploy": {"kind": kind, "changed": changed or [], "resume": False},
    }


def _pending_commit_diff(
    *, state_sha256: str = "b" * 64, release_id: str = "release-a"
) -> dict[str, object]:
    """The diff a successful `rollout deploy` leaves behind.

    Everything is applied and quick validation passed; the commit is the only
    outstanding step, which is why `next_deploy` calls it a resume and names the
    release the evidence has to match.
    """

    return {
        "mode": "release-diff",
        "state_sha256": state_sha256,
        "next_deploy": {
            "kind": "CONTROL_PLANE_ONLY",
            "changed": ["control_plane_wheel"],
            "resume": True,
            "action": "upgrade",
            "pending_commit": True,
            "release_id": release_id,
        },
    }


def _stability_report() -> dict[str, object]:
    return {
        "mode": "stability",
        "healthy": True,
        "window_seconds": 120,
        "sample_count": 5,
    }


def _site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    rollout = root / "deploy/control-plane/regional/rollout-regional-release.sh"
    rollout.parent.mkdir(parents=True)
    rollout.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    rollout.chmod(0o755)
    release_dir = root / "dist/release-a"
    release_dir.mkdir(parents=True)
    release = {"schema_version": 2, "release_id": "release-a", "components": {}}
    content = json.dumps(release, indent=2, sort_keys=True) + "\n"
    (root / "dist/current-release.json").write_text(content, encoding="utf-8")
    (release_dir / "release.json").write_text(content, encoding="utf-8")
    (root / "config").mkdir()
    (root / "config/profile.yaml").write_text(
        "cluster_id: placeholder\n"
        "environment: hyperpod-eks\n"
        "profile_version: profile-v1\n"
        "claims: []\n"
        "observed: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_deploy,
        "read_live_release_state",
        lambda _site: {
            "phase": "rolled-back",
            "rollback_result": {"status": "PASSED"},
            "previous": {},
        },
    )
    monkeypatch.setattr(
        release_failure_recovery,
        "reconcile_rollback_management",
        lambda _site, _state, **_kwargs: root,
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
                "templateSource": "config/profile.yaml",
                "version": "profile-v1",
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
                "runtime": "registry.example/runtime:v1",
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
    site = tmp_path / "site.yaml"
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    site.chmod(0o600)
    monkeypatch.setattr(release_deploy, "ROOT", root)
    monkeypatch.setattr(
        release_deploy,
        "compute_agent_config_digest",
        lambda *_args, **_kwargs: "b" * 64,
    )
    return site


def test_live_release_state_distinguishes_not_found_from_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    missing = (
        "Error from server (NotFound): configmaps "
        '"gpu-fault-regional-release-state" not found'
    )

    with pytest.raises(release_deploy.ReleaseStateNotFound):
        READ_LIVE_RELEASE_STATE(
            site,
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 1, "", missing
            ),
        )

    with pytest.raises(release_deploy.ReleaseDeployError, match="Forbidden"):
        READ_LIVE_RELEASE_STATE(
            site,
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 1, "", "Error from server (Forbidden)"
            ),
        )


def test_site_resolution_is_explicit_or_environment_driven(tmp_path: Path) -> None:
    site = tmp_path / "site.yaml"
    site.write_text("site", encoding="utf-8")
    site.chmod(0o600)

    assert release_deploy.resolve_site_file(site, {}) == site.resolve()
    assert (
        release_deploy.resolve_site_file(None, {release_deploy.SITE_ENV: str(site)})
        == site.resolve()
    )
    with pytest.raises(release_deploy.ReleaseDeployError, match="never guesses"):
        release_deploy.resolve_site_file(None, {})


def test_prepare_site_release_updates_declared_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    original = site.read_bytes()
    profile = tmp_path / "repo/config/profile.yaml"
    profile_plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest()
    )

    prepared = release_deploy.prepare_site_release(site, profile_plan=profile_plan)

    value = yaml.safe_load(site.read_text(encoding="utf-8"))
    assert value["spec"]["release"] == {
        "manifest": "dist/current-release.json",
        "agentConfigDigest": "b" * 64,
    }
    runtime_profile = value["spec"]["runtimeProfile"]
    assert runtime_profile["version"] == "profile-v1"
    assert runtime_profile["templateSource"] == "config/profile.yaml"
    assert runtime_profile["source"].endswith("profiles/profile-v1.yaml"), (
        "Profile source was not moved to the immutable site snapshot"
    )
    assert prepared.site_changed is True, "new release metadata did not update site"
    assert prepared.release_id == "release-a"
    assert (prepared.state_dir / "site.before.yaml").read_bytes() == original
    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "PREPARED"
    assert state["desired"]["agent_config_digest"] == "b" * 64

    snapshot = Path(runtime_profile["source"])
    repeated_plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    )
    repeated = release_deploy.prepare_site_release(site, profile_plan=repeated_plan)
    assert repeated.site_changed is False, (
        "identical release preparation was not idempotent"
    )


def test_profile_plan_site_identity_uses_stable_cpu_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    profile = tmp_path / "repo/config/profile.yaml"
    live_sha = hashlib.sha256(profile.read_bytes()).hexdigest()
    original = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    added = dict(document["spec"]["clusters"][0])
    added.update(
        {
            "clusterId": "gpu-b",
            "context": "gpu-b",
            "hyperpodClusterName": "hp-gpu-b",
            "eksClusterArn": ("arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"),
        }
    )
    document["spec"]["clusters"].append(added)
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    expanded = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    assert expanded.site_identity == {
        "site_name": "test-site",
        "aws_region": REGION,
        "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
    }
    assert expanded.site_identity_sha256 == original.site_identity_sha256

    document["spec"]["cpu"]["eksArn"] = (
        "arn:aws:eks:us-east-1:123456789012:cluster/control-replacement"
    )
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    replaced_control_plane = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=live_sha
    )

    assert replaced_control_plane.site_identity_sha256 != (
        original.site_identity_sha256
    )


def test_profile_plan_recovers_live_baseline_from_matching_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    template = tmp_path / "repo/config/profile.yaml"
    initial = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(template.read_bytes()).hexdigest()
    )
    release_deploy.prepare_site_release(site, profile_plan=initial)
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    snapshot = Path(document["spec"]["runtimeProfile"]["source"])
    live_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    document["spec"]["runtimeProfile"]["source"] = str(template)
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    recovered = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    assert recovered.change_kind == "UNCHANGED"
    assert recovered.approval_required is False
    assert recovered.active_source == snapshot
    prepared = release_deploy.prepare_site_release(site, profile_plan=recovered)
    repaired = yaml.safe_load(site.read_text(encoding="utf-8"))
    assert repaired["spec"]["runtimeProfile"]["source"] == str(snapshot)
    assert prepared.site_changed is True, "matching live snapshot did not repair site"


def test_execute_release_uses_verified_noop_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        release_deploy,
        "_run",
        lambda arguments, **_kwargs: calls.append(("run", list(arguments))),
    )

    def run_json(arguments, **_kwargs):
        command = list(arguments)
        calls.append(("json", command))
        if "verify" in command:
            return _verification_report()
        if "release-diff" in command:
            return _release_diff("NOOP", ["release_delivery"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)

    profile = tmp_path / "repo/config/profile.yaml"
    prepared = release_deploy.execute_release(
        site,
        live_state={
            "release_id": "release-previous",
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
    )

    assert calls[0][1][-1] == "check"
    assert calls[1][1][1] == "release-diff"
    assert calls[2][1][1] == "stage-noop"
    assert calls[3][1][3] == "verify"
    assert calls[4][1][1] == "commit"
    assert calls[5][1][1] == "release-summary"
    assert len(calls) == 6
    assert all("deploy" not in command for _kind, command in calls), (
        "verified NOOP release still invoked the deploy command"
    )
    assert all("status" not in command for _kind, command in calls), (
        "release-deploy repeated the full status command after verify"
    )

    verification_path = prepared.state_dir / release_deploy.VERIFICATION_REPORT
    summary_path = prepared.state_dir / release_deploy.RELEASE_SUMMARY_REPORT
    assert json.loads(verification_path.read_text())["healthy"] is True
    assert json.loads(summary_path.read_text())["next_deploy"]["kind"] == "NOOP"
    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED"
    assert state["deployment"]["status"] == "SKIPPED_NOOP"
    assert state["deployment"]["fast_path"] is True
    assert state["verification"]["status"] == "PASSED"
    assert state["stability"]["status"] == "SKIPPED_NOOP"
    assert state["release_summary"]["status"] == "AVAILABLE"
    assert state["completion_warnings"] == []


def test_execute_release_falls_back_to_deploy_for_non_noop_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        release_deploy,
        "_run",
        lambda arguments, **_kwargs: calls.append(("run", list(arguments))),
    )

    def run_json(arguments, **_kwargs):
        command = list(arguments)
        calls.append(("json", command))
        if "verify" in command:
            return _verification_report()
        if "stability" in command:
            return _stability_report()
        if "release-diff" in command:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    prepared = release_deploy.execute_release(
        site,
        run_checks=False,
        live_state={
            "release_id": "release-a",
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
    )

    assert calls[0][1][1] == "release-diff"
    assert calls[1][1][3] == "deploy"
    # verify and the stability window are independent read-only phases and now
    # run on two threads, so which of them reaches the fake first is not part of
    # the contract. That both precede the commit is.
    assert sorted(
        "stability" if "stability" in command else "verify"
        for _kind, command in calls[2:4]
    ) == ["stability", "verify"]
    assert calls[4][0] == "run"
    assert calls[4][1][1] == "commit"
    assert calls[5][1][1] == "release-summary"
    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["deployment"]["status"] == "APPLIED"
    assert state["deployment"]["fast_path"] is False
    assert state["stability"]["status"] == "PASSED"
    assert state["release_summary"]["next_deploy"]["kind"] == "NOOP"


def test_execute_release_rejects_out_of_band_admin_email_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)

    with pytest.raises(release_deploy.ReleaseDeployError, match="IaC and site.yaml"):
        release_deploy.execute_release(
            site, admin_email="ops@example.com", run_checks=False, live_state={}
        )


def test_execute_release_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)

    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def fail_verify(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", fail_verify)

    with pytest.raises(release_deploy.ReleaseDeployError, match="verify failed"):
        profile = tmp_path / "repo/config/profile.yaml"
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    state_path = site.parent / "release-deploy/release-a/state.json"
    state = json.loads(state_path.read_text())
    assert state["phase"] == "FAILED"
    assert "verify failed" in state["error"]
    assert state["rollback"]["status"] == "PASSED"
    assert state["rollback"]["management_state_synced"] is True


def test_execute_release_rolls_back_after_stability_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    commands = []
    monkeypatch.setattr(
        release_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)),
    )

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            return _verification_report()
        if "stability" in arguments:
            return {"mode": "stability", "healthy": False, "window_seconds": 120}
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError, match="stability report"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "release_id": "release-a",
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest(),
            },
        )

    assert any(command[1] == "rollback" for command in commands), (
        "stability failure did not invoke the low-level rollback"
    )
    assert any(command[1] == "sync-state" for command in commands), (
        "successful rollback did not reconcile the management release state"
    )
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["phase"] == "FAILED"
    assert "stability report" in state["error"]
    assert state["rollback"]["status"] == "PASSED"
    assert state["rollback"]["management_state_synced"] is True
    assert state["rollback"]["started_at"]
    assert state["rollback"]["completed_at"]


def test_execute_release_honors_disabled_automatic_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    document["spec"]["autoRollback"] = False
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        release_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)),
    )

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError, match="verify failed"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    assert not any(command[1] == "rollback" for command in commands), (
        "autoRollback=false still invoked rollback"
    )
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["rollback"]["status"] == "SKIPPED_POLICY"


def test_execute_release_aligns_a_completed_low_level_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    def run(arguments, **_kwargs):
        command = list(arguments)
        commands.append(command)
        if "gpu_fault.admin.cli" in command and "deploy" in command:
            raise release_deploy.ReleaseDeployError("low-level deploy failed")

    monkeypatch.setattr(release_deploy, "_run", run)
    monkeypatch.setattr(
        release_deploy,
        "_run_json",
        lambda arguments, **_kwargs: (
            _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
            if "release-diff" in arguments
            else _release_summary()
        ),
    )
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError, match="low-level deploy"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    assert any(command[1] == "sync-state" for command in commands), (
        "completed low-level rollback was not aligned to management state"
    )
    assert not any(command[1] == "rollback" for command in commands), (
        "completed low-level rollback was executed a second time"
    )
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["rollback"]["status"] == "PASSED"
    assert state["rollback"]["management_state_synced"] is True


def test_execute_release_does_not_rollback_a_committed_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    def run(arguments, **_kwargs):
        command = list(arguments)
        commands.append(command)
        if len(command) > 1 and command[1] == "commit":
            raise release_deploy.ReleaseDeployError("backup cleanup failed")

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            return _verification_report()
        if "stability" in arguments:
            return _stability_report()
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run", run)
    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    monkeypatch.setattr(
        release_deploy,
        "read_live_release_state",
        lambda _site: {
            "phase": "complete",
            "transaction_committed": True,
            "commit_cleanup_completed": False,
        },
    )
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError, match="backup cleanup"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    assert not any(command[1] == "rollback" for command in commands), (
        "committed release was rolled back after cleanup failure"
    )
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["rollback"]["status"] == "SKIPPED_COMMITTED"


def test_rollback_cleanup_finishes_before_management_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    prepared = release_deploy.PreparedRelease(
        site_file=site,
        release_id="release-a",
        runtime_profile_version="profile-v1",
        agent_config_digest="a" * 64,
        profile_change_kind="UNCHANGED",
        profile_approval=None,
        state_dir=site.parent / "release-deploy/release-a",
        site_changed=False,
    )
    states = iter(
        (
            {
                "phase": "rolled-back",
                "rollback_cleanup_completed": False,
                "rollback_result": {"status": "PASSED"},
            },
            {
                "phase": "rolled-back",
                "rollback_cleanup_completed": True,
                "rollback_result": {"status": "PASSED"},
            },
        )
    )
    modes: list[str] = []
    monkeypatch.setattr(
        release_failure_recovery,
        "reconcile_rollback_management",
        lambda *_args, **_kwargs: site.parent,
    )

    result = release_failure_recovery.recover_release_failure(
        prepared,
        site_file=site,
        root=site.parent,
        environment={},
        failure_error="upgrade failed",
        failed_at="2026-09-03T00:00:00+00:00",
        deployment_succeeded=False,
        commit_started=False,
        automatic_rollback=True,
        run_release_mode=lambda _site, *, mode, **_kwargs: modes.append(mode),
        read_live_state=lambda _site: next(states),
        update_phase=lambda *_args, **_kwargs: None,
    )

    assert modes == ["rollback", "sync-state"], (
        "management alignment ran before rollback cleanup completed"
    )
    assert result is not None and result["status"] == "PASSED"


def test_rollback_cleanup_failure_preserves_pending_checkpoint(tmp_path: Path) -> None:
    prepared = SimpleNamespace(release_id="release-a", state_dir=tmp_path / "release-a")
    modes: list[str] = []

    def fail_cleanup(_site, *, mode, **_kwargs):
        modes.append(mode)
        raise release_deploy.ReleaseDeployError("cleanup still failed")

    result = release_failure_recovery.recover_release_failure(
        prepared,
        site_file=tmp_path / "site.yaml",
        root=tmp_path,
        environment={},
        failure_error="upgrade failed",
        failed_at="2026-09-03T00:00:00+00:00",
        deployment_succeeded=False,
        commit_started=False,
        automatic_rollback=True,
        run_release_mode=fail_cleanup,
        read_live_state=lambda _site: {
            "phase": "rolled-back",
            "rollback_cleanup_completed": False,
            "rollback_result": {"status": "PASSED"},
        },
        update_phase=lambda *_args, **_kwargs: None,
    )

    assert modes == ["rollback"]
    assert result is not None and result["status"] == "CLEANUP_PENDING"
    assert result["management_state_synced"] is False


def test_execute_release_persists_failure_before_interrupted_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    state_path = site.parent / "release-deploy/release-a/state.json"
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def fail_verify(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    observed: dict[str, Any] = {}

    def interrupt_rollback(_site_file, *, mode, **_kwargs):
        assert mode == "rollback"
        observed.update(json.loads(state_path.read_text(encoding="utf-8")))
        raise KeyboardInterrupt("deployment session ended")

    monkeypatch.setattr(release_deploy, "_run_json", fail_verify)
    monkeypatch.setattr(release_deploy, "_run_release_mode", interrupt_rollback)

    with pytest.raises(KeyboardInterrupt, match="session ended"):
        profile = tmp_path / "repo/config/profile.yaml"
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    assert observed["phase"] == "FAILED"
    assert "verify failed" in str(observed["error"])
    assert observed["rollback"]["status"] == "IN_PROGRESS"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state == observed


def test_release_summary_failure_does_not_fail_verified_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            return _verification_report()
        if "stability" in arguments:
            return _stability_report()
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        raise release_deploy.ReleaseDeployError("summary endpoint unavailable")

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    prepared = release_deploy.execute_release(
        site,
        run_checks=False,
        live_state={
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()
        },
    )

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED"
    assert state["verification"]["status"] == "PASSED"
    assert state["release_summary"]["status"] == "UNAVAILABLE"
    assert "summary endpoint unavailable" in state["completion_warnings"][0]


def test_profile_change_requires_approval_and_generates_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    profile = tmp_path / "repo/config/profile.yaml"
    initial = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest()
    )
    prepared_initial = release_deploy.prepare_site_release(site, profile_plan=initial)
    active_profile = Path(
        yaml.safe_load(site.read_text(encoding="utf-8"))["spec"]["runtimeProfile"][
            "source"
        ]
    )
    live_sha = hashlib.sha256(active_profile.read_bytes()).hexdigest()
    value = yaml.safe_load(profile.read_text(encoding="utf-8"))
    value["claims"] = [
        {"capability": "gpuReset", "mode": "OBSERVE", "owner": "gpu-fault-node-agent"}
    ]
    profile.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    plan = release_deploy.plan_runtime_profile(site, live_profile_sha256=live_sha)

    assert plan.approval_required is True, "Profile change did not require approval"
    assert plan.change_kind == "EXPANSIVE"
    assert plan.desired_version.startswith("regional-hyperpod-"), (
        "Profile version was not derived from the normalized policy digest"
    )
    with pytest.raises(release_deploy.ReleaseDeployError, match="approve-profile"):
        release_deploy.execute_release(
            site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
        )

    pending = json.loads(
        admin_profile_approval.profile_plan_path(site.parent).read_text(
            encoding="utf-8"
        )
    )
    assert pending["plan_sha256"] == release_deploy.profile_plan_sha256(pending)
    assert pending["site_identity"] == {
        "site_name": "test-site",
        "aws_region": REGION,
        "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
    }
    assert len(pending["site_identity_sha256"]) == 64
    approval = admin_profile_approval.approve_profile(
        site.parent,
        reference="CHG-12345",
        expected_plan_sha256=str(pending["plan_sha256"]),
    )
    assert approval["plan_sha256"] == pending["plan_sha256"]

    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            return _verification_report()
        if "stability" in arguments:
            return _stability_report()
        if "release-diff" in arguments:
            return _release_diff("FULL", ["runtime_profile"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    prepared = release_deploy.execute_release(
        site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
    )
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    assert document["spec"]["runtimeProfile"]["version"] == plan.desired_version
    assert prepared.profile_approval == "CHG-12345"
    assert prepared_initial.runtime_profile_version == "profile-v1"
    assert not admin_profile_approval.profile_plan_path(site.parent).exists(), (
        "successful release retained its active Profile plan"
    )
    assert not admin_profile_approval.profile_approval_path(site.parent).exists(), (
        "successful release retained its active Profile approval"
    )
    archive = admin_profile_approval.profile_approval_archive_path(
        site.parent, pending["plan_sha256"]
    )
    consumed = json.loads((archive / "consumed.json").read_text(encoding="utf-8"))
    assert consumed["status"] == "CONSUMED"
    assert consumed["relation"] == "EXACT"


def test_profile_approval_survives_failed_release_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    profile = tmp_path / "repo/config/profile.yaml"
    initial = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest()
    )
    release_deploy.prepare_site_release(site, profile_plan=initial)
    active_profile = Path(
        yaml.safe_load(site.read_text(encoding="utf-8"))["spec"]["runtimeProfile"][
            "source"
        ]
    )
    live_sha = hashlib.sha256(active_profile.read_bytes()).hexdigest()
    value = yaml.safe_load(profile.read_text(encoding="utf-8"))
    value["claims"] = [
        {"capability": "gpuReset", "mode": "OBSERVE", "owner": "gpu-fault-node-agent"}
    ]
    profile.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    with pytest.raises(release_deploy.ReleaseDeployError, match="approve-profile"):
        release_deploy.execute_release(
            site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
        )
    pending = json.loads(
        admin_profile_approval.profile_plan_path(site.parent).read_text()
    )
    admin_profile_approval.approve_profile(
        site.parent,
        reference="CHG-12345",
        expected_plan_sha256=str(pending["plan_sha256"]),
    )
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def fail_verify(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("FULL", ["runtime_profile"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", fail_verify)
    with pytest.raises(release_deploy.ReleaseDeployError, match="verify failed"):
        release_deploy.execute_release(
            site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
        )

    assert admin_profile_approval.profile_approval_path(site.parent).is_file(), (
        "failed release discarded the resumable Profile approval"
    )
    assert admin_profile_approval.profile_plan_path(site.parent).is_file(), (
        "failed release discarded the approved Profile plan"
    )
    failed_state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert (
        failed_state["profile_change"]["approval_plan_sha256"]
        == (pending["plan_sha256"])
    )
    assert failed_state["profile_change"]["approval_relation"] == "EXACT"

    def succeed(arguments, **_kwargs):
        if "verify" in arguments:
            return _verification_report()
        if "stability" in arguments:
            return _stability_report()
        if "release-diff" in arguments:
            return _release_diff("FULL", ["runtime_profile"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", succeed)
    prepared = release_deploy.execute_release(
        site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
    )

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED"
    assert state["profile_change"]["approval_plan_sha256"] == pending["plan_sha256"]
    assert state["profile_change"]["approval_relation"] == "PREPARED_RESUME"
    assert state["profile_approval_audit"]["relation"] == "PREPARED_RESUME"
    archive = admin_profile_approval.profile_approval_archive_path(
        site.parent, str(pending["plan_sha256"])
    )
    consumed = json.loads((archive / "consumed.json").read_text())
    assert consumed["relation"] == "PREPARED_RESUME"


def test_verify_and_stability_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cli verify` and `rollout stability` run at the same time.

    Both are read-only, neither reads anything the other writes, and on
    production they cost 44 s and 128 s -- the stability window being almost
    entirely sleep. Run serially that is nearly three minutes of the release
    spent waiting twice. Each fake below refuses to return until it has seen the
    other start, so a serial driver cannot get past this case.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    site = _site(tmp_path, monkeypatch)
    verify_started = threading.Event()
    stability_started = threading.Event()
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_json(arguments, **_kwargs):
        command = list(arguments)
        if "verify" in command:
            verify_started.set()
            assert stability_started.wait(timeout=10), (
                "the stability window had not started while verify was running"
            )
            return _verification_report()
        if "stability" in command:
            stability_started.set()
            assert verify_started.wait(timeout=10), (
                "verify had not started while the stability window was running"
            )
            return _stability_report()
        if "release-diff" in command:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    prepared = release_deploy.execute_release(
        site,
        run_checks=False,
        live_state={
            "release_id": "release-a",
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
    )

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED", "the release must reach COMPLETED phase"
    assert state["verification"]["status"] == "PASSED", "verification must pass"
    assert state["stability"]["status"] == "PASSED", "stability must pass"
    assert (prepared.state_dir / release_deploy.VERIFICATION_REPORT).is_file(), (
        "verification report must be written"
    )
    assert (prepared.state_dir / release_deploy.STABILITY_REPORT).is_file(), (
        "stability report must be written"
    )


@pytest.mark.parametrize("failing", ["verify", "stability"])
def test_a_concurrent_phase_failure_rolls_back_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    """Two phases in flight, one rollback.

    Running them together must not turn one failure into two recoveries, and the
    surviving phase must not be able to drive the release forward past a failure
    in the other.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    site = _site(tmp_path, monkeypatch)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        release_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)),
    )

    def run_json(arguments, **_kwargs):
        command = list(arguments)
        if "verify" in command:
            if failing == "verify":
                raise release_deploy.ReleaseDeployError("verify failed")
            return _verification_report()
        if "stability" in command:
            if failing == "stability":
                return {"mode": "stability", "healthy": False, "window_seconds": 120}
            return _stability_report()
        if "release-diff" in command:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "release_id": "release-a",
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest(),
            },
        )

    assert [command[1] for command in commands].count("rollback") == 1
    assert not any(command[1] == "commit" for command in commands), (
        "a failed phase let the release commit"
    )
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["phase"] == "FAILED"
    assert state["rollback"]["status"] == "PASSED"


def test_a_stability_failure_still_records_the_verification_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify produced a result, so the failed release keeps it.

    Serially the verification report was always on disk by the time the stability
    window ran, so a stability failure left an operator able to see that the fleet
    had verified. Running the two phases together must not cost that: the report
    the release paid 44 s for is written, and the VERIFIED phase recorded, before
    the stability failure is re-raised.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_json(arguments, **_kwargs):
        command = list(arguments)
        if "verify" in command:
            return _verification_report()
        if "stability" in command:
            return {"mode": "stability", "healthy": False, "window_seconds": 120}
        if "release-diff" in command:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError, match="stability"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "release_id": "release-a",
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest(),
            },
        )

    state_dir = site.parent / "release-deploy/release-a"
    assert (state_dir / release_deploy.VERIFICATION_REPORT).is_file(), (
        "the verification report the release produced was thrown away"
    )
    state = json.loads((state_dir / "state.json").read_text())
    assert state["verification"]["status"] == "PASSED", (
        "verification must pass before stability fails"
    )
    assert state["phase"] == "FAILED", (
        "the release must fail after stability check fails"
    )
    assert not (state_dir / release_deploy.STABILITY_REPORT).exists(), (
        "a failed stability window must not leave a stability report"
    )


def test_recovery_names_the_accepted_schema_change_instead_of_the_site_policy(
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(release_id="release-a", state_dir=tmp_path)

    result = release_failure_recovery.recover_release_failure(
        prepared,
        site_file=tmp_path / "site.yaml",
        root=tmp_path,
        environment={release_failure_recovery.ACCEPT_SCHEMA_CHANGE_ENV: "snapshot"},
        failure_error="RuntimeError: verify failed",
        failed_at="2026-09-07T10:00:00+00:00",
        deployment_succeeded=True,
        commit_started=False,
        automatic_rollback=True,
        deployment={"next_deploy": {"changed": ["database_schema"]}},
        run_release_mode=lambda *_args, **_kwargs: None,
        read_live_state=lambda _site: {"phase": "failed"},
        update_phase=lambda *_args, **_kwargs: None,
    )

    assert result is not None
    assert result["status"] == "SKIPPED_POLICY"
    assert "accept-schema-change" in result["reason"]


def test_execute_release_runs_an_accepted_schema_change_fail_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``autoRollback: true`` in the site, the acceptance in the environment: a
    verify failure records SKIPPED_POLICY with the acceptance as the reason and
    never invokes the rollback the engine would refuse."""

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setenv(release_failure_recovery.ACCEPT_SCHEMA_CHANGE_ENV, "snapshot")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        release_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)),
    )

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("FULL", ["database_schema", "control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError, match="verify failed"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    assert not any(command[1] == "rollback" for command in commands), (
        "an accepted schema change invoked the rollback the engine refuses"
    )
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["rollback"]["status"] == "SKIPPED_POLICY"
    assert "accept-schema-change" in state["rollback"]["reason"]


def test_the_driver_and_the_engine_agree_on_the_refusal_exit_code() -> None:
    from gpu_fault_release import regional_release_store_preflight as STORE

    assert (
        release_failure_recovery.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
        == STORE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
    )
    assert release_failure_recovery.refused_inflight_installs(
        release_deploy.ReleaseDeployError("command failed (3): rollout.sh")
    ), "the engine's refusal exit code is what the driver classifies on"
    assert not release_failure_recovery.refused_inflight_installs(
        release_deploy.ReleaseDeployError("command failed (2): rollout.sh")
    ), "an ordinary engine error stays a rollback failure"


def test_an_engine_refusal_is_recorded_as_refused_not_as_a_failed_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine refused before touching anything (an install step is in
    flight), so the driver must not say the rollback failed: nothing was rolled
    back, and the operator's next step is to wait, not to repair a half-restore.
    The automatic rollback is also told it is automatic, so a control plane that
    cannot answer the store read does not wedge it."""

    site = _site(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    def run(arguments, **_kwargs):
        commands.append(list(arguments))
        if "rollback" in arguments:
            raise release_deploy.ReleaseDeployError(
                f"command failed (3): {arguments[0]}"
            )

    monkeypatch.setattr(release_deploy, "_run", run)

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError) as failure:
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    assert "verify failed" in str(failure.value)
    assert "rollback also failed" not in str(failure.value)
    rollback = next(command for command in commands if command[1] == "rollback")
    assert "--automatic" in rollback
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["phase"] == "FAILED"
    assert state["rollback"]["status"] == "REFUSED_INFLIGHT_INSTALLS"
    assert "nothing was rolled back" in state["rollback"]["reason"]
    assert "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS" in state["rollback"]["reason"]
    verdict = state["rollback"]["inflight_installs"]
    assert verdict["verdict"] == "refused"
    assert verdict["checked"] is True, "a refusal is the check having run"
    assert "rollback log" in verdict["reason"], "the steps live in the engine log"


def _verify_failure_after_deploy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: Path
) -> dict[str, Any]:
    """Run a release whose verify fails and whose engine rollback passes; return
    the driver's persisted state."""

    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"
    with pytest.raises(release_deploy.ReleaseDeployError, match="verify failed"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )
    return json.loads((site.parent / "release-deploy/release-a/state.json").read_text())


def test_the_rollback_record_carries_the_engines_inflight_install_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An automatic rollback that proceeded ``inflight-installs-unchecked`` (no
    Running control-plane Pod could run the probe) used to leave one stderr
    line behind. The engine now writes the verdict into its transaction state
    before ``rollback-started``; the driver copies it into its own rollback
    record, so the release history says whether the rollback was checked and
    on what (fix round 3, MEDIUM-3)."""

    site = _site(tmp_path, monkeypatch)
    engine_verdict = {
        "checked": False,
        "verdict": "unchecked",
        "reason": "automatic rollback: no Running control-plane Pod could run the probe",
        "steps": [],
    }
    monkeypatch.setattr(
        release_deploy,
        "read_live_release_state",
        lambda _site: {
            "phase": "rolled-back",
            "rollback_result": {"status": "PASSED"},
            "previous": {},
            "inflight_installs": dict(engine_verdict),
        },
    )

    state = _verify_failure_after_deploy(monkeypatch, tmp_path, site)

    assert state["rollback"]["status"] == "PASSED"
    assert state["rollback"]["inflight_installs"] == engine_verdict


def test_a_rollback_record_says_when_the_engine_left_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine that predates the verdict leaves no ``inflight_installs`` in
    its state; the record must say so rather than imply the check passed. It
    must not blame a rollback re-entered after ``rollback-cpu-restored``: that
    re-entry skips the gate but the FIRST attempt's verdict stays in the state
    (with its ``checked_at``), so the state is not empty there (fix round 4,
    LOW-4)."""

    site = _site(tmp_path, monkeypatch)

    state = _verify_failure_after_deploy(monkeypatch, tmp_path, site)

    verdict = state["rollback"]["inflight_installs"]
    assert verdict["checked"] is False
    assert verdict["verdict"] == "unrecorded"
    assert "inflight_installs" in verdict["reason"], (
        "the reason names the missing engine-state key"
    )
    assert "predates" in verdict["reason"], "the one cause that leaves no verdict"
    assert "re-enter" not in verdict["reason"], (
        "a re-entry keeps the first attempt's verdict; it is not unrecorded"
    )
