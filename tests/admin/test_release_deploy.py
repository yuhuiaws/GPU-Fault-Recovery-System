from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import release_deploy

REGION = "us-east-1"


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
        site,
        live_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        approval=None,
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
        site,
        live_profile_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        approval=None,
    )
    repeated = release_deploy.prepare_site_release(site, profile_plan=repeated_plan)
    assert repeated.site_changed is False, (
        "identical release preparation was not idempotent"
    )


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
        return _verification_report() if "verify" in command else _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)

    profile = tmp_path / "repo/config/profile.yaml"
    prepared = release_deploy.execute_release(
        site,
        live_state={
            "release_id": "release-a",
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
    )

    assert calls[0][1][-1] == "check"
    assert calls[1][1][1] == "release-summary"
    assert calls[2][1][3] == "verify"
    assert calls[3][1][1] == "commit"
    assert len(calls) == 4
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
    summaries = iter(
        (
            _release_summary("CONTROL_PLANE_ONLY", ["control_plane_wheel"]),
            _release_summary(),
        )
    )
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
        return next(summaries)

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

    assert calls[0][1][1] == "release-summary"
    assert calls[1][1][3] == "deploy"
    assert calls[2][1][3] == "verify"
    assert calls[3][1][1] == "stability"
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


def test_execute_release_rolls_back_after_stability_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    commands = []
    summaries = iter((_release_summary("CONTROL_PLANE_ONLY", ["control_plane_wheel"]),))
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
        return next(summaries)

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
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["rollback"]["status"] == "PASSED"


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
    monkeypatch.delenv(release_deploy.PROFILE_APPROVAL_ENV, raising=False)
    site = _site(tmp_path, monkeypatch)
    profile = tmp_path / "repo/config/profile.yaml"
    initial = release_deploy.plan_runtime_profile(
        site,
        live_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        approval=None,
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

    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=live_sha, approval=None
    )

    assert plan.approval_required is True, "Profile change did not require approval"
    assert plan.change_kind == "EXPANSIVE"
    assert plan.desired_version.startswith("regional-hyperpod-"), (
        "Profile version was not derived from the normalized policy digest"
    )
    with pytest.raises(release_deploy.ReleaseDeployError, match="PROFILE_APPROVAL"):
        release_deploy.execute_release(
            site, run_checks=False, live_state={"runtime_profile_sha256": live_sha}
        )

    approved = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=live_sha, approval="CHG-12345"
    )
    prepared = release_deploy.prepare_site_release(site, profile_plan=approved)
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    assert document["spec"]["runtimeProfile"]["version"] == (approved.desired_version)
    assert prepared.profile_approval == "CHG-12345"
    assert prepared_initial.runtime_profile_version == "profile-v1"
