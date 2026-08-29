from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    "rollout_regional_release_architecture",
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py",
)
FLEET_MODULE = lazy_script_module(
    "regional_release_fleet_rollout_architecture",
    ROOT / "deploy/control-plane/regional/regional_release_fleet_rollout.py",
)
VALIDATION_MODULE = lazy_script_module(
    "regional_release_validation_architecture",
    ROOT / "deploy/control-plane/regional/regional_release_validation.py",
)


def test_node_runtime_rollout_uses_fleet_waves(tmp_path: Path, monkeypatch) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    deploy_calls = []
    wait_calls = []
    fleet_calls = []
    deployment_reads = iter(({"status": "IN_PROGRESS"}, {"status": "SUCCEEDED"}))
    waves = iter(({"node_ids": ["node-a", "node-b"]}, {"node_ids": ["node-c"]}))
    monkeypatch.setenv("GPU_FAULT_INSTALLER_MAX_UNAVAILABLE", "2")
    monkeypatch.setattr(
        release, "_target_node_names", lambda _target: ("node-a", "node-b", "node-c")
    )
    monkeypatch.setattr(
        release,
        "_deploy_reconciler",
        lambda *_args, **kwargs: (
            deploy_calls.append(kwargs.get("allowed_node_names"))
            or ("b" * 64, "t" * 64)
        ),
    )
    monkeypatch.setattr(
        release,
        "_wait_agents",
        lambda *_args, **kwargs: wait_calls.append(kwargs.get("node_names")),
    )

    def fleet_command(operation, payload):
        fleet_calls.append((operation, payload))
        if operation == "create":
            return {"status": "PLANNED"}
        if operation == "next-wave":
            return next(waves)
        return next(deployment_reads)

    monkeypatch.setattr(release, "_fleet_command", fleet_command)

    identity = FLEET_MODULE.roll_node_runtime(
        release,
        target,
        phase="upgrade",
        wheel_cm=release.executor_wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.node_wheel_sha,
        config_digest=release.config.agent_config_digest,
    )

    assert identity == ("b" * 64, "t" * 64)
    assert deploy_calls == [(), ("node-a", "node-b"), ("node-c",), None]
    assert wait_calls == [("node-a", "node-b"), ("node-c",), None]
    request = fleet_calls[0][1]["request"]
    assert request["max_unavailable"] == 2
    assert request["desired_bundle_sha256"] == "b" * 64
    assert request["desired_template_sha256"] == "t" * 64


def test_secret_backup_returns_only_reference(tmp_path: Path, monkeypatch) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    calls = []
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
    )
    monkeypatch.setattr(
        release,
        "_get_json",
        lambda _args: {"type": "Opaque", "data": {"sensitive-key": "encoded-value"}},
    )
    monkeypatch.setattr(
        release.runner, "run", lambda args, **kwargs: calls.append((args, kwargs)) or ""
    )

    reference = FLEET_MODULE.backup_secret(
        release,
        ["kubectl"],
        source="gpu-fault-email",
        backup="gpu-fault-email-rollback-release-a",
        required=True,
    )

    assert reference == "gpu-fault-email-rollback-release-a"
    assert calls[0][1]["sensitive"] is True
    assert "encoded-value" in calls[0][1]["input_text"]


def test_profile_finalize_rejects_old_profile_activity(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="profile-v2")
    )
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    responses = iter(
        ("cpu-pod", json.dumps({"workflow_count": 1, "workload_count": 0}))
    )
    monkeypatch.setattr(
        release.runner, "run", lambda *_args, **_kwargs: next(responses)
    )

    with pytest.raises(MODULE.ReleaseError, match="old-profile activity"):
        VALIDATION_MODULE.ensure_profile_transition_safe(release, "profile-v1")


def test_rollback_rejects_non_transactional_endpoint_change(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    release.state = {
        "release_diff": {"kind": "DATA_PLANE_COMPATIBLE", "changed": ["endpoint"]}
    }

    with pytest.raises(MODULE.ReleaseError, match="not transactional"):
        release.rollback(state={"metadata": {}, "cpu_wheel": "old-wheel"})


def test_profile_finalize_allows_drained_old_profile(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="profile-v2")
    )
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    responses = iter(
        ("cpu-pod", json.dumps({"workflow_count": 0, "workload_count": 0}))
    )
    monkeypatch.setattr(
        release.runner, "run", lambda *_args, **_kwargs: next(responses)
    )

    VALIDATION_MODULE.ensure_profile_transition_safe(release, "profile-v1")


def test_stability_window_accepts_steady_samples(tmp_path: Path, monkeypatch) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    snapshot = {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {"count": 0, "alerts": []},
    }
    snapshots = iter((snapshot, snapshot))
    times = iter((0.0, 0.0, 0.0, 120.0))
    monkeypatch.setattr(release, "_stability_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    report = release.validate_stability_window(window_seconds=120, sample_seconds=120)

    assert report["healthy"] is True
    assert report["sample_count"] == 2


def test_stability_window_rejects_new_restarts(tmp_path: Path, monkeypatch) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    baseline = {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {"count": 0, "alerts": []},
    }
    restarted = {**baseline, "restarts": {"cpu/pod/container": 1}}
    snapshots = iter((baseline, restarted))
    times = iter((0.0, 0.0, 0.0))
    monkeypatch.setattr(release, "_stability_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    with pytest.raises(MODULE.ReleaseError, match="observed a restart"):
        release.validate_stability_window(window_seconds=120, sample_seconds=120)
