"""A CONTROL_PLANE_ONLY release may close its stability window at 60 s.

The configured window (120..300 s) waits out what a data-plane restart sets in
motion; a release that rolled only the control-plane role ConfigMaps/replicas
restarted no data plane, so once two clean 30 s samples have been seen and every
control-plane Deployment reports converged there is nothing left to wait for.
Anything else -- another release kind, a critical alert in any sample, a
Deployment still rolling -- keeps the full configured window or fails as before.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release import rollout as MODULE
from scripts import release_deploy_evidence as EVIDENCE
from tests.regional._release_orchestrator_support import config_file

CONTROL_PLANE_ONLY = "CONTROL_PLANE_ONLY"


def _snapshot(*, restarts: int = 0, alerts: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "restarts": {"cpu/pod/container": restarts},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {
            "count": len(alerts),
            "alerts": [{"alertname": name, "severity": "critical"} for name in alerts],
        },
    }


def _deployment(name: str, *, ready: int = 2, generation: int = 3) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "generation": generation},
        "spec": {"replicas": 2},
        "status": {
            "observedGeneration": 3,
            "replicas": 2,
            "updatedReplicas": 2,
            "readyReplicas": ready,
        },
    }


def _deployment_listing(**overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [
            overrides.get(name, _deployment(name)) for name in inventory.CPU_DEPLOYMENTS
        ]
    }


def _release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    samples: list[dict[str, Any]],
    *,
    deployments: dict[str, Any] | None = None,
):
    """A release whose window reads `samples` (the last one repeating), whose
    clock advances on sleep, and whose Deployment listing is `deployments`."""

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.delenv("GPU_FAULT_RELEASE_QUEUE_GROWTH_MIN_DEPTH", raising=False)
    monkeypatch.delenv("GPU_FAULT_RELEASE_STABILITY_GRACE_ALERTS", raising=False)
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    pending = list(samples)

    def next_sample() -> dict[str, Any]:
        return pending.pop(0) if len(pending) > 1 else pending[0]

    monkeypatch.setattr(release, "_stability_snapshot", next_sample)
    listing = deployments if deployments is not None else _deployment_listing()
    reads: list[list[str]] = []

    def get_json(args: list[str]) -> dict[str, Any]:
        reads.append(list(args))
        return listing

    monkeypatch.setattr(release, "_get_json", get_json)
    clock = {"now": 0.0}
    monkeypatch.setattr(VALIDATION.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        VALIDATION.time,
        "sleep",
        lambda seconds: clock.update(now=clock["now"] + seconds),
    )
    return release, clock, reads


def test_control_plane_only_window_closes_after_two_clean_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, clock, reads = _release(tmp_path, monkeypatch, [_snapshot()])

    report = release.validate_stability_window(
        window_seconds=120, sample_seconds=30, release_kind=CONTROL_PLANE_ONLY
    )

    assert clock["now"] == 60.0, "the window closed at the 60 s floor"
    assert report["healthy"] is True, "an early close is a healthy report"
    assert report["early_exit"] is True, "the report says the window closed early"
    assert report["window_seconds"] == 60, "the applied window is the floor"
    assert report["configured_window_seconds"] == 120, "the configured value is kept"
    assert report["release_kind"] == CONTROL_PLANE_ONLY, "the kind is recorded"
    assert report["sample_count"] == 3, "baseline plus two clean samples"
    assert "2 clean samples" in report["early_exit_reason"], (
        "the reason names the clean samples"
    )
    assert len(reads) == 1, "the Deployment listing is read once, at the floor"
    assert reads[0][-2:] == ["get", "deployment"], "it is one `get deployment`"


def test_full_release_keeps_the_configured_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, clock, reads = _release(tmp_path, monkeypatch, [_snapshot()])

    report = release.validate_stability_window(
        window_seconds=120, sample_seconds=30, release_kind="FULL"
    )

    assert clock["now"] == 120.0, "a FULL release runs the whole window"
    assert report["early_exit"] is False, "no early exit for a FULL release"
    assert report["window_seconds"] == 120, "the configured window applied"
    assert report["sample_count"] == 5, "baseline plus four samples"
    assert "not CONTROL_PLANE_ONLY" in report["early_exit_reason"], (
        "the reason says why the full window applied"
    )
    assert reads == [], "no Deployment listing is read for a full-length window"


def test_unknown_release_kind_keeps_the_configured_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, clock, _reads = _release(tmp_path, monkeypatch, [_snapshot()])

    report = release.validate_stability_window(window_seconds=120, sample_seconds=30)

    assert clock["now"] == 120.0, "no kind means the full window"
    assert report["early_exit"] is False, "no kind never exits early"
    assert report["release_kind"] is None, "the absent kind is recorded as None"


def test_a_restart_still_fails_the_control_plane_only_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, _clock, _reads = _release(
        tmp_path, monkeypatch, [_snapshot(), _snapshot(), _snapshot(restarts=1)]
    )

    with pytest.raises(MODULE.ReleaseError, match="observed a restart"):
        release.validate_stability_window(
            window_seconds=120, sample_seconds=30, release_kind=CONTROL_PLANE_ONLY
        )


def test_a_graced_alert_in_a_sample_keeps_the_full_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alert the restart grace excuses is still an alert: no early close."""

    release, clock, reads = _release(
        tmp_path,
        monkeypatch,
        [_snapshot(), _snapshot(alerts=("GpuFaultCollectorSilent",)), _snapshot()],
    )
    release.state = {
        "cluster_attempts": {
            "gpu-a": {"state": "CONVERGED", "converged_at_epoch": time.time() - 30}
        }
    }

    report = release.validate_stability_window(
        window_seconds=120, sample_seconds=30, release_kind=CONTROL_PLANE_ONLY
    )

    assert report["healthy"] is True, "the graced alert does not fail the window"
    assert clock["now"] == 120.0, "the full window is kept after a non-clean sample"
    assert report["early_exit"] is False, "a graced alert forbids the early close"
    assert report["window_seconds"] == 120, "the configured window applied"
    assert "a sample had critical alerts" in report["early_exit_reason"], (
        "the reason names the non-clean observation"
    )
    assert reads == [], "the Deployments are never asked once a sample was not clean"


def test_unconverged_deployments_keep_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A control-plane Deployment still rolling holds the window open."""

    lagging = inventory.CPU_DEPLOYMENTS[0]
    release, clock, reads = _release(
        tmp_path,
        monkeypatch,
        [_snapshot()],
        deployments=_deployment_listing(**{lagging: _deployment(lagging, ready=1)}),
    )

    report = release.validate_stability_window(
        window_seconds=120, sample_seconds=30, release_kind=CONTROL_PLANE_ONLY
    )

    assert clock["now"] == 120.0, "the window ran its configured length"
    assert report["early_exit"] is False, "an unconverged Deployment forbids the close"
    assert lagging in report["early_exit_reason"], "the reason names the Deployment"
    assert len(reads) == 2, (
        "the Deployments are asked at 60 and 90 s; the deadline itself closes 120 s"
    )


def _fake_release(listing: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: ["kubectl", *args],
        _get_json=lambda _args: listing,
    )


def test_an_unobserved_generation_is_not_converged() -> None:
    listing = _deployment_listing()
    listing["items"][0] = _deployment(inventory.CPU_DEPLOYMENTS[0], generation=4)
    reason = VALIDATION.control_plane_deployments_converged(_fake_release(listing))
    assert reason is not None and "observedGeneration" in reason, (
        "a Deployment whose generation is not yet observed is not converged"
    )


def test_an_absent_control_plane_deployment_is_not_converged() -> None:
    reason = VALIDATION.control_plane_deployments_converged(
        _fake_release({"items": []})
    )
    assert reason is not None and "is absent" in reason, (
        "a missing control-plane Deployment is reported by name"
    )


def test_release_kind_comes_from_the_persisted_diff() -> None:
    persisted = SimpleNamespace(
        _load_state=lambda: {"release_diff": {"kind": CONTROL_PLANE_ONLY}}
    )
    assert VALIDATION.stability_release_kind(persisted) == CONTROL_PLANE_ONLY, (
        "the persisted release_diff.kind is what the window is told"
    )
    unreadable = SimpleNamespace(
        _load_state=lambda: (_ for _ in ()).throw(
            MODULE.ReleaseError("regional release state is missing")
        )
    )
    assert VALIDATION.stability_release_kind(unreadable) is None, (
        "no readable state means the full-length window"
    )
    without_diff = SimpleNamespace(_load_state=lambda: {"phase": "complete"})
    assert VALIDATION.stability_release_kind(without_diff) is None, (
        "a state without a diff means the full-length window"
    )


def test_deploy_evidence_accepts_the_control_plane_only_floor() -> None:
    early = {
        "mode": "stability",
        "healthy": True,
        "window_seconds": 60,
        "early_exit": True,
        "release_kind": CONTROL_PLANE_ONLY,
    }
    EVIDENCE.validate_stability_report(early)

    with pytest.raises(EVIDENCE.ReleaseDeployError, match="invalid window"):
        EVIDENCE.validate_stability_report({**early, "release_kind": "FULL"})
    with pytest.raises(EVIDENCE.ReleaseDeployError, match="invalid window"):
        EVIDENCE.validate_stability_report({**early, "early_exit": False})
    EVIDENCE.validate_stability_report(
        {"mode": "stability", "healthy": True, "window_seconds": 120}
    )
