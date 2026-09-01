from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ADMIN = lazy_script_module(
    "regional_admin_commands_test",
    ROOT / "deploy/control-plane/regional/regional_admin_commands.py",
)


def _admin_module():
    return ADMIN._load()


def test_full_status_keeps_health_when_release_summary_is_missing(monkeypatch) -> None:
    module = _admin_module()
    release = SimpleNamespace(config=SimpleNamespace(site_name="test-site"))
    monkeypatch.setattr(
        module,
        "build_health_report",
        lambda _release, *, mode: {
            "mode": mode,
            "healthy": False,
            "summary": {"FAIL": 1},
            "checks": [],
        },
    )

    def broken_summary(_release):
        raise RuntimeError("deployment is missing")

    monkeypatch.setattr(module, "build_release_status", broken_summary)

    report = module.build_full_status(release)

    assert report["healthy"] is False
    assert report["release_status_error"] == "deployment is missing"
    assert report["health"]["mode"] == "status"


def test_release_summary_does_not_repeat_health_checks(monkeypatch) -> None:
    module = _admin_module()
    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site"), _load_state=lambda: {}
    )
    monkeypatch.setattr(
        module,
        "build_health_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("release summary repeated health checks")
        ),
    )
    monkeypatch.setattr(
        module, "build_release_status", lambda _release: {"site_name": "test-site"}
    )
    monkeypatch.setattr(
        module,
        "classify_release",
        lambda _release, _state: SimpleNamespace(
            as_dict=lambda: {"kind": "NOOP", "changed": []}
        ),
    )

    report = module.build_release_summary(release)

    assert report["mode"] == "release-summary"
    assert report["next_deploy"] == {"kind": "NOOP", "changed": []}


def test_failed_release_diff_is_reused_for_resume() -> None:
    diff = ADMIN.stored_release_diff(
        {
            "phase": "failed",
            "release_diff": {
                "kind": "DATA_PLANE_COMPATIBLE",
                "changed": ["executor_wheel"],
            },
        }
    )

    assert diff is not None
    assert diff.as_dict() == {
        "kind": "DATA_PLANE_COMPATIBLE",
        "changed": ["executor_wheel"],
    }


def test_retry_diff_restores_physical_artifact_changes(monkeypatch) -> None:
    module = _admin_module()
    release = SimpleNamespace(
        wheel_cm="new-control",
        executor_wheel_cm="new-executor",
        bundle_cm="new-bundle",
        node_wheel_sha="n" * 64,
        executor_wheel_sha="e" * 64,
        config=SimpleNamespace(clusters=(SimpleNamespace(cluster_id="gpu-a"),)),
    )
    state = {
        "phase": "failed",
        "release_diff": {"kind": "DATA_PLANE_COMPATIBLE", "changed": ["node_bundle"]},
        "previous": {
            "cpu_wheel": "old-control",
            "metadata": {
                "required-agent-artifact-sha256": "a" * 64,
                "required-regional-executor-artifact-sha256": "b" * 64,
            },
            "clusters": {
                "gpu-a": {
                    "wheel": "old-executor",
                    "reconciler_wheel": "old-executor",
                    "bundle": "old-bundle",
                }
            },
        },
    }
    monkeypatch.setattr(
        module,
        "classify_release",
        lambda _release, _state: module.diff_from_changed(()),
    )

    diff = module.retry_release_diff(release, state)

    assert diff.kind is module.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    assert diff.changed == {
        "control_plane_wheel",
        "executor_wheel",
        "node_runtime_wheel",
        "node_bundle",
    }


@pytest.mark.parametrize(
    ("phase", "expected_resume"),
    (
        ("failed", True),
        ("registry-staged", True),
        ("data-plane-progress", True),
        ("rolled-back", False),
    ),
)
def test_deploy_only_resumes_an_unrolled_back_release(
    monkeypatch, phase: str, expected_resume: bool
) -> None:
    module = _admin_module()
    expected_diff = module.diff_from_changed({"control_plane_wheel"})
    calls: list[dict[str, object]] = []
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=("gpu-a",)),
        _cpu=lambda *args: ["kubectl", *args],
        _load_state=lambda: {"phase": phase, "release_id": "previous-candidate"},
        upgrade=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr(
        module, "retry_release_diff", lambda _release, _state: expected_diff
    )

    module.run_deploy(release)

    assert calls == [{"resume": expected_resume, "diff": expected_diff}]


def test_explicit_resume_reuses_the_checkpoint_retry_diff(monkeypatch) -> None:
    module = _admin_module()
    expected_diff = module.diff_from_changed({"control_plane_wheel"})
    calls: list[dict[str, object]] = []
    state = {"phase": "registry-staged", "release_id": "candidate-release"}
    release = SimpleNamespace(
        _load_state=lambda: state, upgrade=lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        module, "retry_release_diff", lambda _release, value: expected_diff
    )

    module.run_resume(release)

    assert calls == [{"resume": True, "diff": expected_diff}]


def test_explicit_resume_rejects_a_terminal_release() -> None:
    module = _admin_module()
    release = SimpleNamespace(_load_state=lambda: {"phase": "complete"})

    with pytest.raises(module.ReleaseError, match="incomplete upgrade transaction"):
        module.run_resume(release)


@pytest.mark.parametrize(
    ("phase", "expected_resume"),
    (
        ("failed", True),
        ("registry-staged", True),
        ("data-plane-progress", True),
        ("rolled-back", False),
    ),
)
def test_release_summary_reports_retry_transaction_mode(
    monkeypatch, phase: str, expected_resume: bool
) -> None:
    module = _admin_module()
    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site"),
        _load_state=lambda: {"phase": phase},
    )
    monkeypatch.setattr(
        module, "build_release_status", lambda _release: {"site_name": "test-site"}
    )
    monkeypatch.setattr(
        module,
        "retry_release_diff",
        lambda _release, _state: module.diff_from_changed({"control_plane_wheel"}),
    )

    report = module.build_release_summary(release)

    assert report["next_deploy"]["resume"] is expected_resume


@pytest.mark.parametrize(
    ("state_exists", "phase"),
    [
        (False, None),
        (True, "bootstrap-cleaned"),
        (True, "bootstrap-data-plane-progress"),
    ],
)
def test_deploy_rejects_empty_cluster_set_during_bootstrap(
    monkeypatch, state_exists: bool, phase: str | None
) -> None:
    module = _admin_module()
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=()),
        _cpu=lambda *args: ["kubectl", *args],
        _load_state=lambda: {"phase": phase},
        bootstrap=lambda: calls.append("bootstrap"),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0 if state_exists else 1
        ),
    )

    with pytest.raises(
        module.ReleaseError,
        match="initial regional bootstrap requires at least one GPU cluster",
    ):
        module.run_deploy(release)

    assert calls == []


def test_deploy_allows_empty_cluster_set_after_completed_state(monkeypatch) -> None:
    module = _admin_module()
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=()),
        _cpu=lambda *args: ["kubectl", *args],
        _load_state=lambda: {"phase": "complete"},
        noop=lambda _diff: calls.append("noop"),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr(
        module,
        "classify_release",
        lambda _release, _state: module.diff_from_changed(()),
    )

    module.run_deploy(release)

    assert calls == ["noop"]
