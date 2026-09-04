from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ADMIN = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_admin_commands.py"
)
CHECKS = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_admin_checks.py"
)
STATE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_state.py"
)


def _admin_module():
    return ADMIN.load()


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


def test_full_status_reads_the_cluster_once_for_both_halves(monkeypatch) -> None:
    """The health report and the release summary share one snapshot.

    `status` is the command an administrator runs while watching something go
    wrong, so its cost matters. Both halves read the same Deployments and the
    same ConfigMaps; the health report opens a read snapshot around its checks,
    and until `build_full_status` opened one first, everything the summary read
    afterwards was read against a cache that had already been torn down.
    """

    module = _admin_module()
    calls: list[tuple[str, ...]] = []

    class Runner:
        @staticmethod
        def run(arguments, **_kwargs):
            calls.append(tuple(arguments))
            return json.dumps({"data": {"state.json": "{}"}})

    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site"), runner=Runner()
    )
    release._get_json = lambda args: STATE.get_json(release, args)
    release._read_snapshot = lambda: STATE.read_snapshot(release)
    release._load_state = lambda: {}
    command = ["cpu", "-n", "gpu-fault-system", "get", "configmap", "release-metadata"]

    # Stand in for the two builders, each reading what the other reads. The
    # health report opens its own snapshot the way the real one does, which is
    # the nesting this case is about.
    def health(_release, *, mode):
        with CHECKS._read_snapshot(release):
            release._get_json(command)
        return {"mode": mode, "healthy": True, "summary": {"FAIL": 0}, "checks": []}

    monkeypatch.setattr(module, "build_health_report", health)
    monkeypatch.setattr(
        module,
        "build_release_status",
        lambda _release: {"release_metadata": release._get_json(command)},
    )
    monkeypatch.setattr(
        module,
        "classify_release",
        lambda _release, _state: SimpleNamespace(as_dict=lambda: {"kind": "NOOP"}),
    )

    report = module.build_full_status(release)

    assert report["healthy"] is True
    assert len(calls) == 1


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
        ("partial-convergence", True),
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
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: {"phase": phase, "release_id": "previous-candidate"},
        upgrade=lambda **kwargs: calls.append(kwargs),
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


@pytest.mark.parametrize(
    "phase", ("rollback-data-progress", "rollback-verifying", "rollback-failed")
)
def test_deploy_and_resume_continue_the_existing_rollback(
    monkeypatch, phase: str
) -> None:
    module = _admin_module()
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=("gpu-a",)),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: {"phase": phase, "release_id": "candidate-release"},
        rollback=lambda: calls.append("rollback"),
    )

    with pytest.raises(module.ReleaseError, match="rollback recovery completed"):
        module.run_deploy(release)
    module.run_resume(release)

    assert calls == ["rollback", "rollback"]


def test_deploy_reports_completed_rollback_cleanup_as_recovery(monkeypatch) -> None:
    module = _admin_module()
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=("gpu-a",)),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: {
            "phase": "rolled-back",
            "rollback_cleanup_completed": False,
            "rollback_result": {"status": "PASSED"},
        },
        rollback=lambda: calls.append("rollback"),
    )

    with pytest.raises(module.ReleaseError, match="rollback recovery completed"):
        module.run_deploy(release)

    assert calls == ["rollback"]


def test_explicit_resume_rejects_a_terminal_release() -> None:
    module = _admin_module()
    release = SimpleNamespace(_load_state=lambda: {"phase": "complete"})

    with pytest.raises(
        module.ReleaseError, match="incomplete upgrade or rollback transaction"
    ):
        module.run_resume(release)


@pytest.mark.parametrize(
    ("phase", "expected_resume"),
    (
        ("failed", True),
        ("partial-convergence", True),
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


def test_release_summary_reports_rollback_as_the_next_action(monkeypatch) -> None:
    module = _admin_module()
    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site"),
        _load_state=lambda: {"phase": "rollback-data-progress"},
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

    assert report["next_deploy"]["action"] == "rollback"
    assert report["next_deploy"]["resume"] is True


@pytest.mark.parametrize(
    ("state_exists", "phase"),
    [
        (False, None),
        (True, "bootstrap-cleaned"),
        (True, "bootstrap-cleanup-progress"),
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
        runner=SimpleNamespace(probe=lambda _args: state_exists),
        _load_state=lambda: {"phase": phase},
        bootstrap=lambda: calls.append("bootstrap"),
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
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: {"phase": "complete"},
        noop=lambda _diff: calls.append("noop"),
    )
    monkeypatch.setattr(
        module,
        "classify_release",
        lambda _release, _state: module.diff_from_changed(()),
    )

    module.run_deploy(release)

    assert calls == ["noop"]
