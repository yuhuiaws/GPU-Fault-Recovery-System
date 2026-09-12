from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_release_state as STATE

ROOT = Path(__file__).resolve().parents[2]


def _admin_module():
    return ADMIN


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
        # The candidate node preflight is joined after the control-plane phases
        # have run, so it is a phase a crashed release can be sitting on with
        # mutations already applied. Resuming is what preserves them; a fresh
        # upgrade would discard `completed_phases` and repeat the lot.
        ("candidate-preflight-ready", True),
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
        # The same release as the recorded transaction: a `failed` phase only
        # resumes its own candidate (a different one is refused, see
        # tests/regional/test_release_supersede_failed_transaction.py).
        release_id="previous-candidate",
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=("gpu-a",)),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: {"phase": phase, "release_id": "previous-candidate"},
        pin_approved_manifest_plan=lambda _digest: None,
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
        _load_state=lambda: state,
        pin_approved_manifest_plan=lambda _digest: None,
        upgrade=lambda **kwargs: calls.append(kwargs),
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
        # The candidate node preflight is joined after the control-plane phases
        # have run, so it is a phase a crashed release can be sitting on with
        # mutations already applied. Resuming is what preserves them; a fresh
        # upgrade would discard `completed_phases` and repeat the lot.
        ("candidate-preflight-ready", True),
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


def _status_release(reads: list[str]) -> SimpleNamespace:
    def load_state():
        reads.append("state")
        return {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        }

    return SimpleNamespace(
        config=SimpleNamespace(site_name="test-site"), _load_state=load_state
    )


def _stub_summary(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module,
        "build_release_status",
        lambda _release: {
            "site_name": "test-site",
            "configured_release": {"release_id": "release-a"},
        },
    )
    monkeypatch.setattr(
        module,
        "classify_release",
        lambda _release, _state: SimpleNamespace(
            as_dict=lambda: {"kind": "NOOP", "changed": []}
        ),
    )


def test_status_reads_the_release_state_once(monkeypatch) -> None:
    """A pre-read state serves the health baseline and the summary together.

    ``status`` used to read the release-state ConfigMap for the rolled-back
    baseline, again for the summary, and the admin CLI once more before either.
    The engine now reads it once and hands it down.
    """

    module = _admin_module()
    reads: list[str] = []
    release = _status_release(reads)
    _stub_summary(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "build_quick_health_report",
        lambda _release: {
            "mode": "status",
            "healthy": True,
            "summary": {"PASS": 2, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [],
            "scope": "quick",
        },
    )
    monkeypatch.setattr(
        module,
        "build_health_report",
        lambda _release, *, mode: {
            "mode": mode,
            "healthy": True,
            "summary": {"PASS": 12, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [],
        },
    )
    state = {"release_id": "release-a", "phase": "complete"}

    quick = module.build_quick_status(release, state=state)
    full = module.build_full_status(release, state=state)

    assert reads == [], "a state the caller read is not read again"
    assert quick["health_scope"] == "quick" and full["health_scope"] == "full"
    for report in (quick, full):
        assert report["mode"] == "status"
        assert report["healthy"] is True
        assert report["live_release"]["release_id"] == "release-a"
        assert report["next_deploy"] == {"kind": "NOOP", "changed": []}
        assert set(report) >= {
            "mode",
            "healthy",
            "health",
            "live_release",
            "configured_release",
            "next_deploy",
        }, "quick and full status share one document shape"
    # Without a pre-read state each builder reads once, as before.
    module.build_quick_status(release)
    assert reads == ["state"]


def test_status_header_names_verdict_release_next_deploy_and_failures() -> None:
    module = _admin_module()
    report = {
        "healthy": False,
        "health_scope": "quick",
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        },
        "next_deploy": {"kind": "CONTROL_PLANE_ONLY", "resume": False},
        "health": {
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 1, "SKIP": 0},
            "checks": [
                {"name": "cpu_workloads", "status": "PASS"},
                {"name": "control_api", "status": "FAIL", "summary": "healthz"},
            ],
        },
    }

    lines = module.status_header_lines(report, json_destination="stdout")

    assert lines == [
        "healthy: NO (quick health, 2 checks)",
        "live release: release-a phase=complete committed=yes",
        "next deploy: CONTROL_PLANE_ONLY",
        "failing checks: control_api",
        "full JSON report: stdout",
    ]
    resumed = module.status_header_lines(
        {**report, "next_deploy": {"kind": "FULL", "resume": True, "action": "commit"}},
        json_destination="stdout (also in /var/log/status.log)",
    )
    assert resumed[2] == "next deploy: FULL (resume commit)"
    assert resumed[4] == "full JSON report: stdout (also in /var/log/status.log)"
    # A summary that failed before it could name the release still gets a header.
    bare = module.status_header_lines(
        {"healthy": True, "next_deploy_error": "state missing"},
        json_destination="stdout",
    )
    assert bare[0] == "healthy: yes (full health, 0 checks)"
    assert bare[1] == "live release: unknown phase=unknown committed=no"
    assert bare[2] == "next deploy: state missing"
    assert bare[3] == "failing checks: none"


def test_compact_report_keeps_failures_and_names_the_passes() -> None:
    module = _admin_module()
    report = {
        "mode": "preflight",
        "healthy": False,
        "summary": {"PASS": 2, "WARN": 0, "FAIL": 1, "SKIP": 0},
        "checks": [
            {"name": "tools", "status": "PASS", "summary": "ok", "details": {"a": 1}},
            {"name": "aurora", "status": "FAIL", "summary": "down", "details": None},
            {"name": "monitoring", "status": "PASS", "summary": "ok", "details": {}},
        ],
    }

    compact = module.compact_report(report)

    assert compact["checks"] == [report["checks"][1]]
    assert compact["passed_checks"] == ["tools", "monitoring"]
    assert compact["healthy"] is False and compact["summary"] == report["summary"]
    assert report["checks"][0]["details"] == {"a": 1}, "the source report is untouched"


def test_full_report_is_requested_by_environment() -> None:
    module = _admin_module()

    assert module.full_report_requested({}) is False
    assert module.full_report_requested({module.FULL_REPORT_ENV: "0"}) is False
    assert module.full_report_requested({module.FULL_REPORT_ENV: "1"}) is True
    assert module.full_report_requested({module.FULL_REPORT_ENV: " true "}) is True


# --- a complete, uncommitted live release and a different candidate ----------------------


def _pending_commit_state(release_id: str = "live-release") -> dict[str, object]:
    return {
        "phase": "complete",
        "release_id": release_id,
        "transaction_committed": False,
        "release_lifecycle": "COMMITTED",
    }


def test_a_different_candidate_commits_the_live_release_then_upgrades(
    monkeypatch,
) -> None:
    """Deploy #29 (2026-09-09) was refused with the engine's raw
    ``resume release_id does not match the candidate``: the live release had
    completed, its verify had failed, and no candidate but itself could ever
    commit it. A different candidate now commits it as the baseline first."""

    module = _admin_module()
    expected_diff = module.diff_from_changed({"control_plane_wheel"})
    events: list[object] = []
    states = [
        _pending_commit_state(),
        {**_pending_commit_state(), "transaction_committed": True},
    ]
    release = SimpleNamespace(
        release_id="candidate-release",
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=("gpu-a",)),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: dict(states.pop(0)),
        pin_approved_manifest_plan=lambda _digest: events.append(("pin", _digest)),
        upgrade=lambda **kwargs: events.append(("upgrade", kwargs)),
    )
    monkeypatch.setattr(
        module,
        "commit_live_release",
        lambda _release, state: events.append(("commit", state["release_id"])),
    )
    monkeypatch.setattr(module, "narrate_step", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module, "classify_release", lambda _release, _state: expected_diff
    )

    module.run_deploy(release)

    assert events == [
        ("commit", "live-release"),
        ("upgrade", {"diff": expected_diff}),
    ], events


def test_the_same_candidate_still_resumes_into_its_own_commit(monkeypatch) -> None:
    module = _admin_module()
    expected_diff = module.diff_from_changed({"control_plane_wheel"})
    events: list[object] = []
    release = SimpleNamespace(
        release_id="live-release",
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=("gpu-a",)),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: _pending_commit_state(),
        pin_approved_manifest_plan=lambda _digest: None,
        upgrade=lambda **kwargs: events.append(("upgrade", kwargs)),
    )
    monkeypatch.setattr(
        module,
        "commit_live_release",
        lambda _release, _state: events.append(("commit", None)),
    )
    monkeypatch.setattr(
        module, "retry_release_diff", lambda _release, _state: expected_diff
    )

    module.run_deploy(release)

    assert events == [("upgrade", {"resume": True, "diff": expected_diff})], events


def test_next_deploy_names_the_live_release_a_different_candidate_will_commit(
    monkeypatch,
) -> None:
    module = _admin_module()
    expected_diff = module.diff_from_changed({"control_plane_wheel"})
    seen: list[dict[str, object]] = []

    def classify(_release, state):
        seen.append(state)
        return expected_diff

    monkeypatch.setattr(module, "classify_release", classify)
    release = SimpleNamespace(release_id="candidate-release")

    report = module.next_deploy(release, _pending_commit_state())

    assert report["action"] == "upgrade"
    assert report["resume"] is False
    assert report["commits_live_release_id"] == "live-release"
    assert seen[0]["transaction_committed"] is True, (
        "the diff is classified against the release as it will be once committed"
    )


def test_deploy_leaves_an_uncommitted_bootstrap_for_verify_and_commit() -> None:
    """Live 2026-09-12 trace: a first bootstrap whose verify or stability failed
    sits at complete/uncommitted with no previous release. Resuming it as an
    upgrade refuses ("resume release diff does not match"), because bootstrap
    records no diff or plan; there is nothing to re-apply, so deploy returns and
    the driver verifies and commits it."""

    module = _admin_module()
    calls: list[str] = []
    release = SimpleNamespace(
        release_id="release-a",
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=()),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: {
            "phase": "complete",
            "release_id": "release-a",
            "transaction_committed": False,
            "previous": None,
        },
        upgrade=lambda **_kwargs: calls.append("upgrade"),
        noop=lambda _diff: calls.append("noop"),
        commit_release=lambda: calls.append("commit"),
        pin_approved_manifest_plan=lambda _digest: calls.append("pin"),
    )

    module.run_deploy(release)

    assert calls == [], (
        "an uncommitted bootstrap must not be re-applied or committed by deploy"
    )


@pytest.mark.parametrize(
    ("recorded_release", "expected_pins"),
    [("candidate-release", ["plan-digest"]), ("older-release", [])],
)
def test_a_resumed_bootstrap_pins_the_plan_only_for_the_same_candidate(
    recorded_release: str, expected_pins: list[str]
) -> None:
    """Live 2026-09-12: a bootstrap interrupted on tree A was rerun with the
    fixed tree B; the pin from A refused B's manifests, and that refusal tore
    the partial site down. Only the same candidate resumes under the pin; a new
    candidate re-plans, since a bootstrap has no baseline to protect."""

    module = _admin_module()
    pins: list[str] = []
    calls: list[str] = []
    release = SimpleNamespace(
        release_id="candidate-release",
        config=SimpleNamespace(namespace="gpu-fault-system", clusters=("gpu-a",)),
        _cpu=lambda *args: ["kubectl", *args],
        runner=SimpleNamespace(probe=lambda _args: True),
        _load_state=lambda: {
            "phase": "bootstrap-started",
            "release_id": recorded_release,
            "approved_manifest_sha256": "plan-digest",
        },
        pin_approved_manifest_plan=lambda digest: pins.append(digest),
        bootstrap=lambda: calls.append("bootstrap"),
    )

    module.run_deploy(release)

    assert pins == expected_pins, "the plan pin belongs to the candidate that made it"
    assert calls == ["bootstrap"], "the bootstrap itself always runs"
