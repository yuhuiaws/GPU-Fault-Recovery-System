"""The upgrade and rollback preflights run their independent checks at once.

The Aurora credential refresh Job, the two store probes and the previous-state
capture read and write disjoint things, so they overlap; the failure semantics
of the serial form are kept: every lane runs to its end, the first failure in
declared order is raised unchanged, and nothing is written before all passed.

The blocking-`Event` pattern is deliberate -- a serialized implementation does
not fail an assertion, it fails to finish, so each overlap is expressed as steps
that can only all complete if they really run together.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_aurora_refresh as REFRESH
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_preflight_concurrency as PREFLIGHT
from gpu_fault_release.regional_release_config import ReleaseError

OVERLAP_TIMEOUT = 10.0


@pytest.fixture(autouse=True)
def _no_ambient_admin_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Narration must never reach a real operator log from a test."""

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)


class _Stop(RuntimeError):
    """Raised by the first mutation after the preflight, to end the test there."""


def _control_plane_diff() -> DIFF.ReleaseDiff:
    return DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"cpu_worker_manifests"}),
    )


def _upgrade_double(**overrides) -> SimpleNamespace:
    fields = {
        "config": SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        "state": {},
        "_ensure_contexts": lambda: None,
        "_require_cpu_secrets": lambda: None,
        "_apply_rds_ca_bundle": lambda: None,
        "_refresh_aurora_credentials": lambda: None,
        "_remote_commands_are_idle": lambda: True,
        "_require_no_inflight_installs": lambda **_kwargs: {"verdict": "clear"},
        "_capture_previous": lambda **_kwargs: {"metadata": {}},
        "_backup_release_secrets": lambda: (_ for _ in ()).throw(_Stop()),
        "_save_state": lambda *_args, **_kwargs: pytest.fail(
            "state was written before the preflight had finished"
        ),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _await(event: threading.Event, what: str) -> None:
    if not event.wait(timeout=OVERLAP_TIMEOUT):
        raise AssertionError(f"{what} never overlapped the other preflight lanes")


# --- the upgrade ---------------------------------------------------------------


def test_upgrade_probes_and_capture_overlap_after_refresh() -> None:
    """Read-only lanes overlap only after credentials have been prepared."""

    refresh_started = threading.Event()
    probe_started = threading.Event()
    capture_started = threading.Event()
    order: list[str] = []
    lock = threading.Lock()

    def note(name: str) -> None:
        with lock:
            order.append(name)

    def refresh() -> dict:
        assert not probe_started.is_set(), (
            "store probes must wait until the credential refresh has completed"
        )
        assert not capture_started.is_set(), (
            "the previous snapshot must wait until credentials are refreshed"
        )
        refresh_started.set()
        note("refresh")
        return {"status": "refreshed"}

    def idle() -> bool:
        probe_started.set()
        _await(refresh_started, "the idle probe")
        _await(capture_started, "the idle probe")
        note("idle")
        return True

    def capture(**_kwargs) -> dict:
        capture_started.set()
        _await(refresh_started, "the capture")
        _await(probe_started, "the capture")
        note("capture")
        return {"metadata": {"release_id": "previous"}}

    def backup() -> dict:
        note("backup")
        raise _Stop()

    release = _upgrade_double(
        _refresh_aurora_credentials=refresh,
        _remote_commands_are_idle=idle,
        _capture_previous=capture,
        _backup_release_secrets=backup,
    )

    with pytest.raises(_Stop):
        ORCHESTRATION.upgrade_release(release, diff=_control_plane_diff())

    assert order[0] == "refresh"
    assert sorted(order[1:3]) == ["capture", "idle"]
    assert order[3] == "backup", "the Secret backup is the first step after the lanes"


def test_upgrade_capture_from_the_lane_is_the_previous_state(monkeypatch) -> None:
    """What the concurrent lane captured is what the transaction records."""

    saves: list[dict] = []

    def save_state(phase, **updates):
        saves.append({"phase": phase, **updates})

    def run_phases(*_args, **_kwargs) -> None:
        raise _Stop()

    def recover(_release, *, error, **_kwargs) -> None:
        raise error

    previous = {
        "metadata": {"release_id": "old"},
        "aurora_refresh": {
            "schema_version": 1,
            "namespace": "gpu-fault-system",
            "objects": [],
            "absent": [
                {"resource": resource, "name": REFRESH.CRONJOB_NAME}
                for resource in sorted(REFRESH.OBJECT_RESOURCES)
            ],
            "absence_verified": True,
        },
    }
    release = _upgrade_double(
        _capture_previous=lambda **_kwargs: previous,
        _backup_release_secrets=lambda: {"cpu": None, "clusters": {}},
        _save_state=save_state,
    )
    monkeypatch.setattr(ORCHESTRATION, "run_upgrade_phases", run_phases)
    monkeypatch.setattr(ORCHESTRATION, "recover_failed_upgrade", recover)

    with pytest.raises(_Stop):
        ORCHESTRATION.upgrade_release(release, diff=_control_plane_diff())

    assert len(saves) == 1, "exactly the preflight checkpoint is written"
    assert saves[0]["phase"] == "preflight", "the first checkpoint is the preflight"
    assert saves[0]["previous"]["metadata"] == {"release_id": "old"}, (
        "the lane's capture is the recorded previous state"
    )
    assert saves[0]["previous"]["secret_backups"] == {"cpu": None, "clusters": {}}, (
        "the Secret backups are attached after the lanes"
    )
    assert release.state["inflight_installs"] == {"verdict": "clear"}, (
        "the store probe's verdict rides in the state"
    )


def test_a_busy_store_still_refuses_the_upgrade_with_the_same_error() -> None:
    """The idle probe's refusal is the serial one, raised before any write."""

    refreshed = threading.Event()
    release = _upgrade_double(
        _refresh_aurora_credentials=lambda: refreshed.set(),
        _remote_commands_are_idle=lambda: False,
        _require_no_inflight_installs=lambda **_kwargs: pytest.fail(
            "the in-flight install gate ran after the idle probe refused"
        ),
        _backup_release_secrets=lambda: pytest.fail(
            "Secret backups were taken after a preflight lane failed"
        ),
    )

    with pytest.raises(
        ReleaseError, match="remote commands are PENDING/LEASED/WAITING"
    ):
        ORCHESTRATION.upgrade_release(release, diff=_control_plane_diff())

    assert refreshed.is_set(), "the refresh lane still ran to its end"
    assert release.state == {}, "no state is written when a preflight lane fails"


def test_the_first_failure_in_declared_order_wins(capsys) -> None:
    """Independent read failures retain their order after all workers finish."""

    with pytest.raises(ReleaseError, match="store failed"):
        PREFLIGHT.run_preflight_lanes(
            SimpleNamespace(),
            (
                PREFLIGHT.PreflightLane(
                    "store-probes",
                    lambda: (_ for _ in ()).throw(ReleaseError("store failed")),
                ),
                PREFLIGHT.PreflightLane(
                    "previous-capture",
                    lambda: (_ for _ in ()).throw(ReleaseError("capture failed")),
                ),
            ),
            phase="upgrade-preflight",
        )

    captured = capsys.readouterr()
    assert "previous-capture also failed" in captured.err, (
        "the second failure is reported beside the first"
    )
    assert "preflight-concurrent" in captured.err, (
        "the lane timing line is narrated even when a lane fails"
    )


def test_failed_refresh_prevents_store_reads_and_previous_capture() -> None:
    def refused_read(*_args, **_kwargs):
        pytest.fail("a dependent read started before credentials were prepared")

    release = _upgrade_double(
        _refresh_aurora_credentials=lambda: (_ for _ in ()).throw(
            ReleaseError("credential refresh failed")
        ),
        _remote_commands_are_idle=refused_read,
        _capture_previous=refused_read,
    )
    with pytest.raises(ReleaseError, match="credential refresh failed"):
        ORCHESTRATION.upgrade_release(release, diff=_control_plane_diff())
    assert release.state == {}


def test_a_refused_validation_never_pays_for_the_capture() -> None:
    """Validation stays ahead of the capture on its lane, as it did serially."""

    release = _upgrade_double(
        config=SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        _capture_previous=lambda **_kwargs: pytest.fail(
            "the previous release was captured for a refused transaction"
        ),
    )
    # A cluster-set change is not transactional under autoRollback, so the
    # validation refuses this diff before anything else happens on its lane.
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.FULL, changed=frozenset({"clusters"})
    )

    with pytest.raises(ReleaseError, match="not yet transactional"):
        ORCHESTRATION.upgrade_release(release, diff=diff)

    assert release.state == {}, "a refused transaction writes nothing"


def test_a_resume_captures_nothing_on_the_lane() -> None:
    """A resumed transaction keeps its persisted previous state."""

    release = _upgrade_double(
        _capture_previous=lambda **_kwargs: pytest.fail(
            "a resume must not re-capture the previous release"
        ),
        _load_state=lambda: (_ for _ in ()).throw(_Stop()),
    )

    with pytest.raises(_Stop):
        ORCHESTRATION.upgrade_release(release, resume=True)


# --- the runner ----------------------------------------------------------------


def test_lanes_return_their_results_by_name(capsys) -> None:
    results = PREFLIGHT.run_preflight_lanes(
        SimpleNamespace(),
        (
            PREFLIGHT.PreflightLane("one", lambda: 1),
            PREFLIGHT.PreflightLane("two", lambda: "two"),
        ),
        phase="test-preflight",
    )

    assert results == {"one": 1, "two": "two"}, "every lane's result is keyed by name"
    line = next(
        line
        for line in capsys.readouterr().err.splitlines()
        if "preflight-concurrent" in line
    )
    for field in ("phase=test-preflight", "one=", "two=", "wall=", "serial=", "saved="):
        assert field in line, f"the timing line carries {field}"


def test_a_dry_run_keeps_the_declared_order() -> None:
    """A dry run's trace stays readable: the lanes run one after another."""

    order: list[str] = []
    started = threading.Event()

    def first() -> None:
        order.append("first")
        started.set()

    def second() -> None:
        assert started.is_set(), "the second lane started before the first finished"
        order.append("second")

    PREFLIGHT.run_preflight_lanes(
        SimpleNamespace(runner=SimpleNamespace(dry_run=True)),
        (
            PREFLIGHT.PreflightLane("first", first),
            PREFLIGHT.PreflightLane("second", second),
        ),
        phase="test-preflight",
    )

    assert order == ["first", "second"], "a dry run runs the lanes in declared order"


def test_a_failing_lane_does_not_interrupt_the_others() -> None:
    finished: list[str] = []

    def slow() -> None:
        threading.Event().wait(0.05)
        finished.append("slow")

    with pytest.raises(ReleaseError, match="fast failed"):
        PREFLIGHT.run_preflight_lanes(
            SimpleNamespace(),
            (
                PREFLIGHT.PreflightLane(
                    "fast", lambda: (_ for _ in ()).throw(ReleaseError("fast failed"))
                ),
                PREFLIGHT.PreflightLane("slow", slow),
            ),
            phase="test-preflight",
        )

    assert finished == ["slow"], "the slow lane ran to its end beside the failure"


# --- the rollback --------------------------------------------------------------


def _rollback_double(**overrides) -> SimpleNamespace:
    fields = {
        "config": SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=True, clusters=()
        ),
        "state": {},
        "_refresh_aurora_credentials": lambda: None,
        "_require_no_inflight_installs": lambda **_kwargs: None,
        "_save_state": lambda *_args, **_kwargs: pytest.fail(
            "state was written before the rollback preflight had finished"
        ),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_rollback_refresh_finishes_before_install_gate() -> None:
    order: list[str] = []

    def refresh() -> None:
        assert order == []
        order.append("refresh")

    def gate(**kwargs) -> dict:
        assert order == ["refresh"]
        order.append(f"gate:{kwargs['action']}:{kwargs['unreadable']}")
        return {"verdict": "clear"}

    release = _rollback_double(
        _refresh_aurora_credentials=refresh, _require_no_inflight_installs=gate
    )

    verdict = PREFLIGHT.run_rollback_preflight(
        release, check_installs=True, automatic=True
    )
    assert order == ["refresh", "gate:rollback:proceed"]
    assert verdict == {"verdict": "clear"}


def test_rollback_skips_the_install_gate_once_the_control_plane_is_restored() -> None:
    """A cleanup re-entry runs the refresh alone, on the calling thread."""

    calls: list[str] = []
    release = _rollback_double(
        # The rollback reads its own completed phases from the live state.
        state={"rollback_completed_phases": ["rollback-cpu-restored"]},
        _refresh_aurora_credentials=lambda: calls.append(
            f"refresh:{threading.current_thread().name}"
        ),
        _require_no_inflight_installs=lambda **_kwargs: pytest.fail(
            "the install gate ran after the control plane was already restored"
        ),
    )

    assert (
        PREFLIGHT.run_rollback_preflight(release, check_installs=False, automatic=False)
        is None
    )

    assert calls == [f"refresh:{threading.current_thread().name}"], (
        "a single lane runs inline, without a thread pool"
    )
