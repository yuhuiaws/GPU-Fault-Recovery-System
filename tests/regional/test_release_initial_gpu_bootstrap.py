from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin.execution import current_deadline, deadline_scope
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_registry as REGISTRY
from gpu_fault_release import rollout as ROLLOUT

ROOT = Path(__file__).resolve().parents[2]
OVERLAP_TIMEOUT = 5.0


def test_initial_registry_is_written_once_as_a_complete_set() -> None:
    desired = [{"cluster_id": "gpu-a"}, {"cluster_id": "gpu-b"}]
    writes: list[list[dict[str, str]]] = []

    REGISTRY.initialize_registry(
        SimpleNamespace(),
        load=lambda _release: ([{"cluster_id": "gpu-a"}], None),
        desired=lambda _release: desired,
        write=lambda _release, value: writes.append(value),
    )

    assert writes == [desired], (
        "initial registry was not replaced by one complete atomic cluster set"
    )


def test_initial_gpu_bootstrap_is_bounded_parallel() -> None:
    targets = [SimpleNamespace(cluster_id=f"gpu-{index}") for index in range(6)]
    active = 0
    maximum = 0
    lock = threading.Lock()
    first_wave = threading.Event()
    saved: list[list[str]] = []

    def bootstrap_target(_release, _target) -> None:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                first_wave.set()
        assert first_wave.wait(timeout=2), (
            "four bootstrap workers did not start concurrently"
        )
        time.sleep(0.01)
        with lock:
            active -= 1

    release = SimpleNamespace(
        config=SimpleNamespace(clusters=targets),
        _save_state=lambda _phase, **updates: saved.append(
            list(updates["completed_cluster_ids"])
        ),
    )
    completed = {"gpu-0"}

    ORCHESTRATION.bootstrap_gpu_clusters(release, completed, bootstrap=bootstrap_target)

    assert maximum == 4, "initial GPU bootstrap did not enforce the four-cluster limit"
    assert completed == {target.cluster_id for target in targets}
    assert saved[-1] == sorted(completed)


def test_initial_gpu_bootstrap_checkpoints_successes_before_failure() -> None:
    targets = [SimpleNamespace(cluster_id=name) for name in ("gpu-a", "gpu-b", "gpu-c")]
    saved: list[list[str]] = []

    def bootstrap_target(_release, target) -> None:
        if target.cluster_id == "gpu-b":
            raise RuntimeError("failed")

    release = SimpleNamespace(
        config=SimpleNamespace(clusters=targets),
        _save_state=lambda _phase, **updates: saved.append(
            list(updates["completed_cluster_ids"])
        ),
    )
    completed: set[str] = set()

    with pytest.raises(ORCHESTRATION.ReleaseError, match="gpu-b"):
        ORCHESTRATION.bootstrap_gpu_clusters(
            release, completed, bootstrap=bootstrap_target
        )

    assert completed == {"gpu-a", "gpu-c"}, (
        "successful clusters were not retained after another bootstrap failed"
    )
    assert saved[-1] == ["gpu-a", "gpu-c"]


@pytest.fixture
def initial_release(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    calls: list[str] = []
    saved: list[dict[str, object]] = []

    def recorder(name: str):
        def step(*_arguments, **_kwargs) -> None:
            calls.append(name)

        return step

    def save(phase: str, **updates: object) -> None:
        calls.append(f"state:{phase}")
        release.state.update(phase=phase, **updates)
        saved.append(dict(release.state))

    release = SimpleNamespace(
        calls=calls,
        saved=saved,
        record_state=save,
        state={},
        release_id="release-1",
        config=SimpleNamespace(namespace="gpu-fault-system", auto_rollback=False),
        runner=SimpleNamespace(run=lambda *_a, **_k: ""),
        _cpu=lambda *arguments: list(arguments),
        _save_state=save,
        _ensure_contexts=recorder("contexts"),
        _apply_rds_ca_bundle=recorder("rds-ca-bundle"),
        _require_cpu_secrets=recorder("cpu-secrets"),
        _initialize_registry=recorder("initialize-registry"),
        _update_registry=lambda *_a, **_k: pytest.fail(
            "bootstrap updated the registry incrementally"
        ),
        _upload_release=recorder("upload"),
        _prepare_bootstrap_workflows=recorder("aurora-refresh"),
        _ensure_schema=recorder("schema"),
        _apply_cpu=recorder("cpu"),
        _prepare_nlb=recorder("prepare-nlb"),
        _wait_nlb=recorder("nlb"),
        _apply_control_plane_observability=recorder("observability"),
        _apply_dataplane_expected_rules=recorder("expected-rules"),
        _validate_release=recorder("validate"),
        _cleanup_bootstrap=recorder("cleanup"),
    )
    monkeypatch.setattr(ROLLOUT, "ensure_runtime_profile", recorder("runtime-profile"))
    monkeypatch.setattr(
        ROLLOUT,
        "bootstrap_gpu_clusters",
        lambda _release, completed: calls.append(f"gpu:{sorted(completed)}"),
    )
    return release


def test_initial_bootstrap_uses_atomic_registry_then_parallel_gpu_workers(
    initial_release: SimpleNamespace,
) -> None:
    """The complete registry, CPU, profile and endpoint gate both branches."""
    ROLLOUT.RegionalRelease.bootstrap(initial_release)

    calls = initial_release.calls
    assert calls[:-5] == [
        "contexts",
        "rds-ca-bundle",
        "state:bootstrap-started",
        "cpu-secrets",
        "initialize-registry",
        "cpu-secrets",
        "prepare-nlb",
        "upload",
        "aurora-refresh",
        "schema",
        "cpu",
        "runtime-profile",
        "state:bootstrap-cpu-ready",
        "nlb",
        "state:bootstrap-endpoint-ready",
    ]
    assert sorted(calls[-5:-3]) == ["gpu:[]", "observability"]
    assert calls[-3:] == ["expected-rules", "validate", "state:complete"]


@pytest.mark.parametrize("auto_rollback", [False, True])
def test_bootstrap_prerequisite_refusal_preserves_original_candidate_before_writes(
    initial_release: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    auto_rollback: bool,
) -> None:
    release = initial_release
    release.config.auto_rollback = auto_rollback
    release.state = {"phase": "bootstrap-cleaned", "release_id": "old-candidate"}
    original = dict(release.state)

    def reject(instance: SimpleNamespace) -> None:
        assert instance.state == original, "bootstrap replaced the predecessor identity"
        raise RuntimeError("prerequisite entry refused")

    monkeypatch.setattr(ROLLOUT, "prepare_bootstrap_prerequisite_entry", reject)
    release.runner.run = lambda *_args, **_kwargs: pytest.fail(
        "bootstrap mutated resources before validating the pending prerequisite"
    )
    with pytest.raises(RuntimeError, match="prerequisite entry refused"):
        ROLLOUT.RegionalRelease.bootstrap(release)
    assert release.state == original
    assert release.saved == [], (
        "refusal must not stamp the new candidate's failed state"
    )
    assert release.calls == ["contexts"], "refusal entered application cleanup"


def test_bootstrap_checks_prerequisite_before_initial_checkpoint_and_resources(
    initial_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = initial_release

    def checked(instance: SimpleNamespace) -> None:
        assert instance.saved == [], (
            "candidate state was written before prerequisite check"
        )
        instance.calls.append("prerequisite-entry")

    def apply(*_args: object, **_kwargs: object) -> str:
        assert "prerequisite-entry" in release.calls, (
            "bootstrap resources preceded the pending-repair check"
        )
        return ""

    monkeypatch.setattr(ROLLOUT, "prepare_bootstrap_prerequisite_entry", checked)
    release.runner.run = apply
    ROLLOUT.RegionalRelease.bootstrap(release)
    assert release.calls.index("prerequisite-entry") < release.calls.index(
        "state:bootstrap-started"
    )


@pytest.mark.parametrize("first", ["gpu", "observability"])
def test_initial_bootstrap_overlaps_configuration_but_joins_before_validation(
    initial_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, first: str
) -> None:
    release = initial_release
    started = {name: threading.Event() for name in ("gpu", "observability")}
    finish = {name: threading.Event() for name in started}
    finished = {name: threading.Event() for name in started}

    def branch(name: str) -> None:
        assert release.state["phase"] == "bootstrap-endpoint-ready"
        started[name].set()
        assert finish[name].wait(OVERLAP_TIMEOUT), (
            "test did not release the bootstrap branch"
        )
        finished[name].set()

    def gpu(_release, completed: set[str]) -> None:
        branch("gpu")
        completed.add("gpu-a")
        release.record_state(
            "bootstrap-data-plane-progress", completed_cluster_ids=sorted(completed)
        )

    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", gpu)
    monkeypatch.setattr(
        release, "_apply_control_plane_observability", lambda: branch("observability")
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ROLLOUT.RegionalRelease.bootstrap, release)
        try:
            assert all(event.wait(OVERLAP_TIMEOUT) for event in started.values()), (
                "GPU bootstrap and observability configuration did not overlap"
            )
            finish[first].set()
            assert finished[first].wait(OVERLAP_TIMEOUT), (
                "selected bootstrap branch did not finish"
            )
            assert not future.done(), "bootstrap returned before both branches drained"
            assert "expected-rules" not in release.calls
            assert "validate" not in release.calls
            assert "state:complete" not in release.calls
        finally:
            for event in finish.values():
                event.set()
        future.result(timeout=OVERLAP_TIMEOUT)

    assert release.calls[-3:] == ["expected-rules", "validate", "state:complete"]
    assert release.state["completed_cluster_ids"] == ["gpu-a"]
    assert release.state["transaction_committed"] is False


@pytest.mark.parametrize("failed_branch", ["gpu", "observability"])
@pytest.mark.parametrize("auto_rollback", [False, True])
def test_initial_bootstrap_drains_both_branches_before_failure_or_cleanup(
    initial_release: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failed_branch: str,
    auto_rollback: bool,
) -> None:
    release = initial_release
    release.config.auto_rollback = auto_rollback
    started = {name: threading.Event() for name in ("gpu", "observability")}
    finished = {name: threading.Event() for name in started}
    drain = threading.Event()
    error = RuntimeError(f"{failed_branch} rejected candidate")

    def branch(name: str) -> None:
        started[name].set()
        try:
            assert all(event.wait(OVERLAP_TIMEOUT) for event in started.values()), (
                "bootstrap branches did not both start"
            )
            if name == failed_branch:
                raise error
            assert drain.wait(OVERLAP_TIMEOUT), (
                "test did not release the surviving branch"
            )
        finally:
            finished[name].set()

    def gpu(_release, completed: set[str]) -> None:
        try:
            branch("gpu")
        finally:
            completed.add("gpu-a")
            release.record_state(
                "bootstrap-data-plane-progress", completed_cluster_ids=sorted(completed)
            )

    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", gpu)
    monkeypatch.setattr(
        release, "_apply_control_plane_observability", lambda: branch("observability")
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ROLLOUT.RegionalRelease.bootstrap, release)
        try:
            assert finished[failed_branch].wait(OVERLAP_TIMEOUT), (
                "failing bootstrap branch did not finish"
            )
            assert not future.done(), "failure returned while another branch was active"
            assert "state:bootstrap-failed" not in release.calls
            assert "cleanup" not in release.calls
            assert "expected-rules" not in release.calls
            assert "validate" not in release.calls
        finally:
            drain.set()
        with pytest.raises(RuntimeError) as raised:
            future.result(timeout=OVERLAP_TIMEOUT)

    assert raised.value is error
    assert all(event.is_set() for event in finished.values()), (
        "bootstrap failure left a branch running"
    )
    assert release.state["phase"] == "bootstrap-failed"
    assert release.state["completed_cluster_ids"] == ["gpu-a"]
    assert release.state["transaction_committed"] is False
    assert release.calls.count("cleanup") == int(auto_rollback)
    assert "expected-rules" not in release.calls
    assert "validate" not in release.calls
    assert "state:complete" not in release.calls
    if failed_branch == "observability":
        assert release.state["resume_phase"] == "bootstrap-data-plane-progress"
    expected_tail = ["state:bootstrap-failed"] + (["cleanup"] if auto_rollback else [])
    assert release.calls[-len(expected_tail) :] == expected_tail


def test_initial_bootstrap_retains_gpu_failure_when_observability_also_fails(
    initial_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_error = RuntimeError("GPU bootstrap rejected candidate")
    obs_started = threading.Event()
    gpu_started = threading.Event()

    def observability() -> None:
        obs_started.set()
        assert gpu_started.wait(OVERLAP_TIMEOUT), (
            "GPU branch never entered the failure barrier"
        )
        raise RuntimeError("AMP rejected candidate")

    def gpu(_release, _completed) -> None:
        gpu_started.set()
        assert obs_started.wait(OVERLAP_TIMEOUT), (
            "monitoring branch never entered the failure barrier"
        )
        raise gpu_error

    monkeypatch.setattr(
        initial_release, "_apply_control_plane_observability", observability
    )
    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", gpu)
    with pytest.raises(RuntimeError) as raised:
        ROLLOUT.RegionalRelease.bootstrap(initial_release)

    assert raised.value is gpu_error
    assert any(
        "control-plane observability also failed: RuntimeError: AMP rejected candidate"
        in note
        for note in getattr(raised.value, "__notes__", [])
    ), "secondary monitoring failure was omitted from diagnostics"
    assert initial_release.state["phase"] == "bootstrap-failed"


def test_initial_bootstrap_interruption_still_drains_observability(
    initial_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    obs_started = threading.Event()
    interrupted = threading.Event()
    drain = threading.Event()
    obs_finished = threading.Event()

    def observability() -> None:
        obs_started.set()
        assert drain.wait(OVERLAP_TIMEOUT), (
            "test did not release interrupted monitoring"
        )
        obs_finished.set()

    def gpu(_release, _completed) -> None:
        assert obs_started.wait(OVERLAP_TIMEOUT), (
            "monitoring did not start before interruption"
        )
        interrupted.set()
        raise KeyboardInterrupt

    monkeypatch.setattr(
        initial_release, "_apply_control_plane_observability", observability
    )
    monkeypatch.setattr(ROLLOUT, "bootstrap_gpu_clusters", gpu)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ROLLOUT.RegionalRelease.bootstrap, initial_release)
        try:
            assert interrupted.wait(OVERLAP_TIMEOUT), (
                "GPU branch did not reach interruption"
            )
            assert not future.done(), "interruption returned before monitoring drained"
        finally:
            drain.set()
        with pytest.raises(KeyboardInterrupt):
            future.result(timeout=OVERLAP_TIMEOUT)

    assert obs_finished.is_set(), "interruption left monitoring running"
    assert "validate" not in initial_release.calls
    assert "cleanup" not in initial_release.calls
    assert "expected-rules" not in initial_release.calls
    assert "state:complete" not in initial_release.calls


@pytest.mark.parametrize("gate", ["cpu", "profile", "endpoint"])
def test_initial_bootstrap_gate_failure_never_launches_independent_branches(
    initial_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    def refused(*_args, **_kwargs) -> None:
        raise RuntimeError(f"{gate} gate refused")

    if gate == "profile":
        monkeypatch.setattr(ROLLOUT, "ensure_runtime_profile", refused)
    else:
        monkeypatch.setattr(
            initial_release, "_apply_cpu" if gate == "cpu" else "_wait_nlb", refused
        )
    with pytest.raises(RuntimeError, match="gate refused"):
        ROLLOUT.RegionalRelease.bootstrap(initial_release)

    assert "observability" not in initial_release.calls
    assert "expected-rules" not in initial_release.calls
    assert "gpu:[]" not in initial_release.calls
    assert "validate" not in initial_release.calls
    assert initial_release.state["phase"] == "bootstrap-failed"


def test_initial_bootstrap_final_validation_can_still_reject_the_release(
    initial_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def validate() -> None:
        assert "observability" in initial_release.calls
        assert "gpu:[]" in initial_release.calls
        assert "expected-rules" in initial_release.calls
        raise RuntimeError("collector health is not verified")

    monkeypatch.setattr(initial_release, "_validate_release", validate)
    with pytest.raises(RuntimeError, match="collector health"):
        ROLLOUT.RegionalRelease.bootstrap(initial_release)

    assert initial_release.state["phase"] == "bootstrap-failed"
    assert "state:complete" not in initial_release.calls


def test_initial_bootstrap_preserves_the_callers_deadline_in_observability(
    initial_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = []
    monkeypatch.setattr(
        initial_release,
        "_apply_control_plane_observability",
        lambda: observed.append(current_deadline()),
    )

    with deadline_scope("initial-bootstrap-test", 30) as deadline:
        ROLLOUT.RegionalRelease.bootstrap(initial_release)

    assert observed == [deadline]


def test_initial_bootstrap_resume_reconfigures_observability_before_validation(
    initial_release: SimpleNamespace,
) -> None:
    initial_release.state.update(
        phase="bootstrap-failed",
        release_id="release-1",
        resume_phase="bootstrap-data-plane-progress",
        completed_cluster_ids=["gpu-a"],
    )

    ROLLOUT.RegionalRelease.bootstrap(initial_release)

    assert "schema" not in initial_release.calls
    assert "cpu" not in initial_release.calls
    assert "runtime-profile" in initial_release.calls
    assert "nlb" in initial_release.calls
    assert "gpu:['gpu-a']" in initial_release.calls
    assert initial_release.calls.count("observability") == 1
    assert initial_release.calls[-3:] == [
        "expected-rules",
        "validate",
        "state:complete",
    ]
    assert initial_release.state["completed_cluster_ids"] == ["gpu-a"]
