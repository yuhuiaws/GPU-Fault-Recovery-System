from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.admin import deadlines
from gpu_fault_release import regional_release_gpu_rollout as GPU
from gpu_fault_release import regional_release_images as IMAGES
from gpu_fault_release import regional_release_online_registry as REGISTRY
from gpu_fault_release import regional_release_preflight_concurrency as PREFLIGHT
from gpu_fault_release import regional_release_progress as PROGRESS
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout as ROLLOUT

CPU_IMAGE = "registry.example/cpu@sha256:" + "a" * 64
EXECUTOR_IMAGE = "registry.example/executor@sha256:" + "b" * 64


def test_bootstrap_completion_consumes_cluster_iterators_once() -> None:
    state = PROGRESS.bootstrap_completion_state(iter(("gpu-b", "gpu-a")), now=10)
    assert state["completed_cluster_ids"] == ["gpu-a", "gpu-b"]
    assert set(state["cluster_attempts"]) == {"gpu-a", "gpu-b"}
    assert state["cluster_attempts"]["gpu-a"]["converged_at_epoch"] == 10


@pytest.mark.parametrize("capture_gpu", [False, True])
def test_empty_gpu_capture_keeps_the_recorded_split_executor(capture_gpu: bool) -> None:
    state = {
        "release_manifest_schema_version": 4,
        "runtime_image": CPU_IMAGE,
        "executor_image": EXECUTOR_IMAGE,
    }
    assert IMAGES.capture_previous_image_identity(
        state,
        cpu_images={"cpu": CPU_IMAGE},
        executor_images={},
        capture_gpu=capture_gpu,
    ) == (CPU_IMAGE, CPU_IMAGE, EXECUTOR_IMAGE)


def test_split_images_are_not_reinterpreted_as_legacy_adoption() -> None:
    state = {
        "release_manifest_schema_version": 4,
        "runtime_image": CPU_IMAGE,
        "executor_image": EXECUTOR_IMAGE,
        "adopted_live_runtime_image": CPU_IMAGE,
    }
    assert IMAGES.capture_previous_image_identity(
        state,
        cpu_images={"cpu": CPU_IMAGE},
        executor_images={"gpu": EXECUTOR_IMAGE},
        capture_gpu=True,
    ) == (CPU_IMAGE, CPU_IMAGE, EXECUTOR_IMAGE)
    with pytest.raises(IMAGES.ReleaseError, match="drifted"):
        IMAGES.capture_previous_image_identity(
            state,
            cpu_images={"cpu": CPU_IMAGE},
            executor_images={"gpu": CPU_IMAGE},
            capture_gpu=True,
        )


def test_empty_gpu_capture_never_invents_a_v4_executor() -> None:
    with pytest.raises(IMAGES.ReleaseError, match="Executor image"):
        IMAGES.capture_previous_image_identity(
            {"release_manifest_schema_version": 4, "runtime_image": CPU_IMAGE},
            cpu_images={"cpu": CPU_IMAGE},
            executor_images={},
            capture_gpu=True,
        )


@pytest.mark.parametrize("scope", ["site", "cluster:gpu-a"])
def test_join_sync_accepts_a_scoped_capture_without_overriding_the_reader(
    scope,
) -> None:
    calls: list[tuple[str, dict]] = []
    release = SimpleNamespace(
        state={"rollback_completed_phases": ["rollback-verified"]},
        config=SimpleNamespace(
            clusters=(SimpleNamespace(cluster_id="gpu-a"),),
            release_manifest_schema_version=4,
        ),
        _capture_previous=lambda: pytest.fail("scoped sync repeated a full capture"),
        _save_state=lambda phase, **updates: calls.append((phase, updates)),
    )
    ROLLOUT.sync_release_state(
        release,
        cluster_id="gpu-a",
        captured_previous={
            "capture_scope": scope,
            "live_runtime_image": CPU_IMAGE,
            "executor_image": EXECUTOR_IMAGE,
        },
    )
    phase, written = calls[0]
    assert phase == "complete"
    assert release.state == {}
    assert written["adopted_live_runtime_image"] is None
    assert written["completed_cluster_ids"] == ["gpu-a"]
    assert written["cluster_attempts"]["gpu-a"]["state"] == "CONVERGED"
    assert written["transaction_committed"] is True


@pytest.mark.parametrize("cluster_id", [None, "other", "gpu-a"])
def test_join_sync_refuses_an_unbound_capture_before_writing(cluster_id) -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(SimpleNamespace(cluster_id="gpu-a"),)),
        _save_state=lambda *_args, **_kwargs: pytest.fail("unbound sync wrote state"),
    )
    with pytest.raises(ROLLOUT.ReleaseError, match="scope differs|unknown cluster_id"):
        ROLLOUT.sync_release_state(
            release,
            cluster_id=cluster_id,
            captured_previous={"capture_scope": "cluster:unbound"},
        )


def test_snapshot_cache_cannot_answer_a_node_mutation_safety_read() -> None:
    release = SimpleNamespace()
    with STATE.read_snapshot(release):
        with pytest.raises(GPU.ReleaseError, match="fresh node safety"):
            GPU.gpu_node_items(release, SimpleNamespace(cluster_id="gpu-a"), fresh=True)


def test_preflight_workers_inherit_the_callers_deadline() -> None:
    def expiry() -> float:
        active = deadlines.current_deadline()
        assert active is not None
        return active.expires

    with deadlines.deadline_scope("preflight-test", 5) as deadline:
        result = PREFLIGHT.run_preflight_lanes(
            SimpleNamespace(),
            (
                PREFLIGHT.PreflightLane("one", expiry),
                PREFLIGHT.PreflightLane("two", expiry),
            ),
            phase="test",
        )
    assert result == {"one": deadline.expires, "two": deadline.expires}


@pytest.mark.parametrize("operation", ["run", "condition", "probe_output"])
def test_command_preparation_consumes_the_same_time_budget(
    monkeypatch, operation
) -> None:
    clock = [100.0]
    observed: list[float] = []
    monkeypatch.setattr(deadlines.time, "monotonic", lambda: clock[0])

    def prepare() -> None:
        clock[0] += 3

    def execute(_arguments, **kwargs):
        observed.append(kwargs["timeout_seconds"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ROLLOUT, "run_command", execute)
    runner = ROLLOUT.Runner(before_command=prepare)
    getattr(runner, operation)(["kubectl", "wait"], timeout_seconds=5)
    assert observed == [2]


def test_failed_join_transition_cannot_authorize_purge(monkeypatch) -> None:
    mutations: list[str] = []
    release = SimpleNamespace(
        _update_registry=lambda *_args, **_kwargs: mutations.append("secret")
    )
    monkeypatch.setattr(
        REGISTRY,
        "transition_join_registry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            REGISTRY.ReleaseError("registry identity differs")
        ),
    )
    monkeypatch.setattr(
        REGISTRY, "purge_registry_cluster", lambda *_args: mutations.append("registry")
    )
    with pytest.raises(REGISTRY.ReleaseError, match="identity differs"):
        REGISTRY.purge_failed_join(release, SimpleNamespace(cluster_id="gpu-a"))
    assert mutations == []
