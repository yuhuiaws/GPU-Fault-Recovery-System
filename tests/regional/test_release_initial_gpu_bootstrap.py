from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ROLLOUT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
)
ORCHESTRATION = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_orchestration.py"
)
REGISTRY = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_registry.py"
)


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


def test_initial_bootstrap_uses_atomic_registry_then_parallel_gpu_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first install writes the registry once and then fans out.

    The registry has to be complete before any GPU cluster is touched, because
    a cluster that comes up against a half-written registry sees itself as
    unknown and fails closed. Incremental per-cluster registry updates would
    reintroduce exactly that window, so ``_update_registry`` must stay unused
    here.
    """

    calls: list[str] = []

    def recorder(name: str):
        def step(*_arguments, **_kwargs) -> None:
            calls.append(name)

        return step

    release = SimpleNamespace(
        state={},
        release_id="release-1",
        config=SimpleNamespace(namespace="gpu-fault-system", auto_rollback=False),
        runner=SimpleNamespace(run=lambda *_a, **_k: ""),
        _cpu=lambda *arguments: list(arguments),
        _save_state=lambda phase, **_updates: calls.append(f"state:{phase}"),
        _ensure_contexts=recorder("contexts"),
        _require_cpu_secrets=recorder("cpu-secrets"),
        _initialize_registry=recorder("initialize-registry"),
        _update_registry=lambda *_a, **_k: pytest.fail(
            "bootstrap updated the registry incrementally"
        ),
        _upload_release=recorder("upload"),
        _ensure_schema=recorder("schema"),
        _apply_cpu=recorder("cpu"),
        _apply_nlb=recorder("nlb"),
        _validate_release=recorder("validate"),
    )
    monkeypatch.setattr(ROLLOUT, "ensure_runtime_profile", recorder("runtime-profile"))
    monkeypatch.setattr(
        ROLLOUT,
        "bootstrap_gpu_clusters",
        lambda _release, completed: calls.append(f"gpu:{sorted(completed)}"),
    )

    ROLLOUT.RegionalRelease.bootstrap(release)

    assert calls == [
        "contexts",
        "state:bootstrap-started",
        "cpu-secrets",
        "initialize-registry",
        "cpu-secrets",
        "upload",
        "schema",
        "cpu",
        "runtime-profile",
        "state:bootstrap-cpu-ready",
        "nlb",
        "state:bootstrap-endpoint-ready",
        "gpu:[]",
        "validate",
        "state:complete",
    ]
