"""The GPU stage overlaps what does not need the endpoint with the gate that proves it.

Live 2026-09-12 (join-cluster, four GPU nodes): the GPU half of the release ran
strictly serially -- DNS/TLS gate, then DCGM, then the ADOT collector and the
two artifact ConfigMaps, then one Deployment wave after another -- and about two
minutes of it was stacked waiting on independent objects. These tests pin the
shape that replaced it: DCGM, the collector and the uploads are applied while
the gate probe is still running, and nothing that talks to the control plane
(the Executor, watcher and collector Deployments) is applied until the gate has
passed. A gate stub that only completes once the independent steps have been
recorded is what makes the overlap observable; a gate that finished at once
would let a serial implementation pass by accident.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_gpu_rollout as GPU_ROLLOUT
from gpu_fault_release import regional_release_gpu_stage as STAGE
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import rollout as MODULE
from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ReleaseError,
)

TARGET = SimpleNamespace(cluster_id="gpu-a")
Component = DIFF.ReleaseComponent
WAIT_SECONDS = 5.0


class _Recorder:
    """Records step names in the order they happen, from any thread.

    ``gate`` blocks until every name in ``gate_waits_for`` has been recorded,
    so the recorded order proves the other steps ran *while* the gate was
    still open rather than merely before it.
    """

    def __init__(self, gate_waits_for: tuple[str, ...]) -> None:
        self.calls: list[str] = []
        self.threads: dict[str, str] = {}
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._gate_waits_for = gate_waits_for

    def record(self, name: str) -> None:
        with self._changed:
            self.calls.append(name)
            self.threads[name] = threading.current_thread().name
            self._changed.notify_all()

    def gate(self, _target: Any) -> None:
        self.record("gate-started")
        with self._changed:
            satisfied = self._changed.wait_for(
                lambda: all(name in self.calls for name in self._gate_waits_for),
                timeout=WAIT_SECONDS,
            )
        assert satisfied, (
            f"the gate waited for {self._gate_waits_for} but only saw "
            f"{self.calls}: the independent steps must not wait for the gate"
        )
        self.record("gate-passed")

    def before(self, earlier: str, later: str) -> bool:
        return self.calls.index(earlier) < self.calls.index(later)


def _release_attributes(recorder: _Recorder) -> dict[str, Any]:
    return {
        "executor_wheel_cm": "wheel",
        "bundle_cm": "bundle",
        "bundle_sha": "b" * 64,
        "executor_wheel_sha": "e" * 64,
        "node_wheel_sha": "n" * 64,
        "config": SimpleNamespace(
            agent_config_digest="c" * 64,
            executor_wheel=Path("executor.whl"),
            bundle=Path("bundle.tar.gz"),
        ),
        "_ensure_gpu_namespace": lambda _target: recorder.record("namespace"),
        "_ensure_connection_secret": lambda _target: recorder.record("secret"),
        "_quiesce_gpu_executor": lambda _target: recorder.record("quiesce"),
        "_verify_gpu_control_plane_endpoint": recorder.gate,
        "_apply_gpu_dcgm_exporter": lambda _target, **_kwargs: recorder.record("dcgm"),
        "_apply_gpu_adot_collector": lambda _target, **_kwargs: recorder.record("adot"),
        "_apply_gpu_deployments": lambda *_args, **_kwargs: recorder.record(
            "deployments"
        ),
        "_roll_node_runtime": lambda _target, **_kwargs: recorder.record(
            "node-runtime"
        ),
        "_gpu": lambda _target, *arguments: ["kubectl", *arguments],
        "_upload_config_map": lambda _kubectl, name, *_args, **_kwargs: (
            recorder.record(f"upload:{name}")
        ),
    }


def _assert_deployments_only_after_the_gate(recorder: _Recorder) -> None:
    assert recorder.before("gate-passed", "deployments"), recorder.calls
    assert recorder.before("dcgm", "deployments"), recorder.calls
    assert recorder.before("adot", "deployments"), recorder.calls
    assert recorder.before("deployments", "node-runtime"), recorder.calls


def test_join_applies_dcgm_adot_and_uploads_while_the_gate_is_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder(("dcgm", "adot", "upload:wheel", "upload:bundle"))
    for name in ("prepare_join_registry", "ensure_runtime_profile"):
        monkeypatch.setattr(MODULE, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "join_target", lambda _self, _cluster_id: TARGET)
    release = SimpleNamespace(
        **_release_attributes(recorder),
        _ensure_contexts=lambda: None,
        _update_registry=lambda _target, *, remove: None,
        _config_map_data=lambda _name: {
            "required-agent-artifact-sha256": "n" * 64,
            "required-regional-executor-artifact-sha256": "e" * 64,
        },
        _apply_observability=lambda **_kwargs: recorder.record("observability"),
        state={},
    )

    MODULE.RegionalRelease.join_cluster(release, "gpu-a")

    assert recorder.calls[:3] == ["namespace", "secret", "quiesce"], recorder.calls
    for name in ("dcgm", "adot", "upload:wheel", "upload:bundle"):
        assert recorder.before("gate-started", name), recorder.calls
        assert recorder.before(name, "gate-passed"), (
            f"{name} must run while the gate is open: {recorder.calls}"
        )
    _assert_deployments_only_after_the_gate(recorder)
    assert recorder.calls[-1] == "observability", recorder.calls


def test_bootstrap_applies_dcgm_and_adot_while_the_gate_is_open() -> None:
    recorder = _Recorder(("dcgm", "adot"))

    ORCHESTRATION.bootstrap_gpu_target(
        SimpleNamespace(**_release_attributes(recorder)), TARGET
    )

    assert recorder.calls[:3] == ["namespace", "secret", "quiesce"], recorder.calls
    for name in ("dcgm", "adot"):
        assert recorder.before(name, "gate-passed"), recorder.calls
    assert "upload:wheel" not in recorder.calls, (
        "a bootstrap uploads its artifacts in its own phase, not in the stage"
    )
    _assert_deployments_only_after_the_gate(recorder)


def test_upgrade_stages_the_gate_beside_dcgm_and_the_collector() -> None:
    recorder = _Recorder(("dcgm", "adot"))
    progress: list[tuple[tuple[Component, ...], str]] = []

    GPU_ROLLOUT.upgrade_gpu_target(
        SimpleNamespace(**_release_attributes(recorder)),
        TARGET,
        DIFF.ReleaseDiff(kind=DIFF.ReleaseChangeKind.FULL, changed=frozenset()),
        DIFF.ReleaseExecutionPlan(
            nodes=(
                Component.ENDPOINT,
                Component.DCGM,
                Component.OBSERVABILITY,
                Component.EXECUTOR,
                Component.WATCHER,
                Component.COLLECTOR,
            )
        ),
        progress=lambda components, status, _details: progress.append(
            (components, status)
        ),
    )

    for name in ("dcgm", "adot"):
        assert recorder.before(name, "gate-passed"), recorder.calls
    assert recorder.before("secret", "gate-started"), recorder.calls
    assert recorder.before("gate-passed", "deployments"), recorder.calls
    # Every stage component records its own progress, and the Deployment wave
    # is not marked STARTED until the whole stage has completed.
    deployments = (Component.EXECUTOR, Component.WATCHER, Component.COLLECTOR)
    wave_started = progress.index((deployments, "STARTED"))
    for component in (Component.ENDPOINT, Component.DCGM, Component.OBSERVABILITY):
        assert progress.index(((component,), "STARTED")) < progress.index(
            ((component,), "COMPLETED")
        )
        assert progress.index(((component,), "COMPLETED")) < wave_started
    assert progress[-1] == (deployments, "COMPLETED")


def test_stage_failure_lets_the_other_steps_finish_and_skips_the_deployments() -> None:
    """A failed step never interrupts its neighbours and never lets the wave run.

    The rollback snapshot was captured before the stage began, so the surviving
    steps leave exactly the objects a rollback puts back; the failure keeps its
    own type so the orchestrator still tells a cluster-local pause from a
    release-wide failure.
    """

    recorder = _Recorder(("adot",))
    progress: list[tuple[tuple[Component, ...], str]] = []
    attributes = _release_attributes(recorder)

    def failing_dcgm(_target: Any, **_kwargs: Any) -> None:
        recorder.record("dcgm")
        raise ClusterLocalReleaseError("gpu-a DCGM DaemonSet did not roll out")

    attributes["_apply_gpu_dcgm_exporter"] = failing_dcgm

    with pytest.raises(ClusterLocalReleaseError, match="DCGM DaemonSet") as raised:
        GPU_ROLLOUT.upgrade_gpu_target(
            SimpleNamespace(**attributes),
            TARGET,
            DIFF.ReleaseDiff(kind=DIFF.ReleaseChangeKind.FULL, changed=frozenset()),
            DIFF.ReleaseExecutionPlan(
                nodes=(
                    Component.ENDPOINT,
                    Component.DCGM,
                    Component.OBSERVABILITY,
                    Component.EXECUTOR,
                )
            ),
            progress=lambda components, status, _details: progress.append(
                (components, status)
            ),
        )

    assert type(raised.value) is ClusterLocalReleaseError
    assert "gate-passed" in recorder.calls and "adot" in recorder.calls, (
        f"the surviving steps must run to their end: {recorder.calls}"
    )
    assert "deployments" not in recorder.calls, recorder.calls
    assert ((Component.DCGM,), "FAILED") in progress
    assert ((Component.ENDPOINT,), "COMPLETED") in progress
    assert ((Component.OBSERVABILITY,), "COMPLETED") in progress
    assert not any(
        components == (Component.EXECUTOR,) for components, _status in progress
    ), progress


def test_gate_failure_still_lets_dcgm_and_adot_finish() -> None:
    """The gate failing is the case the stage exists for: nothing that talks to
    the control plane is applied, and the exporters that were already going up
    finish so the failure report describes a settled cluster."""

    recorder = _Recorder(())
    attributes = _release_attributes(recorder)

    def failing_gate(_target: Any) -> None:
        recorder.record("gate-started")
        raise ReleaseError("gpu-a: GPU DNS/TLS check failed: NXDOMAIN")

    attributes["_verify_gpu_control_plane_endpoint"] = failing_gate

    with pytest.raises(ReleaseError, match="DNS/TLS"):
        ORCHESTRATION.bootstrap_gpu_target(SimpleNamespace(**attributes), TARGET)

    assert {"dcgm", "adot"} <= set(recorder.calls), recorder.calls
    assert "deployments" not in recorder.calls, recorder.calls
    assert "node-runtime" not in recorder.calls, recorder.calls


def test_first_failure_in_declared_order_wins_and_the_rest_are_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(message: str):
        def action() -> None:
            raise ReleaseError(message)

        return action

    steps = [
        STAGE.GpuStageStep("gate", fail("gate broke")),
        STAGE.GpuStageStep("dcgm", lambda: None),
        STAGE.GpuStageStep("adot", fail("adot broke")),
    ]

    with pytest.raises(ReleaseError, match="gate broke"):
        STAGE.run_gpu_stage(SimpleNamespace(), "gpu-a", steps)

    assert "gpu-a: adot also failed while the stage ran: adot broke" in (
        capsys.readouterr().err
    )


def test_dry_run_keeps_the_stage_serial_in_declared_order() -> None:
    """A dry run prints the plan; it has to read in one fixed order."""

    seen: list[tuple[str, str]] = []

    def step(name: str) -> STAGE.GpuStageStep:
        return STAGE.GpuStageStep(
            name, lambda: seen.append((name, threading.current_thread().name))
        )

    release = SimpleNamespace(runner=SimpleNamespace(dry_run=True))
    STAGE.run_gpu_stage(release, "gpu-a", [step("gate"), step("dcgm"), step("adot")])

    main = threading.main_thread().name
    assert seen == [("gate", main), ("dcgm", main), ("adot", main)]


def test_a_stage_of_one_step_runs_inline() -> None:
    seen: list[str] = []
    STAGE.run_gpu_stage(
        SimpleNamespace(runner=SimpleNamespace(dry_run=False)),
        "gpu-a",
        [
            STAGE.GpuStageStep(
                "gate", lambda: seen.append(threading.current_thread().name)
            )
        ],
    )
    assert seen == [threading.main_thread().name]
    STAGE.run_gpu_stage(SimpleNamespace(), "gpu-a", [])


def test_join_upload_steps_carry_the_release_artifacts() -> None:
    uploads: list[tuple[Any, ...]] = []
    release = SimpleNamespace(
        executor_wheel_cm="wheel-cm",
        executor_wheel_sha="e" * 64,
        bundle_cm="bundle-cm",
        bundle_sha="b" * 64,
        config=SimpleNamespace(
            executor_wheel=Path("/artifacts/executor.whl"),
            bundle=Path("/artifacts/bundle.tar.gz"),
        ),
        _gpu=lambda _target: ["kubectl", "--context", "gpu-a"],
        _upload_config_map=lambda *args, **kwargs: uploads.append((args, kwargs)),
    )

    for step in STAGE.join_gpu_upload_steps(release, TARGET):
        step.action()

    assert uploads == [
        (
            (
                ["kubectl", "--context", "gpu-a"],
                "wheel-cm",
                "executor.whl",
                Path("/artifacts/executor.whl"),
                "e" * 64,
            ),
            {"compress": True},
        ),
        (
            (
                ["kubectl", "--context", "gpu-a"],
                "bundle-cm",
                "bundle.tar.gz",
                Path("/artifacts/bundle.tar.gz"),
                "b" * 64,
            ),
            {},
        ),
    ]
