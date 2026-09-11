"""BOOT-020 runner contracts from the 2026-09-07 review.

The executor-only stage accepted any data-plane change, said nothing about the
two-phase pin window, let a resume roll the CPU again unnoticed and accepted an
empty rollback plan. (The one-list-per-context snapshot read is covered by
``test_boot020_live_snapshot_reads_once_per_context_inside_one_snapshot`` in
``test_regional_acceptance_fixtures``.)
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_boot020_release_rolling as boot020
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder

EXECUTOR = boot020.EXECUTOR_DEPLOYMENT
PREVIOUS = "a" * 64
CANDIDATE = "b" * 64


class FakeBackend:
    """A live-shaped backend: pins, real Deployment names, global components."""

    def __init__(self) -> None:
        self.completed: set[str] = set()
        self.live = {
            "cpu_wheel": "cpu-v1",
            "runtime_profile_version": "profile-v1",
            "clusters": {
                "cluster-a": {
                    "wheel": "executor-v1",
                    "reconciler_wheel": "node-v1",
                    "bundle": "bundle-v1",
                }
            },
        }
        self.cpu_generations = {"gpu-fault-api-ha": 1, "gpu-fault-worker": 1}
        self.gpu_generations = {
            "cluster-a": {EXECUTOR: 1, "gpu-fault-completion-watcher": 1}
        }
        self.resume_rolls_cpu = False
        self.executor_rolls_watcher = False
        # Per-phase pin window a test wants the deploy result to carry instead
        # of the correct one; ``None`` drops the ``pins`` key altogether.
        self.pin_overrides: dict[str, dict[str, Any] | None] = {}
        # Per-scenario rollback plan replacing the correct one.
        self.plan_overrides: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[Any, ...]] = []

    def classify(self, scenario: str) -> dict[str, Any]:
        kind = (
            "NOOP" if scenario in self.completed else boot020.EXPECTED_KINDS[scenario]
        )
        return {"kind": kind, "changed": [] if kind == "NOOP" else [scenario]}

    def snapshot(self, scenario: str, *, live: bool = True) -> dict[str, Any]:
        return {
            "phase": "complete",
            "release_id": scenario,
            "live": copy.deepcopy(self.live),
            "cpu_generations": dict(self.cpu_generations),
            "gpu_generations": copy.deepcopy(self.gpu_generations),
            "next_deploy": self.classify(scenario),
        }

    def _with_pins(self, result: dict[str, Any], phase: str) -> dict[str, Any]:
        correct = {
            "rolled-back": {"required": PREVIOUS, "compatible": []},
            "staged": {"required": PREVIOUS, "compatible": [CANDIDATE]},
            "finalized": {"required": CANDIDATE, "compatible": []},
        }[phase] | {"candidate": CANDIDATE, "previous_required": PREVIOUS}
        pins = self.pin_overrides.get(phase, correct)
        if pins is not None:
            result["pins"] = pins
        return result

    def deploy(
        self,
        scenario: str,
        *,
        diff: dict[str, Any],
        fault_phase: str | None = None,
        resume: bool = False,
        auto_rollback: bool | None = None,
    ) -> dict[str, Any]:
        self.calls.append((scenario, fault_phase, resume, auto_rollback, diff["kind"]))
        plans = {
            "control_plane": {
                "clusters": {},
                "global_components": ["cpu_stage", "cpu_finalize"],
                "restores_data_plane": False,
                "needs_controller": False,
            },
            "executor": {
                "clusters": {"cluster-a": ["collector", "executor", "reconciler"]},
                "global_components": [],
                "restores_data_plane": True,
                "needs_controller": False,
            },
            "agent": {
                "clusters": {"cluster-a": ["reconciler", "agent"]},
                "global_components": [],
                "restores_data_plane": True,
                "needs_controller": True,
            },
            "full": {
                "clusters": {"cluster-a": ["executor", "reconciler", "agent"]},
                "global_components": ["cpu_stage"],
                "restores_data_plane": True,
                "needs_controller": True,
            },
        }
        if fault_phase and auto_rollback:
            return self._with_pins(
                {
                    "phase": "rolled-back",
                    "injected_failure": fault_phase,
                    "rollback_plan": self.plan_overrides.get(scenario, plans[scenario]),
                    "rollback_timing": {"t_safe_seconds": 30.0, "t_full_seconds": 45.0},
                    "operation_duration_seconds": 45.0,
                },
                "rolled-back",
            )
        if fault_phase:
            return self._with_pins(
                {
                    "phase": "failed",
                    "injected_failure": fault_phase,
                    "operation_duration_seconds": 1.0,
                },
                "staged",
            )
        if scenario == "control_plane":
            self.live["cpu_wheel"] = "cpu-v2"
            self.cpu_generations = {"gpu-fault-api-ha": 2, "gpu-fault-worker": 2}
        elif scenario == "executor":
            self.live["clusters"]["cluster-a"]["wheel"] = "executor-v2"
            self.gpu_generations["cluster-a"][EXECUTOR] = 2
            if self.executor_rolls_watcher:
                self.gpu_generations["cluster-a"]["gpu-fault-completion-watcher"] = 2
            if resume and self.resume_rolls_cpu:
                self.cpu_generations = {"gpu-fault-api-ha": 3, "gpu-fault-worker": 3}
        elif scenario == "agent":
            self.live["clusters"]["cluster-a"]["reconciler_wheel"] = "node-v2"
            self.gpu_generations["cluster-a"]["gpu-fault-completion-watcher"] = 3
        elif scenario == "full":
            self.live["cpu_wheel"] = "cpu-v3"
            self.live["clusters"]["cluster-a"]["wheel"] = "executor-v3"
        self.completed.add(scenario)
        return self._with_pins(
            {
                "phase": "complete",
                "injected_failure": None,
                "operation_duration_seconds": 10.0,
            },
            "finalized",
        )


def _recorder(tmp_path: Path) -> EvidenceRecorder:
    return EvidenceRecorder(
        tmp_path / "GF-REGIONAL-BOOT-020.json",
        case_id="GF-REGIONAL-BOOT-020",
        inputs={"configs": "test"},
    )


def test_live_shaped_backend_completes_all_stages(tmp_path: Path) -> None:
    result = boot020.run_release_rolling(FakeBackend(), _recorder(tmp_path))

    assert result["status"] == "COMPLETED"
    assert "executor_interrupted_snapshot" in result["stages"]


def test_executor_stage_rejects_a_resume_that_rolls_the_cpu_again(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    backend.resume_rolls_cpu = True

    with pytest.raises(boot020.AcceptanceCheckError, match="rolled the CPU"):
        boot020.run_release_rolling(backend, _recorder(tmp_path))


def test_executor_stage_rejects_non_executor_generation_changes(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.executor_rolls_watcher = True

    with pytest.raises(boot020.AcceptanceCheckError, match="non-executor Deployments"):
        boot020.run_release_rolling(backend, _recorder(tmp_path))


def _pins(required: str, compatible: list[str]) -> dict[str, Any]:
    return {
        "required": required,
        "compatible": compatible,
        "candidate": CANDIDATE,
        "previous_required": PREVIOUS,
    }


@pytest.mark.parametrize(
    ("phase", "pins", "message"),
    [
        ("rolled-back", _pins(PREVIOUS, [CANDIDATE]), "rollback kept the candidate"),
        ("staged", _pins(PREVIOUS, []), "staged window lacks"),
        ("finalized", _pins(PREVIOUS, []), "finalized required pin"),
        ("finalized", None, "carries no pins"),
    ],
)
def test_executor_stage_checks_the_pin_window_at_each_checkpoint(
    tmp_path: Path, phase: str, pins: dict[str, Any] | None, message: str
) -> None:
    """A deploy result whose pins do not match its phase fails the executor stage.

    The correct window at every checkpoint (previous required + empty after a
    rollback, previous required + candidate compatible while staged, candidate
    required + empty once finalized) is what the live-shaped backend returns,
    and ``test_live_shaped_backend_completes_all_stages`` proves it passes.
    """

    backend = FakeBackend()
    backend.pin_overrides[phase] = pins

    with pytest.raises(boot020.AcceptanceCheckError, match=message):
        boot020.run_release_rolling(backend, _recorder(tmp_path))


@pytest.mark.parametrize(
    ("scenario", "plan", "message"),
    [
        ("executor", {"clusters": {}, "global_components": []}, "names no components"),
        (
            "control_plane",
            {"clusters": {"c": ["executor"]}, "restores_data_plane": False},
            "no global components",
        ),
    ],
)
def test_rollback_scope_requires_components(
    tmp_path: Path, scenario: str, plan: dict[str, Any], message: str
) -> None:
    backend = FakeBackend()
    backend.plan_overrides[scenario] = plan

    with pytest.raises(boot020.AcceptanceCheckError, match=message):
        boot020.run_release_rolling(backend, _recorder(tmp_path))


def test_pin_window_reads_state_previous_and_the_live_metadata() -> None:
    release = SimpleNamespace(
        _config_map_data=lambda name: {
            boot020.REQUIRED_EXECUTOR_PIN: PREVIOUS,
            boot020.COMPATIBLE_EXECUTOR_PINS: f"{CANDIDATE}, {PREVIOUS}",
        },
        executor_wheel_sha=CANDIDATE,
    )
    state = {"previous": {"metadata": {boot020.REQUIRED_EXECUTOR_PIN: PREVIOUS}}}

    pins = boot020.LiveReleaseRollingBackend({})._pin_window(release, state)

    assert pins == {
        "required": PREVIOUS,
        "compatible": [CANDIDATE, PREVIOUS],
        "candidate": CANDIDATE,
        "previous_required": PREVIOUS,
    }
