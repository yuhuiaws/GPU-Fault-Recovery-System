"""Phase contracts are checked against the real release and reject bad callers."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Unpack

import pytest

from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseConfig
from gpu_fault_release.regional_release_diff import (
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
    diff_from_changed,
)
from gpu_fault_release.regional_release_interfaces import (
    CpuRollout,
    GpuRollout,
    GpuRolloutOptions,
    Snapshot,
)
from gpu_fault_release.regional_release_orchestration import upgrade_gpu_clusters
from gpu_fault_release.regional_release_progress import take_phase_checkpoint
from tests.regional._release_orchestrator_support import config_file

ROOT = Path(__file__).resolve().parents[2]

TYPE_IMPORTS = """\
from typing import Any, assert_type
from gpu_fault_release.rollout import RegionalRelease
from gpu_fault_release.regional_release_config import ClusterTarget
from gpu_fault_release.regional_release_diff import ReleaseDiff, ReleaseExecutionPlan
from gpu_fault_release.regional_release_interfaces import (
    AgentNodeSet, Snapshot, CpuRollout, GpuRollout, RollbackTargetArguments,
    UpgradeOptions, RollbackCheckpoint,
)
from gpu_fault_release.regional_release_gpu_rollout import ProgressCallback
from gpu_fault_release.regional_release_orchestration import (
    inherit_superseded_previous, _apply_upgrade_cpu, upgrade_gpu_clusters,
)
from gpu_fault_release.regional_release_rollback_context import rollback_target_arguments
from gpu_fault_release.regional_release_rollback_target import rollback_target
"""

VALID_CALLERS = """\
def check(
    release: RegionalRelease,
    target: ClusterTarget,
    diff: ReleaseDiff,
    plan: ReleaseExecutionPlan,
    progress: ProgressCallback,
) -> None:
    snapshot: Snapshot = release
    cpu: CpuRollout = release
    gpu: GpuRollout = release
    previous = snapshot._capture_previous(plan)
    assert_type(previous, dict[str, Any])
    nodes = cpu._capture_active_agent_node_sets()
    assert_type(nodes, dict[str, AgentNodeSet])
    cpu._wait_candidate_cpu_agent_heartbeats(
        nodes, required_identity=cpu._candidate_agent_pin_identity()
    )
    cpu._apply_cpu(finalize=False, force_restart=True, diff=diff)
    gpu._upgrade_gpu_target(
        target, diff, plan, progress=progress, candidate_preflighted=True
    )
    release.upgrade()
    release.upgrade(resume=True, diff=diff, supersede=None)
    release.rollback()
    release.rollback(state=previous, automatic=True)
    arguments = rollback_target_arguments(
        release, previous=previous, metadata={}, artifact="artifact",
        config_digest="config", profile="profile", executor_artifact="executor",
        executor_compatibility="compatibility", runtime_image="image",
    )
    assert_type(arguments, RollbackTargetArguments)
    rollback_target(release, target, previous=previous, components=frozenset(), **arguments)
"""

INVALID_CALLERS = """\
def wrong_progress() -> None:
    pass

def invalid(
    release: RegionalRelease, snapshot: Snapshot, cpu: CpuRollout, gpu: GpuRollout,
    target: ClusterTarget, diff: ReleaseDiff, plan: ReleaseExecutionPlan,
    checkpoint: RollbackCheckpoint,
) -> None:
    release.upgrade(resume="yes")  # expect: arg-type
    release.upgrade(checkpoint="old")  # expect: call-arg
    release.rollback(automatic="yes")  # expect: arg-type
    snapshot._capture_previous(plan={})  # expect: arg-type
    cpu._apply_cpu(finalize="true")  # expect: arg-type
    gpu._upgrade_gpu_target(target, diff, plan, progress=wrong_progress)  # expect: arg-type
    gpu._upgrade_gpu_target(target, diff, candidate_preflighted="yes")  # expect: arg-type
    checkpoint("restored", timing_name=17)  # expect: arg-type
    options: UpgradeOptions = {"resume": "yes"}  # expect: typeddict-item
    nodes: AgentNodeSet = {"node_ids": "node-a"}  # expect: typeddict-item
    arguments: RollbackTargetArguments = {}  # expect: typeddict-item
    inherit_superseded_previous(object(), {})  # expect: arg-type
    _apply_upgrade_cpu(object(), diff=diff, previous={}, finalize=True)  # expect: arg-type
    upgrade_gpu_clusters(object(), diff=diff, plan=plan, previous={}, completed_phases=set(), completed_clusters=set(), registry_staged=False)  # expect: arg-type

def missing_contract(release: object) -> GpuRollout:
    return release  # expect: return-value
"""


@pytest.fixture(scope="module")
def type_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("release-types")
    results: dict[str, str] = {}
    for name, callers in (("valid", VALID_CALLERS), ("invalid", INVALID_CALLERS)):
        source = root / f"{name}.py"
        source.write_text(TYPE_IMPORTS + "\n" + callers, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--follow-imports=silent",
                "--cache-dir",
                str(root / "cache"),
                "--config-file",
                str(ROOT / "pyproject.toml"),
                str(source),
            ],
            cwd=ROOT,
            env={**os.environ, "MYPYPATH": str(ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode in {0, 1}, result.stdout + result.stderr
        assert result.returncode == (0 if name == "valid" else 1), (
            result.stdout + result.stderr
        )
        results[name] = result.stdout + result.stderr
    return results


def test_concrete_release_satisfies_all_phase_contracts(
    type_results: dict[str, str],
) -> None:
    assert "Success: no issues found" in type_results["valid"]


def test_wrong_phase_inputs_are_rejected_not_treated_as_any(
    type_results: dict[str, str],
) -> None:
    source = TYPE_IMPORTS + "\n" + INVALID_CALLERS
    expected = {
        (index, line.split("# expect: ", 1)[1])
        for index, line in enumerate(source.splitlines(), start=1)
        if "# expect: " in line
    }
    actual = {
        (int(line), code)
        for line, code in re.findall(
            r"invalid\.py:(\d+): error: .* \[([a-z-]+)\]", type_results["invalid"]
        )
    }

    assert actual == expected, type_results["invalid"]


@pytest.fixture
def release(tmp_path: Path) -> rollout.RegionalRelease:
    config = rollout.ReleaseConfig.load(config_file(tmp_path))
    return rollout.RegionalRelease(config, rollout.Runner(dry_run=True))


def test_real_release_exposes_the_narrow_protocol_operations(
    release: rollout.RegionalRelease,
) -> None:
    assert isinstance(release, Snapshot), (
        "the release must implement snapshot operations"
    )
    assert isinstance(release, CpuRollout), (
        "the release must implement CPU rollout operations"
    )
    assert isinstance(release, GpuRollout), (
        "the release must implement GPU rollout operations"
    )


def test_public_phase_delegates_preserve_supplied_inputs_and_defaults(
    release: rollout.RegionalRelease, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object, dict[str, Any]]] = []

    def upgrade(receiver: object, **options: Any) -> None:
        calls.append(("upgrade", receiver, options))

    def rollback(receiver: object, **options: Any) -> None:
        calls.append(("rollback", receiver, options))

    monkeypatch.setattr(rollout, "upgrade_release", upgrade)
    monkeypatch.setattr(rollout, "rollback_release", rollback)
    previous = {"phase": "failed"}
    diff = diff_from_changed({"cpu_worker_manifests"})

    release.upgrade()
    release.upgrade(resume=True, diff=diff, supersede=previous)
    release.rollback()
    release.rollback(state=previous, automatic=True)

    assert calls == [
        ("upgrade", release, {}),
        ("upgrade", release, {"resume": True, "diff": diff, "supersede": previous}),
        ("rollback", release, {}),
        ("rollback", release, {"state": previous, "automatic": True}),
    ]
    assert calls[1][2]["supersede"] is previous
    assert calls[3][2]["state"] is previous


@dataclass
class GpuPhaseRecorder:
    config: ReleaseConfig
    state: dict[str, Any] = field(default_factory=dict)
    visited: list[str] = field(default_factory=list)

    def _save_state(self, phase: str, **updates: object) -> None:
        self.state.update(phase=phase, **updates)

    def _upgrade_gpu_target(
        self,
        target: ClusterTarget,
        diff: ReleaseDiff,
        plan: ReleaseExecutionPlan | None = None,
        **options: Unpack[GpuRolloutOptions],
    ) -> None:
        assert options["candidate_preflighted"] is True
        progress = options.get("progress")
        assert progress is not None
        self.visited.append(target.cluster_id)
        progress(ReleaseComponent.EXECUTOR, "STARTED", None)
        progress(ReleaseComponent.EXECUTOR, "COMPLETED", None)


def test_gpu_phase_needs_only_its_contract_not_the_entire_release(
    release: rollout.RegionalRelease,
) -> None:
    config = replace(
        release.config,
        clusters=release.config.clusters[:1],
        upgrade_max_parallel_clusters=1,
    )
    recorder = GpuPhaseRecorder(config)
    completed: set[str] = set()
    diff = diff_from_changed({"executor_wheel"})
    plan = ReleaseExecutionPlan((ReleaseComponent.EXECUTOR,))

    upgrade_gpu_clusters(
        recorder,
        diff=diff,
        plan=plan,
        previous={},
        completed_phases=set(),
        completed_clusters=completed,
        registry_staged=False,
    )

    assert recorder.visited == [config.clusters[0].cluster_id]
    assert completed == set(recorder.visited)
    assert recorder.state["completed_cluster_ids"] == recorder.visited
    assert (
        recorder.state["cluster_attempts"][recorder.visited[0]]["state"] == "CONVERGED"
    )


@pytest.mark.parametrize(
    ("current", "pending_names", "stamp", "carried"),
    [
        ("cpu", ("schema",), "cpu", ["schema"]),
        ("schema", ("cpu", "preflight"), "cpu", ["preflight"]),
        ("cpu", ("cpu",), "cpu", []),
        ("cpu", (), "cpu", []),
        ("resuming", (), "resuming", []),
        ("", (), "preflight", []),
        ("", ("unranked",), "preflight", []),
    ],
)
def test_checkpoint_stamp_does_not_regress_and_only_carries_other_ranked_phases(
    current: str, pending_names: tuple[str, ...], stamp: str, carried: list[str]
) -> None:
    pending = {name: {name: True} for name in pending_names}

    actual_stamp, merged, actual_carried = take_phase_checkpoint(
        pending, current=current, order=("preflight", "schema", "cpu"), updates={}
    )

    assert actual_stamp == stamp, "the checkpoint must retain the furthest known phase"
    assert actual_carried == carried, (
        "carried narration must be ordered and exclude the stamp"
    )
    assert merged == {name: True for name in pending_names}, (
        "unranked pending metadata must not silently disappear"
    )
    assert pending == {}, "the checkpoint must consume the caller's pending updates"


def test_checkpoint_merge_preserves_insertion_order_and_explicit_update_precedence() -> (
    None
):
    nested = {"unchanged": ["evidence"]}
    pending = {
        "cpu": {"shared": "first", "pending_order": "first", "nested": nested},
        "schema": {"shared": "second", "pending_order": "second", "only_pending": True},
    }
    explicit = {"shared": "explicit", "component_progress": {"status": "STARTED"}}

    stamp, merged, carried = take_phase_checkpoint(
        pending,
        current="preflight",
        order=("preflight", "schema", "cpu"),
        updates=explicit,
    )

    assert stamp == "cpu" and carried == ["schema"], (
        "checkpoint ranking must be independent of metadata insertion order"
    )
    assert merged == {
        "shared": "explicit",
        "pending_order": "second",
        "nested": nested,
        "only_pending": True,
        "component_progress": {"status": "STARTED"},
    }, "explicit component progress must override pending values"
    assert merged["nested"] is nested, (
        "checkpoint assembly must not rewrite retained metadata"
    )
    assert explicit["shared"] == "explicit", "the explicit input must remain unchanged"
    assert pending == {}, (
        "consumed pending phases must not be replayed by a later checkpoint"
    )
