from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATION = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_orchestration.py"
)
GPU_ROLLOUT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_gpu_rollout.py"
)


def test_upgrade_ensures_schema_before_rolling_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The database has to accept the candidate before any CPU Pod runs it.

    Rolling the CPU role first would put new code in front of an old schema,
    which is exactly the window a release must never open.
    """

    calls: list[str] = []
    monkeypatch.setattr(
        ORCHESTRATION,
        "preflight_upgrade_mutations",
        lambda _self, _plan: calls.append("preflight"),
    )
    release: SimpleNamespace = SimpleNamespace(
        state={"phase": "preflight"},
        config=SimpleNamespace(clusters=(), upgrade_max_parallel_clusters=1),
        _upload_release=lambda _diff: calls.append("upload"),
        _ensure_schema=lambda: calls.append("schema"),
        _capture_active_agent_node_sets=lambda: {},
        _wait_candidate_cpu_agent_heartbeats=lambda _expected, **_kwargs: None,
        _apply_cpu=lambda **_kwargs: calls.append("cpu"),
        _validate_release_quick=lambda _plan: calls.append("verify"),
        _save_state=lambda phase, **updates: release.state.update(
            {"phase": phase, **updates}
        ),
    )

    ORCHESTRATION.run_upgrade_phases(
        release,
        diff=ORCHESTRATION.ReleaseDiff(
            kind=ORCHESTRATION.ReleaseChangeKind.FULL,
            changed=frozenset({"database_schema"}),
        ),
        plan=ORCHESTRATION.ReleaseExecutionPlan(
            nodes=(
                ORCHESTRATION.ReleaseComponent.SCHEMA,
                ORCHESTRATION.ReleaseComponent.CPU_STAGE,
                ORCHESTRATION.ReleaseComponent.VERIFY,
            )
        ),
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )

    assert calls == ["upload", "preflight", "schema", "cpu", "verify"]


def test_completed_component_progress_is_flushed_by_phase_checkpoint() -> None:
    saves = []
    release = SimpleNamespace(
        state={"phase": "preflight"},
        config=SimpleNamespace(clusters=()),
        _upload_release=lambda _diff: None,
        _ensure_schema=lambda: None,
        _validate_release_quick=lambda _plan: None,
        _save_state=lambda phase, **updates: (
            release.state.update({"phase": phase, **updates}),
            saves.append(json.loads(json.dumps(release.state))),
        ),
    )
    diff = ORCHESTRATION.ReleaseDiff(
        kind=ORCHESTRATION.ReleaseChangeKind.FULL,
        changed=frozenset({"database_schema"}),
    )
    plan = ORCHESTRATION.ReleaseExecutionPlan(
        nodes=(
            ORCHESTRATION.ReleaseComponent.SCHEMA,
            ORCHESTRATION.ReleaseComponent.VERIFY,
        )
    )

    ORCHESTRATION.run_upgrade_phases(
        release,
        diff=diff,
        plan=plan,
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )

    schema_started = next(
        item
        for item in saves
        if ((item.get("component_progress") or {}).get("global") or {})
        .get("schema", {})
        .get("status")
        == "STARTED"
    )
    schema_completed = next(
        item
        for item in saves
        if ((item.get("component_progress") or {}).get("global") or {})
        .get("schema", {})
        .get("status")
        == "COMPLETED"
    )
    assert schema_started["phase"] == "candidate-preflight-ready"
    assert schema_completed["phase"] == "schema-ready"


def test_same_gpu_deployment_wave_writes_one_started_checkpoint() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    saves = []
    release: SimpleNamespace

    def save_state(phase, **updates):
        release.state.update({"phase": phase, **updates})
        saves.append(json.loads(json.dumps(release.state)))

    def upgrade(active_target, diff, plan, *, progress, candidate_preflighted):
        assert candidate_preflighted is True
        GPU_ROLLOUT.upgrade_gpu_target(
            release,
            active_target,
            diff,
            plan,
            progress=progress,
            candidate_preflighted=candidate_preflighted,
        )

    release = SimpleNamespace(
        state={"phase": "data-plane-progress"},
        config=SimpleNamespace(clusters=(target,), upgrade_max_parallel_clusters=1),
        executor_wheel_cm="wheel",
        _apply_gpu_deployments=lambda *_args, **_kwargs: None,
        _save_state=save_state,
        _upgrade_gpu_target=upgrade,
    )
    diff = ORCHESTRATION.ReleaseDiff(
        kind=ORCHESTRATION.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"executor_wheel"}),
    )
    components = (
        ORCHESTRATION.ReleaseComponent.EXECUTOR,
        ORCHESTRATION.ReleaseComponent.WATCHER,
        ORCHESTRATION.ReleaseComponent.COLLECTOR,
    )
    plan = ORCHESTRATION.ReleaseExecutionPlan(nodes=components)

    ORCHESTRATION.upgrade_gpu_clusters(
        release,
        diff=diff,
        plan=plan,
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )

    started = [
        item
        for item in saves
        if {
            name: entry.get("status")
            for name, entry in (
                ((item.get("component_progress") or {}).get("clusters") or {}).get(
                    "gpu-a", {}
                )
            ).items()
        }
        == {"executor": "STARTED", "watcher": "STARTED", "collector": "STARTED"}
    ]
    partial_started = [
        item
        for item in saves
        if 0
        < sum(
            entry.get("status") == "STARTED"
            for entry in (
                ((item.get("component_progress") or {}).get("clusters") or {}).get(
                    "gpu-a", {}
                )
            ).values()
        )
        < 3
    ]

    assert len(started) == 1
    assert partial_started == []


def test_agent_progress_starts_only_after_node_barrier() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    events = []
    config = SimpleNamespace(agent_config_digest="config-a")
    plan = ORCHESTRATION.ReleaseExecutionPlan(
        nodes=(
            ORCHESTRATION.ReleaseComponent.RECONCILER,
            ORCHESTRATION.ReleaseComponent.AGENT,
        )
    )
    diff = ORCHESTRATION.ReleaseDiff(
        kind=ORCHESTRATION.ReleaseChangeKind.FULL,
        changed=frozenset({"node_runtime_wheel"}),
    )

    def blocked_rollout(*_args, **_kwargs):
        raise RuntimeError("pre-node barrier rejected")

    blocked = SimpleNamespace(
        config=config,
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="a" * 64,
        _roll_node_runtime=blocked_rollout,
    )
    with pytest.raises(RuntimeError, match="pre-node barrier rejected"):
        GPU_ROLLOUT.upgrade_gpu_target(
            blocked,
            target,
            diff,
            plan,
            progress=lambda selection, status, details: events.append(
                (selection, status, details)
            ),
        )
    assert events == []

    def completed_rollout(*_args, mutation_started, **_kwargs):
        mutation_started()

    completed = SimpleNamespace(
        config=config,
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="a" * 64,
        _roll_node_runtime=completed_rollout,
    )
    GPU_ROLLOUT.upgrade_gpu_target(
        completed,
        target,
        diff,
        plan,
        progress=lambda selection, status, details: events.append(
            (selection, status, details)
        ),
    )

    assert [status for _selection, status, _details in events] == [
        "STARTED",
        "COMPLETED",
    ]


def test_candidate_host_preflight_runs_before_schema_and_cpu() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    calls = []
    release = SimpleNamespace(
        state={"phase": "preflight"},
        config=SimpleNamespace(clusters=(target,), agent_config_digest="config-a"),
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="a" * 64,
        _upload_release=lambda _diff: calls.append("upload"),
        _preflight_gpu_dcgm_exporter=lambda _target: calls.append("dcgm-preflight"),
        _preflight_gpu_deployments=lambda *_args, **_kwargs: calls.append(
            "gpu-preflight"
        ),
        _preflight_node_runtime=lambda *_args, **_kwargs: calls.append(
            "host-preflight"
        ),
        _ensure_schema=lambda: (
            calls.append("schema"),
            (_ for _ in ()).throw(RuntimeError("stop after schema")),
        ),
        _save_state=lambda phase, **updates: release.state.update(
            {"phase": phase, **updates}
        ),
    )
    diff = ORCHESTRATION.ReleaseDiff(
        kind=ORCHESTRATION.ReleaseChangeKind.FULL,
        changed=frozenset({"database_schema", "node_runtime_wheel"}),
    )
    plan = ORCHESTRATION.ReleaseExecutionPlan(
        nodes=(
            ORCHESTRATION.ReleaseComponent.SCHEMA,
            ORCHESTRATION.ReleaseComponent.DCGM,
            ORCHESTRATION.ReleaseComponent.EXECUTOR,
            ORCHESTRATION.ReleaseComponent.RECONCILER,
            ORCHESTRATION.ReleaseComponent.AGENT,
        )
    )

    with pytest.raises(RuntimeError, match="stop after schema"):
        ORCHESTRATION.run_upgrade_phases(
            release,
            diff=diff,
            plan=plan,
            previous={},
            completed_phases=set(),
            completed_clusters=set(),
            registry_staged=False,
        )

    assert calls == [
        "upload",
        "dcgm-preflight",
        "gpu-preflight",
        "host-preflight",
        "schema",
    ]


def test_resume_revalidates_candidate_preflight_before_pending_mutations() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    calls = []
    release = SimpleNamespace(
        state={"phase": "candidate-preflight-ready"},
        config=SimpleNamespace(clusters=(target,), agent_config_digest="config-a"),
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="a" * 64,
        _upload_release=lambda _diff: pytest.fail(
            "resume repeated the completed upload"
        ),
        _preflight_node_runtime=lambda *_args, **_kwargs: calls.append(
            "host-preflight"
        ),
        _ensure_schema=lambda: (
            calls.append("schema"),
            (_ for _ in ()).throw(RuntimeError("stop after schema")),
        ),
        _save_state=lambda phase, **updates: release.state.update(
            {"phase": phase, **updates}
        ),
    )
    diff = ORCHESTRATION.ReleaseDiff(
        kind=ORCHESTRATION.ReleaseChangeKind.FULL,
        changed=frozenset({"database_schema", "node_runtime_wheel"}),
    )
    plan = ORCHESTRATION.ReleaseExecutionPlan(
        nodes=(
            ORCHESTRATION.ReleaseComponent.SCHEMA,
            ORCHESTRATION.ReleaseComponent.AGENT,
        )
    )

    with pytest.raises(RuntimeError, match="stop after schema"):
        ORCHESTRATION.run_upgrade_phases(
            release,
            diff=diff,
            plan=plan,
            previous={},
            completed_phases={"uploaded", "candidate-preflight-ready"},
            completed_clusters=set(),
            registry_staged=False,
        )

    assert calls == ["host-preflight", "schema"]
