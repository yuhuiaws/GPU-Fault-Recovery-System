from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module
from tests.regional._release_orchestrator_support import phase_release

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATION = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_orchestration.py"
)
GPU_ROLLOUT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_gpu_rollout.py"
)
PROGRESS = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_progress.py"
)


def test_upgrade_ensures_schema_before_rolling_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The database has to accept the candidate before any CPU Pod runs it.

    Rolling the CPU role first would put new code in front of an old schema,
    which is exactly the window a release must never open.
    """

    calls: list[str] = []
    # The preflight runs in the background now, so it is not part of the order
    # under test here; `test_release_orchestration_concurrency.py` owns that.
    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
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

    assert calls == ["upload", "schema", "cpu", "verify"]


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
    # The schema no longer waits behind the candidate node preflight, so the
    # write that carries its STARTED marker is stamped with the last phase that
    # really completed before it.
    assert schema_started["phase"] == "uploaded"
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


def test_candidate_host_preflight_covers_every_mutation_it_gates() -> None:
    """Each cluster is preflighted in dependency order before it is mutated.

    The preflight now runs beside the control-plane phases instead of in front of
    them (see `test_release_orchestration_concurrency.py` for the overlap and the
    join), so the schema here waits for the preflight to finish before it fails:
    the ordering under test is the one *inside* the preflight, and a race would
    otherwise decide it.
    """

    target = SimpleNamespace(cluster_id="gpu-a")
    calls = []
    preflighted = threading.Event()

    def ensure_schema():
        assert preflighted.wait(timeout=10), "the candidate preflight never ran"
        calls.append("schema")
        raise RuntimeError("stop after schema")

    def preflight_node_runtime(*_args, **_kwargs):
        calls.append("host-preflight")
        preflighted.set()

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
        _preflight_node_runtime=preflight_node_runtime,
        _ensure_schema=ensure_schema,
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
    """A resumed release re-proves the candidate on the nodes it will mutate.

    `candidate-preflight-ready` being recorded does not make it true any more:
    the fleet may have changed between the two attempts, and the data plane is
    told it was already preflighted. The schema waits for the preflight here for
    the same reason as in the test above -- the two now run concurrently.
    """

    target = SimpleNamespace(cluster_id="gpu-a")
    calls = []
    preflighted = threading.Event()

    def ensure_schema():
        assert preflighted.wait(timeout=10), "resume skipped the candidate preflight"
        calls.append("schema")
        raise RuntimeError("stop after schema")

    def preflight_node_runtime(*_args, **_kwargs):
        calls.append("host-preflight")
        preflighted.set()

    release = SimpleNamespace(
        state={"phase": "candidate-preflight-ready"},
        config=SimpleNamespace(clusters=(target,), agent_config_digest="config-a"),
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="a" * 64,
        _upload_release=lambda _diff: pytest.fail(
            "resume repeated the completed upload"
        ),
        _preflight_node_runtime=preflight_node_runtime,
        _ensure_schema=ensure_schema,
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


CPU_SIDE_PLAN = ("SCHEMA", "REGISTRY", "CPU_STAGE", "VERIFY")


def test_started_marker_shares_the_phase_checkpoint_write() -> None:
    """Each component's STARTED marker rides on the previous phase checkpoint.

    Every state write is a ConfigMap apply of the whole transaction (~2.5 s on
    the production control plane), and the pair "phase x complete" / "component y
    started" describes one instant. Writing them separately doubled the writes
    without recording anything a resume can tell apart.
    """

    calls: list[str] = []
    saves: list[dict] = []
    release = phase_release(calls, saves)
    plan = ORCHESTRATION.ReleaseExecutionPlan(
        nodes=tuple(
            getattr(ORCHESTRATION.ReleaseComponent, name) for name in CPU_SIDE_PLAN
        )
    )

    ORCHESTRATION.run_upgrade_phases(
        release,
        diff=ORCHESTRATION.ReleaseDiff(
            kind=ORCHESTRATION.ReleaseChangeKind.FULL,
            changed=frozenset({"database_schema", "control_plane_wheel"}),
        ),
        plan=plan,
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )

    # `uploaded`, one merged write per planned component, the write that opens
    # the data plane, and `complete`.
    assert len(saves) == 1 + len(CPU_SIDE_PLAN) + 2, [item["phase"] for item in saves]
    registry_started = next(
        item
        for item in saves
        if ((item.get("component_progress") or {}).get("global") or {})
        .get("registry", {})
        .get("status")
        == "STARTED"
    )
    assert registry_started["phase"] == "schema-ready"
    assert "schema-ready" in registry_started["completed_phases"]


def test_schema_ready_not_written_without_schema_component() -> None:
    """A plan without a schema change must not claim the schema is ready.

    The claim used to be written unconditionally, so a control-plane-only
    release recorded a phase it never ran, and every consumer of
    `completed_phases` had to treat that stamp as meaningless.
    """

    calls: list[str] = []
    saves: list[dict] = []
    release = phase_release(calls, saves)

    ORCHESTRATION.run_upgrade_phases(
        release,
        diff=ORCHESTRATION.ReleaseDiff(
            kind=ORCHESTRATION.ReleaseChangeKind.CONTROL_PLANE_ONLY,
            changed=frozenset({"control_plane_wheel"}),
        ),
        plan=ORCHESTRATION.ReleaseExecutionPlan(
            nodes=(
                ORCHESTRATION.ReleaseComponent.CPU_STAGE,
                ORCHESTRATION.ReleaseComponent.VERIFY,
            )
        ),
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )

    assert "schema" not in calls, "schema must not be loaded without a SCHEMA component"
    assert [item for item in saves if item["phase"] == "schema-ready"] == [], (
        "schema-ready must not be saved without a SCHEMA component"
    )
    assert all("schema-ready" not in item["completed_phases"] for item in saves), (
        "schema-ready must not be checkpointed without a SCHEMA component"
    )


def test_resume_without_schema_component_needs_no_schema_checkpoint() -> None:
    """A resumed release must not wait for a checkpoint nobody will write.

    Treating "the phase's component is not in the plan" as satisfied is what
    lets the phases after the schema run at all once the unconditional
    `schema-ready` checkpoint is gone.
    """

    calls: list[str] = []
    saves: list[dict] = []
    release = phase_release(calls, saves)

    ORCHESTRATION.run_upgrade_phases(
        release,
        diff=ORCHESTRATION.ReleaseDiff(
            kind=ORCHESTRATION.ReleaseChangeKind.CONTROL_PLANE_ONLY,
            changed=frozenset({"control_plane_wheel"}),
        ),
        plan=ORCHESTRATION.ReleaseExecutionPlan(
            nodes=(
                ORCHESTRATION.ReleaseComponent.CPU_STAGE,
                ORCHESTRATION.ReleaseComponent.VERIFY,
            )
        ),
        previous={},
        completed_phases={"uploaded", "candidate-preflight-ready"},
        completed_clusters=set(),
        registry_staged=False,
    )

    assert calls[0] == "cpu-stage"
    assert "barrier" in calls
    assert calls.index("cpu-stage") < calls.index("verify")
    assert release.state["phase"] == "complete"
    assert release.state["release_lifecycle"] == "COMMITTED"


def test_cluster_attempts_with_rejects_reserved_fields() -> None:
    """The fields parameter cannot smuggle validated control keys."""
    state = {"cluster_attempts": {}}
    for forbidden_key in ("state", "attempt_generation", "updated_at_epoch"):
        with pytest.raises(ValueError, match=f"reserved keys: \\['{forbidden_key}'\\]"):
            PROGRESS.cluster_attempts_with(
                state, "cluster-a", "PENDING", fields={forbidden_key: "smuggled"}
            )
    # Allowed fields go through fine.
    result = PROGRESS.cluster_attempts_with(
        state, "cluster-a", "PENDING", fields={"converged_at_epoch": 1234.5}
    )
    assert result["cluster-a"]["converged_at_epoch"] == 1234.5
