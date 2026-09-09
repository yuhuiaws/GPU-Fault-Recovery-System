from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault_release import regional_release_agent_convergence as CONVERGENCE_MODULE
from gpu_fault_release import regional_release_diff as DIFF_MODULE
from gpu_fault_release import regional_release_fleet_rollout as FLEET_MODULE
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION_MODULE
from tests.regional._release_orchestrator_support import (
    DNS_MODULE,
    REGION,
    config_file,
    phase_release,
)
from tests.regional._release_orchestrator_support import RELEASE_MODULE as MODULE


def test_gpu_upgrade_is_serial_and_persists_per_cluster_attempts() -> None:
    targets = tuple(
        SimpleNamespace(cluster_id=cluster_id)
        for cluster_id in ("gpu-a", "gpu-b", "gpu-c")
    )
    calls = []
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=targets, upgrade_max_parallel_clusters=1),
        state={
            "cluster_attempts": {
                target.cluster_id: {"state": "PENDING", "attempt_generation": 1}
                for target in targets
            }
        },
    )

    def save_state(phase, **updates):
        release.state.update({"phase": phase, **updates})

    def upgrade(target, _diff, _plan, *, progress, candidate_preflighted):
        assert candidate_preflighted is True
        calls.append(target.cluster_id)
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "STARTED", None)
        if target.cluster_id == "gpu-b":
            raise ORCHESTRATION_MODULE.ClusterLocalReleaseError("capacity")
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "COMPLETED", None)

    setattr(release, "_save_state", save_state)
    setattr(release, "_upgrade_gpu_target", upgrade)
    diff = ORCHESTRATION_MODULE.ReleaseDiff(
        kind=ORCHESTRATION_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"executor_wheel"}),
    )
    plan = ORCHESTRATION_MODULE.ReleaseExecutionPlan(
        nodes=(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR,)
    )
    completed = set()

    with pytest.raises(
        ORCHESTRATION_MODULE.PartialClusterRolloutError, match="gpu-b rollout failed"
    ):
        ORCHESTRATION_MODULE.upgrade_gpu_clusters(
            release,
            diff=diff,
            plan=plan,
            previous={},
            completed_phases=set(),
            completed_clusters=completed,
            registry_staged=False,
        )

    assert calls == ["gpu-a", "gpu-b"]
    assert completed == {"gpu-a"}
    assert release.state["release_lifecycle"] == "PAUSED"
    assert release.state["cluster_attempts"]["gpu-a"]["state"] == "CONVERGED"
    assert release.state["cluster_attempts"]["gpu-b"]["state"] == "FAILED"
    assert release.state["cluster_attempts"]["gpu-c"]["state"] == "PENDING"


def test_converged_cluster_records_when_it_stopped_being_restarted() -> None:
    """A converged cluster records the epoch, and does not claim more than that.

    `converged_at_epoch` is the only thing that tells a later reader how long ago
    this release restarted the data plane, which is what bounds the window where
    restart-shaped alerts are excused. `identity_verified` stays false because
    nothing here has proven the full candidate pin: per-cluster convergence
    compares the artifact and config digests, not the protocol version or the
    compatibility digest, so the pre-finalize heartbeat barrier is still the only
    evidence that may stand in for itself.
    """

    target = SimpleNamespace(cluster_id="gpu-a")
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(target,), upgrade_max_parallel_clusters=1),
        state={},
    )
    setattr(
        release,
        "_save_state",
        lambda phase, **updates: release.state.update({"phase": phase, **updates}),
    )
    setattr(
        release,
        "_upgrade_gpu_target",
        lambda _target, _diff, _plan, *, progress, candidate_preflighted: progress(
            ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "COMPLETED", None
        ),
    )
    before = time.time()

    ORCHESTRATION_MODULE.upgrade_gpu_clusters(
        release,
        diff=ORCHESTRATION_MODULE.ReleaseDiff(
            kind=ORCHESTRATION_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
            changed=frozenset({"executor_wheel"}),
        ),
        plan=ORCHESTRATION_MODULE.ReleaseExecutionPlan(
            nodes=(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR,)
        ),
        previous={},
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )

    attempt = release.state["cluster_attempts"]["gpu-a"]
    assert attempt["state"] == "CONVERGED"
    assert before <= float(attempt["converged_at_epoch"]) <= time.time()
    assert attempt["identity_verified"] is False


def test_gpu_upgrade_honors_bounded_cluster_parallelism() -> None:
    # Five clusters: the canary rolls alone (see
    # `test_parallel_clusters_start_after_first_converges`) and the four that
    # follow it pair up twice, which is what the barrier below counts.
    targets = tuple(
        SimpleNamespace(cluster_id=cluster_id)
        for cluster_id in ("gpu-a", "gpu-b", "gpu-c", "gpu-d", "gpu-e")
    )
    barrier = threading.Barrier(2, timeout=30)
    concurrent = []
    active = []
    active_lock = threading.Lock()
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=targets, upgrade_max_parallel_clusters=2),
        state={},
    )

    def save_state(phase, **updates):
        release.state.update({"phase": phase, **updates})

    def upgrade(target, _diff, _plan, *, progress, candidate_preflighted):
        with active_lock:
            active.append(target.cluster_id)
            concurrent.append(len(active))
        # Two clusters must genuinely overlap, so each one waits for a partner;
        # a serial implementation would time out here instead of pairing up. The
        # canary has no partner by design.
        if target.cluster_id != "gpu-a":
            barrier.wait()
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "STARTED", None)
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "COMPLETED", None)
        with active_lock:
            active.remove(target.cluster_id)

    setattr(release, "_save_state", save_state)
    setattr(release, "_upgrade_gpu_target", upgrade)
    diff = ORCHESTRATION_MODULE.ReleaseDiff(
        kind=ORCHESTRATION_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"executor_wheel"}),
    )
    plan = ORCHESTRATION_MODULE.ReleaseExecutionPlan(
        nodes=(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR,)
    )
    completed = set()

    ORCHESTRATION_MODULE.upgrade_gpu_clusters(
        release,
        diff=diff,
        plan=plan,
        previous={},
        completed_phases=set(),
        completed_clusters=completed,
        registry_staged=False,
    )

    assert completed == {"gpu-a", "gpu-b", "gpu-c", "gpu-d", "gpu-e"}
    assert max(concurrent) == 2
    assert sorted(release.state["completed_cluster_ids"]) == [
        "gpu-a",
        "gpu-b",
        "gpu-c",
        "gpu-d",
        "gpu-e",
    ]
    attempts = release.state["cluster_attempts"]
    assert {cluster_id: entry["state"] for cluster_id, entry in attempts.items()} == {
        target.cluster_id: "CONVERGED" for target in targets
    }
    progress = release.state["component_progress"]["clusters"]
    assert sorted(progress) == ["gpu-a", "gpu-b", "gpu-c", "gpu-d", "gpu-e"]
    assert {entry["executor"]["status"] for entry in progress.values()} == {"COMPLETED"}


def test_parallel_clusters_start_after_first_converges() -> None:
    """With parallelism on, the first cluster still rolls alone.

    A failure the engine cannot attribute to one cluster rolls back every cluster
    that already converged, so the widest blast radius is bought by starting
    every cluster at once. The first pending cluster is the canary: it proves the
    candidate against a real cluster before the rest may overlap.
    """

    targets = tuple(
        SimpleNamespace(cluster_id=cluster_id)
        for cluster_id in ("gpu-a", "gpu-b", "gpu-c")
    )
    # Only the clusters after the canary pair up, so the canary must not join
    # this barrier -- and the two that follow it must.
    paired = threading.Barrier(2, timeout=30)
    others_started = threading.Event()
    order: list[str] = []
    order_lock = threading.Lock()
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=targets, upgrade_max_parallel_clusters=3),
        state={},
    )

    def save_state(phase, **updates):
        release.state.update({"phase": phase, **updates})

    def upgrade(target, _diff, _plan, *, progress, candidate_preflighted):
        with order_lock:
            order.append(target.cluster_id)
        if target.cluster_id == "gpu-a":
            # Long enough that a rollout which starts everything at once is
            # caught by the assertion below instead of racing past it.
            others_started.wait(timeout=0.2)
        else:
            attempts = release.state.get("cluster_attempts") or {}
            assert attempts.get("gpu-a", {}).get("state") == "CONVERGED", (
                "a cluster started before the canary converged"
            )
            others_started.set()
            paired.wait()
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "STARTED", None)
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "COMPLETED", None)

    setattr(release, "_save_state", save_state)
    setattr(release, "_upgrade_gpu_target", upgrade)
    diff = ORCHESTRATION_MODULE.ReleaseDiff(
        kind=ORCHESTRATION_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"executor_wheel"}),
    )
    plan = ORCHESTRATION_MODULE.ReleaseExecutionPlan(
        nodes=(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR,)
    )
    completed: set[str] = set()

    ORCHESTRATION_MODULE.upgrade_gpu_clusters(
        release,
        diff=diff,
        plan=plan,
        previous={},
        completed_phases=set(),
        completed_clusters=completed,
        registry_staged=False,
    )

    assert order[0] == "gpu-a"
    assert completed == {"gpu-a", "gpu-b", "gpu-c"}


def test_parallel_gpu_upgrade_failure_stops_unstarted_clusters() -> None:
    targets = tuple(
        SimpleNamespace(cluster_id=cluster_id)
        for cluster_id in ("gpu-a", "gpu-b", "gpu-c", "gpu-d")
    )
    started = []
    started_lock = threading.Lock()
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=targets, upgrade_max_parallel_clusters=2),
        state={},
    )

    def save_state(phase, **updates):
        release.state.update({"phase": phase, **updates})

    def upgrade(target, _diff, _plan, *, progress, candidate_preflighted):
        with started_lock:
            started.append(target.cluster_id)
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "STARTED", None)
        raise ORCHESTRATION_MODULE.ClusterLocalReleaseError(f"{target.cluster_id} sick")

    setattr(release, "_save_state", save_state)
    setattr(release, "_upgrade_gpu_target", upgrade)
    diff = ORCHESTRATION_MODULE.ReleaseDiff(
        kind=ORCHESTRATION_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"executor_wheel"}),
    )
    plan = ORCHESTRATION_MODULE.ReleaseExecutionPlan(
        nodes=(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR,)
    )

    with pytest.raises(ORCHESTRATION_MODULE.PartialClusterRolloutError) as failure:
        ORCHESTRATION_MODULE.upgrade_gpu_clusters(
            release,
            diff=diff,
            plan=plan,
            previous={},
            completed_phases=set(),
            completed_clusters=set(),
            registry_staged=False,
        )

    # The first failure closes the gate, so at most the clusters already in
    # flight are touched and every cluster that never started is reported.
    assert len(started) <= 2
    failed_cluster_ids = release.state["failed_cluster_ids"]
    not_started_cluster_ids = release.state["not_started_cluster_ids"]
    assert sorted(failed_cluster_ids) == sorted(started)
    assert set(failed_cluster_ids).isdisjoint(not_started_cluster_ids), (
        "a cluster was reported as both failed and never started"
    )
    assert set(failed_cluster_ids) | set(not_started_cluster_ids) == {
        target.cluster_id for target in targets
    }
    assert release.state["failure_scope"] == "cluster-local"
    assert release.state["release_lifecycle"] == "PAUSED"
    assert release.state["phase"] == "data-plane-paused"
    attempts = release.state["cluster_attempts"]
    assert {attempts[cluster_id]["state"] for cluster_id in failed_cluster_ids} == {
        "FAILED"
    }
    assert f"not_started={not_started_cluster_ids}" in str(failure.value)
    if len(failed_cluster_ids) > 1:
        assert "also failed: " + ", ".join(failed_cluster_ids[1:]) in str(failure.value)


def test_cluster_local_upgrade_failure_pauses_without_auto_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(auto_rollback=True, clusters=()),
        state={},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _apply_rds_ca_bundle=lambda: None,
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
        _remote_commands_are_idle=lambda: True,
        rollback=lambda **_kwargs: pytest.fail(
            "cluster-local failure triggered automatic rollback"
        ),
    )

    def save_state(phase, **updates):
        release.state.update({"phase": phase, **updates})

    setattr(release, "_save_state", save_state)
    previous = {"metadata": {}}
    monkeypatch.setattr(
        ORCHESTRATION_MODULE,
        "_validate_upgrade_transaction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ORCHESTRATION_MODULE,
        "_upgrade_context",
        lambda *_args, **_kwargs: (previous, set(), set(), False),
    )
    monkeypatch.setattr(
        ORCHESTRATION_MODULE,
        "run_upgrade_phases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ORCHESTRATION_MODULE.PartialClusterRolloutError("gpu-a paused")
        ),
    )
    diff = ORCHESTRATION_MODULE.ReleaseDiff(
        kind=ORCHESTRATION_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"executor_wheel"}),
    )

    with pytest.raises(
        ORCHESTRATION_MODULE.PartialClusterRolloutError, match="gpu-a paused"
    ):
        ORCHESTRATION_MODULE.upgrade_release(release, diff=diff)

    assert release.state["phase"] == "partial-convergence"
    assert release.state["partial_convergence"] is True
    assert release.state["release_lifecycle"] == "PAUSED"


def test_agent_convergence_timeout_pauses_only_its_own_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cluster whose fleet never converges must not revert the others.

    A convergence timeout is the most common way a rollout stops: one node's
    installer is stuck, or a node is cordoned. It says nothing about the
    candidate, so treating it as a global failure rolled back every cluster that
    had already converged -- an outage bought for a single stuck node. The real
    `wait_agents` timeout is driven here, so the test is bound to the exception
    the engine actually raises rather than to a hand-made one.
    """

    calls: list[str] = []
    targets = tuple(
        SimpleNamespace(cluster_id=cluster_id, hyperpod_cluster_name=f"hp-{cluster_id}")
        for cluster_id in ("gpu-a", "gpu-b")
    )

    def wait_for_agents(target) -> None:
        CONVERGENCE_MODULE.wait_agents(
            SimpleNamespace(
                runner=SimpleNamespace(dry_run=False),
                bundle_sha="b" * 64,
                node_template_sha="e" * 64,
                config=SimpleNamespace(
                    agent_config_digest="c" * 64,
                    runtime_profile_version="profile-v1",
                    release_manifest_schema_version=3,
                ),
                _agent_heartbeats_converged=lambda *_args, **_kwargs: False,
            ),
            target,
            "a" * 64,
            node_names=("node-b",),
            # The wait is over before it starts: this test is about what the
            # expiry raises, not about how long it waits.
            timeout_seconds=0,
        )

    def upgrade_target(target, _diff, _plan, *, progress, candidate_preflighted):
        calls.append(f"gpu:{target.cluster_id}")
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "STARTED", None)
        if target.cluster_id == "gpu-b":
            wait_for_agents(target)
        progress(ORCHESTRATION_MODULE.ReleaseComponent.EXECUTOR, "COMPLETED", None)

    release = phase_release(
        calls,
        clusters=targets,
        _upgrade_gpu_target=upgrade_target,
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
        rollback=lambda **_kwargs: pytest.fail(
            "a single cluster's convergence timeout rolled the release back"
        ),
    )
    release.config.auto_rollback = True
    monkeypatch.setattr(
        ORCHESTRATION_MODULE, "preflight_upgrade_mutations", lambda *_args: None
    )
    monkeypatch.setattr(
        ORCHESTRATION_MODULE,
        "_validate_upgrade_transaction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ORCHESTRATION_MODULE,
        "_upgrade_context",
        lambda *_args, **_kwargs: ({"metadata": {}}, set(), set(), False),
    )

    with pytest.raises(
        ORCHESTRATION_MODULE.PartialClusterRolloutError,
        match="gpu-b agents did not converge",
    ):
        ORCHESTRATION_MODULE.upgrade_release(
            release,
            diff=ORCHESTRATION_MODULE.ReleaseDiff(
                kind=ORCHESTRATION_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
                changed=frozenset({"executor_wheel"}),
            ),
        )

    assert calls.count("gpu:gpu-a") == 1
    attempts = release.state["cluster_attempts"]
    assert attempts["gpu-a"]["state"] == "CONVERGED"
    assert attempts["gpu-b"]["state"] == "FAILED"
    assert release.state["failure_scope"] == "cluster-local"
    assert release.state["release_lifecycle"] == "PAUSED"
    assert release.state["partial_convergence"] is True


def test_control_plane_only_upgrade_skips_schema_and_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    calls = []
    monkeypatch.setattr(release, "_ensure_contexts", lambda: None)
    monkeypatch.setattr(release, "_require_cpu_secrets", lambda: None)
    monkeypatch.setattr(release, "_apply_rds_ca_bundle", lambda: None)
    monkeypatch.setattr(release, "_refresh_aurora_credentials", lambda: None)
    monkeypatch.setattr(
        release, "_require_no_inflight_installs", lambda **_kwargs: None
    )
    monkeypatch.setattr(release, "_remote_commands_are_idle", lambda: True)
    monkeypatch.setattr(
        release, "_capture_previous", lambda **_kwargs: {"metadata": {}}
    )
    monkeypatch.setattr(release, "_backup_release_secrets", lambda: {})
    monkeypatch.setattr(
        release, "_save_state", lambda phase, **_updates: calls.append(phase)
    )
    monkeypatch.setattr(
        release,
        "_upload_release",
        lambda diff: calls.append(("upload", diff.kind.value)),
    )
    monkeypatch.setattr(
        release,
        "_ensure_schema",
        lambda: pytest.fail("control-only upgrade ran schema"),
    )
    monkeypatch.setattr(
        release,
        "_apply_cpu",
        lambda *, finalize, **_kwargs: calls.append(("cpu", finalize)),
    )
    monkeypatch.setattr(
        release,
        "_capture_active_agent_node_sets",
        lambda: {"gpu-a": {"node_ids": ["node-a"]}},
    )
    monkeypatch.setattr(
        release,
        "_wait_candidate_cpu_agent_heartbeats",
        lambda identities, **kwargs: calls.append(
            ("agent-heartbeats", identities, bool(kwargs.get("required_identity")))
        ),
    )
    monkeypatch.setattr(release, "_stage_registry", lambda: False)
    monkeypatch.setattr(
        release,
        "_upgrade_gpu_target",
        lambda *_args: pytest.fail("control-only upgrade touched GPU"),
    )
    monkeypatch.setattr(
        release, "_validate_release_quick", lambda _plan: calls.append("verify")
    )
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    release.upgrade(diff=diff)

    assert ("cpu", True) in calls
    assert ("cpu", False) not in calls
    # A plan with no schema change writes no `schema-ready` checkpoint at all:
    # claiming a phase this release never ran is what made `completed_phases`
    # unreadable for the gates that consume it.
    assert "schema-ready" not in calls
    assert "verify" in calls


def test_finalize_proves_fleet_pin_before_closing_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin barrier has to run before the promotion, not after.

    Promoting the pin closes the compatibility window irreversibly: rollback
    cannot reopen it and a staged resume no longer matches the live window. When
    the barrier ran afterwards, a fleet that had not reached the candidate left
    the transaction wedged between the two. Ordering is the whole fix, so it is
    asserted on the interleaving rather than on the calls in isolation.
    """
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    calls: list[object] = []
    monkeypatch.setattr(release, "_ensure_contexts", lambda: None)
    monkeypatch.setattr(release, "_require_cpu_secrets", lambda: None)
    monkeypatch.setattr(release, "_remote_commands_are_idle", lambda: True)
    monkeypatch.setattr(
        release, "_capture_previous", lambda **_kwargs: {"metadata": {}}
    )
    monkeypatch.setattr(release, "_backup_release_secrets", lambda: {})
    monkeypatch.setattr(release, "_save_state", lambda _phase, **_updates: None)
    monkeypatch.setattr(release, "_upload_release", lambda _diff: None)
    monkeypatch.setattr(release, "_apply_rds_ca_bundle", lambda: None)
    monkeypatch.setattr(release, "_refresh_aurora_credentials", lambda: None)
    monkeypatch.setattr(
        release, "_require_no_inflight_installs", lambda **_kwargs: None
    )
    monkeypatch.setattr(release, "_ensure_schema", lambda: None)
    monkeypatch.setattr(release, "_stage_registry", lambda: False)
    monkeypatch.setattr(release, "_validate_release_quick", lambda _plan: None)
    monkeypatch.setattr(
        release,
        "_capture_active_agent_node_sets",
        lambda: {"gpu-a": {"node_ids": ["node-a"]}},
    )
    monkeypatch.setattr(
        release,
        "_apply_cpu",
        lambda *, finalize, **_kwargs: calls.append(("cpu", finalize)),
    )
    monkeypatch.setattr(
        release,
        "_wait_candidate_cpu_agent_heartbeats",
        lambda _identities, **kwargs: calls.append(
            ("barrier", kwargs.get("required_identity"))
        ),
    )
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    release.upgrade(diff=diff)

    promotion = calls.index(("cpu", True))
    pin_barriers = [
        index
        for index, call in enumerate(calls)
        if isinstance(call, tuple) and call[0] == "barrier" and call[1] is not None
    ]
    assert pin_barriers, "finalize never proved the fleet reached the candidate pin"
    assert min(pin_barriers) < promotion, (
        "the pin barrier ran after the window closed, which is the ordering the "
        f"fix is about: {calls}"
    )
    assert calls[pin_barriers[0]][1] == FLEET_MODULE.candidate_agent_pin_identity(
        release
    ), "the barrier that gates the promotion waited on some other identity"
    # The liveness barrier after the cutover is still wanted; it just is not the
    # one that guards the irreversible step.
    assert any(
        isinstance(call, tuple) and call[0] == "barrier" and call[1] is None
        for call in calls[promotion:]
    ), f"finalize no longer checks the fleet survived the cutover: {calls}"


def test_new_upgrade_discards_stale_rollback_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    release.state = {
        "release_id": "previous-release",
        "rollback_completed_phases": [
            "rollback-controller-staged",
            "rollback-data-restored",
            "rollback-cpu-restored",
            "rollback-verified",
        ],
        "rollback_completed_cluster_ids": ["gpu-a"],
        "rollback_result": {"status": "PASSED"},
    }
    saves: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(release, "_ensure_contexts", lambda: None)
    monkeypatch.setattr(release, "_require_cpu_secrets", lambda: None)
    monkeypatch.setattr(release, "_remote_commands_are_idle", lambda: True)
    monkeypatch.setattr(
        release, "_capture_previous", lambda **_kwargs: {"metadata": {}}
    )
    monkeypatch.setattr(release, "_backup_release_secrets", lambda: {})
    monkeypatch.setattr(
        release,
        "_save_state",
        lambda phase, **_updates: saves.append((phase, dict(release.state))),
    )
    monkeypatch.setattr(release, "_upload_release", lambda _diff: None)
    monkeypatch.setattr(release, "_apply_rds_ca_bundle", lambda: None)
    monkeypatch.setattr(release, "_refresh_aurora_credentials", lambda: None)
    monkeypatch.setattr(
        release, "_require_no_inflight_installs", lambda **_kwargs: None
    )
    monkeypatch.setattr(release, "_ensure_schema", lambda: None)
    monkeypatch.setattr(release, "_apply_cpu", lambda **_kwargs: None)
    monkeypatch.setattr(
        release,
        "_capture_active_agent_node_sets",
        lambda: {"gpu-a": {"node_ids": ["node-a"]}},
    )
    monkeypatch.setattr(
        release,
        "_wait_candidate_cpu_agent_heartbeats",
        lambda _identities, **_kwargs: None,
    )
    monkeypatch.setattr(release, "_stage_registry", lambda: False)
    monkeypatch.setattr(release, "_validate_release_quick", lambda _plan: None)
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.CONTROL_PLANE_ONLY,
        changed=frozenset({"control_plane_wheel"}),
    )

    release.upgrade(diff=diff)

    assert saves[0] == ("preflight", {})


def test_endpoint_only_upgrade_skips_executor_and_node_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    calls = []
    monkeypatch.setattr(
        release, "_ensure_connection_secret", lambda _target: calls.append("secret")
    )
    monkeypatch.setattr(
        release,
        "_verify_gpu_control_plane_endpoint",
        lambda _target: calls.append("endpoint"),
    )
    monkeypatch.setattr(
        release, "_apply_gpu_dcgm_exporter", lambda _target: calls.append("dcgm")
    )
    monkeypatch.setattr(
        release, "_apply_gpu_deployments", lambda *_args: calls.append("executor")
    )
    monkeypatch.setattr(
        release, "_deploy_reconciler", lambda *_args, **_kwargs: calls.append("node")
    )
    diff = DIFF_MODULE.ReleaseDiff(
        kind=DIFF_MODULE.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
        changed=frozenset({"endpoint"}),
    )

    MODULE.upgrade_gpu_target(release, target, diff)

    assert calls == ["secret", "endpoint"]


def bootstrap_recorder() -> tuple[SimpleNamespace, list[str]]:
    """A GPU bootstrap whose only effect is the order of its own steps."""

    calls: list[str] = []
    release = SimpleNamespace(
        executor_wheel_cm="wheel",
        bundle_cm="bundle",
        node_wheel_sha="a" * 64,
        config=SimpleNamespace(agent_config_digest="c" * 64),
        _ensure_gpu_namespace=lambda _target: calls.append("namespace"),
        _ensure_connection_secret=lambda _target: calls.append("secret"),
        _quiesce_gpu_executor=lambda _target: calls.append("quiesce"),
        _verify_gpu_control_plane_endpoint=lambda _target: calls.append("endpoint"),
        _apply_gpu_dcgm_exporter=lambda _target: calls.append("dcgm"),
        _apply_gpu_adot_collector=lambda _target: calls.append("adot"),
        _apply_gpu_deployments=lambda _target, _wheel: calls.append("executor"),
        _roll_node_runtime=lambda _target, **_kwargs: calls.append("node-runtime"),
    )
    return release, calls


def test_bootstrap_gates_each_gpu_step_on_its_dependency() -> None:
    """A bootstrapping cluster brings its dependencies up in dependency order.

    The connection Secret and a verified control-plane endpoint must exist before
    the Executor is allowed to reach the region, and DCGM has to be exporting
    before the node Installer rolls, or the first nodes come up without the
    metrics every policy decision reads.
    """

    release, calls = bootstrap_recorder()

    ORCHESTRATION_MODULE.bootstrap_gpu_target(
        release, SimpleNamespace(cluster_id="gpu-a")
    )

    assert calls == [
        "namespace",
        "secret",
        "quiesce",
        "endpoint",
        "dcgm",
        "adot",
        "executor",
        "node-runtime",
    ]


def test_dcgm_exporter_is_ready_before_node_installer(tmp_path: Path) -> None:
    runtime_image = "registry.example/dcgm@sha256:" + "b" * 64
    path = config_file(tmp_path)
    config = MODULE.ReleaseConfig.load(path)

    class RecordingRunner:
        dry_run = False

        def __init__(self) -> None:
            self.calls = []

        def run(self, arguments, **kwargs):
            self.calls.append((arguments, kwargs))
            if "create" in arguments and "configmap" in arguments:
                return (
                    "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: test\n"
                    "data:\n  gpu-fault-counters.csv: test\n"
                )
            return ""

    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    release.dcgm_exporter_image = runtime_image
    release._apply_gpu_dcgm_exporter(config.clusters[0])

    rendered = [kwargs.get("input_text", "") for _arguments, kwargs in runner.calls]
    assert any("gpu-fault-counters.csv" in value for value in rendered), (
        "DCGM counters ConfigMap was not rendered"
    )
    assert any(runtime_image in value for value in rendered), (
        "configured DCGM image was not rendered"
    )
    assert any(
        "daemonset/gpu-fault-dcgm-exporter" in arguments and "rollout" in arguments
        for arguments, _kwargs in runner.calls
    ), "DCGM DaemonSet readiness was not awaited"


def test_node_installer_is_pinned_to_the_release_it_bootstraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Installer roll carries the same pins the release was built from.

    ``test_bootstrap_gates_each_gpu_step_on_its_dependency`` covers the ordering;
    this covers what the last step is handed, because an Installer rolled with a
    stale wheel or config digest would deploy an Agent the region rejects.
    """

    rolls: list[dict] = []
    release, _calls = bootstrap_recorder()
    monkeypatch.setattr(
        release, "_roll_node_runtime", lambda _target, **kwargs: rolls.append(kwargs)
    )

    ORCHESTRATION_MODULE.bootstrap_gpu_target(
        release, SimpleNamespace(cluster_id="gpu-a")
    )

    assert rolls == [
        {
            "phase": "bootstrap",
            "wheel_cm": "wheel",
            "bundle_cm": "bundle",
            "artifact_sha": "a" * 64,
            "config_digest": "c" * 64,
        }
    ]


def test_nlb_service_is_created_after_dns_and_certificate_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hosted zone and certificate are checked before the NLB is applied.

    Applying the Service first would publish a load balancer that no name
    resolves to and no certificate matches, and the CNAME must only follow once
    that Service exists.
    """

    dns_module = DNS_MODULE
    calls: list[str] = []
    for name in ("verify_control_plane_dns_prerequisites", "ensure_control_plane_dns"):
        monkeypatch.setattr(
            dns_module, name, lambda _release, _name=name: calls.append(_name)
        )
    release = SimpleNamespace(
        runner=SimpleNamespace(
            dry_run=False, run=lambda _arguments, **_kwargs: calls.append("apply-nlb")
        ),
        config=SimpleNamespace(
            nlb={
                "name": "gpu-fault-regional-test",
                "public_subnets": "subnet-a,subnet-b",
                "security_group": "sg-a",
                "certificate_arn": (
                    "arn:aws:acm:us-east-1:123456789012:certificate/control-plane"
                ),
            },
            aws_region=REGION,
            namespace="gpu-fault-system",
        ),
        _cpu=lambda *arguments: ["kubectl", *arguments],
    )

    dns_module.apply_control_plane_nlb(release)

    assert calls == [
        "verify_control_plane_dns_prerequisites",
        "apply-nlb",
        "ensure_control_plane_dns",
    ]


def test_dns_gate_executes_the_strict_runtime_sequence(monkeypatch) -> None:
    dns_module = DNS_MODULE
    events: list[str] = []
    certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/control-plane"
    raw_hostname = "internal-nlb.elb.us-east-1.amazonaws.com"

    class DnsRunner:
        dry_run = False

        def run(self, arguments, **_kwargs):
            command = " ".join(arguments)
            if "route53 get-hosted-zone" in command:
                events.append("hosted-zone")
                return json.dumps(
                    {
                        "HostedZone": {
                            "Id": "/hostedzone/Z123",
                            "Name": "site.gpu-fault.internal.",
                            "Config": {"PrivateZone": True},
                        }
                    }
                )
            if "acm describe-certificate" in command:
                events.append("certificate")
                return json.dumps(
                    {
                        "Certificate": {
                            "Status": "ISSUED",
                            "NotAfter": "2100-01-01T00:00:00+00:00",
                            "SubjectAlternativeNames": ["api.site.gpu-fault.internal"],
                        }
                    }
                )
            if "get service gpu-fault-api-nlb" in command:
                events.append("service-hostname")
                return raw_hostname
            if "elbv2 describe-load-balancers" in command:
                events.append("nlb-active")
                return json.dumps(
                    {
                        "LoadBalancers": [
                            {
                                "LoadBalancerArn": "arn:aws:elasticloadbalancing:"
                                "us-east-1:123456789012:loadbalancer/net/test/1",
                                "DNSName": raw_hostname,
                                "State": {"Code": "active"},
                            }
                        ]
                    }
                )
            if "elbv2 describe-listeners" in command:
                events.append("tls-listener")
                return json.dumps(
                    {
                        "Listeners": [
                            {
                                "Protocol": "TLS",
                                "Port": 443,
                                "Certificates": [{"CertificateArn": certificate_arn}],
                            }
                        ]
                    }
                )
            if "get deployment gpu-fault-api-ha" in command:
                events.append("expected-targets")
                return "3"
            if "elbv2 describe-target-groups" in command:
                events.append("target-group")
                return json.dumps(
                    {"TargetGroups": [{"TargetGroupArn": "target-group-arn"}]}
                )
            if "elbv2 describe-target-health" in command:
                events.append("targets-healthy")
                return json.dumps(
                    {
                        "TargetHealthDescriptions": [
                            {"TargetHealth": {"State": "healthy"}},
                            {"TargetHealth": {"State": "healthy"}},
                            {"TargetHealth": {"State": "healthy"}},
                        ]
                    }
                )
            if "route53 list-resource-record-sets" in command:
                events.append("cname-read")
                return json.dumps({"ResourceRecordSets": []})
            if "route53 change-resource-record-sets" in command:
                events.append("cname-upsert")
                return json.dumps({"ChangeInfo": {"Id": "/change/C123"}})
            if "route53 wait resource-record-sets-changed" in command:
                events.append("route53-wait")
                return ""
            if "route53 get-change" in command:
                events.append("route53-insync")
                return json.dumps({"ChangeInfo": {"Status": "INSYNC"}})
            raise AssertionError(f"unexpected command: {command}")

    release = SimpleNamespace(
        runner=DnsRunner(),
        config=SimpleNamespace(
            aws_region=REGION,
            namespace="gpu-fault-system",
            nlb={"name": "gpu-fault-regional-test", "certificate_arn": certificate_arn},
            dns=SimpleNamespace(
                hosted_zone_id="Z123", hostname="api.site.gpu-fault.internal"
            ),
        ),
        _cpu=lambda *arguments: ["kubectl", *arguments],
    )

    def resolve(hostname, port):
        assert hostname == raw_hostname
        assert port == 443
        events.append("raw-nlb-dns")
        return [(2, 1, 6, "", ("192.0.2.10", port))]

    monkeypatch.setattr(dns_module.socket, "getaddrinfo", resolve)

    dns_module.verify_control_plane_dns_prerequisites(release)
    dns_module.ensure_control_plane_dns(release)

    assert events == [
        "hosted-zone",
        "certificate",
        "service-hostname",
        "nlb-active",
        "tls-listener",
        "raw-nlb-dns",
        "expected-targets",
        "target-group",
        "targets-healthy",
        "cname-read",
        "cname-upsert",
        "route53-wait",
        "route53-insync",
    ]


def _cname_record(value: str, *, ttl: int = 60) -> dict:
    return {
        "Name": "api.site.gpu-fault.internal.",
        "Type": "CNAME",
        "TTL": ttl,
        "ResourceRecords": [{"Value": value}],
    }


def test_dns_upsert_is_skipped_only_when_the_live_cname_already_matches() -> None:
    """A no-op UPSERT still costs Route53's propagation wait, so read first."""

    dns_module = DNS_MODULE
    raw_hostname = "internal-nlb.elb.us-east-1.amazonaws.com"

    def release_with(record: dict | None) -> tuple[SimpleNamespace, list[str]]:
        events: list[str] = []

        class DnsRunner:
            dry_run = False

            def run(self, arguments, **_kwargs):
                command = " ".join(arguments)
                if "route53 list-resource-record-sets" in command:
                    events.append("cname-read")
                    return json.dumps(
                        {"ResourceRecordSets": [record] if record else []}
                    )
                if "route53 change-resource-record-sets" in command:
                    events.append("cname-upsert")
                    return json.dumps({"ChangeInfo": {"Id": "/change/C123"}})
                if "route53 wait resource-record-sets-changed" in command:
                    events.append("route53-wait")
                    return ""
                if "route53 get-change" in command:
                    events.append("route53-insync")
                    return json.dumps({"ChangeInfo": {"Status": "INSYNC"}})
                raise AssertionError(f"unexpected command: {command}")

        release = SimpleNamespace(
            runner=DnsRunner(),
            config=SimpleNamespace(
                aws_region=REGION,
                dns=SimpleNamespace(
                    hosted_zone_id="Z123", hostname="api.site.gpu-fault.internal"
                ),
            ),
        )
        return release, events

    # Identical record, fully qualified the way Route53 returns it: no change.
    release, events = release_with(_cname_record(raw_hostname + "."))
    dns_module.ensure_cname_points_at(release, raw_hostname)
    assert events == ["cname-read"], events

    # Anything the release owns that differs still goes through the full
    # submit-and-wait sequence: the value, the TTL, or a missing record.
    for record in (
        _cname_record("previous-nlb.elb.us-east-1.amazonaws.com"),
        _cname_record(raw_hostname, ttl=300),
        None,
    ):
        release, events = release_with(record)
        dns_module.ensure_cname_points_at(release, raw_hostname)
        assert events == [
            "cname-read",
            "cname-upsert",
            "route53-wait",
            "route53-insync",
        ], record

    assert not dns_module.cname_already_points_at(
        {**_cname_record(raw_hostname), "Type": "A"}, raw_hostname
    ), "a non-CNAME record must never satisfy the CNAME check"


ROLLOUT_TARGET = SimpleNamespace(cluster_id="gpu-a")
ROLLOUT_GATE = ("endpoint", "dcgm", "deployments")


def rollout_gate_recorder(calls: list[str]) -> dict:
    """The three steps every Executor rollout path has to order identically."""

    return {
        "executor_wheel_cm": "wheel",
        "_ensure_connection_secret": lambda _target: None,
        "_verify_gpu_control_plane_endpoint": lambda _target: calls.append("endpoint"),
        "_apply_gpu_dcgm_exporter": lambda _target, **_kwargs: calls.append("dcgm"),
        "_apply_gpu_adot_collector": lambda _target, **_kwargs: None,
        "_apply_gpu_deployments": lambda *_args, **_kwargs: calls.append("deployments"),
    }


def test_upgrade_requires_gpu_dns_and_tls_before_the_executor() -> None:
    """An upgrade proves the GPU cluster can reach the region before rolling it.

    Rolling the Executor first would restart the only component that reports
    faults into a region it may not be able to authenticate to, and the cluster
    would go dark instead of failing the release.
    """

    calls: list[str] = []
    release = SimpleNamespace(**rollout_gate_recorder(calls))

    MODULE.upgrade_gpu_target(
        release,
        ROLLOUT_TARGET,
        DIFF_MODULE.ReleaseDiff(
            kind=DIFF_MODULE.ReleaseChangeKind.FULL, changed=frozenset({"endpoint"})
        ),
        DIFF_MODULE.ReleaseExecutionPlan(
            nodes=(
                DIFF_MODULE.ReleaseComponent.ENDPOINT,
                DIFF_MODULE.ReleaseComponent.DCGM,
                DIFF_MODULE.ReleaseComponent.EXECUTOR,
            )
        ),
    )

    assert tuple(calls) == ROLLOUT_GATE


def test_rollback_requires_gpu_dns_and_tls_before_the_executor() -> None:
    """A rollback re-checks the endpoint after restoring the previous Secret."""

    calls: list[str] = []
    release = SimpleNamespace(
        **rollout_gate_recorder(calls),
        config=SimpleNamespace(executor_wheel=Path("executor.whl")),
        runtime_image="candidate-runtime",
        _gpu=lambda _target, *arguments: ["kubectl", *arguments],
        _restore_secret=lambda _arguments, *, source, backup: None,
        _config_map_sha=lambda *_args: "d" * 64,
    )

    ORCHESTRATION_MODULE.rollback_target(
        release,
        ROLLOUT_TARGET,
        previous={
            "secret_backups": {
                "clusters": {
                    "gpu-a": {
                        "source": "gpu-fault-regional-connection",
                        "backup": "gpu-fault-regional-connection-previous",
                    }
                }
            },
            "clusters": {
                "gpu-a": {
                    "wheel": "previous-wheel",
                    "wheel_key": "executor.whl",
                    "dcgm_image": "registry.example/dcgm:previous",
                }
            },
        },
        artifact="artifact",
        config_digest="config",
        runtime_profile_version="profile-v1",
        executor_artifact="executor",
        executor_compatibility="executor",
        node_compatibility="artifact",
        runtime_image="previous-runtime",
        node_installer_image="registry.example/installer:previous",
        components=frozenset(
            {
                DIFF_MODULE.ReleaseComponent.ENDPOINT,
                DIFF_MODULE.ReleaseComponent.DCGM,
                DIFF_MODULE.ReleaseComponent.EXECUTOR,
            }
        ),
    )

    assert tuple(calls) == ROLLOUT_GATE


def test_join_requires_gpu_dns_and_tls_before_the_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A joining cluster is proven reachable before it is given an Executor."""

    calls: list[str] = []
    for name in ("prepare_join_registry", "ensure_runtime_profile"):
        monkeypatch.setattr(MODULE, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        MODULE, "join_target", lambda _self, _cluster_id: ROLLOUT_TARGET
    )
    release = SimpleNamespace(
        **rollout_gate_recorder(calls),
        bundle_cm="bundle",
        bundle_sha="b" * 64,
        executor_wheel_sha="e" * 64,
        node_wheel_sha="n" * 64,
        config=SimpleNamespace(
            executor_wheel=Path("executor.whl"),
            bundle=Path("bundle.tar.gz"),
            agent_config_digest="c" * 64,
        ),
        _ensure_contexts=lambda: None,
        _update_registry=lambda _target, *, remove: None,
        _ensure_gpu_namespace=lambda _target: None,
        _quiesce_gpu_executor=lambda _target: None,
        _gpu=lambda _target, *arguments: ["kubectl", *arguments],
        _upload_config_map=lambda *_args, **_kwargs: None,
        _config_map_data=lambda _name: {
            "required-agent-artifact-sha256": "n" * 64,
            "required-regional-executor-artifact-sha256": "e" * 64,
        },
        _roll_node_runtime=lambda _target, **_kwargs: None,
    )

    MODULE.RegionalRelease.join_cluster(release, "gpu-a")

    assert tuple(calls) == ROLLOUT_GATE


def test_gpu_endpoint_probe_checks_dns_ca_tls_and_health(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["dns"] = {"hosted_zone_id": "Z123", "hostname": "api.site.gpu-fault.internal"}
    path.write_text(json.dumps(value))
    config = MODULE.ReleaseConfig.load(path)

    class ProbeRunner:
        dry_run = False

        def __init__(self) -> None:
            self.manifest = ""

        def run(self, arguments, **kwargs):
            command = " ".join(arguments)
            if "apply -f -" in command:
                self.manifest = kwargs["input_text"]
            if "jsonpath={.status.phase}" in command:
                return "Succeeded"
            if " logs " in f" {command} ":
                return '{"tls": "verified", "status": 200}'
            return ""

    runner = ProbeRunner()
    release = MODULE.RegionalRelease(config, runner)
    release._verify_gpu_control_plane_endpoint(config.clusters[0])
    pod = yaml.safe_load(runner.manifest)
    script = pod["spec"]["containers"][0]["command"][2]

    assert "socket.getaddrinfo" in script
    assert "ssl.create_default_context" in script
    assert "urllib.request.urlopen" in script
    assert '"/healthz"' in script
    assert {
        item["name"]: item.get("value") for item in pod["spec"]["containers"][0]["env"]
    }["EXPECTED_HOSTNAME"] == "api.site.gpu-fault.internal"
    assert {
        (item["key"], item["operator"], item["effect"])
        for item in pod["spec"]["tolerations"]
    } == {
        ("gpu-fault.io/quarantined", "Exists", "NoSchedule"),
        ("node.kubernetes.io/unschedulable", "Exists", "NoSchedule"),
    }


def test_agent_convergence_uses_hyperpod_cluster_name(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    items = [
        {
            "metadata": {
                "name": "node-a",
                "uid": "uid-a",
                "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"},
                "annotations": {
                    "gpu-fault.io/installer-state": "Succeeded",
                    "gpu-fault.io/installer-artifact-sha256": hashlib.sha256(
                        b"wheel"
                    ).hexdigest(),
                    "gpu-fault.io/installer-config-digest": "config-a",
                    "gpu-fault.io/installer-node-uid": "uid-a",
                },
            }
        },
        {
            "metadata": {
                "labels": {"sagemaker.amazonaws.com/cluster-name": "another-hyperpod"},
                "annotations": {},
            }
        },
    ]

    assert MODULE.agents_converged(
        items,
        config.clusters[0],
        hashlib.sha256(b"wheel").hexdigest(),
        config_digest="config-a",
        require_node_uid=True,
    ), "agent convergence ignored the configured HyperPod cluster name"
    items[0]["metadata"]["annotations"]["gpu-fault.io/installer-node-uid"] = "stale"
    assert not MODULE.agents_converged(
        items,
        config.clusters[0],
        hashlib.sha256(b"wheel").hexdigest(),
        config_digest="config-a",
        require_node_uid=True,
    ), "agent convergence accepted a stale node UID annotation"
