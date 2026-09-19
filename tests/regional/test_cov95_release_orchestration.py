from __future__ import annotations

import copy
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_orchestration as orchestration
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ReleaseError,
    canonical_sha256,
)
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import (
    ReleaseExecutionPlan,
    build_execution_plan,
    diff_from_changed,
)
from tests.regional._cov95_release_engine import EngineRelease
from tests.regional._cov95_release_support import ResourceRelease
from tests.regional._release_orchestrator_support import phase_release


@pytest.fixture
def release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EngineRelease:
    release = EngineRelease(tmp_path)
    monkeypatch.setattr(
        orchestration, "prepare_upgrade_credentials", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestration, "preflight_upgrade_mutations", lambda *_args: None
    )
    return release


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("release", "release_id does not match"),
        ("diff", "release diff does not match"),
        ("plan", "execution plan does not match"),
        ("previous", "previous release state is unavailable"),
        ("missing-digest", "baseline digest is missing"),
        ("drifted-digest", "baseline digest drifted"),
    ],
)
def test_resume_rejects_journal_binding_errors_before_any_phase(
    release: EngineRelease, fault: str, problem: str
) -> None:
    diff = diff_from_changed({"cpu_worker_manifests"})
    plan = build_execution_plan(diff)
    loaded = {
        "phase": "failed",
        "release_id": release.release_id,
        "release_diff": diff.as_dict(),
        "execution_plan": plan.as_dict(),
        "previous": release.previous,
        "previous_snapshot_sha256": canonical_sha256(release.previous),
    }
    if fault == "release":
        loaded["release_id"] = "other"
    elif fault == "diff":
        loaded["release_diff"] = {}
    elif fault == "plan":
        loaded["execution_plan"] = {}
    elif fault == "previous":
        loaded["previous"] = {}
    elif fault == "missing-digest":
        loaded.pop("previous_snapshot_sha256")
    else:
        loaded["previous_snapshot_sha256"] = "other"
    release.state = loaded
    with pytest.raises(ReleaseError, match=problem):
        release.upgrade(resume=True, diff=diff)
    assert release.saves == []
    assert release.runner.calls == []
    assert "resume-checkpoint" not in [name for name, _details in release.effects]


def test_automatic_endpoint_compensation_requires_previous_capture(
    release: EngineRelease,
) -> None:
    with pytest.raises(ReleaseError, match="predates endpoint rollback capture"):
        release.upgrade(diff=diff_from_changed({"endpoint"}))
    assert release.saves == []


def test_unspecified_upgrade_refuses_unaccepted_schema_change(
    release: EngineRelease,
) -> None:
    with pytest.raises(ReleaseError, match="schema change"):
        release.upgrade()
    assert release.saves == []


def test_completed_gpu_and_bootstrap_sets_do_not_repeat_actions() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    calls: list[Any] = []
    release = phase_release(calls, clusters=(target,))
    orchestration.upgrade_gpu_clusters(
        release,
        diff=diff_from_changed({"executor_wheel"}),
        plan=ReleaseExecutionPlan((Component.EXECUTOR,)),
        previous={},
        completed_phases=set(),
        completed_clusters={"gpu-a"},
        registry_staged=False,
    )
    orchestration.bootstrap_gpu_clusters(release, {"gpu-a"})
    assert calls == []


def test_parallel_failures_are_recorded_after_successful_canary() -> None:
    targets = tuple(
        SimpleNamespace(cluster_id=name) for name in ("gpu-a", "gpu-b", "gpu-c")
    )
    calls: list[str] = []
    saves: list[Any] = []
    barrier = threading.Barrier(2)

    def upgrade(target: Any, *_args: Any, **_kwargs: Any) -> None:
        calls.append(target.cluster_id)
        if target.cluster_id != "gpu-a":
            barrier.wait(timeout=5)
            raise ClusterLocalReleaseError("fixture local failure")

    release = phase_release(
        [],
        saves,
        clusters=targets,
        max_parallel_clusters=2,
        _upgrade_gpu_target=upgrade,
    )
    with pytest.raises(
        orchestration.PartialClusterRolloutError, match="also failed: gpu-c"
    ):
        orchestration.upgrade_gpu_clusters(
            release,
            diff=diff_from_changed({"executor_wheel"}),
            plan=ReleaseExecutionPlan((Component.EXECUTOR,)),
            previous={},
            completed_phases=set(),
            completed_clusters=set(),
            registry_staged=False,
        )
    assert calls[0] == "gpu-a"
    assert set(calls[1:]) == {"gpu-b", "gpu-c"}
    assert saves[-1]["failed_cluster_ids"] == ["gpu-b", "gpu-c"]
    assert saves[-1]["completed_cluster_ids"] == ["gpu-a"]
    assert saves[-1]["failure_scope"] == "cluster-local"


def cpu_rollback_state(release: EngineRelease) -> None:
    release.state = {
        "phase": "failed",
        "execution_plan": {"nodes": [Component.CPU_FINALIZE.value]},
        "component_progress": {
            "schema_version": 1,
            "global": {Component.CPU_FINALIZE.value: {"status": "STARTED"}},
            "clusters": {},
        },
    }


@pytest.mark.parametrize(
    "snapshot",
    [
        ["invalid"],
        {"customer-config": {}},
        {17: {}},
        {"gpu-fault-config-core": []},
        {"gpu-fault-config-core": {"GPU_FAULT_EXECUTION_TOKEN": "example-only"}},
    ],
)
def test_cpu_rollback_refuses_invalid_configmaps_before_role_apply(
    release: EngineRelease, snapshot: Any
) -> None:
    cpu_rollback_state(release)
    previous = {**release.previous, "cpu_role_config_maps": snapshot}
    with pytest.raises(ReleaseError, match="ConfigMap snapshot|sensitive-looking"):
        release.rollback(state=previous)
    assert release.maps == []
    assert release.environments == []
    assert "rollback-cpu-restored" not in release.state["rollback_completed_phases"]


@pytest.mark.parametrize("captured", [False, True])
def test_cpu_rollback_replays_captured_config_and_preserves_restore_flags(
    release: EngineRelease, captured: bool
) -> None:
    cpu_rollback_state(release)
    release.registry_restored = True
    previous = copy.deepcopy(release.previous)
    if captured:
        previous["cpu_role_config_maps"] = {
            "gpu-fault-api-ha-config-core": {"SAFE_EXAMPLE": 7}
        }
    release.rollback(state=previous)
    assert release.state["phase"] == "rolled-back"
    assert release.state["rollback_cleanup_completed"] is True
    assert len(release.maps) == int(captured)
    if captured:
        assert release.maps[0]["data"] == {"SAFE_EXAMPLE": "7"}
    assert len(release.environments) == 2
    assert (
        release.environments[-1]["GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS"]
        == str(captured).lower()
    )
    assert release.environments[-1]["GPU_FAULT_FORCE_ROLE_RESTART"] == "true"
    names = [name for name, _details in release.effects]
    assert names.index("publish-restored") < names.index("roles")
    assert names.index("verify-rollback") < names.index("delete-backups")


def test_cpu_apply_grants_release_metadata_reads_before_rendering_roles(
    release: EngineRelease, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control-plane Pods poll the pin ConfigMap themselves (fleet pin hot
    reload), so every CPU apply re-applies the read grant before the role-split
    apply can bring up Pods that poll -- an upgraded site has it before its
    first hot-reloading Pods start, and bootstrap is not the only path."""

    monkeypatch.setattr(
        rollout, "build_cpu_apply_environment", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        rollout,
        "render_and_apply_cpu_roles",
        lambda _release, _environment: release.effects.append(("roles", None)),
    )
    orchestration.run_upgrade_phases(
        release,
        diff=diff_from_changed({"regional_cluster_registry"}),
        plan=ReleaseExecutionPlan((Component.CPU_STAGE, Component.VERIFY)),
        previous=release.previous,
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )
    names = [name for name, _details in release.effects]
    assert names.count("release-metadata-rbac") == 1
    assert names.index("release-metadata-rbac") < names.index("roles")
    manifest = release.rbac_manifests[0]
    assert "name: gpu-fault-control-plane-release-metadata" in manifest
    assert 'resourceNames: ["gpu-fault-release-metadata"]' in manifest
    assert "name: gpu-fault-control-plane\n" in manifest, (
        "the grant must bind the control-plane ServiceAccount"
    )
    arguments, _kwargs = next(
        call for call in release.runner.calls if call[1].get("input_text") == manifest
    )
    assert arguments[:3] == ["kubectl", "--kubeconfig", "/secure/cpu.kubeconfig"]


def test_cpu_rollback_requires_readable_previous_artifact(
    release: EngineRelease,
) -> None:
    cpu_rollback_state(release)
    release.old_artifact = b""
    with pytest.raises(ReleaseError, match="has no binaryData"):
        release.rollback(state=release.previous)
    assert release.environments == []
    assert release.state["phase"] != "rolled-back"


def test_noop_delegates_both_cpu_and_gpu_verification(
    release: EngineRelease, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = []
    monkeypatch.setattr(
        rollout,
        "validate_release_components",
        lambda _release, **kwargs: checks.append(kwargs),
    )
    release.noop(diff_from_changed(set()))
    assert checks == [{"cpu": True, "data_plane": True}]
    assert release.state["phase"] == "complete"


def test_staged_registry_forwards_forced_restart_to_cpu_role_renderer(
    release: EngineRelease, monkeypatch: pytest.MonkeyPatch
) -> None:
    release.registry_staged = True
    environments = []
    monkeypatch.setattr(
        rollout, "build_cpu_apply_environment", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        rollout,
        "render_and_apply_cpu_roles",
        lambda _release, environment: environments.append(dict(environment)),
    )
    orchestration.run_upgrade_phases(
        release,
        diff=diff_from_changed({"regional_cluster_registry"}),
        plan=ReleaseExecutionPlan(
            (Component.REGISTRY, Component.CPU_STAGE, Component.VERIFY)
        ),
        previous=release.previous,
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )
    assert environments == [
        {
            "GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS": "spool,worker,ingress",
            "GPU_FAULT_FORCE_ROLE_RESTART": "true",
        }
    ]
    assert release.state["phase"] == "complete"
    assert ("commit-registry", None) in release.effects


@pytest.mark.parametrize("drift", [False, True])
def test_resumed_refresher_checkpoint_requires_fresh_current_proof(
    release: EngineRelease, drift: bool
) -> None:
    release.aurora_drift = drift
    completed = {"uploaded", "data-converged", "aurora-refresh-ready"}
    arguments = {
        "diff": diff_from_changed({"aurora_refresh_drift"}),
        "plan": ReleaseExecutionPlan((Component.AURORA_REFRESH, Component.VERIFY)),
        "previous": release.previous,
        "completed_phases": completed,
        "completed_clusters": set(),
        "registry_staged": False,
    }
    if drift:
        with pytest.raises(ReleaseError, match="refresher has drifted"):
            orchestration.run_upgrade_phases(release, **arguments)
        assert release.saves == []
    else:
        orchestration.run_upgrade_phases(release, **arguments)
        assert ("refresh", {"required": True}) in release.effects
        assert ("apply-refresher", None) not in release.effects


def test_accepted_schema_metadata_is_checkpointed_with_schema_completion(
    release: EngineRelease,
) -> None:
    acceptance = {"mode": "no-snapshot", "database_schema_version": 17}
    release.state["schema_change_acceptance"] = acceptance
    orchestration.run_upgrade_phases(
        release,
        diff=diff_from_changed({"database_schema"}),
        plan=ReleaseExecutionPlan((Component.SCHEMA, Component.VERIFY)),
        previous=release.previous,
        completed_phases=set(),
        completed_clusters=set(),
        registry_staged=False,
    )
    assert release.state["schema_change_acceptance"] == acceptance
    assert "schema-ready" in release.state["completed_phases"]
    assert ("schema", None) in release.effects


class ExpectedRulesFailure(EngineRelease):
    def _apply_dataplane_expected_rules(self) -> None:
        raise ReleaseError("expected rules rejected")


def test_joined_component_is_failed_when_its_postcheck_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = ExpectedRulesFailure(tmp_path)
    monkeypatch.setattr(
        orchestration, "preflight_upgrade_mutations", lambda *_args: None
    )
    with pytest.raises(ReleaseError, match="expected rules rejected"):
        orchestration.run_upgrade_phases(
            release,
            diff=diff_from_changed({"observability_manifests"}),
            plan=ReleaseExecutionPlan((Component.OBSERVABILITY, Component.VERIFY)),
            previous=release.previous,
            completed_phases=set(),
            completed_clusters=set(),
            registry_staged=False,
        )
    assert (
        release.state["component_progress"]["global"]["observability"]["status"]
        == "FAILED"
    )
    assert "observability-ready" not in release.state["completed_phases"]


def test_rollback_stops_after_first_target_failure_and_keeps_failed_timing(
    release: EngineRelease, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets = ResourceRelease(("gpu-a", "gpu-b")).config.clusters
    release.config = replace(release.config, clusters=targets)
    release.state = {
        "phase": "failed",
        "execution_plan": {"nodes": [Component.EXECUTOR.value]},
        "component_progress": {
            "schema_version": 1,
            "global": {},
            "clusters": {
                target.cluster_id: {
                    Component.EXECUTOR.value: {
                        "status": "STARTED",
                        "updated_at_epoch": index,
                    }
                }
                for index, target in enumerate(targets, start=1)
            },
        },
    }
    calls = []

    def restore(_release: Any, target: Any, **_kwargs: Any) -> None:
        calls.append(target.cluster_id)
        raise ReleaseError("target restore failed")

    monkeypatch.setattr(orchestration, "rollback_target", restore)
    with pytest.raises(ReleaseError, match="gpu-b rollback failed"):
        release.rollback(state=release.previous)
    assert calls == ["gpu-b"]
    # An executor rollback re-stages the control plane for the previous pin
    # window before any cluster is restored, and needs no Agent identities to
    # do so (the previous snapshot here records none).
    previous_artifact = release.previous["metadata"][
        "required-regional-executor-artifact-sha256"
    ]
    assert release.environments, "the controller was not staged before the restore"
    assert all(
        env["GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256"] == previous_artifact
        for env in release.environments
    ), "every staged control-plane render must carry the previous executor pin"
    assert release.state["rollback_timing"]["clusters"]["gpu-b"]["status"] == "FAILED"
    assert release.state["rollback_completed_cluster_ids"] == []
    assert "rollback-data-restored" not in release.state["rollback_completed_phases"]


def test_completed_rollback_resume_does_not_repeat_restores_or_cleanup(
    release: EngineRelease,
) -> None:
    cpu_rollback_state(release)
    release.rollback(state=release.previous)
    completed = copy.deepcopy(release.state)
    effects = list(release.effects)
    saves = copy.deepcopy(release.saves)
    release.rollback()
    assert release.state == completed
    assert release.saves == saves
    assert release.effects[len(effects) :] == [("refresh", {})]
    assert len(release.environments) == 2
    assert release.state["rollback_result"]["status"] == "PASSED"


class BootstrapCleanupRelease(EngineRelease):
    def _cleanup_bootstrap(self) -> None:
        self.effects.append(("bootstrap-cleanup", None))
        self.state["phase"] = "bootstrap-cleaned"


def test_rollback_without_previous_uses_bootstrap_cleanup_and_stops(
    tmp_path: Path,
) -> None:
    release = BootstrapCleanupRelease(tmp_path)
    release.state = {"phase": "bootstrap-failed"}
    release.rollback()
    assert release.effects == [("bootstrap-cleanup", None)]
    assert release.state == {"phase": "bootstrap-cleaned"}
    assert release.environments == []
    assert release.runner.calls == []


@pytest.mark.parametrize("present", [False, True])
def test_legacy_cpu_snapshot_cannot_restore_an_uncaptured_live_refresher(
    release: EngineRelease, monkeypatch: pytest.MonkeyPatch, present: bool
) -> None:
    cpu_rollback_state(release)
    previous = copy.deepcopy(release.previous)
    previous.pop("aurora_refresh")
    reads = []

    def read(_release: Any) -> dict[str, Any] | None:
        reads.append("refresher")
        return {"kind": "CronJob"} if present else None

    monkeypatch.setattr(orchestration, "read_aurora_refresh_cronjob", read)
    if present:
        with pytest.raises(
            ReleaseError, match="predates full Aurora refresher capture"
        ):
            release.rollback(state=previous)
        assert release.effects == []
        assert release.saves == []
        assert release.environments == []
    else:
        release.rollback(state=previous)
        assert release.state["phase"] == "rolled-back"
        assert len(release.environments) == 2
    assert reads == ["refresher"]
