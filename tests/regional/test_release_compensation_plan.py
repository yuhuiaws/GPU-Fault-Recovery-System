from __future__ import annotations

from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
PROGRESS = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_progress.py"
)
DIFF = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_diff.py"
)


def _state(*components, completed_phases=(), completed_clusters=()):
    return {
        "execution_plan": {"nodes": [component.value for component in components]},
        "completed_phases": list(completed_phases),
        "completed_cluster_ids": list(completed_clusters),
    }


def test_artifact_only_failure_has_no_compensation() -> None:
    state = _state(DIFF.ReleaseComponent.CPU_FINALIZE, DIFF.ReleaseComponent.VERIFY)

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a"])

    assert plan.empty, "non-mutating failure unexpectedly scheduled rollback work"
    assert plan.conservative is True


def test_explicit_legacy_rollback_without_transaction_evidence_is_full() -> None:
    plan = PROGRESS.build_rollback_compensation_plan({}, ["gpu-a"])

    assert plan.restores_cpu is True
    assert plan.restores_data_plane is True
    assert plan.needs_controller is True


def test_control_plane_only_rollback_does_not_touch_gpu() -> None:
    state = _state(
        DIFF.ReleaseComponent.CPU_FINALIZE,
        DIFF.ReleaseComponent.VERIFY,
        completed_phases=("uploaded", "schema-ready", "cpu-finalized"),
    )

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a"])

    assert plan.global_components == frozenset({DIFF.ReleaseComponent.CPU_FINALIZE})
    assert plan.for_cluster("gpu-a") == frozenset()
    assert plan.needs_controller is False


def test_observability_rules_are_compensated_without_cpu_or_gpu_rollout() -> None:
    state = _state(
        DIFF.ReleaseComponent.OBSERVABILITY,
        DIFF.ReleaseComponent.VERIFY,
        completed_phases=("uploaded", "schema-ready"),
    )
    state["component_progress"] = PROGRESS.update_component_progress(
        state,
        DIFF.ReleaseComponent.OBSERVABILITY,
        PROGRESS.PROGRESS_STARTED,
        observed_at_epoch=10.0,
    )

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a"])

    assert plan.restores_observability is True
    assert plan.restores_cpu is False
    assert plan.restores_data_plane is False


def test_precise_executor_progress_rolls_back_only_started_cluster() -> None:
    state = _state(
        DIFF.ReleaseComponent.EXECUTOR,
        DIFF.ReleaseComponent.VERIFY,
        completed_phases=("uploaded", "schema-ready"),
    )
    state["component_progress"] = PROGRESS.update_component_progress(
        state,
        DIFF.ReleaseComponent.EXECUTOR,
        PROGRESS.PROGRESS_STARTED,
        cluster_id="gpu-b",
        observed_at_epoch=10.0,
    )

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a", "gpu-b"])

    assert plan.for_cluster("gpu-a") == frozenset()
    assert plan.for_cluster("gpu-b") == frozenset({DIFF.ReleaseComponent.EXECUTOR})
    assert plan.needs_controller is False
    assert plan.conservative is False


def test_agent_not_started_does_not_stage_rollback_controller() -> None:
    state = _state(
        DIFF.ReleaseComponent.EXECUTOR,
        DIFF.ReleaseComponent.AGENT,
        DIFF.ReleaseComponent.VERIFY,
        completed_phases=("uploaded", "schema-ready"),
    )
    state["component_progress"] = PROGRESS.update_component_progress(
        state,
        DIFF.ReleaseComponent.EXECUTOR,
        PROGRESS.PROGRESS_COMPLETED,
        cluster_id="gpu-a",
        observed_at_epoch=20.0,
    )

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a"])

    assert plan.for_cluster("gpu-a") == frozenset({DIFF.ReleaseComponent.EXECUTOR})
    assert plan.needs_controller is False


def test_legacy_gpu_failure_falls_back_to_full_planned_cluster_scope() -> None:
    state = _state(
        DIFF.ReleaseComponent.CPU_STAGE,
        DIFF.ReleaseComponent.EXECUTOR,
        DIFF.ReleaseComponent.AGENT,
        DIFF.ReleaseComponent.CPU_FINALIZE,
        DIFF.ReleaseComponent.VERIFY,
        completed_phases=("uploaded", "schema-ready", "cpu-staged"),
    )

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a", "gpu-b"])

    expected = frozenset({DIFF.ReleaseComponent.EXECUTOR, DIFF.ReleaseComponent.AGENT})
    assert plan.for_cluster("gpu-a") == expected
    assert plan.for_cluster("gpu-b") == expected
    assert plan.needs_controller is True
    assert plan.conservative is True


def test_component_progress_records_duration_and_details() -> None:
    state: dict[str, object] = {}
    started = PROGRESS.update_component_progress(
        state,
        DIFF.ReleaseComponent.AGENT,
        PROGRESS.PROGRESS_STARTED,
        cluster_id="gpu-a",
        observed_at_epoch=10.0,
        details={"wave": 0},
    )
    state["component_progress"] = started
    completed = PROGRESS.update_component_progress(
        state,
        DIFF.ReleaseComponent.AGENT,
        PROGRESS.PROGRESS_COMPLETED,
        cluster_id="gpu-a",
        observed_at_epoch=14.5,
        details={"nodes": ["node-a"]},
    )

    entry = completed["clusters"]["gpu-a"]["agent"]
    assert entry["duration_seconds"] == 4.5
    assert entry["details"] == {"wave": 0, "nodes": ["node-a"]}


def test_replayed_manifests_records_templates_the_rollback_cannot_restore() -> None:
    """A template edit is re-applied from the candidate tree, so it is recorded.

    Rollback re-renders the CPU and data-plane manifests from the checkout it is
    running out of, with the previous release's pins substituted in. Anything the
    template edit itself changed therefore survives the rollback, and the plan
    says which deliveries that applies to instead of leaving it to be inferred.
    """
    state = _state(
        DIFF.ReleaseComponent.CPU_FINALIZE, completed_phases=("cpu-finalized",)
    )
    state["release_diff"] = {
        "changed": [
            "cpu_manifests",
            "executor_manifests",
            "observability_manifests",
            "endpoint_manifests",
            "control_plane_wheel",
        ]
    }

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a"])

    assert plan.replayed_manifests == frozenset(
        {"cpu_manifests", "executor_manifests"}
    ), "snapshot-restored or non-manifest changes leaked into the replayed set"
    assert plan.as_dict()["replayed_manifests"] == [
        "cpu_manifests",
        "executor_manifests",
    ], "the rollback plan persisted to state omits the replayed deliveries"


def test_pin_only_rollback_replays_no_manifests() -> None:
    state = _state(
        DIFF.ReleaseComponent.CPU_FINALIZE, completed_phases=("cpu-finalized",)
    )
    state["release_diff"] = {"changed": ["control_plane_wheel", "runtime_image"]}

    plan = PROGRESS.build_rollback_compensation_plan(state, ["gpu-a"])

    assert plan.replayed_manifests == frozenset(), (
        "a pin-only release was reported as carrying template drift"
    )
