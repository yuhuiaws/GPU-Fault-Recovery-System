from pathlib import Path
from types import SimpleNamespace

from scripts.component_wheels import COMPONENTS, dependency_closure, entrypoint_modules
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
DIFF = lazy_script_module(
    "regional_release_diff",
    ROOT / "deploy/control-plane/regional/regional_release_diff.py",
)


def _release() -> SimpleNamespace:
    return SimpleNamespace(
        wheel_sha="a" * 64,
        executor_wheel_sha="b" * 64,
        node_wheel_sha="c" * 64,
        bundle_sha="d" * 64,
        runtime_profile_sha="e" * 64,
        runtime_profile_template_sha="4" * 64,
        runtime_profile_policy_sha="5" * 64,
        endpoint_digest="f" * 64,
        dcgm_digest="1" * 64,
        notification_digest="3" * 64,
        cluster_registry_digest="6" * 64,
        config=SimpleNamespace(
            database_schema_version=6,
            agent_protocol_version=3,
            executor_protocol_version=2,
            agent_config_digest="2" * 64,
            runtime_profile_version="hyperpod-v1",
            component_digests={
                "control_plane": "a" * 64,
                "executor": "b" * 64,
                "node_runtime": "c" * 64,
            },
            clusters=[SimpleNamespace(cluster_id="gpu-a")],
        ),
    )


def _state() -> dict:
    release = _release()
    return {
        "wheel_sha256": release.wheel_sha,
        "executor_wheel_sha256": release.executor_wheel_sha,
        "node_wheel_sha256": release.node_wheel_sha,
        "bundle_sha256": release.bundle_sha,
        "component_digests": release.config.component_digests,
        "database_schema_version": 6,
        "agent_protocol_version": 3,
        "executor_protocol_version": 2,
        "agent_config_digest": release.config.agent_config_digest,
        "runtime_profile_sha256": release.runtime_profile_sha,
        "runtime_profile_policy_sha256": release.runtime_profile_policy_sha,
        "runtime_profile_version": "hyperpod-v1",
        "endpoint_digest": release.endpoint_digest,
        "dcgm_digest": release.dcgm_digest,
        "notification_digest": release.notification_digest,
        "cluster_registry_digest": release.cluster_registry_digest,
        "cluster_ids": ["gpu-a"],
    }


def test_release_diff_classifies_noop_and_component_scopes() -> None:
    release = _release()
    state = _state()

    assert DIFF.classify_release(release, state).kind is DIFF.ReleaseChangeKind.NOOP
    state["wheel_sha256"] = "9" * 64
    assert (
        DIFF.classify_release(release, state).kind
        is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    )
    state = _state()
    state["notification_digest"] = "8" * 64
    assert (
        DIFF.classify_release(release, state).kind
        is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    )
    state = _state()
    state["executor_wheel_sha256"] = "9" * 64
    assert (
        DIFF.classify_release(release, state).kind
        is DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    )
    state = _state()
    state["agent_protocol_version"] = 2
    assert DIFF.classify_release(release, state).kind is DIFF.ReleaseChangeKind.FULL
    state = _state()
    state["cluster_registry_digest"] = "7" * 64
    assert DIFF.classify_release(release, state).kind is DIFF.ReleaseChangeKind.FULL


def test_packaging_only_artifact_changes_are_not_classified_as_noop() -> None:
    release = _release()
    state = _state()
    state.update(
        {
            "wheel_sha256": "6" * 64,
            "executor_wheel_sha256": "7" * 64,
            "node_wheel_sha256": "8" * 64,
            "bundle_sha256": "9" * 64,
        }
    )

    diff = DIFF.classify_release(release, state)

    assert diff.kind is DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    assert diff.changed == {
        "control_plane_wheel",
        "executor_wheel",
        "node_runtime_wheel",
        "node_bundle",
    }


def test_profile_path_only_migration_is_not_a_release_change() -> None:
    release = _release()
    state = _state()
    state.pop("runtime_profile_policy_sha256")
    state["runtime_profile_sha256"] = release.runtime_profile_template_sha

    diff = DIFF.classify_release(release, state)

    assert diff.kind is DIFF.ReleaseChangeKind.NOOP
    assert "runtime_profile" not in diff.changed


def test_profile_policy_change_remains_full() -> None:
    release = _release()
    state = _state()
    state["runtime_profile_policy_sha256"] = "9" * 64

    diff = DIFF.classify_release(release, state)

    assert diff.kind is DIFF.ReleaseChangeKind.FULL
    assert "runtime_profile" in diff.changed


def test_component_dependency_closures_are_runtime_specific() -> None:
    closures = {
        name: dependency_closure({*component.roots, *entrypoint_modules(component)})
        for name, component in COMPONENTS.items()
    }

    assert "gpu_fault.app.factory" in closures["control_plane"]
    assert "gpu_fault.app.factory" not in closures["executor"]
    assert "gpu_fault.app.factory" not in closures["node_runtime"]
    assert "gpu_fault.cluster_executor" in closures["executor"]
    assert "gpu_fault.cluster_executor" not in closures["control_plane"]
    assert "gpu_fault.cluster_executor" not in closures["node_runtime"]
    assert "gpu_fault.node_agent.app" in closures["node_runtime"]
    assert "gpu_fault.node_agent.app" not in closures["control_plane"]
    assert "gpu_fault.node_agent.app" not in closures["executor"]
