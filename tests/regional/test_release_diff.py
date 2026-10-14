from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin.config import default_admin_config
from gpu_fault_release import regional_release_config as CONFIG
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_node_scope as SCOPE
from scripts import component_wheels
from scripts.component_wheels import (
    COMPONENTS,
    component_data_files,
    component_definition,
    component_modules,
    dependency_closure,
    entrypoint_modules,
)
from scripts.release_identity import file_set_identity

ROOT = Path(__file__).resolve().parents[2]


_NO_PROBE = object()


def _release(
    *,
    drift: bool = False,
    probes: list[str] | None = None,
    pod_profile: str | None | object = _NO_PROBE,
) -> SimpleNamespace:
    admin_config = default_admin_config()
    extra: dict = {}
    if pod_profile is not _NO_PROBE:
        # The live probe: what a Running api-ha replica loaded at start-up.
        extra["_cpu_runtime_profile_drift"] = lambda: pod_profile
    return SimpleNamespace(
        **extra,
        _observability_drift=lambda: (
            probes.append("read-only-probe") if probes is not None else None,
            drift,
        )[1],
        _aurora_refresh_drift=lambda: False,
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
        observability_rules_digest="8" * 64,
        observability_adot_digest="9" * 64,
        cluster_registry_digest="6" * 64,
        admin_config_role_digests=admin_config.role_sha256(),
        rendered_manifest_digest="7" * 64,
        node_template_sha="8" * 64,
        runtime_image="runtime@sha256:" + "1" * 64,
        executor_image="runtime@sha256:" + "1" * 64,
        node_installer_image="installer@sha256:" + "2" * 64,
        dcgm_exporter_image="dcgm@sha256:" + "3" * 64,
        adot_image="adot@sha256:" + "4" * 64,
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
            release_delivery_sha256="9" * 64,
            release_delivery_identity={},
            delivery_component_digests={
                "cpu": "a" * 64,
                "cpu_ingress": "5" * 64,
                "cpu_spool": "6" * 64,
                "cpu_worker": "7" * 64,
                "dcgm": "b" * 64,
                "endpoint": "c" * 64,
                "executor": "d" * 64,
                "watcher": "1" * 64,
                "collector": "2" * 64,
                "node": "e" * 64,
                "observability": "f" * 64,
                "schema": "0" * 64,
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
        "admin_config_role_sha256": release.admin_config_role_digests,
        "cluster_ids": ["gpu-a"],
        "release_delivery_sha256": release.config.release_delivery_sha256,
        "cpu_manifest_sha256": release.config.delivery_component_digests["cpu"],
        "cpu_ingress_manifest_sha256": (
            release.config.delivery_component_digests["cpu_ingress"]
        ),
        "cpu_worker_manifest_sha256": (
            release.config.delivery_component_digests["cpu_worker"]
        ),
        "cpu_spool_manifest_sha256": (
            release.config.delivery_component_digests["cpu_spool"]
        ),
        "dcgm_manifest_sha256": release.config.delivery_component_digests["dcgm"],
        "endpoint_manifest_sha256": (
            release.config.delivery_component_digests["endpoint"]
        ),
        "executor_manifest_sha256": (
            release.config.delivery_component_digests["executor"]
        ),
        "watcher_manifest_sha256": (
            release.config.delivery_component_digests["watcher"]
        ),
        "collector_manifest_sha256": (
            release.config.delivery_component_digests["collector"]
        ),
        "node_manifest_sha256": release.config.delivery_component_digests["node"],
        "observability_manifest_sha256": (
            release.config.delivery_component_digests["observability"]
        ),
        "observability_rules_sha256": release.observability_rules_digest,
        "observability_adot_sha256": release.observability_adot_digest,
        "schema_manifest_sha256": (release.config.delivery_component_digests["schema"]),
        "rendered_manifest_sha256": release.rendered_manifest_digest,
        "node_template_sha256": release.node_template_sha,
        "runtime_image": release.runtime_image,
        "node_installer_image": release.node_installer_image,
        "dcgm_image": release.dcgm_exporter_image,
        "adot_image": release.adot_image,
    }


def test_cpu_image_change_does_not_roll_any_gpu_component() -> None:
    release = _release()
    state = _state()
    release.runtime_image = "runtime@sha256:" + "a" * 64
    diff = DIFF.classify_release(release, state)
    assert diff.changed == frozenset({"runtime_image"})
    assert diff.kind is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    plan = DIFF.build_execution_plan(diff)
    assert plan.has(DIFF.ReleaseComponent.CPU_FINALIZE), (
        "CPU image changes must update the CPU runtime"
    )
    assert not plan.has(
        DIFF.ReleaseComponent.EXECUTOR,
        DIFF.ReleaseComponent.RECONCILER,
        DIFF.ReleaseComponent.AGENT,
        DIFF.ReleaseComponent.WATCHER,
        DIFF.ReleaseComponent.COLLECTOR,
    ), "a CPU-only image change selected a GPU rollout"


def test_executor_image_change_does_not_roll_cpu_or_node_runtime() -> None:
    release = _release()
    release.executor_image = "executor@sha256:" + "b" * 64
    diff = DIFF.classify_release(release, _state())
    assert diff.changed == frozenset({"executor_image"})
    plan = DIFF.build_execution_plan(diff)
    assert plan.has(DIFF.ReleaseComponent.EXECUTOR, DIFF.ReleaseComponent.RECONCILER), (
        "Executor image changes must update its consumers"
    )
    assert not plan.has(
        DIFF.ReleaseComponent.CPU_FINALIZE, DIFF.ReleaseComponent.AGENT
    ), "unchanged CPU and host runtimes were selected"


def test_rebinding_same_node_dependency_image_to_new_source_does_not_roll_nodes() -> (
    None
):
    release = _release()
    state = _state()
    dependency = {
        "reference": "registry.example/node@sha256:" + "c" * 64,
        "wheelhouse_sha256": "d" * 64,
        "source_identity_sha256": "e" * 64,
    }
    state["node_dependencies"] = dependency
    release.config.release_delivery_identity = {
        "images": {
            "node_dependencies": {**dependency, "source_identity_sha256": "f" * 64}
        }
    }
    assert DIFF.classify_release(release, state).kind is DIFF.ReleaseChangeKind.NOOP


def test_release_diff_classifies_noop_and_component_scopes() -> None:
    release = _release()
    state = _state()

    assert DIFF.classify_release(release, state).kind is DIFF.ReleaseChangeKind.NOOP
    state["wheel_sha256"] = "9" * 64
    assert (
        DIFF.classify_release(release, state).kind
        is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    )


def test_live_monitoring_drift_selects_repair_even_with_unchanged_artifacts() -> None:
    calls: list[str] = []
    release = _release(drift=True, probes=calls)

    diff = DIFF.classify_release(release, _state())
    plan = DIFF.build_execution_plan(diff)

    assert calls == ["read-only-probe"]
    assert diff.changed == {"observability_drift"}
    assert diff.kind is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    assert plan.has(DIFF.ReleaseComponent.OBSERVABILITY), "drift has no repair node"
    assert not plan.has(DIFF.ReleaseComponent.CPU_STAGE, DIFF.ReleaseComponent.AGENT), (
        "monitoring-only repair unnecessarily restarts CPU or nodes"
    )


def test_legacy_observability_digest_migrates_adot_but_reapplies_rules() -> None:
    release = _release()
    state = _state()
    state.pop("observability_rules_sha256")
    state.pop("observability_adot_sha256")

    diff = DIFF.classify_release(release, state)

    assert "observability_rules" in diff.changed
    assert "observability_adot" not in diff.changed
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


def test_delivery_component_changes_drive_component_scope() -> None:
    release = _release()

    state = _state()
    state["cpu_manifest_sha256"] = "1" * 64
    diff = DIFF.classify_release(release, state)
    assert diff.kind is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    assert "cpu_manifests" in diff.changed

    state = _state()
    state["node_template_sha256"] = "2" * 64
    diff = DIFF.classify_release(release, state)
    assert diff.kind is DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    assert "node_template" in diff.changed

    state = _state()
    state["schema_manifest_sha256"] = "3" * 64
    diff = DIFF.classify_release(release, state)
    assert diff.kind is DIFF.ReleaseChangeKind.FULL
    assert "schema_manifests" in diff.changed

    state = _state()
    state["watcher_manifest_sha256"] = "4" * 64
    diff = DIFF.classify_release(release, state)
    assert diff.kind is DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    assert diff.changed == {"watcher_manifests"}

    state = _state()
    state["collector_manifest_sha256"] = "5" * 64
    diff = DIFF.classify_release(release, state)
    assert diff.kind is DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    assert diff.changed == {"collector_manifests"}


def test_cpu_role_manifest_change_targets_only_that_role() -> None:
    release = _release()
    state = _state()
    state["cpu_worker_manifest_sha256"] = "0" * 64

    diff = DIFF.classify_release(release, state)
    plan = DIFF.build_execution_plan(diff)

    assert diff.kind is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    assert diff.changed == {"cpu_worker_manifests"}
    assert plan.nodes == (
        DIFF.ReleaseComponent.CPU_FINALIZE,
        DIFF.ReleaseComponent.VERIFY,
    )
    assert DIFF.control_plane_role_targets(diff) == ("worker",)


def test_legacy_cpu_manifest_digest_maps_to_all_roles() -> None:
    release = _release()
    raw = {
        name: {"sha256": value}
        for name, value in release.config.delivery_component_digests.items()
        if name not in {"cpu_ingress", "cpu_worker", "cpu_spool"}
    }
    delivery = {
        "schema_version": 1,
        "runtime_prebuilt": True,
        "components": raw,
        "images": {
            name: {"reference": f"registry/{name}@sha256:" + "a" * 64}
            for name in ("runtime", "node_installer", "dcgm_exporter", "adot")
        },
        "node_template_inputs": {"sha256": "b" * 64},
    }
    delivery["sha256"] = CONFIG.canonical_sha256(delivery)
    manifest = {
        "deployable": True,
        "delivery": delivery,
        "components": {"node_bundle": {"template_sha256": "b" * 64}},
        "database": {"rollback_compatible": False},
    }

    _, _, digests, _, _ = CONFIG.parse_delivery_identity(
        manifest, manifest["components"]
    )

    assert digests["cpu_ingress"] == digests["cpu"]
    assert digests["cpu_worker"] == digests["cpu"]
    assert digests["cpu_spool"] == digests["cpu"]


def test_execution_plan_selects_only_changed_component_dependencies() -> None:
    endpoint = DIFF.build_execution_plan(
        DIFF.ReleaseDiff(
            kind=DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
            changed=frozenset({"endpoint", "endpoint_manifests"}),
        )
    )
    assert endpoint.nodes == (
        DIFF.ReleaseComponent.ENDPOINT,
        DIFF.ReleaseComponent.VERIFY,
    )

    node = DIFF.build_execution_plan(
        DIFF.ReleaseDiff(
            kind=DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
            changed=frozenset({"node_bundle", "node_template"}),
        )
    )
    assert node.has(
        DIFF.ReleaseComponent.REGISTRY,
        DIFF.ReleaseComponent.CPU_STAGE,
        DIFF.ReleaseComponent.RECONCILER,
        DIFF.ReleaseComponent.AGENT,
        DIFF.ReleaseComponent.CPU_FINALIZE,
        DIFF.ReleaseComponent.VERIFY,
    ), "node identity changes did not select the required DAG dependencies"
    assert not node.has(
        DIFF.ReleaseComponent.ENDPOINT,
        DIFF.ReleaseComponent.DCGM,
        DIFF.ReleaseComponent.EXECUTOR,
        DIFF.ReleaseComponent.WATCHER,
        DIFF.ReleaseComponent.COLLECTOR,
    ), "node-only change unnecessarily selected unrelated GPU components"

    watcher = DIFF.build_execution_plan(
        DIFF.ReleaseDiff(
            kind=DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
            changed=frozenset({"watcher_manifests"}),
        )
    )
    assert watcher.nodes == (
        DIFF.ReleaseComponent.WATCHER,
        DIFF.ReleaseComponent.VERIFY,
    )


def test_admin_config_change_is_control_plane_only_and_skips_registry() -> None:
    release = _release()
    state = _state()
    state["admin_config_role_sha256"] = {
        **release.admin_config_role_digests,
        "worker": "9" * 64,
    }

    diff = DIFF.classify_release(release, state)
    plan = DIFF.build_execution_plan(diff)

    assert diff.kind is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    assert diff.changed == {"admin_config_worker"}
    assert plan.nodes == (
        DIFF.ReleaseComponent.CPU_FINALIZE,
        DIFF.ReleaseComponent.VERIFY,
    )
    assert DIFF.control_plane_role_targets(diff) == ("worker",)


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

    deploy_host = component_modules("deploy_host")
    assert component_definition("deploy_host").distribution == "gpu-fault-deploy-host"
    assert "gpu_fault.admin.cli" in deploy_host
    assert "gpu_fault.admin.workflow_reconcile" in deploy_host
    assert "gpu_fault.release_state_snapshot" in deploy_host
    # The reconcile engine mutates the Store, so it belongs to the control-plane
    # wheel; the deploy host only execs it there. No control-plane module imports
    # it, which is why this asks what the wheel contains rather than what the
    # import graph reaches.
    assert "gpu_fault.workflow_reconcile" in component_modules("control_plane")
    assert "gpu_fault.workflow_reconcile" not in deploy_host
    assert "gpu_fault.admin.cli" not in closures["control_plane"]
    assert "deploy-host-tools.json" not in {
        path.name for path in component_data_files("control_plane")
    }
    assert "deploy-host-tools.json" in {
        path.name for path in component_data_files("deploy_host")
    }


def test_admin_source_change_only_changes_deploy_host_digest() -> None:
    control_before = component_wheels.component_source_digest("control_plane")
    deploy_before = component_wheels.component_source_digest("deploy_host")
    source = component_wheels.MODULES["gpu_fault.admin.cli"]
    override = {
        source.relative_to(component_wheels.SOURCE).as_posix(): (
            source.read_bytes() + b"\n# deployment-only change\n"
        )
    }

    assert (
        component_wheels.component_source_digest(
            "control_plane", source_overrides=override
        )
        == control_before
    )
    assert (
        component_wheels.component_source_digest(
            "deploy_host", source_overrides=override
        )
        != deploy_before
    )


def test_a_replica_on_another_profile_version_is_a_cpu_only_change() -> None:
    """Live 2026-09-26: ConfigMap hyperpod-v1, Pods hyperpod-v1-boot020 (BOOT-020
    restored the ConfigMap without a roll). The classifier used to see nothing,
    skip the CPU roll and fail its own control_api verification."""

    release = _release(pod_profile="hyperpod-v1-boot020")
    state = _state()
    assert state["runtime_profile_version"] == release.config.runtime_profile_version, (
        "the ConfigMap already agrees; only the replicas drift"
    )

    diff = DIFF.classify_release(release, state)

    assert DIFF.CPU_POD_PROFILE_DRIFT in diff.changed, diff
    assert diff.kind is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY, diff
    plan = DIFF.build_execution_plan(diff)
    assert plan.has(
        DIFF.ReleaseComponent.CPU_STAGE, DIFF.ReleaseComponent.CPU_FINALIZE
    ), plan
    assert not plan.has(
        DIFF.ReleaseComponent.RUNTIME_PROFILE,
        DIFF.ReleaseComponent.AGENT,
        DIFF.ReleaseComponent.EXECUTOR,
    ), "an env-only drift rolls the CPU roles, not the data plane"


@pytest.mark.parametrize("pod_profile", ["hyperpod-v1", None, _NO_PROBE])
def test_an_agreeing_or_unknown_replica_profile_is_not_a_change(pod_profile) -> None:
    release = _release(pod_profile=pod_profile)

    diff = DIFF.classify_release(release, _state())

    assert DIFF.CPU_POD_PROFILE_DRIFT not in diff.changed, diff
    assert diff.kind is DIFF.ReleaseChangeKind.NOOP, (
        "matching, unknown (no Running replica) or unprobed never invents a roll"
    )


def test_a_state_recorded_with_the_pin_folded_membership_digest_is_unchanged(
    monkeypatch,
) -> None:
    """The join's sync-state recorded `cluster_registry_digest` with the profile
    pins folded in; the next deploy's membership digest no longer folds them.
    Same clusters: no `clusters` change. Different clusters: still flagged."""

    release = _release()
    state = _state()
    state["cluster_registry_digest"] = "legacy-" + "6" * 57
    monkeypatch.setattr(
        DIFF, "legacy_registry_config_digest", lambda _clusters: "legacy-" + "6" * 57
    )
    assert "clusters" not in DIFF.classify_release(release, state).changed, (
        "a legacy digest for the same membership reads as unchanged"
    )
    state["cluster_registry_digest"] = "other-" + "6" * 58
    assert "clusters" in DIFF.classify_release(release, state).changed, (
        "a genuinely different membership stays a clusters change"
    )


def _consumer_plan() -> tuple[DIFF.ReleaseComponent, ...]:
    return (
        DIFF.ReleaseComponent.CONNECTION,
        DIFF.ReleaseComponent.EXECUTOR,
        DIFF.ReleaseComponent.WATCHER,
        DIFF.ReleaseComponent.RECONCILER,
        DIFF.ReleaseComponent.VERIFY,
    )


def test_a_recorded_node_scope_change_rolls_the_secret_and_its_consumers() -> None:
    """``node-scope --converge`` rewrote a cluster's selector in site.yaml; the
    release state still carries the digest of the scopes the last release
    rolled, so the deploy that follows is a data-plane release, not a NOOP."""

    release = _release()
    state = _state()
    state[SCOPE.RECORDED_SCOPES_DIGEST_STATE_KEY] = "0" * 64

    diff = DIFF.classify_release(release, state)

    assert diff.changed == {SCOPE.RECORDED_SCOPES_INPUT}, diff
    assert diff.kind is DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    plan = DIFF.build_execution_plan(diff)
    assert plan.nodes == _consumer_plan(), (
        "the Secret is re-rendered and exactly the Deployments reading it roll"
    )
    assert not plan.has(
        DIFF.ReleaseComponent.ENDPOINT,
        DIFF.ReleaseComponent.COLLECTOR,
        DIFF.ReleaseComponent.AGENT,
        DIFF.ReleaseComponent.CPU_STAGE,
        DIFF.ReleaseComponent.CPU_FINALIZE,
        DIFF.ReleaseComponent.REGISTRY,
    ), "no NLB/DNS apply, no collector or node Agent wave, no CPU roll"


def test_a_state_recorded_before_node_scopes_were_an_input_is_not_a_change() -> None:
    release = _release()
    state = _state()
    assert SCOPE.RECORDED_SCOPES_DIGEST_STATE_KEY not in state, (
        "the fixture predates it"
    )
    assert DIFF.classify_release(release, state).kind is DIFF.ReleaseChangeKind.NOOP, (
        "an older state says nothing about what it rolled; the served-scope "
        "probe decides for it"
    )
    state[SCOPE.RECORDED_SCOPES_DIGEST_STATE_KEY] = SCOPE.cluster_node_scopes_digest(
        release.config
    )
    assert (
        SCOPE.RECORDED_SCOPES_INPUT not in DIFF.classify_release(release, state).changed
    )


def test_a_secret_serving_another_scope_is_drift_the_plan_repairs(monkeypatch) -> None:
    """Live: both converged clusters' Secrets still served the join-time
    selector while the recorded digest was unknown to the state."""

    release = _release()
    monkeypatch.setattr(
        DIFF,
        "node_scope_drift",
        lambda _release: {
            "gpu-a": {
                "recorded": {
                    "selector": "gpu-fault.io/control-plane=site",
                    "source": "label-convention",
                },
                "served": {
                    "selector": "eks.amazonaws.com/nodegroup=gpu",
                    "source": "flag-legacy",
                },
            }
        },
    )

    diff = DIFF.classify_release(release, _state())

    assert diff.changed == {SCOPE.SERVED_SCOPE_DRIFT_INPUT}, diff
    assert diff.kind is DIFF.ReleaseChangeKind.DATA_PLANE_COMPATIBLE
    assert DIFF.build_execution_plan(diff).nodes == _consumer_plan()


def test_the_served_scope_is_not_probed_without_a_committed_state(monkeypatch) -> None:
    def probe(_release):
        raise AssertionError("a bootstrap has no Secret to compare against")

    monkeypatch.setattr(DIFF, "node_scope_drift", probe)
    assert (
        SCOPE.SERVED_SCOPE_DRIFT_INPUT
        not in DIFF.classify_release(_release(), {}).changed
    )


def test_an_amp_installer_change_alone_reapplies_the_observability_component(
    tmp_path: Path,
) -> None:
    """The release runs ``deploy/observability/install-amp-monitoring.sh`` to
    apply the AMP rules, the Alertmanager definition and the control-plane
    collector, but only when the observability digest moves. While the
    installer was in no ``component_inputs.observability`` pattern, a fix to it
    alone moved no digest: the plan skipped observability and the site kept
    what the old installer had applied."""

    identity = yaml.safe_load(
        (ROOT / "config/release-identity.yaml").read_text(encoding="utf-8")
    )
    patterns = tuple(identity["component_inputs"]["observability"])
    for pattern in patterns:
        for source in sorted(ROOT.glob(pattern)):
            target = tmp_path / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    installer = tmp_path / "deploy/observability/install-amp-monitoring.sh"
    assert installer.is_file(), (
        f"no observability component input matches the AMP installer: {patterns}"
    )
    before = file_set_identity(tmp_path, patterns)["sha256"]
    installer.write_text(
        installer.read_text(encoding="utf-8") + "\n# an installer fix\n",
        encoding="utf-8",
    )
    after = file_set_identity(tmp_path, patterns)["sha256"]
    assert after != before, "an installer edit left the observability digest alone"

    # The live release recorded the old digest; the new one moves the release
    # delivery identity too, which scopes nothing on its own.
    release = _release()
    release.config.delivery_component_digests["observability"] = after
    state = _state()
    state["observability_manifest_sha256"] = before
    state["release_delivery_sha256"] = "0" * 64
    diff = DIFF.classify_release(release, state)

    assert diff.changed == {"observability_manifests", "release_delivery"}, diff
    assert diff.kind is DIFF.ReleaseChangeKind.CONTROL_PLANE_ONLY
    assert DIFF.build_execution_plan(diff).nodes == (
        DIFF.ReleaseComponent.OBSERVABILITY,
        DIFF.ReleaseComponent.VERIFY,
    )
